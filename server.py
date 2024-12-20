import os
import io
import PyPDF2
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_milvus import Milvus
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Dict, List
from typing_extensions import TypedDict
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# <--------------- MongoDB Connection --------------->
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId

mongo_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongo_uri, server_api=ServerApi('1'))
db = client["doc_bot"]
ud_db = db["user_details"]


URI = "http://localhost:19530/doc_bot"
api_key = os.getenv("GROQ_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGroq(api_key=api_key, model="llama3-70b-8192")
vector_store = None

class State(TypedDict):
    messages: Annotated[List[Dict], add_messages]
    document_id: str

graph_builder = StateGraph(State)
memory_store = {}

prompt_template = PromptTemplate(
    input_variables=["context", "question", "history"],
    template = (
        """
        You are a highly professional and friendly customer support assistant named DIGIBOT. 
        Your role is to provide clear, concise, and company-specific answers only and they should be strictly based on the context provided below. 
        Your responses must follow these rules:
        1.Only answer questions directly related to the provided context. 
        If the question is outside the given context, seems irrelevant, or asks for external information (e.g., jokes, searches, general trivia, etc.), 
        respond only with: 'Sorry, I am unable to answer your question.😓'
        2.Never mention, imply, or refer to the existence of any external documents, files, training sources,give hints about the source of your knowledge, or elaborate beyond the context provided. 
        3.Limit your answers to a maximum of 3 concise lines. 
        5.For questions that:
        - Reference a document or ask, "What is this document about?"
        - Request a joke, trivia, or external search etc. anything
        - Cannot be answered based on the context
        Simply reply: 'Sorry, I am unable to understand you Question.😓'

        **Context**: {context}  
        **Previous Interactions**: {history}  
        **Question**: {question}  
        """
    ),
)

# <--------------- METHODS ---------------> 

def extract_pdf_text(pdf) -> str:
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def generate_file_ids(chunks:list[str], file_name:str):
    file_id = [f"{file_name}_{i}" for i in range(len(chunks))]
    return file_id

def create_vector_store(chunks: list[str], document_id: str, file_id: list[str]):
    if document_id is None:
        raise HTTPException(status_code=400, detail="Document ID is required")
    else:
        collection_name = f"id_{document_id}"
        global vector_store
        vector_store = Milvus.from_texts(
            texts=chunks,
            embedding=embeddings,
            connection_args={"uri": URI},
            collection_name=collection_name,
            ids=file_id,
            drop_old=False,
        )
        return({"message": "Vector store created successfully."})

def chatbot(state: State):
    document_id = state.get('document_id')
    if not document_id:
        return {"messages": "Document ID is missing."}
    
    user_message = state['messages'][-1]
    user_input = user_message.content
    session_key = f"{document_id}"
    
    if session_key not in memory_store:
        memory_store[session_key] = []
    memory_store[session_key].append({"role": "user", "content": user_input})
    
    if user_input.lower() in ["exit", "quit", "finish"]:
        memory_store[session_key] = []
        return {"messages": "Goodbye! 😊"}
    
    vector_store = Milvus(
        collection_name=document_id,
        connection_args={"uri": URI},
        embedding_function=embeddings
    )
    
    docs = vector_store.similarity_search(user_input)

    if docs:
        context = "\n\n".join([doc.page_content for doc in docs])
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in memory_store[session_key]])

        try:
            response = llm.invoke(prompt_template.format(
                context=context, question=user_input, history=history
            ))
            assistant_response = response.content

            memory_store[session_key].append({"role": "assistant", "content": assistant_response})
            return {"messages": assistant_response}
        except Exception as e:
            return {"messages": "There was an error processing your request."}
    else:
        return {"messages": "No relevant documents found for your question."}

graph_builder.add_node("Chatbot", chatbot)
graph_builder.add_edge(START, "Chatbot")
graph_builder.add_edge("Chatbot", END)
graph = graph_builder.compile()

# <---------- API ENDPOINTS ---------->

@app.post("/chat")
async def chat_with_bot(document_id: str, user_input: str): 
    state = {
        'messages': [HumanMessage(content=user_input)],
        'document_id': document_id
    }
    try:
        for event in graph.stream(state):
            for value in event.values():
                return {"response": value["messages"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), document_id: str = Form(...)):
    try:
        file_name = file.filename
        user_data = ud_db.find_one({"_id": ObjectId(document_id)})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        if file.filename in user_data["user_files"]:
            pass
        else:
            updated_data = ud_db.find_one_and_update(
                {"_id": ObjectId(document_id)},
                {"$addToSet": {"user_files": file_name}},
            )
            if updated_data:
                content = await file.read()
                pdf_text = extract_pdf_text(io.BytesIO(content))
                text_chunks = get_text_chunks(pdf_text)
                ud_db.find_one_and_update(
                    {"_id": ObjectId(document_id)},
                    {"$push": {"file_ids": len(text_chunks)}}
                )
                file_ids = generate_file_ids(text_chunks, file_name)
                create_vector_store(text_chunks, document_id,file_ids)
            
        return {"message": "Operation completed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
@app.delete("/delete")
async def delete_file(document_id: str, file_name: str):
    try:
        result = ud_db.find_one({"_id": ObjectId(document_id)})
        file_index = result["user_files"].index(file_name)
        file_id = result["file_ids"][file_index]
        unique_id = [f"{file_name}_{i}" for i in range(file_id)]
        vector_store = Milvus(
            collection_name=f"id_{document_id}",
            connection_args={"uri": URI},
            embedding_function=embeddings
        )
        vector_store.delete(unique_id)
        
        # Remove File name
        ud_db.find_one_and_update(
            {"_id": ObjectId(document_id)},
            {"$pull": {"user_files": file_name}},
        )
        # Set file_id to null
        ud_db.find_one_and_update(
            {"_id": ObjectId(document_id)},
            {"$unset": {f"file_ids.{file_index}": 1}},
        )
        # remove null value
        ud_db.find_one_and_update(
            {"_id": ObjectId(document_id)},
            {"$pull": {f"file_ids":None }},
        )
        return {"message": "File deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

# <--------------- MAIN --------------->
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="192.168.29.21", port=8000, reload=True) 