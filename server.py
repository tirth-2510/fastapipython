import os
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pymilvus import MilvusClient
from groq import Groq
from langchain_milvus import Zilliz
import PyPDF2
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from bson.objectid import ObjectId
import io
from dotenv import load_dotenv

load_dotenv()

app= FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mongo_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongo_uri)
db = client["doc_bot"]
ud_db = db["user_details"]

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

vector_store = None
memory_store = []

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
        vector_store = Zilliz.from_texts(
            texts=chunks,
            embedding=embeddings,
            connection_args={"uri": os.getenv("ZILLIZ_URI_ENDPOINT"), "token":os.getenv("ZILLIZ_TOKEN")},
            collection_name=collection_name,
            ids=file_id,
            drop_old=False,
        )
        return({"message": "Vector store created successfully."})

def chatbot(document_id: str, user_input: str):
    message = [{"role":"user", "content":"""
    You are a highly professional customer support assistant named DIGIBOT. Respond as if you are the company representative, helping with questions and apologize if user directly asks about your knowledge base or if you are unable to assist, with the sentence: 'I am not able to understand your query , could you try to rephrase your queries?' """},
    {"role":"assistant", "content":"Ok I will answer from the context only without mentioning about the context or the document, I will simply respond with 'I am not able to understand your query , could you try to rephrase your query?'. I will answer on behalf of the company and won't answer for anything else."},{"role":"user","content":user_input}]

    vector_store = Zilliz(
        collection_name=f"id_{document_id}",
        connection_args={"uri": os.getenv("ZILLIZ_URI_ENDPOINT"), "token": os.getenv("ZILLIZ_TOKEN")},
        embedding_function=embeddings
    )
    docs = vector_store.similarity_search(query=user_input, k=1)
    context = docs[0].page_content
    if docs is not None:
        message.append({"role": "system", "content": context})
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=message,
            n=1
        )
        return(completion.choices[0].message.content)

@app.get("/")
async def root():
    return {"message": "Connection Successfull"}

@app.post("/chat")
async def chat(data: dict = Body(...)):
    user_input = data.get("user_input")
    document_id = data.get("document_id")
    if user_input.lower() in ["exit", "quit", "finish"]:
        return {"Goodbye!"}
    return{"response": chatbot(document_id, user_input)}

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
async def delete_file(data: dict = Body(...)):
    document_id = data.get("document_id")
    file_name = data.get("file_name")
    try:
        result = ud_db.find_one({"_id": ObjectId(document_id)})
        file_index = result["user_files"].index(file_name)
        file_id = result["file_ids"][file_index]
        unique_id = [f"{file_name}_{i}" for i in range(file_id)]
        vector_store = Zilliz(
            collection_name=f"id_{document_id}",
            connection_args={"uri": os.getenv("ZILLIZ_URI_ENDPOINT"), "token":os.getenv("ZILLIZ_TOKEN")},
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
        client = MilvusClient(uri: os.getenv("ZILLIZ_URI_ENDPOINT"), token: os.getenv("ZILLIZ_TOKEN"))
        collection = client.get_collection_stats(f"id_{document_id}")
        if collection["row_count"] == 0:
            client.drop_collection(f"id_{document_id}")
        
        return {"message": "File deleted successfully."} 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
