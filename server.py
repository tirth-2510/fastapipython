import os
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from langchain_milvus import Zilliz
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

memory_store = []

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
    response = chatbot(document_id, user_input)
    return {"response": response}
