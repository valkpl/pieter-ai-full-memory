# Pieter AI Memory API
# Version: April 30, 2025
# Compatible with:
# - llama-index==0.10.28
# - llama-index-vector-stores-pinecone==0.1.2
# - llama-index-embeddings-openai==0.1.3
# - pinecone-client==3.2.2

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.service_context import ServiceContext
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Set up FastAPI app
app = FastAPI()

# Enable CORS (required for OpenAI plugin access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve plugin manifest + OpenAPI schema
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="plugin-manifest")
app.mount("/", StaticFiles(directory=".", html=True), name="root-files")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index("pieter-ai-full-memory")
except Exception as e:
    raise RuntimeError(f"❌ Pinecone initialization failed: {str(e)}")

# Create vector store and context
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4"),
    embed_model=OpenAIEmbedding()
)

# Load vector index
try:
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        service_context=service_context
    )
except Exception as e:
    raise RuntimeError(f"❌ Index loading failed: {str(e)}")

# Categorize prompt intent
def classify_intent(prompt):
    prompt = prompt.lower()
    if any(w in prompt for w in ["instagram", "caption", "social"]):
        return "social"
    elif any(w in prompt for w in ["article", "piece"]):
        return "article"
    elif any(w in prompt for w in ["pitch", "pitching"]):
        return "pitch"
    elif any(w in prompt for w in ["sermon", "talk", "message", "teaching", "seminar"]):
        return "sermon"
    return "general"

# Run query
def chat_with_pieter_ai(question: str) -> str:
    if not index:
        return "⚠️ The vector index is not initialized."

    intent = classify_intent(question)
    filter_map = {
        "social": ["social_media", "blogs", "book"],
        "article": ["blogs", "book"],
        "pitch": ["blogs", "book", "social_media"],
        "sermon": ["blogs", "book", "transcripts"]
    }

    filters = {"source": {"$in": filter_map.get(intent, [])}} if intent in filter_map else {}

    try:
        query_engine = index.as_query_engine(similarity_top_k=5, filters=filters)
        response = query_engine.query(question)
        if not response or not str(response).strip():
            return "⚠️ No answer found. Try rephrasing your question or asking something else."
        return str(response)
    except Exception as e:
        return f"❌ An error occurred while processing your query: {str(e)}"

# API route
@app.post("/predict/")
async def predict(body: dict = Body(...)):
    try:
        question = body.get("data", [""])[0]
        if not question:
            return JSONResponse(status_code=400, content={"result": "⚠️ Please include a question."})
        result = chat_with_pieter_ai(question)
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"result": f"❌ Internal server error: {str(e)}"})
