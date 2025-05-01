# Pieter AI Memory API — Stable for llama-index==0.10.28

import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from llama_index.core import VectorStoreIndex, StorageContext  # ✅ CORRECT for 0.10.28
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.indices.service_context import ServiceContext
from llama_index.schema import MetadataFilter, MetadataFilters  # ✅ CORRECT for 0.10.28
from pinecone import Pinecone
import llama_index

# ✅ Log actual llama-index version to Render logs
logging.basicConfig(level=logging.INFO)
logging.info(f"📦 llama-index version: {llama_index.__version__}")

# Load environment variables
load_dotenv()

# FastAPI app setup
app = FastAPI()

# CORS for plugin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for plugin manifest
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="well-known")
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Initialize Pinecone (SDK v3)
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index("pieter-ai-full-memory")
except Exception as e:
    raise RuntimeError(f"❌ Pinecone initialization failed: {str(e)}")

# Setup vector store and context
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4"),
    embed_model=OpenAIEmbedding()
)

# Load index
try:
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        service_context=service_context
    )
except Exception as e:
    raise RuntimeError(f"❌ Index loading failed: {str(e)}")

# Intent classifier
def classify_intent(prompt):
    prompt = prompt.lower()
    if any(w in prompt for w in ["instagram", "caption", "social"]):
        return "social"
    if any(w in prompt for w in ["article", "piece"]):
        return "article"
    if any(w in prompt for w in ["pitch", "pitching"]):
        return "pitch"
    if any(w in prompt for w in ["sermon", "talk", "message", "teaching", "seminar"]):
        return "sermon"
    return "general"

# Chat engine
def chat_with_pieter_ai(question: str) -> str:
    if not index:
        return "⚠️ The vector index is not initialized."

    filter_map = {
        "social": ["social_media", "blogs", "book"],
        "article": ["blogs", "book"],
        "pitch": ["blogs", "book", "social_media"],
        "sermon": ["blogs", "book", "transcripts"]
    }

    intent = classify_intent(question)
    filters = None
    sources = filter_map.get(intent)

    if sources:
        try:
            filters = MetadataFilters(
                filters=[MetadataFilter(key="source", operator="in", value=sources)]
            )
        except Exception as e:
            return f"❌ Filter construction failed: {str(e)}"

    try:
        query_engine = index.as_query_engine(similarity_top_k=5, filters=filters)
        response = query_engine.query(question)
        return str(response) if str(response).strip() else "⚠️ No answer found."
    except Exception as e:
        return f"❌ Query engine error: {str(e)}"

# POST endpoint
@app.post("/predict/")
async def predict(body: dict = Body(...)):
    try:
        question = body.get("data", [""])[0]
        if not question.strip():
            return JSONResponse(status_code=400, content={"result": "⚠️ Please include a question."})
        result = chat_with_pieter_ai(question)
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"result": f"❌ Internal server error: {str(e)}"})
