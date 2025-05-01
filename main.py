# Pieter AI Memory API - Clean for llama-index==0.10.28

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.indices.service_context import ServiceContext
from llama_index.schema import MetadataFilter, MetadataFilters  # ✅ Correct for 0.10.28
from pinecone import Pinecone

# Print llama-index version to confirm runtime match
import llama_index
print("📦 llama-index version:", llama_index.__version__)

# Load env vars
load_dotenv()

# FastAPI app
app = FastAPI()

# CORS config for plugin use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static serving for plugin manifest and OpenAPI
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="well-known")
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Pinecone setup
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index("pieter-ai-full-memory")
except Exception as e:
    raise RuntimeError(f"❌ Pinecone init failed: {str(e)}")

# Vector store setup
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
    raise RuntimeError(f"❌ Index load failed: {str(e)}")

# Intent classification
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

# Core chat logic
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

    if intent in filter_map:
        try:
            filters = MetadataFilters(
                filters=[MetadataFilter(key="source", operator="in", value=filter_map[intent])]
            )
        except Exception as e:
            return f"❌ Metadata filter error: {str(e)}"

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
        return JSONResponse(status_code=500, content={"result": f"❌ Server error: {str(e)}"})
