# Pieter AI Memory API - Optimized for Full Document Use in CustomGPT

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

# Optional: log llama-index version
import llama_index

# Load env variables
load_dotenv()

# FastAPI app setup
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static/manifest routes
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="well-known")
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index("pieter-ai-full-memory")
except Exception as e:
    raise RuntimeError(f"❌ Pinecone init failed: {str(e)}")

# Set up vector store and index
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4"),
    embed_model=OpenAIEmbedding()
)

try:
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        service_context=service_context
    )
except Exception as e:
    raise RuntimeError(f"❌ Index load failed: {str(e)}")

# Intent classifier
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

# Query handler
def chat_with_pieter_ai(question: str, mode: str = "full", debug: bool = False, token_limit: int = 4000) -> str:
    if not index:
        return "⚠️ Index not initialized."

    try:
        query_engine = index.as_query_engine(similarity_top_k=8, response_mode="no_text")

        response = query_engine.query(question)
        raw_nodes = response.source_nodes

        stitched_text = "\n\n---\n\n".join([node.get_text() for node in raw_nodes])
        stitched_text = stitched_text[:token_limit]  # Optional safeguard

        if mode == "summary":
            return str(response)

        output = stitched_text
        if debug:
            debug_info = f"[debug] Retrieved {len(raw_nodes)} nodes, {len(stitched_text.split())} words"
            output = f"{debug_info}\n\n{stitched_text}"

        return output

    except Exception as e:
        return f"❌ Query error: {str(e)}"

# POST endpoint
@app.post("/predict/")
async def predict(body: dict = Body(...)):
    try:
        question = body.get("data")
        if not question or not isinstance(question, list) or not question[0].strip():
            return JSONResponse(status_code=400, content={"result": "⚠️ Please include a valid question."})

        # Extract optional mode/debug/token_limit from request
        mode = body.get("mode", "full")
        debug = body.get("debug", False)
        token_limit = int(body.get("token_limit", 4000))

        result = chat_with_pieter_ai(question[0], mode=mode, debug=debug, token_limit=token_limit)
        return JSONResponse(content={"result": result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"result": f"❌ Internal error: {str(e)}"})
