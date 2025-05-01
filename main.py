# Pieter AI Memory API - Enhanced for CustomGPT Integration (llama-index==0.10.28)

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
# print("📦 llama-index version:", llama_index.__version__)

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

# Intent classifier (optional utility)
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

# Helper to estimate tokens
def estimate_tokens(text):
    return int(len(text) / 4)  # crude estimate: 4 chars ≈ 1 token

# Query handler
def chat_with_pieter_ai(question: str, mode: str = "summary", debug: bool = False, max_tokens: int = 12000) -> dict:
    if not index:
        return {"result": "⚠️ Index not initialized."}

    try:
        if mode == "full":
            query_engine = index.as_query_engine(similarity_top_k=10, response_mode="no_text")
            response = query_engine.query(question)

            stitched_sources = []
            total_tokens = 0
            for node in response.source_nodes:
                node_text = node.get_text()
                tokens = estimate_tokens(node_text)
                if total_tokens + tokens > max_tokens:
                    break
                stitched_sources.append(node_text)
                total_tokens += tokens

            result = "\n\n---\n\n".join(stitched_sources) or "⚠️ No sources returned."
        else:
            query_engine = index.as_query_engine(similarity_top_k=5, response_mode="tree_summarize")
            response = query_engine.query(question)
            result = str(response)

        output = {"result": result}

        if debug:
            output["debug"] = {
                "mode": mode,
                "source_count": len(response.source_nodes) if hasattr(response, "source_nodes") else 0,
                "token_estimate": estimate_tokens(result)
            }

        return output

    except Exception as e:
        return {"result": f"❌ Query error: {str(e)}"}

# POST endpoint
@app.post("/predict/")
async def predict(body: dict = Body(...)):
    try:
        question = body.get("data")
        mode = body.get("mode", "summary")
        debug = body.get("debug", False)
        max_tokens = int(body.get("max_tokens", 12000))

        if not question or not isinstance(question, list) or not question[0].strip():
            return JSONResponse(status_code=400, content={"result": "⚠️ Please include a valid question."})

        response_data = chat_with_pieter_ai(question[0], mode=mode, debug=debug, max_tokens=max_tokens)
        return JSONResponse(content=response_data)

    except Exception as e:
        return JSONResponse(status_code=500, content={"result": f"❌ Internal error: {str(e)}"})
