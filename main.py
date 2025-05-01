# Pieter AI Memory API - Full Source Output for CustomGPT Integration (llama-index==0.10.28)

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

# Query handler
def chat_with_pieter_ai(question: str) -> str:
    if not index:
        return "⚠️ Index not initialized."

    try:
        query_engine = index.as_query_engine(similarity_top_k=5, response_mode="no_text")
        response = query_engine.query(question)

        if not response or not response.source_nodes:
            return "⚠️ No matching sources found. Try rephrasing your question."

        # Return full unfiltered content
        sources = "\n\n---\n\n".join([f"SOURCE {i+1}:\n{node.get_text()}" for i, node in enumerate(response.source_nodes)])
        return f"[retrieved full source chunks]\n\n{sources}"
    except Exception as e:
        return f"❌ Query error: {str(e)}"

# POST endpoint
@app.post("/predict/")
async def predict(body: dict = Body(...)):
    try:
        question = body.get("data")
        if not question or not isinstance(question, list) or not question[0].strip():
            return JSONResponse(status_code=400, content={"result": "⚠️ Please include a valid question."})
        result = chat_with_pieter_ai(question[0])
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"result": f"❌ Internal error: {str(e)}"})
