# Pieter AI Memory API - FULLY UPDATED SCRIPT

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
from llama_index.schema import MetadataFilter, MetadataFilters
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Set up FastAPI app
app = FastAPI()

# Enable CORS for OpenAI plugin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static routing
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="well-known")
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("pieter-ai-full-memory")

# Vector store + service context
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4"),
    embed_model=OpenAIEmbedding()
)

# Load vector index
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    service_context=service_context
)

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

# Main query logic with increased similarity_top_k
def chat_with_pieter_ai(question: str, debug: bool = False, mode: str = "full") -> str:
    if not index:
        return "⚠️ Index not initialized."

    intent = classify_intent(question)
    filter_map = {
        "social": ["social_media", "blogs", "book"],
        "article": ["blogs", "book"],
        "pitch": ["blogs", "book", "social_media"],
        "sermon": ["blogs", "book", "transcripts"]
    }

    sources = filter_map.get(intent)
    metadata_filters = (
        MetadataFilters(filters=[MetadataFilter(key="source", operator="in", value=sources)])
        if sources else None
    )

    try:
        query_engine = index.as_query_engine(similarity_top_k=20, filters=metadata_filters,
                                             response_mode="no_text" if mode == "full" else "tree_summarize")
        response = query_engine.query(question)
        if not response or not str(response).strip():
            return "⚠️ No answer found. Try rephrasing your question."

        if debug:
            debug_output = "\n\n[sources used with metadata]\n"
            for node in response.source_nodes:
                source_file = node.node.metadata.get("file_name", "UNKNOWN FILE")
                score = node.score
                debug_output += f"• {source_file} (score={score:.2f})\n"
            return f"{str(response)}\n{debug_output}"
        else:
            return str(response)
    except Exception as e:
        return f"❌ Query error: {str(e)}"

# API endpoint for plugin access
@app.post("/predict/")
async def predict(body: dict = Body(...)):
    try:
        question = body.get("data", [""])[0]
        debug = body.get("debug", False)
        mode = body.get("mode", "summary")
        if not question.strip():
            return JSONResponse(status_code=400, content={"result": "⚠️ Please include a valid question."})
        result = chat_with_pieter_ai(question, debug=debug, mode=mode)
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"result": f"❌ Internal server error: {str(e)}"})
