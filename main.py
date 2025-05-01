
# Pieter AI Memory API - Updated with Debug + Full Source Metadata (filenames included)

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

# FastAPI app setup
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file routes
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="well-known")
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Pinecone initialization
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index("pieter-ai-full-memory")
except Exception as e:
    raise RuntimeError(f"Pinecone init failed: {str(e)}")

# LlamaIndex service and vector store
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4"),
    embed_model=OpenAIEmbedding()
)

try:
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)
except Exception as e:
    raise RuntimeError(f"Index load failed: {str(e)}")

# Query handler
def chat_with_pieter_ai(question: str, debug: bool = False, mode: str = "full") -> str:
    if not index:
        return "⚠️ Index not initialized."

    try:
        query_engine = index.as_query_engine(similarity_top_k=5, response_mode="no_text" if mode == "full" else "tree_summarize")
        response = query_engine.query(question)

        if not response or not str(response).strip():
            return "⚠️ No answer found. Try rephrasing your question."

        if debug:
            debug_output = "\n\n[sources used with metadata]\n"
            for node in response.source_nodes:
                node_text = node.node.get_text()
                source_file = node.node.metadata.get("file_name", "UNKNOWN FILE")
                score = node.score
                debug_output += f"• {source_file} (score={score:.2f})\n"
            return f"{str(response)}\n{debug_output}"
        else:
            return str(response)
    except Exception as e:
        return f"❌ Query error: {str(e)}"

# API endpoint
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
        return JSONResponse(status_code=500, content={"result": f"❌ Internal error: {str(e)}"})
