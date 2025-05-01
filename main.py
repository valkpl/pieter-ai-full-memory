# Pieter AI Memory API - Full Retrieval w/ Mode Toggle, Debug Output, Token Limit

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

# Intent classifier (currently not used for filtering)
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
def chat_with_pieter_ai(question: str, mode: str = "full", token_limit: int = None, debug: bool = False) -> str:
    if not index:
        return "⚠️ Index not initialized."

    try:
        response_mode = "tree_summarize" if mode == "summary" else "no_text"
        query_engine = index.as_query_engine(similarity_top_k=5, response_mode=response_mode)

        response = query_engine.query(question)
        if not response or not str(response).strip():
            return "⚠️ No answer found. Try rephrasing your question."

        result = str(response)

        if mode == "full":
            nodes = response.source_nodes
            if token_limit:
                token_count = 0
                sources = []
                for node in nodes:
                    text = node.get_text()
                    token_count += len(text.split())
                    if token_count <= token_limit:
                        sources.append(f"SOURCE:\n{text}")
                    else:
                        break
            else:
                sources = [f"SOURCE:\n{node.get_text()}" for node in nodes]

            result += "\n\n[sources used]\n" + "\n\n---\n\n".join(sources)

        if debug:
            result = "[debug] Response retrieved\n\n" + result

        return result

    except Exception as e:
        return f"❌ Query error: {str(e)}"

# POST endpoint
@app.post("/predict/")
async def predict(body: dict = Body(...)):
    try:
        question = body.get("data", [""])[0]
        mode = body.get("mode", "full")
        debug = body.get("debug", False)
        token_limit = body.get("token_limit", None)

        if not question.strip():
            return JSONResponse(status_code=400, content={"result": "⚠️ Please include a valid question."})

        result = chat_with_pieter_ai(question, mode=mode, token_limit=token_limit, debug=debug)
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"result": f"❌ Internal error: {str(e)}"})
