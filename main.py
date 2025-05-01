# Pieter AI Memory API - Full Document Mode + Debug Output (llama-index==0.10.28)

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

import llama_index

# Load env variables
load_dotenv()

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/.well-known", StaticFiles(directory=".well-known"), name="well-known")
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Init Pinecone
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index("pieter-ai-full-memory")
except Exception as e:
    raise RuntimeError(f"❌ Pinecone init failed: {str(e)}")

# Setup vector store and service context
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

# Query engine
def chat_with_pieter_ai(question: str, mode="summary", debug=False, max_tokens=None):
    if not index:
        return {"text": "⚠️ Index not initialized."}

    try:
        response_mode = "tree_summarize" if mode == "summary" else "no_text"
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode=response_mode
        )

        response = query_engine.query(question)

        if not response or not str(response).strip():
            return {"text": "⚠️ No answer found. Try rephrasing your question."}

        debug_log = ""
        sources_text = ""

        if hasattr(response, "source_nodes") and response.source_nodes:
            if mode == "full":
                chunks = [node.get_text() for node in response.source_nodes]
                if max_tokens:
                    total = 0
                    trimmed_chunks = []
                    for chunk in chunks:
                        chunk_tokens = len(chunk.split())
                        if total + chunk_tokens > max_tokens:
                            break
                        trimmed_chunks.append(chunk)
                        total += chunk_tokens
                    chunks = trimmed_chunks
                sources_text = "\n\n---\n\n".join([f"SOURCE:\n{c}" for c in chunks])
            else:
                sources_text = ""

        if debug:
            debug_log += f"\n\n[debug] top_k={len(response.source_nodes)}"
            debug_log += f"\n[debug] response_mode={response_mode}"
            debug_log += f"\n[debug] token_limit={max_tokens or '∞'}"
            debug_log += "\n[debug] source preview:\n"
            for i, node in enumerate(response.source_nodes[:3]):
                preview = node.get_text().strip().split("\n")[0][:100]
                debug_log += f"  {i+1}. {preview}...\n"

        return {
            "text": str(response),
            "sources_text": f"\n\n[sources used]\n{sources_text}" if sources_text else "",
            "debug_log": debug_log if debug else ""
        }

    except Exception as e:
        return {"text": f"❌ Query error: {str(e)}"}

# POST endpoint
@app.post("/predict/")
async def predict(body: dict = Body(...)):
    try:
        question = body.get("data", [""])[0]
        if not question.strip():
            return JSONResponse(status_code=400, content={"result": "⚠️ Please include a valid question."})

        mode = body.get("mode", "summary")
        debug = body.get("debug", False)
        max_tokens = body.get("max_tokens", None)

        result = chat_with_pieter_ai(question, mode=mode, debug=debug, max_tokens=max_tokens)

        # Stitch full return into `result` for CustomGPT display
        combined = result["text"] + result.get("sources_text", "") + result.get("debug_log", "")
        return JSONResponse(content={"result": combined})

    except Exception as e:
        return JSONResponse(status_code=500, content={"result": f"❌ Internal error: {str(e)}"})
