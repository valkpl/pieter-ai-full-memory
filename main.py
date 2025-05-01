# main.py – Pieter AI Memory with debug + full source integration

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

# Load env
load_dotenv()
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/.well-known", StaticFiles(directory=".well-known"), name="well-known")

# Init
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("pieter-ai-full-memory")

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4"),
    embed_model=OpenAIEmbedding()
)
index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

@app.post("/predict/")
async def predict(body: dict = Body(...)):
    question = body.get("data", [""])[0]
    mode = body.get("mode", "full")  # default is full
    debug = body.get("debug", False)

    if not question.strip():
        return JSONResponse(status_code=400, content={"result": "⚠️ Please include a valid question."})

    try:
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="no_text" if mode == "full" else "tree_summarize"
        )
        response = query_engine.query(question)

        result = str(response)
        if debug and hasattr(response, "source_nodes"):
            sources = "\n\n[debug sources used]\n" + "\n\n---\n\n".join([
                f"SOURCE {i+1} (score={node.score:.2f}):\n{node.get_text()[:1000]}"
                for i, node in enumerate(response.source_nodes)
            ])
            result += f"\n\n{sources}"

        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"result": f"❌ Query error: {str(e)}"})
