from dotenv import load_dotenv
import os
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.service_context import ServiceContext
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize Pinecone (correct for SDK v3)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("pieter-ai-full-memory")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Set global LLM and embedding model
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4"),
    embed_model=OpenAIEmbedding()
)

# Load index
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    service_context=service_context
)

# Intent classifier
def classify_intent(prompt):
    prompt = prompt.lower()
    if any(word in prompt for word in ["instagram", "caption", "social"]):
        return "social"
    elif any(word in prompt for word in ["article", "piece"]):
        return "article"
    elif any(word in prompt for word in ["pitch", "pitching"]):
        return "pitch"
    elif any(word in prompt for word in ["sermon", "talk", "message", "teaching", "seminar"]):
        return "sermon"
    return "general"

# Query logic
def chat_with_pieter_ai(question: str) -> str:
    if not index:
        return "⚠️ The vector index is not initialized."

    intent = classify_intent(question)
    filter_map = {
        "social": {"source": {"$in": ["social_media", "blogs", "book"]}},
        "article": {"source": {"$in": ["blogs", "book"]}},
        "pitch": {"source": {"$in": ["blogs", "book", "social_media"]}},
        "sermon": {"source": {"$in": ["blogs", "book", "transcripts"]}},
    }

    filters = filter_map.get(intent)

    try:
        query_engine = index.as_query_engine(similarity_top_k=5, filters=filters)
        response = query_engine.query(question)
        return str(response)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# FastAPI app
app = FastAPI()

@app.post("/predict/")
async def predict(body: dict = Body(...)):
    question = body.get("data", [""])[0]
    result = chat_with_pieter_ai(question)
    return JSONResponse(content={"result": result})
