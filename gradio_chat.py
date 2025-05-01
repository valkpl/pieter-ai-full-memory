from dotenv import load_dotenv
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from pinecone import Pinecone, ServerlessSpec

# --- Load environment variables ---
load_dotenv()

# --- Pinecone setup ---
index_name = "pieter-ai-full-memory"
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- Global LLM + Embeddings ---
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()

# --- Load index ---
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# --- Intent classifier ---
def classify_intent(prompt):
    prompt = prompt.lower()
    if any(word in prompt for word in ["instagram", "social post", "caption", "twitter", "x ", "facebook", "threads", "tiktok", "socials"]):
        return "social"
    elif any(word in prompt for word in ["article", "piece"]):
        return "article"
    elif any(word in prompt for word in ["pitch", "pitching"]):
        return "pitch"
    elif any(word in prompt for word in ["sermon", "talk", "message", "teaching", "seminar"]):
        return "sermon"
    return "general"

# --- Query handler ---
def chat_with_pieter_ai(question):
    if not index:
        return "⚠️ The vector index is not initialized."

    intent = classify_intent(question)
    filters = {
        "social": {"source": {"$in": ["social_media", "blogs", "book"]}},
        "article": {"source": {"$in": ["blogs", "book"]}},
        "pitch": {"source": {"$in": ["blogs", "book", "social_media"]}},
        "sermon": {"source": {"$in": ["blogs", "book", "transcripts"]}},
    }.get(intent, {})

    try:
        query_engine = index.as_query_engine(similarity_top_k=5, filters=filters)
        response = query_engine.query(question)
        return str(response)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# --- FastAPI App ---
app = FastAPI()

# Optional: Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gradio App ---
gradio_app = gr.Interface(
    fn=chat_with_pieter_ai,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about celibacy, vocation, community..."),
    outputs="text",
    title="Pieter AI Assistant",
    description="A chatbot trained on the life, theology, and work of Pieter Valk. Ask away!"
)

# Mount Gradio at root
app = gr.mount_gradio_app(app, gradio_app, path="/")

# --- Custom API endpoint ---
@app.post("/predict/")
async def predict(request: Request):
    body = await request.json()
    question = body.get("data", [""])[0]
    answer = chat_with_pieter_ai(question)
    return {"response": answer}
