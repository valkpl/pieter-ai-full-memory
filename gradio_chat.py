from dotenv import load_dotenv
import os
import gradio as gr

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from pinecone import Pinecone, ServerlessSpec

from fastapi import FastAPI
import uvicorn

# --- Load env vars ---
load_dotenv()

# --- Set up Pinecone ---
index_name = "pieter-ai-full-memory"
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- Set global models ---
Settings.llm = OpenAI(model="gpt-4")
Settings.embed_model = OpenAIEmbedding()

# --- Load vector index ---
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# --- Intent classification ---
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

# --- Chat logic ---
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

# --- Create Gradio interface ---
gradio_app = gr.Interface(
    fn=chat_with_pieter_ai,
    inputs=gr.Textbox(label="Your Question"),
    outputs=gr.Textbox(label="Pieter AI’s Response"),
    allow_flagging="never"
).queue()

# --- Mount with FastAPI ---
app = FastAPI()
app = gr.mount_gradio_app(app, gradio_app, path="/predict")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
