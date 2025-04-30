from dotenv import load_dotenv
import os
import gradio as gr

from llama_index.core import VectorStoreIndex, StorageContext

from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from pinecone import Pinecone, ServerlessSpec

# --- Step 1: Load environment variables ---
load_dotenv()

# --- Step 2: Set up Pinecone ---
index_name = "pieter-ai-full-memory"
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- Step 3: Set global LLM + Embeddings ---
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()

# --- Step 4: Load index from Pinecone ---
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# --- Step 5: Intent classification ---
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

# --- Step 6: Chat function ---
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

# --- Step 7: Gradio UI ---
api = gr.Interface(
    fn=chat_with_pieter_ai,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about celibacy, vocation, community..."),
    outputs="text",
    title="Pieter AI Assistant",
    description="A chatbot trained on the life, theology, and work of Pieter Valk. Ask away!"
)

if __name__ == "__main__":
    api.launch(server_name="0.0.0.0", server_port=7860)
