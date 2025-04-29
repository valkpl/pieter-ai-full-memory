import os
import gradio as gr
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pinecone import Pinecone  # updated import

# --- Step 1: Set your API keys here or use environment variables ---
os.environ["OPENAI_API_KEY"] = "***REMOVED***"

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
    api_key="pcsk_2RtfVK_9Kw55W5k4xejkdHEwpy5ZSFiruSur6WjgbVUvqHUBCLj5nmGQgqSvWqCYrcBRPt",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# --- Step 2: Configure LlamaIndex context and Pinecone store ---
index_name = "pieter-ai-full-memory"
pinecone_index = pc.Index(index_name)

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo"),
    embed_model=OpenAIEmbedding()
)

# --- Step 3: Load the index ---
index = load_index_from_storage(storage_context=storage_context, service_context=service_context)

# --- Step 4: Set up query engine ---
query_engine = index.as_query_engine(similarity_top_k=5, service_context=service_context)

# --- Step 5: Define chat interface ---
def chat_with_pieter_ai(question):
    try:
        response = query_engine.query(question)
        return str(response)
    except Exception as e:
        return f"Error: {str(e)}"

# --- Step 6: Launch Gradio UI ---
iface = gr.Interface(
    fn=chat_with_pieter_ai,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about celibacy, vocation, community..."),
    outputs="text",
    title="Pieter AI Assistant",
    description="A chatbot trained on the life, theology, and work of Pieter Valk. Ask away!"
)

if __name__ == "__main__":
    iface.launch()
