import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.settings import Settings
from pinecone import Pinecone, ServerlessSpec

# --- Step 1: Set your API keys ---
os.environ["OPENAI_API_KEY"] = "***REMOVED***"
os.environ["PINECONE_API_KEY"] = "pcsk_2RtfVK_9Kw55W5k4xejkdHEwpy5ZSFiruSur6WjgbVUvqHUBCLj5nmGQgqSvWqCYrcBRPt"

# --- Step 2: Initialize Pinecone ---
index_name = "pieter-ai-full-memory"
pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"],
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# --- Step 3: Set global LLM and embedding model ---
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()

# --- Step 4: Build index from documents ---
storage_context = StorageContext.from_defaults(vector_store=vector_store)
documents = SimpleDirectoryReader(input_dir="data", recursive=True).load_data()
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

print("✅ Indexing complete. You can now launch your Gradio chatbot.")
