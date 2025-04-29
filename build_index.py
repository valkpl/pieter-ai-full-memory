from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
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

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- Step 3: Set models ---
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()

# --- Step 4: Define sources ---
sources = {
    "blogs": "Highly edited, formal blog articles",
    "book": "Book manuscript and proposal on vocational singleness",
    "slides": "Teaching notes and outlines, often in bullet form",
    "transcripts": "Conversational podcast transcripts with rich ideas",
    "social_media": "Instagram captions with engagement metrics"
}

# --- Step 5: Load and tag documents ---
documents = []
for source, description in sources.items():
    folder = os.path.join("data", source)
    print(f"📂 Loading: {folder}")
    docs = SimpleDirectoryReader(folder, recursive=True).load_data()
    for doc in docs:
        doc.metadata = {"source": source, "description": description}
        documents.append(doc)

# --- Step 6: Index documents ---
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

print("✅ Index built and stored in Pinecone.")
