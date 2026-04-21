import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

def get_embeddings():
    """Returns a fast, free embedding model suitable for RAG."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

def get_llm():
    """Returns the Groq LLM (very fast and free tier)."""
    from langchain_groq import ChatGroq
    
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=2048,
        api_key=os.getenv("GROQ_API_KEY")
    )