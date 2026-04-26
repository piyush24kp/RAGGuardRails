"""
Central config — reads from .env and exposes typed settings.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION = "company_docs"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # runs locally, no API key needed

DATA_DIR = "./DS-RPC-01/data"  # department subfolders live here
