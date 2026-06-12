import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # Only for extraction
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nexus")
    
    # Model configurations
    MISTRAL_MODEL = "mistral-large-latest"
    MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
    
    # Local embedding model (FREE, 1024 dimensions!)
    # Options for 1024 dimensions:
    # - "intfloat/e5-large-v2" (1024 dims, excellent quality)
    # - "sentence-transformers/gtr-t5-large" (1024 dims)
    # - "BAAI/bge-large-en-v1.5" (1024 dims)
    # EMBEDDING_MODEL = "intfloat/e5-large-v2"  # 1024 dimensions, high quality
    
    # File paths
    TOWERS_FILE = "data/towers_inventory.json"
    POLICIES_FILE = "data/regional_policies.txt"
    
    # Validate API keys
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY not found in environment variables")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables")