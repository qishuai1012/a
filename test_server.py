"""
Test startup script that uses ChromaDB by default instead of Milvus
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.integrated_qa import IntegratedQASystem
from src.api.server import run_server

def main():
    print("Starting server with ChromaDB fallback...")

    # Create QA system with disabled enterprise features to use ChromaDB
    qa_system = IntegratedQASystem(
        vector_store_path="./data/vector_store",
        bm25_cache_path="./data/bm25_cache.pkl",
        enable_enterprise_features=False,  # Disable enterprise features to use ChromaDB
        vector_store_provider="chromadb"   # Explicitly use ChromaDB
    )

    print("QA System initialized successfully")
    print("Starting server on http://0.0.0.0:8000")

    # Start the server
    run_server(host="0.0.0.0", port=8000, qa_system=qa_system)

if __name__ == "__main__":
    main()