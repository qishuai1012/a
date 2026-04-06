"""
Enterprise QA System - Microservice Architecture Entry Point
"""

import argparse
import sys
import uvicorn
from typing import Optional

from src.architecture.api_gateway import get_fastapi_app
from src.architecture.microservice_config import MicroserviceArchitecture
from src.integrated_qa import IntegratedQASystem


def run_api_gateway(host: str = "0.0.0.0", port: int = 8000):
    """Run the API gateway service"""
    app = get_fastapi_app()
    uvicorn.run(app, host=host, port=port)


def run_qa_service(host: str = "0.0.0.0", port: int = 8002, vector_store_path: Optional[str] = None):
    """Run the QA service"""
    from src.api.enhanced_server import run_server
    from src.integrated_qa import IntegratedQASystem

    # Initialize with Milvus as the default vector store
    qa_system = IntegratedQASystem(
        vector_store_path=vector_store_path,
        enable_enterprise_features=True,
        vector_store_provider="milvus"  # Use Milvus by default
    )

    # In a real microservice, this would be a separate service
    # Here we're adapting the existing server to work in microservice mode
    run_server(host=host, port=port, qa_system=qa_system)


def run_auth_service(host: str = "0.0.0.0", port: int = 8001):
    """Run the authentication service"""
    from fastapi import FastAPI
    import uvicorn
    from src.security.authentication import AuthenticationService, SecurityConfig
    from src.security.security_config import get_security_manager
    import jwt

    app = FastAPI(title="Authentication Service")

    security_manager = get_security_manager()
    auth_service = AuthenticationService(security_manager.security_config)

    @app.post("/login")
    async def login(username: str, password: str):
        token_payload = auth_service.authenticate_user(username, password)

        if not token_payload:
            return {"error": "Invalid credentials"}

        # Create JWT token string
        token = jwt.encode({
            "user_id": token_payload.user_id,
            "username": token_payload.username,
            "role": token_payload.role,
            "exp": token_payload.exp,
            "iat": token_payload.iat,
            "jti": token_payload.jti
        }, security_manager.settings.jwt_secret_key,
           algorithm=security_manager.settings.jwt_algorithm)

        return {"access_token": token, "token_type": "bearer"}

    @app.post("/validate")
    async def validate_token(token: str):
        payload = auth_service.decode_token(token)
        return {"valid": payload is not None}

    uvicorn.run(app, host=host, port=port)


def run_vector_store_service(host: str = "0.0.0.0", port: int = 8004, persist_path: Optional[str] = None):
    """Run the vector store service"""
    from fastapi import FastAPI
    import uvicorn
    from src.vector_store.enterprise_vector_store import create_enterprise_vector_store, VectorConfig, OptimizedVectorizer

    app = FastAPI(title="Vector Store Service")

    # Initialize vector store with Milvus as the provider
    config = VectorConfig(
        provider="milvus",  # Changed to use Milvus by default
        persist_directory=persist_path,
        collection_name="qa_vectors"
    )
    vector_store = create_enterprise_vector_store(config)
    vectorizer = OptimizedVectorizer()
    vectorizer.set_vector_store(vector_store)

    @app.post("/search")
    async def search(query: str, top_k: int = 5):
        results = vectorizer.search(query, top_k=top_k)
        return {
            "results": [
                {
                    "content": r.content,
                    "score": r.score,
                    "chunk_id": r.chunk_id,
                    "metadata": r.metadata
                } for r in results
            ]
        }

    @app.post("/add")
    async def add_document(content: str, doc_id: str, metadata: dict = None):
        # This would be implemented with proper vectorization
        # For now, we'll just acknowledge the request
        return {"status": "added", "doc_id": doc_id}

    uvicorn.run(app, host=host, port=port)


def run_bm25_cache_service(host: str = "0.0.0.0", port: int = 8005, cache_path: Optional[str] = None):
    """Run the BM25 cache service"""
    from fastapi import FastAPI
    import uvicorn
    from src.bm25_cache.enhanced_cache import MultiLevelCache, CacheConfig
    from src.bm25_cache.cache import BM25Layer

    app = FastAPI(title="BM25 Cache Service")

    # Initialize cache
    cache_config = CacheConfig(persistent_path=cache_path)
    bm25_layer = BM25Layer(
        cache_file=cache_path,
        cache_config=cache_config,
        use_redis=True,  # Enable Redis cache
        redis_config={
            "redis_host": "localhost",
            "redis_port": 6379,
            "default_ttl": 3600
        }
    )

    @app.post("/get")
    async def get_cached_response(query: str):
        response = bm25_layer.query_cache(query)
        return {"cached_response": response}

    @app.post("/cache")
    async def cache_response(query: str, response: str):
        bm25_layer.add_to_cache(query, response)
        return {"status": "cached"}

    uvicorn.run(app, host=host, port=port)


def main():
    """Main entry point for microservice architecture"""
    parser = argparse.ArgumentParser(description="Enterprise QA System - Microservice Mode")

    parser.add_argument(
        "service",
        choices=["gateway", "qa", "auth", "vector_store", "bm25_cache", "all"],
        help="Service to run"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to bind to")
    parser.add_argument("--vector-store-path", help="Path to vector store")
    parser.add_argument("--cache-path", help="Path to cache file")

    args = parser.parse_args()

    if args.service == "gateway":
        port = args.port or 8000
        run_api_gateway(host=args.host, port=port)
    elif args.service == "qa":
        port = args.port or 8002
        run_qa_service(host=args.host, port=port, vector_store_path=args.vector_store_path)
    elif args.service == "auth":
        port = args.port or 8001
        run_auth_service(host=args.host, port=port)
    elif args.service == "vector_store":
        port = args.port or 8004
        run_vector_store_service(host=args.host, port=port, persist_path=args.vector_store_path)
    elif args.service == "bm25_cache":
        port = args.port or 8005
        run_bm25_cache_service(host=args.host, port=port, cache_path=args.cache_path)
    elif args.service == "all":
        # In a real deployment, you'd use docker-compose or kubernetes
        print("Starting all services...")
        print("This is a simplified example. In production, use orchestration tools like:")
        print("- Docker Compose")
        print("- Kubernetes")
        print("- Cloud deployment services")

        # Show configuration
        arch = MicroserviceArchitecture()
        print("\nService Configuration:")
        for name, config in arch.services.items():
            print(f"- {name}: {config.host}:{config.port}")


if __name__ == "__main__":
    main()