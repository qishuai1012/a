"""
Main entry point for the Integrated QA System
"""

import argparse
import sys

from src.integrated_qa import IntegratedQASystem
from src.api.server import run_server


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Integrated QA System - RAG-based Question Answering"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Server command
    server_parser = subparsers.add_parser("server", help="Run API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    server_parser.add_argument(
        "--vector-store",
        help="Path to vector store directory"
    )
    server_parser.add_argument(
        "--bm25-cache",
        help="Path to BM25 cache file"
    )

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="File or directory path to ingest")
    ingest_parser.add_argument(
        "--vector-store",
        help="Path to vector store directory"
    )
    ingest_parser.add_argument(
        "--bm25-cache",
        help="Path to BM25 cache file"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument(
        "--vector-store",
        help="Path to vector store directory"
    )
    query_parser.add_argument(
        "--bm25-cache",
        help="Path to BM25 cache file"
    )
    query_parser.add_argument(
        "--session",
        help="Conversation session ID"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Create QA system
    qa_system = IntegratedQASystem(
        vector_store_path=getattr(args, 'vector_store', None),
        bm25_cache_path=getattr(args, 'bm25_cache', None)
    )

    if args.command == "server":
        run_server(host=args.host, port=args.port, qa_system=qa_system)

    elif args.command == "ingest":
        import os
        if os.path.isdir(args.path):
            chunks = qa_system.ingest_directory(args.path)
            print(f"Ingested directory: {chunks} chunks created")
        else:
            chunks = qa_system.ingest_document(args.path)
            print(f"Ingested document: {chunks} chunks created")

    elif args.command == "query":
        response = qa_system.query(args.query, session_id=args.session)
        print(f"\nAnswer: {response.answer}")
        print(f"Cache hit: {response.cache_hit}")
        print(f"Query type: {response.query_type}")
        print(f"Confidence: {response.confidence:.2f}")
        if response.source_documents:
            print(f"\nSource documents ({len(response.source_documents)}):")
            for i, doc in enumerate(response.source_documents, 1):
                print(f"  {i}. {doc[:100]}...")


if __name__ == "__main__":
    main()
