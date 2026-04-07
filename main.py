"""
Main entry point for the Integrated QA System
Supports both monolithic and microservice architectures
"""

import argparse
import sys

from src.integrated_qa import IntegratedQASystem
from src.api.server import run_server
from microservice_main import main as microservice_main
from src.cli.arguments import create_common_parser


def main():
    """Main entry point"""
    parser = create_common_parser()
    args = parser.parse_args()

    # Handle microservice mode
    if hasattr(args, 'microservice_mode') and args.microservice_mode:
        # Redirect to microservice main
        # Modify sys.argv to pass microservice args
        sys.argv = [sys.argv[0]] + sys.argv[1:]  # Pass through args
        microservice_main()
        return

    if args.command == "microservice":
        # Run specific microservice
        import subprocess
        cmd = [sys.executable, sys.argv[0], "--microservice-mode"] + sys.argv[1:]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

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
