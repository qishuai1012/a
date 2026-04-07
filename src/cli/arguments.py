"""
Shared command line argument parsing for the Integrated QA System
"""

import argparse


def add_server_arguments(parser):
    """Add common server arguments to a parser"""
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--vector-store",
        help="Path to vector store directory"
    )
    parser.add_argument(
        "--bm25-cache",
        help="Path to BM25 cache file"
    )


def add_microservice_arguments(parser):
    """Add microservice-specific arguments to a parser"""
    parser.add_argument(
        "service_type",
        choices=["gateway", "qa", "auth", "vector_store", "bm25_cache", "all"],
        help="Type of service to run"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to bind to")
    parser.add_argument("--vector-store-path", help="Path to vector store")
    parser.add_argument("--cache-path", help="Path to cache file")


def add_ingest_arguments(parser):
    """Add ingest-specific arguments to a parser"""
    parser.add_argument("path", help="File or directory path to ingest")
    parser.add_argument(
        "--vector-store",
        help="Path to vector store directory"
    )
    parser.add_argument(
        "--bm25-cache",
        help="Path to BM25 cache file"
    )


def add_query_arguments(parser):
    """Add query-specific arguments to a parser"""
    parser.add_argument("query", help="Query text")
    parser.add_argument(
        "--vector-store",
        help="Path to vector store directory"
    )
    parser.add_argument(
        "--bm25-cache",
        help="Path to BM25 cache file"
    )
    parser.add_argument(
        "--session",
        help="Conversation session ID"
    )


def create_common_parser():
    """Create a parser with all common arguments"""
    parser = argparse.ArgumentParser(
        description="Integrated QA System - RAG-based Question Answering"
    )

    # Add microservice support
    parser.add_argument(
        "--microservice-mode",
        action="store_true",
        help="Run in microservice mode"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Server command
    server_parser = subparsers.add_parser("server", help="Run API server")
    add_server_arguments(server_parser)

    # Microservice command
    microservice_parser = subparsers.add_parser("microservice", help="Run in microservice mode")
    add_microservice_arguments(microservice_parser)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    add_ingest_arguments(ingest_parser)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    add_query_arguments(query_parser)

    return parser