"""
Integrated QA System - Main Entry Point
Orchestrates all components for the complete QA pipeline
"""

from typing import Optional, List, Dict, AsyncGenerator
from dataclasses import dataclass
import asyncio
import structlog
import logging
import hashlib
from datetime import datetime

from .document_loader import DocumentLoader, Document
from .document_processor import DocumentProcessor, TextChunk
from .vector_store.vectorizer import Vectorizer, ChromaVectorStore
from .bm25_cache.cache import BM25Layer
from .dialogue_history import DialogueManager, DialogueHistory
from .rag_system.classifier import QueryType, QueryClassifier
from .rag_system.retriever import HybridRetriever, SimpleRanker
from .rag_system.prompt_builder import PromptBuilder
from .utils.error_handling import (
    CircuitBreaker,
    CircuitBreakerConfig,
    ComprehensiveErrorHandler
)


@dataclass
class QAResponse:
    """Represents a QA system response"""
    answer: str
    source_documents: List[str]
    confidence: float
    cache_hit: bool
    query_type: str
    conversation_id: str


class IntegratedQASystem:
    """
    Integrated Question Answering System

    Combines:
    - Document loading and processing
    - Vector search and BM25 caching
    - Query classification
    - RAG-based generation
    - Dialogue history management
    """

    def __init__(
        self,
        vector_store_path: Optional[str] = None,
        bm25_cache_path: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        max_history_turns: int = 10,
        enable_enterprise_features: bool = True,
        vector_store_provider: str = "milvus",  # Changed default to Milvus
        business_category: str = "general",
        permission_level: str = "public"
    ):
        """
        Initialize the Integrated QA System

        Args:
            vector_store_path: Path to persist vector store
            bm25_cache_path: Path to persist BM25 cache
            llm_api_key: API key for LLM service
            llm_base_url: Base URL for LLM API
            max_history_turns: Maximum conversation turns to keep
            enable_enterprise_features: Whether to enable enterprise features (vector store, etc.)
            vector_store_provider: Vector store provider ('milvus', 'chromadb', 'pinecone')
            business_category: Default business category for documents
            permission_level: Default permission level for documents
        """
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url or "http://localhost:11434/api"
        self.enable_enterprise_features = enable_enterprise_features
        self.vector_store_provider = vector_store_provider
        self.business_category = business_category
        self.permission_level = permission_level

        # Initialize error handling components
        self.error_handler = ComprehensiveErrorHandler()

        # Initialize circuit breakers for critical components
        self.llm_circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60,
            expected_exception=(ConnectionError, TimeoutError, Exception),
            timeout=30
        )
        self.llm_circuit_breaker = CircuitBreaker(self.llm_circuit_breaker_config)

        # Initialize document pipeline
        self.document_loader = DocumentLoader()
        self.document_processor = DocumentProcessor()

        # Initialize vector store with enterprise features
        if enable_enterprise_features:
            from .vector_store.enterprise_vector_store import OptimizedVectorizer, create_enterprise_vector_store, VectorConfig

            # Default to chromadb if milvus is not available
            vector_config = VectorConfig(
                provider=vector_store_provider,
                persist_directory=vector_store_path
            )

            try:
                self.vector_store = create_enterprise_vector_store(vector_config)
                self.vectorizer = OptimizedVectorizer()
                self.vectorizer.set_vector_store(self.vector_store)
            except Exception as e:
                print(f"Warning: Could not initialize {vector_store_provider}, falling back to ChromaDB: {str(e)}")
                # Fallback to ChromaDB
                from .vector_store.vectorizer import Vectorizer, ChromaVectorStore
                self.vectorizer = Vectorizer()
                self.vector_store = ChromaVectorStore(
                    persist_directory=vector_store_path
                )
                self.vectorizer.set_vector_store(self.vector_store)
        else:
            from .vector_store.vectorizer import Vectorizer, ChromaVectorStore
            self.vectorizer = Vectorizer()
            self.vector_store = ChromaVectorStore(
                persist_directory=vector_store_path
            )
            self.vectorizer.set_vector_store(self.vector_store)

        # Initialize BM25 layer
        self.bm25_layer = BM25Layer(cache_file=bm25_cache_path)

        # Initialize dialogue management
        self.dialogue_manager = DialogueManager()
        self.max_history_turns = max_history_turns

        # Initialize RAG components
        self.classifier = QueryClassifier()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_layer=self.bm25_layer
        )
        self.ranker = SimpleRanker()
        self.prompt_builder = PromptBuilder()

        # Track initialized state
        self._initialized = False

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate document ID based on file path and timestamp"""
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"doc_{path_hash}_{timestamp}"

    def _should_ingest(self, content: str, existing_doc_ids: set) -> bool:
        """Check if document should be ingested based on content similarity"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return content_hash not in existing_doc_ids

    def ingest_document(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        business_category: Optional[str] = None,
        permission_level: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> int:
        """
        Ingest a document into the system with enterprise features

        Args:
            file_path: Path to the document
            doc_id: Optional document ID (auto-generated if not provided)
            business_category: Business category for the document
            permission_level: Permission level for the document
            tags: List of tags for the document

        Returns:
            Number of chunks created
        """
        # Generate doc_id if not provided
        doc_id = doc_id or self._generate_doc_id(file_path)

        # Load document
        doc = self.document_loader.load(file_path)

        # Process and chunk with enterprise metadata
        chunks = self.document_processor.process(
            content=doc.content,
            doc_id=doc_id,
            metadata={
                "file_path": file_path,
                "file_type": doc.file_type,
                "original_length": len(doc.content),
                "processed_at": datetime.now().isoformat(),
                "source_file_hash": hashlib.md5(doc.content.encode()).hexdigest()
            },
            business_category=business_category or self.business_category,
            permission_level=permission_level or self.permission_level,
            tags=tags
        )

        # Check if we have already processed this document
        if not chunks:
            return 0

        # Store in vector store
        contents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "chunk_id": chunk.chunk_id,
                "parent_document": chunk.parent_document,
                "file_type": doc.file_type,
                "business_category": chunk.business_category,
                "permission_level": chunk.permission_level,
                "tags": chunk.tags,
                "update_timestamp": chunk.update_timestamp.isoformat() if chunk.update_timestamp else None,
                "semantic_similarity": chunk.semantic_similarity,
                "parent_id": chunk.parent_id,
                "child_ids": chunk.child_ids
            }
            for chunk in chunks
        ]

        self.vectorizer.process_and_store(
            contents,
            ids=[chunk.chunk_id for chunk in chunks],
            metadatas=metadatas
        )

        # Add to BM25 document store
        self.bm25_layer.add_documents(contents, metadatas)

        return len(chunks)

    def ingest_directory(
        self,
        directory_path: str,
        file_types: Optional[List[str]] = None,
        business_category: Optional[str] = None,
        permission_level: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> int:
        """
        Ingest all documents from a directory

        Args:
            directory_path: Path to directory
            file_types: Optional list of file extensions to process
            business_category: Business category for the documents
            permission_level: Permission level for the documents
            tags: List of tags for the documents

        Returns:
            Total number of chunks created
        """
        docs = self.document_loader.load_directory(directory_path, file_types)
        total_chunks = 0

        for doc in docs:
            chunks = self.document_processor.process(
                content=doc.content,
                doc_id=self._generate_doc_id(doc.file_path),
                metadata={
                    "file_path": doc.file_path,
                    "file_type": doc.file_type,
                    "original_length": len(doc.content),
                    "processed_at": datetime.now().isoformat(),
                    "source_file_hash": hashlib.md5(doc.content.encode()).hexdigest()
                },
                business_category=business_category or self.business_category,
                permission_level=permission_level or self.permission_level,
                tags=tags
            )

            if chunks:
                contents = [chunk.content for chunk in chunks]
                metadatas = [
                    {
                        "chunk_id": chunk.chunk_id,
                        "parent_document": chunk.parent_document,
                        "file_type": doc.file_type,
                        "business_category": chunk.business_category,
                        "permission_level": chunk.permission_level,
                        "tags": chunk.tags,
                        "update_timestamp": chunk.update_timestamp.isoformat() if chunk.update_timestamp else None,
                        "semantic_similarity": chunk.semantic_similarity,
                        "parent_id": chunk.parent_id,
                        "child_ids": chunk.child_ids
                    }
                    for chunk in chunks
                ]

                self.vectorizer.process_and_store(
                    contents,
                    ids=[chunk.chunk_id for chunk in chunks],
                    metadatas=metadatas
                )

                self.bm25_layer.add_documents(contents, metadatas)
                total_chunks += len(chunks)

        return total_chunks

    def query(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        business_category_filter: Optional[str] = None,
        permission_level_filter: Optional[str] = None
    ) -> QAResponse:
        """
        Process a user query with optional filters

        Args:
            query_text: User's question
            session_id: Optional conversation session ID
            top_k: Number of documents to retrieve
            business_category_filter: Filter results by business category
            permission_level_filter: Filter results by permission level

        Returns:
            QAResponse object
        """
        with self.error_handler.error_context("query_processing", query_text=query_text, session_id=session_id):
            # Get or create session
            session = self.dialogue_manager.get_or_create_session(session_id)
            if session_id is None:
                session_id = session.session_id

            # Check BM25 cache first
            cached_answer = self.bm25_layer.query_cache(query_text)
            if cached_answer:
                # Add to history
                session.add_turn(query_text, cached_answer, {"cache_hit": True})

                return QAResponse(
                    answer=cached_answer,
                    source_documents=[],
                    confidence=1.0,
                    cache_hit=True,
                    query_type="cached",
                    conversation_id=session_id
                )

            # Classify query
            query_type = self.classifier.classify(query_text)

            if query_type == QueryType.GENERAL:
                # Direct LLM for general knowledge
                answer = self._query_llm_directly(query_text, session)
                response = QAResponse(
                    answer=answer,
                    source_documents=[],
                    confidence=0.8,
                    cache_hit=False,
                    query_type="general",
                    conversation_id=session_id
                )
            else:
                # RAG for professional queries
                answer, sources = self._query_rag(
                    query_text,
                    session,
                    top_k,
                    business_category_filter,
                    permission_level_filter
                )
                response = QAResponse(
                    answer=answer,
                    source_documents=[s.content for s in sources],
                    confidence=sources[0].combined_score if sources else 0,
                    cache_hit=False,
                    query_type="professional",
                    conversation_id=session_id
                )

            # Cache the response
            self.bm25_layer.add_to_cache(query_text, response.answer)

            return response

    def _query_llm_directly(
        self,
        query: str,
        session: DialogueHistory
    ) -> str:
        """Query LLM directly for general knowledge"""
        messages = session.format_for_llm(self.prompt_builder.system_prompt)

        # Add current query
        messages.append({"role": "user", "content": query})

        # Call LLM (placeholder - implement based on your LLM provider)
        response = self._call_llm(messages)

        # Update history
        session.add_turn(query, response)

        return response

    def _query_rag(
        self,
        query: str,
        session: DialogueHistory,
        top_k: int,
        business_category_filter: Optional[str] = None,
        permission_level_filter: Optional[str] = None
    ) -> tuple:
        """Query using RAG pipeline with optional filters"""
        # Prepare metadata filter
        metadata_filter = {}
        if business_category_filter:
            metadata_filter["business_category"] = business_category_filter
        if permission_level_filter:
            metadata_filter["permission_level"] = permission_level_filter

        # Retrieve documents with optional filtering
        results = self.retriever.retrieve(query, top_k=top_k * 2)

        # Apply permission filtering at the application level as well
        if permission_level_filter:
            filtered_results = []
            for result in results:
                perm_level = result.metadata.get("permission_level", "public")
                # Only allow access if permission level matches or is more public
                if self._has_permission(permission_level_filter, perm_level):
                    filtered_results.append(result)
            results = filtered_results

        # Re-rank
        reranked = self.ranker.rerank(query, results, top_k=top_k)

        # Build prompt with context
        context_docs = [r.content for r in reranked]
        history = session.get_messages(limit=self.max_history_turns * 2)
        history_dicts = [{"role": m.role, "content": m.content} for m in history]

        messages = self.prompt_builder.build(
            query=query,
            documents=context_docs,
            conversation_history=history_dicts
        )

        # Call LLM
        answer = self._call_llm(messages)

        # Update history
        session.add_turn(query, answer, {"sources": [r.chunk_id for r in reranked]})

        return answer, reranked

    def _has_permission(self, required_level: str, available_level: str) -> bool:
        """Check if user has sufficient permission level"""
        permission_hierarchy = {
            "admin": 4,
            "internal": 3,
            "partner": 2,
            "public": 1
        }

        req_level_value = permission_hierarchy.get(required_level, 1)
        avail_level_value = permission_hierarchy.get(available_level, 1)

        return req_level_value <= avail_level_value

    def _call_llm(self, messages: List[dict]) -> str:
        """
        Call LLM API with error handling and circuit breaker
        """
        def _internal_call():
            import requests
            import json

            # Try to use the configured LLM API
            try:
                # Determine if this is an OpenAI-compatible API (like Ollama) or OpenAI
                if "api.openai.com" in self.llm_base_url.lower():
                    # OpenAI API format
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.llm_api_key}"
                    }

                    data = {
                        "model": "gpt-3.5-turbo",  # Default OpenAI model
                        "messages": messages,
                        "stream": False,
                        "temperature": 0.7
                    }

                    response = requests.post(
                        f"{self.llm_base_url}/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=60
                    )
                else:
                    # Assume it's an OpenAI-compatible API like Ollama
                    headers = {
                        "Content-Type": "application/json"
                    }

                    # Format messages for Ollama if not using OpenAI
                    formatted_messages = []
                    for msg in messages:
                        formatted_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })

                    data = {
                        "model": "qwen:7b" if "ollama" in self.llm_base_url.lower() else "llama2",  # Default to qwen if ollama detected
                        "messages": formatted_messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.7
                        }
                    }

                    response = requests.post(
                        f"{self.llm_base_url}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=60
                    )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    # Log error details
                    print(f"LLM API Error: {response.status_code} - {response.text}")
                    # Return a fallback response
                    return "抱歉，目前无法获取答案，请稍后重试。"

            except requests.exceptions.ConnectionError:
                print("无法连接到LLM服务，请确认服务是否正常运行")
                return "抱歉，目前无法获取答案，请稍后重试。"
            except requests.exceptions.Timeout:
                print("LLM服务请求超时")
                return "抱歉，请求处理超时，请稍后重试。"
            except KeyError as e:
                print(f"LLM API响应格式错误: {str(e)} - {response.text if 'response' in locals() else 'No response'}")
                return "抱歉，服务响应格式异常，请稍后重试。"
            except Exception as e:
                # Log the error details
                print(f"LLM调用发生未知错误: {str(e)}")
                return "抱歉，目前无法获取答案，请稍后重试。"

        # Execute with circuit breaker protection
        try:
            return self.llm_circuit_breaker.call(_internal_call)
        except Exception as e:
            # Log the error and return a safe fallback
            self.error_handler.handle_error(e, {"operation": "_call_llm", "messages_length": len(messages)})
            return "抱歉，系统暂时不可用，请稍后重试。"

    async def query_stream(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        top_k: int = 5
    ) -> AsyncGenerator[str, None]:
        """
        Process query with streaming response

        Args:
            query_text: User's question
            session_id: Optional conversation session ID
            top_k: Number of documents to retrieve

        Yields:
            Chunks of the response text
        """
        # Check cache first (no streaming for cache hits)
        cached_answer = self.bm25_layer.query_cache(query_text)
        if cached_answer:
            yield cached_answer
            return

        # Get session
        session = self.dialogue_manager.get_or_create_session(session_id)

        # Classify and route
        query_type = self.classifier.classify(query_text)

        if query_type == QueryType.GENERAL:
            messages = session.format_for_llm(self.prompt_builder.system_prompt)
            messages.append({"role": "user", "content": query_text})
        else:
            results = self.retriever.retrieve(query_text, top_k=top_k * 2)
            reranked = self.ranker.rerank(query_text, results, top_k=top_k)
            context_docs = [r.content for r in reranked]

            history = session.get_messages(limit=self.max_history_turns * 2)
            history_dicts = [{"role": m.role, "content": m.content} for m in history]

            messages = self.prompt_builder.build(
                query=query_text,
                documents=context_docs,
                conversation_history=history_dicts
            )

        # Stream from LLM (placeholder)
        full_response = ""
        async for chunk in self._stream_llm(messages):
            yield chunk
            full_response += chunk

        # Update history
        session.add_turn(query_text, full_response)

        # Cache
        self.bm25_layer.add_to_cache(query_text, full_response)

    async def _stream_llm(
        self,
        messages: List[dict]
    ) -> AsyncGenerator[str, None]:
        """
        Stream from LLM API

        Placeholder implementation
        """
        # Example for streaming:
        # import openai
        # client = openai.OpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url)
        # stream = client.chat.completions.create(
        #     model="your-model",
        #     messages=messages,
        #     stream=True
        # )
        # for chunk in stream:
        #     if chunk.choices[0].delta.content:
        #         yield chunk.choices[0].delta.content

        yield "[流式响应] 这是一个示例流式回答。"

    def save(self) -> None:
        """Save system state (cache, etc.)"""
        self.bm25_layer.save_cache()

    def get_session(self, session_id: str) -> Optional[DialogueHistory]:
        """Get a conversation session"""
        return self.dialogue_manager.get_session(session_id)

    def clear_session(self, session_id: str) -> bool:
        """Clear a conversation session"""
        return self.dialogue_manager.delete_session(session_id)


if __name__ == "__main__":
    # Example usage
    qa_system = IntegratedQASystem(
        vector_store_path="./data/vector_store",
        bm25_cache_path="./data/bm25_cache.pkl"
    )

    # Ingest documents
    # qa_system.ingest_directory("./docs")

    # Query
    response = qa_system.query("什么是 RAG?")
    print(f"Answer: {response.answer}")
    print(f"Cache hit: {response.cache_hit}")
    print(f"Query type: {response.query_type}")
