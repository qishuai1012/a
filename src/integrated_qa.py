"""
Integrated QA System - Main Entry Point
Orchestrates all components for the complete QA pipeline
"""

from typing import Optional, List, Dict, AsyncGenerator
from dataclasses import dataclass
import asyncio

from .document_loader import DocumentLoader, Document
from .document_processor import DocumentProcessor, TextChunk
from .vector_store import Vectorizer, ChromaVectorStore
from .bm25_cache import BM25Layer
from .dialogue_history import DialogueManager, DialogueHistory
from .rag_system.classifier import QueryType, QueryClassifier
from .rag_system.retriever import HybridRetriever, SimpleRanker
from .rag_system.prompt_builder import PromptBuilder


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
        max_history_turns: int = 10
    ):
        """
        Initialize the Integrated QA System

        Args:
            vector_store_path: Path to persist vector store
            bm25_cache_path: Path to persist BM25 cache
            llm_api_key: API key for LLM service
            llm_base_url: Base URL for LLM API
            max_history_turns: Maximum conversation turns to keep
        """
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url or "http://localhost:11434/api"

        # Initialize document pipeline
        self.document_loader = DocumentLoader()
        self.document_processor = DocumentProcessor()

        # Initialize vector store
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

    def ingest_document(self, file_path: str, doc_id: Optional[str] = None) -> int:
        """
        Ingest a document into the system

        Args:
            file_path: Path to the document
            doc_id: Optional document ID (auto-generated if not provided)

        Returns:
            Number of chunks created
        """
        # Load document
        doc = self.document_loader.load(file_path)

        # Process and chunk
        chunks = self.document_processor.process(doc.content, doc_id or file_path)

        # Store in vector store
        contents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "chunk_id": chunk.chunk_id,
                "parent_document": chunk.parent_document,
                "file_type": doc.file_type
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
        file_types: Optional[List[str]] = None
    ) -> int:
        """
        Ingest all documents from a directory

        Args:
            directory_path: Path to directory
            file_types: Optional list of file extensions to process

        Returns:
            Total number of chunks created
        """
        docs = self.document_loader.load_directory(directory_path, file_types)
        total_chunks = 0

        for doc in docs:
            chunks = self.document_processor.process(doc.content, doc.file_path)
            contents = [chunk.content for chunk in chunks]
            metadatas = [
                {
                    "chunk_id": chunk.chunk_id,
                    "parent_document": chunk.parent_document,
                    "file_type": doc.file_type
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
        top_k: int = 5
    ) -> QAResponse:
        """
        Process a user query

        Args:
            query_text: User's question
            session_id: Optional conversation session ID
            top_k: Number of documents to retrieve

        Returns:
            QAResponse object
        """
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
            answer, sources = self._query_rag(query_text, session, top_k)
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
        top_k: int
    ) -> tuple:
        """Query using RAG pipeline"""
        # Retrieve documents
        results = self.retriever.retrieve(query, top_k=top_k * 2)

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

    def _call_llm(self, messages: List[dict]) -> str:
        """
        Call LLM API

        Placeholder implementation - customize for your LLM provider
        """
        # Example for OpenAI-compatible API:
        # import openai
        # client = openai.OpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url)
        # response = client.chat.completions.create(
        #     model="your-model",
        #     messages=messages
        # )
        # return response.choices[0].message.content

        # Placeholder response
        return "[LLM Response] 这是一个示例回答。请配置实际的 LLM API。"

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
