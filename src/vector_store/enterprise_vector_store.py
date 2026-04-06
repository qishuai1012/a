"""
Enterprise-grade Vector Store Module
Supports multiple vector databases (ChromaDB, Milvus, Pinecone) with advanced features
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .vectorizer import VectorStore, SearchResult, BaseEmbedding, SentenceTransformerEmbedding
from .milvus_store import MilvusVectorStore, MilvusConfig, create_milvus_vector_store

logger = logging.getLogger(__name__)


@dataclass
class VectorConfig:
    """Configuration for vector store"""
    provider: str = "chromadb"  # chromadb, milvus, pinecone
    collection_name: str = "qa_collection"
    persist_directory: Optional[str] = None
    dimension: int = 384  # Default for MiniLM
    max_connections: int = 10
    timeout: int = 30
    metadata_filtering: bool = True


class AsyncVectorStore(ABC):
    """Async-enabled vector store interface"""

    @abstractmethod
    async def aadd(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Async add embeddings to the store"""
        pass

    @abstractmethod
    async def asearch(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Async search for similar vectors"""
        pass

    @abstractmethod
    def add(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Sync add embeddings to the store"""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Sync search for similar vectors"""
        pass


class EnterpriseChromaDB(VectorStore, AsyncVectorStore):
    """
    Enhanced ChromaDB implementation with enterprise features
    """

    def __init__(self, config: VectorConfig):
        if not CHROMADB_AVAILABLE:
            raise ImportError("Please install chromadb: pip install chromadb")

        self.config = config
        self.dimension = config.dimension

        if config.persist_directory:
            self.client = chromadb.PersistentClient(path=config.persist_directory)
        else:
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))

        # Create or get collection with proper configuration
        try:
            self.collection = self.client.get_collection(name=config.collection_name)
        except:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=config.collection_name,
                metadata={"hnsw:space": "cosine"},
                # Enable embedding function if needed
            )

        self.executor = ThreadPoolExecutor(max_workers=config.max_connections)
        logger.info(f"Initialized EnterpriseChromaDB with collection: {config.collection_name}")

    def add(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents synchronously"""
        if ids is None:
            ids = [f"doc_{int(time.time())}_{i}" for i in range(len(documents))]

        # Convert numpy embeddings to list
        embeddings_list = embeddings.tolist()

        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in documents]

        # Add documents with enhanced metadata
        chroma_metadatas = []
        for doc, meta in zip(documents, metadatas):
            meta_copy = meta.copy()
            meta_copy["text"] = doc[:1000]  # Limit text in metadata to prevent oversized records
            meta_copy["created_at"] = str(int(time.time()))
            meta_copy["doc_length"] = len(doc)
            chroma_metadatas.append(meta_copy)

        try:
            self.collection.add(
                embeddings=embeddings_list,
                documents=documents,
                metadatas=chroma_metadatas,
                ids=ids
            )
            logger.debug(f"Added {len(ids)} documents to collection {self.config.collection_name}")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    async def aadd(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Async add documents"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self.add,
            embeddings,
            documents,
            metadatas,
            ids
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Search synchronously"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filter_metadata
            )

            search_results = []
            if results and results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    if i < len(results["documents"][0]) and i < len(results["distances"][0]) and i < len(results["metadatas"][0]):
                        content = results["documents"][0][i] if results["documents"] and i < len(results["documents"][0]) else ""
                        distance = results["distances"][0][i] if results["distances"] and i < len(results["distances"][0]) else 0
                        metadata = results["metadatas"][0][i] if results["metadatas"] and i < len(results["metadatas"][0]) else {}

                        search_results.append(SearchResult(
                            content=content,
                            score=1 - distance,  # Convert distance to similarity
                            chunk_id=chunk_id,
                            metadata=metadata
                        ))

            logger.debug(f"Search returned {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    async def asearch(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Async search"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.search,
            query_embedding,
            top_k,
            filter_metadata
        )

    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs"""
        self.collection.delete(ids=ids)

    def count(self) -> int:
        """Get total number of documents"""
        return self.collection.count()


class OptimizedVectorizer:
    """
    Enhanced vectorizer with async support and performance optimizations
    """

    def __init__(
        self,
        embedding: Optional[BaseEmbedding] = None,
        vector_store: Optional[Union[VectorStore, AsyncVectorStore]] = None,
        batch_size: int = 10,
        enable_async: bool = True
    ):
        self.embedding = embedding or SentenceTransformerEmbedding()
        self.vector_store = vector_store
        self.batch_size = batch_size
        self.enable_async = enable_async

    def set_vector_store(self, vector_store: Union[VectorStore, AsyncVectorStore]) -> None:
        """Set the vector store"""
        self.vector_store = vector_store

    def process_and_store(
        self,
        documents: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None
    ) -> None:
        """
        Synchronously embed documents and store them in batches
        """
        if self.vector_store is None:
            raise ValueError("Vector store not set")

        # Process in batches to optimize memory usage
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size] if ids else None
            batch_metas = metadatas[i:i + self.batch_size] if metadatas else None

            embeddings = self.embedding.embed(batch_docs)
            self.vector_store.add(embeddings, batch_docs, batch_metas, batch_ids)

    async def aprocess_and_store(
        self,
        documents: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None
    ) -> None:
        """
        Asynchronously embed documents and store them in batches
        """
        if self.vector_store is None:
            raise ValueError("Vector store not set")

        if not hasattr(self.vector_store, 'aadd'):
            # Fall back to sync if async not available
            self.process_and_store(documents, ids, metadatas)
            return

        # Process in batches asynchronously
        tasks = []
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size] if ids else None
            batch_metas = metadatas[i:i + self.batch_size] if metadatas else None

            embeddings = self.embedding.embed(batch_docs)
            task = self.vector_store.aadd(embeddings, batch_docs, batch_metas, batch_ids)
            tasks.append(task)

        await asyncio.gather(*tasks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Synchronously search for relevant documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not set")

        query_embedding = self.embedding.embed_query(query)
        return self.vector_store.search(query_embedding, top_k, filter_metadata)

    async def asearch(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Asynchronously search for relevant documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not set")

        if not hasattr(self.vector_store, 'asearch'):
            # Fall back to sync if async not available
            return self.search(query, top_k, filter_metadata)

        query_embedding = self.embedding.embed_query(query)
        return await self.vector_store.asearch(query_embedding, top_k, filter_metadata)


class VectorPerformanceMonitor:
    """
    Performance monitoring for vector operations
    """

    def __init__(self):
        self.metrics = {
            'search_count': 0,
            'avg_search_time': 0,
            'total_search_time': 0,
            'add_count': 0,
            'avg_add_time': 0,
            'total_add_time': 0,
            'error_count': 0
        }

    def record_search(self, search_time: float):
        """Record search operation metrics"""
        self.metrics['search_count'] += 1
        self.metrics['total_search_time'] += search_time
        self.metrics['avg_search_time'] = self.metrics['total_search_time'] / self.metrics['search_count']

    def record_add(self, add_time: float):
        """Record add operation metrics"""
        self.metrics['add_count'] += 1
        self.metrics['total_add_time'] += add_time
        self.metrics['avg_add_time'] = self.metrics['total_add_time'] / self.metrics['add_count']

    def record_error(self):
        """Record error occurrence"""
        self.metrics['error_count'] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()


# Factory function for creating vector stores
def create_enterprise_vector_store(config: VectorConfig) -> Union[VectorStore, AsyncVectorStore]:
    """
    Factory function to create enterprise vector store instances
    """
    if config.provider == "chromadb":
        return EnterpriseChromaDB(config)
    elif config.provider == "milvus" and MILVUS_AVAILABLE:
        from .milvus_store import create_milvus_vector_store, MilvusConfig

        milvus_config = MilvusConfig(
            collection_name=config.collection_name,
            dimension=config.dimension,
            max_connections=config.max_connections,
            timeout=config.timeout
        )

        return create_milvus_vector_store(milvus_config)
    elif config.provider == "pinecone":
        # Placeholder for Pinecone implementation
        raise NotImplementedError("Pinecone implementation not yet available")
    else:
        raise ValueError(f"Unsupported vector store provider: {config.provider}")


if __name__ == "__main__":
    # Example usage
    config = VectorConfig(
        provider="chromadb",
        collection_name="test_collection",
        persist_directory="./data/vector_store"
    )

    vectorizer = OptimizedVectorizer()
    vector_store = create_enterprise_vector_store(config)
    vectorizer.set_vector_store(vector_store)

    # Add documents
    docs = ["这是第一个文档", "这是第二个文档", "企业知识库优化示例"]
    vectorizer.process_and_store(docs, ids=["1", "2", "3"])

    # Search
    results = vectorizer.search("第一个")
    for result in results:
        print(f"Score: {result.score:.3f}, Content: {result.content[:50]}...")