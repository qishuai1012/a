"""
Milvus Vector Store Module
Implements enterprise-grade vector storage using Milvus
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from .vectorizer import VectorStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class MilvusConfig:
    """Configuration for Milvus vector store"""
    uri: str = "http://milvus:19530"  # Milvus server address
    token: Optional[str] = None          # Authentication token (if required)
    db_name: str = "default"             # Database name
    collection_name: str = "qa_collection"  # Collection name
    dimension: int = 384                 # Embedding dimension (for MiniLM)
    consistency_level: str = "Strong"    # Consistency level
    max_connections: int = 10
    timeout: int = 30
    recreate_collection: bool = False    # Whether to recreate collection if exists


class MilvusVectorStore(VectorStore):
    """
    Milvus-based vector store implementation
    """

    def __init__(self, config: MilvusConfig):
        self.config = config

        # Connect to Milvus
        connections.connect(
            alias="default",
            uri=config.uri,
            token=config.token,
            db_name=config.db_name
        )

        # Define collection schema
        self.schema = CollectionSchema([
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=config.dimension),
            FieldSchema("content", DataType.VARCHAR, max_length=65535),
            FieldSchema("chunk_id", DataType.VARCHAR, max_length=100),
            FieldSchema("parent_document", DataType.VARCHAR, max_length=1000),
            FieldSchema("metadata", DataType.JSON),
        ], description="QA System Vector Collection")

        # Create or get collection
        if config.recreate_collection and utility.has_collection(config.collection_name):
            utility.drop_collection(config.collection_name)

        if not utility.has_collection(config.collection_name):
            self.collection = Collection(
                name=config.collection_name,
                schema=self.schema,
                consistency_level=config.consistency_level
            )

            # Create index for efficient similarity search
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
        else:
            self.collection = Collection(config.collection_name)

        # Load collection into memory for search
        self.collection.load()

        logger.info(f"Initialized MilvusVectorStore with collection: {config.collection_name}")

    def add(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to Milvus collection"""
        # Prepare data for insertion
        entities = []
        for i, doc in enumerate(documents):
            # Use provided ID or create one based on index
            chunk_id = ids[i] if ids and i < len(ids) else f"doc_{int(time.time())}_{i}"
            parent_doc = metadatas[i]["parent_document"] if metadatas and i < len(metadatas) else "unknown"

            # Metadata
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            metadata["original_content_length"] = len(doc)

            entity = {
                "embedding": embeddings[i].tolist() if isinstance(embeddings[i], np.ndarray) else embeddings[i],
                "content": doc[:65530],  # Truncate to fit VARCHAR limit
                "chunk_id": chunk_id,
                "parent_document": parent_doc,
                "metadata": metadata
            }
            entities.append(entity)

        # Insert into Milvus
        entities_dict = {
            "embedding": [entity["embedding"] for entity in entities],
            "content": [entity["content"] for entity in entities],
            "chunk_id": [entity["chunk_id"] for entity in entities],
            "parent_document": [entity["parent_document"] for entity in entities],
            "metadata": [entity["metadata"] for entity in entities]
        }

        try:
            insert_result = self.collection.insert([
                entities_dict["embedding"],
                entities_dict["content"],
                entities_dict["chunk_id"],
                entities_dict["parent_document"],
                entities_dict["metadata"]
            ])

            # Flush to ensure data is indexed
            self.collection.flush()

            logger.debug(f"Added {len(documents)} documents to Milvus, insert IDs: {insert_result.primary_keys[:5]}")
        except Exception as e:
            logger.error(f"Failed to add documents to Milvus: {str(e)}")
            raise

    async def aadd(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Async add documents"""
        # Milvus operations are typically synchronous, but we provide this interface for compatibility
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            ThreadPoolExecutor(max_workers=1),
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
        """Search in Milvus collection"""
        try:
            # Prepare search parameters
            search_params = {
                "data": [query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding],
                "anns_field": "embedding",
                "param": {"metric_type": "COSINE", "params": {"nprobe": 10}},
                "limit": top_k,
                "expr": None,
                "output_fields": ["content", "chunk_id", "parent_document", "metadata"]
            }

            # Add filter if provided
            if filter_metadata:
                filter_expr_parts = []
                for key, value in filter_metadata.items():
                    if isinstance(value, str):
                        filter_expr_parts.append(f'metadata["{key}"] == "{value}"')
                    elif isinstance(value, (int, float)):
                        filter_expr_parts.append(f'metadata["{key}"] == {value}')

                if filter_expr_parts:
                    search_params["expr"] = " and ".join(filter_expr_parts)

            # Perform search
            results = self.collection.search(**search_params)

            search_results = []
            for i, hits in enumerate(results):
                for hit in hits:
                    search_results.append(SearchResult(
                        content=hit.entity.get("content"),
                        score=hit.distance,  # Milvus returns distance; smaller is better
                        chunk_id=hit.entity.get("chunk_id"),
                        metadata=hit.entity.get("metadata", {})
                    ))

            # Convert distance to similarity (for consistency with other stores)
            for result in search_results:
                result.score = 1.0 - result.score  # Convert distance to similarity

            logger.debug(f"Search returned {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"Search failed in Milvus: {str(e)}")
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
            ThreadPoolExecutor(max_workers=1),
            self.search,
            query_embedding,
            top_k,
            filter_metadata
        )

    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs"""
       #错误修改齐帅 # expr = f"chunk_id in [{','.join([f'\"{id}\"' for id in ids])}]"
        expr = f"chunk_id in [{','.join(repr(id) for id in ids)}]"
        delete_result = self.collection.delete(expr=expr)
        self.collection.flush()
        logger.debug(f"Deleted {delete_result.delete_count} documents from Milvus")

    def count(self) -> int:
        """Get total number of documents"""
        return self.collection.num_entities


class MilvusOptimizedVectorizer:
    """
    Enhanced vectorizer for Milvus with async support and performance optimizations
    """

    def __init__(
        self,
        embedding: Optional['BaseEmbedding'] = None,
        vector_store: Optional[Union[VectorStore, 'AsyncVectorStore']] = None,
        batch_size: int = 10,
        enable_async: bool = True
    ):
        # Import embedding module dynamically to avoid circular import
        from .vectorizer import SentenceTransformerEmbedding
        self.embedding = embedding or SentenceTransformerEmbedding()
        self.vector_store = vector_store
        self.batch_size = batch_size
        self.enable_async = enable_async

    def set_vector_store(self, vector_store: Union[VectorStore, 'AsyncVectorStore']) -> None:
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


def create_milvus_vector_store(config: MilvusConfig) -> MilvusVectorStore:
    """
    Factory function to create Milvus vector store instance
    """
    return MilvusVectorStore(config)


if __name__ == "__main__":
    # Example usage
    config = MilvusConfig(
        # uri="http://localhost:195?30",
        uri="http://milvus:19530",
        collection_name="test_qa_collection",
        dimension=384  # For MiniLM embeddings
    )

    vectorizer = MilvusOptimizedVectorizer()
    vector_store = create_milvus_vector_store(config)
    vectorizer.set_vector_store(vector_store)

    # Add documents
    docs = ["这是第一个文档", "这是第二个文档", "企业知识库优化示例"]
    vectorizer.process_and_store(docs, ids=["1", "2", "3"])

    # Search
    results = vectorizer.search("第一个")
    for result in results:
        print(f"Score: {result.score:.3f}, Content: {result.content[:50]}...")