"""
Vectorization and Vector Store Module
Handles text embeddings and vector storage/retrieval
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class SearchResult:
    """Represents a search result with score"""
    content: str
    score: float
    chunk_id: str
    metadata: dict


class BaseEmbedding(ABC):
    """Base class for text embedding models"""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding for a single query"""
        pass


class SentenceTransformerEmbedding(BaseEmbedding):
    """
    Embedding using sentence-transformers library
    Supports multilingual models including Chinese
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def embed_query(self, text: str) -> np.ndarray:
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0]


class VectorStore(ABC):
    """Base class for vector stores"""

    @abstractmethod
    def add(self, embeddings: np.ndarray, documents: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """Add embeddings to the store"""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        pass


class ChromaVectorStore(VectorStore):
    """
    Vector store using ChromaDB
    Lightweight and suitable for local development
    """

    def __init__(
        self,
        collection_name: str = "qa_collection",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[BaseEmbedding] = None
    ):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("Please install chromadb: pip install chromadb")

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.embedding_function = embedding_function or SentenceTransformerEmbedding()

    def add(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Add embeddings to the store"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        # Convert numpy embeddings to list
        embeddings_list = embeddings.tolist()

        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in documents]

        # Add documents with text content in metadata for Chroma
        chroma_metadatas = []
        for doc, meta in zip(documents, metadatas):
            meta_copy = meta.copy()
            meta_copy["text"] = doc  # Store text in metadata
            chroma_metadatas.append(meta_copy)

        self.collection.add(
            embeddings=embeddings_list,
            documents=documents,
            metadatas=chroma_metadatas,
            ids=ids
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata
        )

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                search_results.append(SearchResult(
                    content=results["documents"][0][i] if results["documents"] else "",
                    score=1 - results["distances"][0][i] if results["distances"] else 0,  # Convert distance to similarity
                    chunk_id=chunk_id,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {}
                ))

        return search_results

    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs"""
        self.collection.delete(ids=ids)

    def count(self) -> int:
        """Get total number of documents"""
        return self.collection.count()


class Vectorizer:
    """
    Unified vectorizer that combines embedding and storage
    """

    def __init__(
        self,
        embedding: Optional[BaseEmbedding] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.embedding = embedding or SentenceTransformerEmbedding()
        self.vector_store = vector_store

    def set_vector_store(self, vector_store: VectorStore) -> None:
        """Set the vector store"""
        self.vector_store = vector_store

    def process_and_store(
        self,
        documents: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None
    ) -> None:
        """
        Embed documents and store them
        """
        if self.vector_store is None:
            raise ValueError("Vector store not set")

        embeddings = self.embedding.embed(documents)
        self.vector_store.add(embeddings, documents, metadatas, ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> List[SearchResult]:
        """
        Search for relevant documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not set")

        query_embedding = self.embedding.embed_query(query)
        return self.vector_store.search(query_embedding, top_k, filter_metadata)


if __name__ == "__main__":
    # Example usage
    vectorizer = Vectorizer()
    vector_store = ChromaVectorStore()
    vectorizer.set_vector_store(vector_store)

    # Add documents
    docs = ["这是第一个文档", "这是第二个文档"]
    vectorizer.process_and_store(docs, ids=["1", "2"])

    # Search
    results = vectorizer.search("第一个")
    for result in results:
        print(f"Score: {result.score}, Content: {result.content}")
