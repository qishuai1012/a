"""
BM25 Cache Layer Module
Provides fast keyword-based retrieval and caching for common queries
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import pickle
import hashlib
from pathlib import Path

from rank_bm25 import BM25Okapi


@dataclass
class CacheEntry:
    """Represents a BM25 cache entry"""
    query: str
    answer: str
    tokens: List[str]
    score: float


class BM25Cache:
    """
    BM25-based cache for quick lookups of similar queries
    """

    def __init__(
        self,
        cache_file: Optional[str] = None,
        score_threshold: float = 0.5
    ):
        """
        Initialize BM25 cache

        Args:
            cache_file: Optional path to persist cache
            score_threshold: Minimum score to consider a cache hit
        """
        self.cache_file = Path(cache_file) if cache_file else None
        self.score_threshold = score_threshold

        # In-memory storage
        self.entries: List[CacheEntry] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25_index: Optional[BM25Okapi] = None

        # Load existing cache if provided
        if self.cache_file and self.cache_file.exists():
            self.load()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25
        Simple tokenization supporting Chinese and English
        """
        # For Chinese, split by characters
        # For English, split by whitespace
        tokens = []
        current_word = ""

        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                if current_word:
                    tokens.append(current_word.lower())
                    current_word = ""
                tokens.append(char)
            elif char.isalnum():
                current_word += char
            else:
                if current_word:
                    tokens.append(current_word.lower())
                    current_word = ""

        if current_word:
            tokens.append(current_word.lower())

        return tokens

    def add(self, query: str, answer: str, score: float = 1.0) -> None:
        """Add a query-answer pair to the cache"""
        tokens = self._tokenize(query)
        entry = CacheEntry(
            query=query,
            answer=answer,
            tokens=tokens,
            score=score
        )

        self.entries.append(entry)
        self.tokenized_corpus.append(tokens)

        # Rebuild BM25 index
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index"""
        if self.tokenized_corpus:
            self.bm25_index = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 1) -> Optional[CacheEntry]:
        """
        Search for similar queries in the cache

        Returns:
            Best matching CacheEntry if score exceeds threshold, None otherwise
        """
        if not self.entries or self.bm25_index is None:
            return None

        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)

        if len(scores) == 0:
            return None

        # Get best match
        best_idx = scores.argmax()
        best_score = scores[best_idx]

        if best_score >= self.score_threshold:
            return self.entries[best_idx]

        return None

    def search_with_results(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[CacheEntry, float]]:
        """
        Search and return multiple results with scores

        Returns:
            List of (CacheEntry, score) tuples
        """
        if not self.entries or self.bm25_index is None:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)

        results = []
        sorted_indices = scores.argsort()[::-1][:top_k]

        for idx in sorted_indices:
            if scores[idx] >= self.score_threshold:
                results.append((self.entries[idx], scores[idx]))

        return results

    def contains(self, query: str) -> bool:
        """Check if a similar query exists in the cache"""
        return self.search(query) is not None

    def get(self, query: str) -> Optional[str]:
        """Get cached answer for a query"""
        result = self.search(query)
        return result.answer if result else None

    def save(self) -> None:
        """Persist cache to disk"""
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.entries, f)

    def load(self) -> None:
        """Load cache from disk"""
        if self.cache_file and self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                self.entries = pickle.load(f)
            self.tokenized_corpus = [entry.tokens for entry in self.entries]
            self._rebuild_index()

    def clear(self) -> None:
        """Clear all cache entries"""
        self.entries = []
        self.tokenized_corpus = []
        self.bm25_index = None

    def __len__(self) -> int:
        return len(self.entries)


class BM25Layer:
    """
    BM25 retrieval layer for the QA system
    Combines caching with direct BM25 retrieval from documents
    """

    def __init__(
        self,
        cache_file: Optional[str] = None,
        score_threshold: float = 0.5,
        use_cache: bool = True
    ):
        self.cache = BM25Cache(cache_file, score_threshold)
        self.use_cache = use_cache

        # Document corpus for direct BM25 retrieval
        self.documents: List[str] = []
        self.doc_metadata: List[dict] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25_index: Optional[BM25Okapi] = None

    def add_to_cache(self, query: str, answer: str) -> None:
        """Add a query-answer pair to the cache"""
        self.cache.add(query, answer)

    def query_cache(self, query: str) -> Optional[str]:
        """Query the cache for an answer"""
        if not self.use_cache:
            return None
        return self.cache.get(query)

    def add_documents(self, documents: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """Add documents for BM25 retrieval"""
        for doc in documents:
            self.documents.append(doc)
            self.tokenized_docs.append(self.cache._tokenize(doc))

        if metadatas:
            self.doc_metadata.extend(metadatas)
        else:
            self.doc_metadata.extend([{} for _ in documents])

        # Rebuild index
        self.bm25_index = BM25Okapi(self.tokenized_docs)

    def search_documents(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, dict]]:
        """
        Search documents using BM25

        Returns:
            List of (document, score, metadata) tuples
        """
        if not self.documents or self.bm25_index is None:
            return []

        query_tokens = self.cache._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)

        results = []
        sorted_indices = scores.argsort()[::-1][:top_k]

        for idx in sorted_indices:
            if scores[idx] > 0:
                results.append((
                    self.documents[idx],
                    scores[idx],
                    self.doc_metadata[idx]
                ))

        return results

    def is_hit(self, query: str) -> bool:
        """Check if query results in a cache hit"""
        return self.cache.contains(query)

    def save_cache(self) -> None:
        """Persist cache to disk"""
        self.cache.save()

    def load_cache(self) -> None:
        """Load cache from disk"""
        self.cache.load()


if __name__ == "__main__":
    # Example usage
    bm25_layer = BM25Layer()

    # Add some cached Q&A pairs
    bm25_layer.add_to_cache("什么是 RAG?", "RAG 是检索增强生成 (Retrieval-Augmented Generation) 的缩写")
    bm25_layer.add_to_cache("如何安装依赖？", "运行 pip install -r requirements.txt")

    # Query cache
    result = bm25_layer.query_cache("什么是 RAG")
    print(f"Cache result: {result}")

    # Check for hit
    print(f"Is hit: {bm25_layer.is_hit('什么是 RAG')}")
