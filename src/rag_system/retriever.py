"""
RAG System - Retrieval Module
Handles hybrid search (vector + BM25) and re-ranking
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class RetrievalResult:
    """Represents a retrieval result with scores"""
    content: str
    vector_score: float
    bm25_score: float
    combined_score: float
    chunk_id: str
    metadata: dict


class HybridRetriever:
    """
    Hybrid retriever combining vector search and BM25
    """

    def __init__(
        self,
        vector_store=None,
        bm25_layer=None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever

        Args:
            vector_store: VectorStore instance for semantic search
            bm25_layer: BM25Layer instance for keyword search
            vector_weight: Weight for vector scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
        """
        self.vector_store = vector_store
        self.bm25_layer = bm25_layer
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid search

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        vector_results = []
        bm25_results = []

        # Get vector search results
        if self.vector_store:
            try:
                vector_results = self.vector_store.search(query, top_k=top_k * 2)
            except Exception:
                vector_results = []

        # Get BM25 results
        if self.bm25_layer:
            try:
                bm25_docs = self.bm25_layer.search_documents(query, top_k=top_k * 2)
                bm25_results = [
                    {"content": doc, "score": score, "metadata": meta}
                    for doc, score, meta in bm25_docs
                ]
            except Exception:
                bm25_results = []

        # Normalize and combine scores
        return self._combine_results(vector_results, bm25_results, top_k)

    def _normalize_scores(self, results: List[dict], score_key: str = "score") -> List[dict]:
        """Normalize scores to 0-1 range"""
        if not results:
            return results

        scores = [r.get(score_key, 0) for r in results]
        if not scores:
            return results

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            for r in results:
                r["normalized_score"] = 1.0 if max_score > 0 else 0.0
        else:
            for r in results:
                r["normalized_score"] = (r.get(score_key, 0) - min_score) / (max_score - min_score)

        return results

    def _combine_results(
        self,
        vector_results: List,
        bm25_results: List[dict],
        top_k: int
    ) -> List[RetrievalResult]:
        """Combine and re-rank results from both sources"""

        # Build result map by content
        result_map: Dict[str, dict] = {}

        # Add vector results
        for vr in vector_results:
            content = vr.content if hasattr(vr, 'content') else vr.get('content', '')
            result_map[content] = {
                "content": content,
                "vector_score": vr.score if hasattr(vr, 'score') else vr.get('score', 0),
                "bm25_score": 0,
                "chunk_id": vr.chunk_id if hasattr(vr, 'chunk_id') else vr.get('chunk_id', ''),
                "metadata": vr.metadata if hasattr(vr, 'metadata') else vr.get('metadata', {})
            }

        # Add BM25 results (merge if content already exists)
        for br in bm25_results:
            content = br.get('content', '')
            if content in result_map:
                result_map[content]["bm25_score"] = br.get('score', 0)
            else:
                result_map[content] = {
                    "content": content,
                    "vector_score": 0,
                    "bm25_score": br.get('score', 0),
                    "chunk_id": br.get('chunk_id', ''),
                    "metadata": br.get('metadata', {})
                }

        # Normalize scores
        vector_scores = [r["vector_score"] for r in result_map.values()]
        bm25_scores = [r["bm25_score"] for r in result_map.values()]

        # Simple normalization
        max_vector = max(vector_scores) if vector_scores else 1
        max_bm25 = max(bm25_scores) if bm25_scores else 1

        results = []
        for content, data in result_map.items():
            norm_vector = data["vector_score"] / max_vector if max_vector > 0 else 0
            norm_bm25 = data["bm25_score"] / max_bm25 if max_bm25 > 0 else 0

            combined = (
                self.vector_weight * norm_vector +
                self.bm25_weight * norm_bm25
            )

            results.append(RetrievalResult(
                content=data["content"],
                vector_score=data["vector_score"],
                bm25_score=data["bm25_score"],
                combined_score=combined,
                chunk_id=data["chunk_id"],
                metadata=data["metadata"]
            ))

        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)

        return results[:top_k]


class CrossEncoderRanker:
    """
    Re-ranker using cross-encoder model
    Provides more accurate re-ranking but slower
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder ranker

        Args:
            model_name: Name of the cross-encoder model
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("Please install sentence-transformers for re-ranking")

        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Re-rank results using cross-encoder

        Args:
            query: Original query
            results: List of RetrievalResult objects
            top_k: Number of results to return after re-ranking

        Returns:
            Re-ranked list of RetrievalResult objects
        """
        if not results:
            return []

        # Prepare pairs for cross-encoder
        pairs = [[query, r.content] for r in results]

        # Get cross-encoder scores
        ce_scores = self.model.predict(pairs)

        # Update results with new scores
        reranked_results = []
        for result, ce_score in zip(results, ce_scores):
            new_result = RetrievalResult(
                content=result.content,
                vector_score=result.vector_score,
                bm25_score=result.bm25_score,
                combined_score=ce_score,  # Use CE score as combined score
                chunk_id=result.chunk_id,
                metadata=result.metadata
            )
            reranked_results.append(new_result)

        # Sort by new combined score
        reranked_results.sort(key=lambda x: x.combined_score, reverse=True)

        return reranked_results[:top_k]


class SimpleRanker:
    """
    Simple rule-based ranker
    Faster alternative to cross-encoder
    """

    def __init__(self, length_penalty: float = 0.1):
        """
        Initialize simple ranker

        Args:
            length_penalty: Penalty for very short/long documents
        """
        self.length_penalty = length_penalty

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Re-rank using simple heuristics

        Args:
            query: Original query
            results: List of RetrievalResult objects
            top_k: Number of results to return

        Returns:
            Re-ranked list
        """
        if not results:
            return []

        query_length = len(query)
        reranked_results = []

        for result in results:
            doc_length = len(result.content)

            # Length penalty (prefer documents with similar length to query * factor)
            ideal_length = query_length * 5
            length_ratio = min(doc_length, ideal_length) / max(doc_length, ideal_length)

            # Adjusted score
            adjusted_score = result.combined_score * (0.8 + 0.2 * length_ratio)

            new_result = RetrievalResult(
                content=result.content,
                vector_score=result.vector_score,
                bm25_score=result.bm25_score,
                combined_score=adjusted_score,
                chunk_id=result.chunk_id,
                metadata=result.metadata
            )
            reranked_results.append(new_result)

        reranked_results.sort(key=lambda x: x.combined_score, reverse=True)

        return reranked_results[:top_k]


if __name__ == "__main__":
    # Example usage
    retriever = HybridRetriever()

    # Create mock results
    mock_results = [
        RetrievalResult("内容 A", 0.9, 0.3, 0.7, "chunk_1", {}),
        RetrievalResult("内容 B", 0.7, 0.8, 0.73, "chunk_2", {}),
        RetrievalResult("内容 C", 0.5, 0.5, 0.5, "chunk_3", {}),
    ]

    ranker = SimpleRanker()
    reranked = ranker.rerank("查询", mock_results, top_k=2)

    for r in reranked:
        print(f"{r.chunk_id}: {r.combined_score}")
