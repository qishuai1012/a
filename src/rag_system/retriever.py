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
    content: str                #检索到的文本内容
    vector_score: float         #向量相似度得分（0.0-1.0），表示语义相关性
    bm25_score: float           #BM25 算法得分，表示关键词匹配程度
    combined_score: float       #综合得分（向量和 BM25 的加权平均）
    chunk_id: str               #文本块的唯一标识符，用于追踪来源
    metadata: dict              #额外元数据，如文档来源、页码、创建时间等

# HybridRetriever（最核心！混合检索）
# 功能：
# 向量检索（懂语义）
# BM25 关键词检索（精准匹配）
# 加权融合分数
# 去重 + 排序
# 返回最相关的 top-k
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
        self.vector_store = vector_store     # 存储向量数据库实例
        self.bm25_layer = bm25_layer         # 存储 BM25 搜索引擎实例
        self.vector_weight = vector_weight   # 向量得分权重（默认 0.7，偏重语义）
        self.bm25_weight = bm25_weight       # BM25 得分权重（默认 0.3，偏重关键词）

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
         # 初始化向量和 BM25 结果列表
        vector_results = []
        bm25_results = []

        # Get vector search results
        # 从向量数据库获取语义相似的文档
        if self.vector_store:
            try:
                # 搜索 top_k * 2，为了后续融合时有足够的候选池
                vector_results = self.vector_store.search(query, top_k=top_k * 2)
            except Exception:
                # 如果向量检索失败，返回空列表，不影响整个系统
                vector_results = []

        # Get BM25 results
        # 从 BM25 搜索引擎获取关键词匹配的文档
        if self.bm25_layer:
            try:
                # 获取 BM25 搜索结果
                bm25_docs = self.bm25_layer.search_documents(query, top_k=top_k * 2)
                # 将 BM25 结果标准化为字典格式，方便后续处理
                bm25_results = [
                    {"content": doc, "score": score, "metadata": meta}
                    for doc, score, meta in bm25_docs
                ]
            except Exception:
                # 如果 BM25 检索失败，返回空列表
                bm25_results = []

        # Normalize and combine scores
         # 合并并排序来自两个系统的检索结果
        return self._combine_results(vector_results, bm25_results, top_k)

    def _normalize_scores(self, results: List[dict], score_key: str = "score") -> List[dict]:
        """Normalize scores to 0-1 range"""
        # 如果结果为空，直接返回
        if not results:
            return results

        # 提取所有分数值
        scores = [r.get(score_key, 0) for r in results]
        # 如果没有分数，直接返回
        if not scores:
            return results

        # 找出最小和最大分数
        min_score = min(scores)
        max_score = max(scores)

        # 如果最大值等于最小值，说明所有分数一样，避免除以零
        if max_score == min_score:
            for r in results:
                # 如果最大分数大于 0，所有归一化分数设为 1.0，否则设为 0.0
                r["normalized_score"] = 1.0 if max_score > 0 else 0.0
        else:
            # 使用最小-最大归一化公式：(x - min) / (max - min)
            for r in results:
                r["normalized_score"] = (r.get(score_key, 0) - min_score) / (max_score - min_score)

        return results

    #合并分数
    def _combine_results(
        self,
        vector_results: List,
        bm25_results: List[dict],
        top_k: int
    ) -> List[RetrievalResult]:
        """Combine and re-rank results from both sources"""

        # 使用字典以内容为键，避免重复文档
        result_map: Dict[str, dict] = {}

        # 处理向量检索结果
        for vr in vector_results:
            # 动态获取内容字段（可能是属性或字典键）
            content = vr.content if hasattr(vr, 'content') else vr.get('content', '')
            # 将向量结果存入字典，BM25 分数初始化为 0
            result_map[content] = {
                "content": content,
                "vector_score": vr.score if hasattr(vr, 'score') else vr.get('score', 0),
                "bm25_score": 0,
                "chunk_id": vr.chunk_id if hasattr(vr, 'chunk_id') else vr.get('chunk_id', ''),
                "metadata": vr.metadata if hasattr(vr, 'metadata') else vr.get('metadata', {})
            }

        # 处理 BM25 检索结果（合并或新增）
        for br in bm25_results:
            content = br.get('content', '')
            if content in result_map:
                # 如果向量搜索也找到了这段内容，更新 BM25 分数
                result_map[content]["bm25_score"] = br.get('score', 0)
            else:
                # 如果向量搜索没找到，新增条目
                result_map[content] = {
                    "content": content,
                    "vector_score": 0,
                    "bm25_score": br.get('score', 0),
                    "chunk_id": br.get('chunk_id', ''),
                    "metadata": br.get('metadata', {})
                }

        # 提取向量和 BM25 分数，用于归一化
        vector_scores = [r["vector_score"] for r in result_map.values()]
        bm25_scores = [r["bm25_score"] for r in result_map.values()]

        # 找出各自的最大值用于归一化（防止除以零）
        max_vector = max(vector_scores) if vector_scores else 1
        max_bm25 = max(bm25_scores) if bm25_scores else 1

        # 存储最终结果
        results = []
        # 遍历合并后的结果，计算综合得分
        for content, data in result_map.items():
            # 使用最大值归一化：当前分数 / 最大分数
            norm_vector = data["vector_score"] / max_vector if max_vector > 0 else 0
            norm_bm25 = data["bm25_score"] / max_bm25 if max_bm25 > 0 else 0

            # 加权求和得到综合得分
            combined = (
                self.vector_weight * norm_vector +  # 向量得分 * 权重
                self.bm25_weight * norm_bm25       # BM25 得分 * 权重
            )

            # 创建 RetrievalResult 对象并添加到结果列表
            results.append(RetrievalResult(
                content=data["content"],
                vector_score=data["vector_score"],
                bm25_score=data["bm25_score"],
                combined_score=combined,
                chunk_id=data["chunk_id"],
                metadata=data["metadata"]
            ))

        # 按综合得分降序排列
        results.sort(key=lambda x: x.combined_score, reverse=True)

        # 返回前 top_k 个结果
        return results[:top_k]

#CrossEncoderRanker（精排模型）
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
            # 尝试导入 sentence-transformers 库
            from sentence_transformers import CrossEncoder
        except ImportError:
            # 如果没有安装，抛出友好错误
            raise ImportError("Please install sentence-transformers for re-ranking")

        # 加载预训练的交叉编码器模型
        self.model = CrossEncoder(model_name)
        # 记录模型名称
        self.model_name = model_name

    #重排序器
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
        # 如果没有结果，直接返回空列表
        if not results:
            return []

        # 准备成对数据：[[query, doc1], [query, doc2], ...]
        # 这是交叉编码器模型的输入格式
        pairs = [[query, r.content] for r in results]

        # 使用模型预测每一对的相关性分数
        ce_scores = self.model.predict(pairs)

        # 更新结果对象，用交叉编码器的分数替换原有的综合分数
        reranked_results = []
        for result, ce_score in zip(results, ce_scores):  # zip 将原结果和新分数配对
            # 创建新的结果对象，保留原有信息，更新综合得分
            new_result = RetrievalResult(
                content=result.content,
                vector_score=result.vector_score,
                bm25_score=result.bm25_score,
                combined_score=ce_score,  # 使用交叉编码器的新分数
                chunk_id=result.chunk_id,
                metadata=result.metadata
            )
            reranked_results.append(new_result)

        # 按新的综合得分重新排序（降序）
        reranked_results.sort(key=lambda x: x.combined_score, reverse=True)

        # 返回前 top_k 个最相关的结果
        return reranked_results[:top_k]

# SimpleRanker（轻量排序）
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
        # 长度惩罚参数，控制对文档长度的偏好程度
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
        # 如果没有结果，直接返回空列表
        if not results:
            return []

        # 计算查询长度，用于确定理想的文档长度
        query_length = len(query)
        # 存储重排序后的结果
        reranked_results = []

        # 遍历每个检索结果
        for result in results:
            # 获取文档长度
            doc_length = len(result.content)

            # 长度惩罚逻辑：假设理想文档长度是查询长度的 5 倍
            ideal_length = query_length * 5
            # 计算长度比率：越接近理想长度，ratio 越接近 1
            length_ratio = min(doc_length, ideal_length) / max(doc_length, ideal_length)

            # 调整分数：原始分 * (0.8 + 0.2 * 长度因子)
            # 如果长度完美匹配，乘数接近 1.0；如果长度差异巨大，乘数接近 0.8
            adjusted_score = result.combined_score * (0.8 + 0.2 * length_ratio)

            # 创建新的结果对象，更新分数
            new_result = RetrievalResult(
                content=result.content,
                vector_score=result.vector_score,
                bm25_score=result.bm25_score,
                combined_score=adjusted_score,  # 使用调整后的分数
                chunk_id=result.chunk_id,
                metadata=result.metadata
            )
            reranked_results.append(new_result)

        # 按调整后的综合得分重新排序（降序）
        reranked_results.sort(key=lambda x: x.combined_score, reverse=True)

        # 返回前 top_k 个结果
        return reranked_results[:top_k]


if __name__ == "__main__":
    # Example usage
    # 创建混合检索器实例
    retriever = HybridRetriever()

    # 创建模拟的检索结果
    mock_results = [
        RetrievalResult("内容 A", 0.9, 0.3, 0.7, "chunk_1", {}),  # 向量分高，BM25分低
        RetrievalResult("内容 B", 0.7, 0.8, 0.73, "chunk_2", {}), # 向量分和BM25分都较高
        RetrievalResult("内容 C", 0.5, 0.5, 0.5, "chunk_3", {}),  # 两项分数均衡
    ]

    # 创建简单重排序器
    ranker = SimpleRanker()
    # 对模拟结果进行重排序，只返回前 2 个
    reranked = ranker.rerank("查询", mock_results, top_k=2)

    # 打印重排序后的结果
    for r in reranked:
        print(f"{r.chunk_id}: {r.combined_score}")