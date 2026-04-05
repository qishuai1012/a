"""
Enhanced BM25 Cache Layer Module
Provides fast keyword-based retrieval and caching for common queries with enterprise features
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import pickle                                             #将缓存保存到文件中
import hashlib                                            #哈希
from pathlib import Path                                  #跨平台解析文件路径
import asyncio
import structlog
import time
from functools import wraps

from rank_bm25 import BM25Okapi                           #NM25关键词检索算法
from .enhanced_cache import MultiLevelCache, CacheConfig, CacheDecorator

#定义一个缓存条目
@dataclass
class CacheEntry:
    """Represents a BM25 cache entry"""
    query: str           #用户提问
    answer: str          #系统给出答案
    tokens: List[str]    #分词后的列表
    score: float         #匹配分数
    created_at: float    #创建时间戳
    access_count: int    #访问次数
    last_accessed: float #最后访问时间


class BM25Cache:
    """
    BM25-based cache for quick lookups of similar queries with enterprise features
    """

    def __init__(
        self,
        cache_file: Optional[str] = None,
        score_threshold: float = 0.5,
        enable_multi_level_cache: bool = True,
        cache_config: Optional[CacheConfig] = None
    ):
        """
        Initialize BM25 cache

        Args:
            cache_file: Optional path to persist cache
            score_threshold: Minimum score to consider a cache hit
            enable_multi_level_cache: Enable multi-level caching
            cache_config: Configuration for multi-level cache
        """
        #处理缓存文件路径
        self.cache_file = Path(cache_file) if cache_file else None
        self.score_threshold = score_threshold

        # In-memory storage
        self.entries: List[CacheEntry] = []
        self.tokenized_corpus: List[List[str]] = []

        # BM25索引引擎核心
        self.bm25_index: Optional[BM25Okapi] = None

        # Multi-level cache support
        self.enable_multi_level_cache = enable_multi_level_cache
        self.cache_config = cache_config or CacheConfig()

        if self.enable_multi_level_cache:
            self.multi_cache = MultiLevelCache(self.cache_config)

        # Initialize logger
        self.logger = structlog.get_logger(__name__)

        # Load existing cache if provided
        if self.cache_file and self.cache_file.exists():
            self.load()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25
        Simple tokenization supporting Chinese and English
        """
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
            score=score,
            created_at=time.time(),
            access_count=0,
            last_accessed=time.time()
        )

        self.entries.append(entry)
        self.tokenized_corpus.append(tokens)

        # Rebuild BM25 index
        self._rebuild_index()

        # Update multi-level cache if enabled
        if self.enable_multi_level_cache:
            cache_key = f"bm25_direct:{hashlib.sha256(query.encode()).hexdigest()}"
            self.multi_cache.set(cache_key, entry)

    def _rebuild_index(self) -> None:
        """重建 BM25 索引"""
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

        best_idx = scores.argmax()
        best_score = scores[best_idx]

        if best_score >= self.score_threshold:
            # Update access statistics
            entry = self.entries[best_idx]
            entry.access_count += 1
            entry.last_accessed = time.time()

            return entry

        return None

    def search_with_results(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[CacheEntry, float]]:
        """
        搜索并返回前 K 个相似的结果（带分数）
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
                # Update access statistics
                entry = self.entries[idx]
                entry.access_count += 1
                entry.last_accessed = time.time()

                results.append((entry, scores[idx]))

        return results

    def contains(self, query: str) -> bool:
        """检查库里有没有类似的问题（直接调用 search）"""
        return self.search(query) is not None

    def get(self, query: str) -> Optional[str]:
        """直接获取答案字符串，如果没找到就返回 None"""
        result = self.search(query)
        return result.answer if result else None

    def save(self) -> None:
        """把内存里的缓存存到硬盘上"""
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.entries, f)

    def load(self) -> None:
        """Load cache from disk"""
        """从硬盘读取缓存到内存"""
        if self.cache_file and self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                self.entries = pickle.load(f)
            self.tokenized_corpus = [entry.tokens for entry in self.entries]
            self._rebuild_index()

    def clear(self) -> None:
        """清空所有数据"""
        self.entries = []
        self.tokenized_corpus = []
        self.bm25_index = None

    def __len__(self) -> int:
        """支持 len(cache) 语法，返回缓存条目数量"""
        return len(self.entries)


class BM25Layer:
    """
    Enhanced BM25 retrieval layer for the QA system
    Combines caching with direct BM25 retrieval from documents
    """

    def __init__(
        self,
        cache_file: Optional[str] = None,
        score_threshold: float = 0.5,
        use_cache: bool = True,
        enable_multi_level_cache: bool = True,
        cache_config: Optional[CacheConfig] = None
    ):
        # 创建"快捷问答本"（缓存对象）
        self.cache = BM25Cache(
            cache_file,
            score_threshold,
            enable_multi_level_cache,
            cache_config
        )
        # 是否允许查缓存？
        self.use_cache = use_cache

        # Document corpus for direct BM25 retrieval
        self.documents: List[str] = []
        self.doc_metadata: List[dict] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25_index: Optional[BM25Okapi] = None

        # Performance metrics
        self.hit_count = 0
        self.miss_count = 0

        # Initialize logger
        self.logger = structlog.get_logger(__name__)

    def add_to_cache(self, query: str, answer: str) -> None:
        """Add a query-answer pair to the cache"""
        self.cache.add(query, answer)

    @CacheDecorator(None)  # Will be set during initialization
    def query_cache(self, query: str) -> Optional[str]:
        """Query the cache for an answer"""
        if not self.use_cache:
            return None

        # Try multi-level cache first if enabled
        if self.cache.enable_multi_level_cache:
            cached_result = self.cache.multi_cache.get(f"bm25_query:{query}")
            if cached_result:
                self.hit_count += 1
                self.logger.info("Cache hit from multi-level cache", query=query)
                return cached_result

        # Check internal cache
        result = self.cache.get(query)
        if result:
            self.hit_count += 1
            self.logger.info("Cache hit from internal cache", query=query)

            # Update multi-level cache if enabled
            if self.cache.enable_multi_level_cache:
                self.cache.multi_cache.set(f"bm25_query:{query}", result)

            return result

        self.miss_count += 1
        self.logger.info("Cache miss", query=query)
        return None

    def add_documents(self, documents: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """把一堆文档扔进'大仓库'，并建立索引"""

        for doc in documents:
            self.documents.append(doc)
            self.tokenized_docs.append(self.cache._tokenize(doc))

        if metadatas:
            self.doc_metadata.extend(metadatas)
        else:
            self.doc_metadata.extend([{} for _ in documents])

        self.bm25_index = BM25Okapi(self.tokenized_docs)

    def search_documents(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, dict]]:
        """
        在'大仓库'里搜文档
        Returns: [(文档内容, 分数, 元数据), ...]
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
        """检查'快捷本'里有没有答案（不返回内容，只返回 True/False）"""
        return self.cache.contains(query)

    def save_cache(self) -> None:
        """把'快捷本'存到硬盘上（持久化）"""
        self.cache.save()

    def load_cache(self) -> None:
        """从硬盘读取'快捷本'（恢复记忆）"""
        self.cache.load()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0

        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate
        }


if __name__ == "__main__":
    # Example usage
    import time

    cache_config = CacheConfig(
        lru_maxsize=500,
        ttl_seconds=1800,
        persistent_path="./data/test_enhanced_cache.pkl"
    )

    bm25_layer = BM25Layer(cache_file="./data/test_cache.pkl", cache_config=cache_config)

    # Add some cached Q&A pairs
    bm25_layer.add_to_cache("什么是 RAG?", "RAG 是检索增强生成 (Retrieval-Augmented Generation) 的缩写")
    bm25_layer.add_to_cache("如何安装依赖？", "运行 pip install -r requirements.txt")

    # Query cache
    start_time = time.time()
    result = bm25_layer.query_cache("什么是 RAG")
    query_time = time.time() - start_time
    print(f"Cache result: {result}")
    print(f"Query time: {query_time:.4f}s")

    # Check for hit
    print(f"Is hit: {bm25_layer.is_hit('什么是 RAG')}")

    # Print cache stats
    stats = bm25_layer.get_cache_stats()
    print(f"Cache stats: {stats}")