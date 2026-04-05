"""
BM25 Cache Layer Module
Provides fast keyword-based retrieval and caching for common queries
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import pickle                                             #将缓存保存到文件中
import hashlib                                            #哈希
from pathlib import Path                                  #跨平台解析文件路径

from rank_bm25 import BM25Okapi                           #NM25关键词检索算法

#定义一个缓存条目
@dataclass
class CacheEntry:
    """Represents a BM25 cache entry"""
    query: str           #用户提问
    answer: str          #系统给出答案
    tokens: List[str]    #分词后的列表
    score: float         #匹配分数

# BM25Cache（核心缓存类）
# 专门管 “问题 - 答案” 缓存
class BM25Cache:
    """
    BM25-based cache for quick lookups of similar queries
    """
    #1、初始化
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
        #1.处理缓存文件路径
        #如果串了路径字符串，就把他变成Path对象（方便跨平台处理文件）；否则就是None
        self.cache_file = Path(cache_file) if cache_file else None
        #2.设置及格线
        #只有相似度分数超过0.5的，才算“命中缓存”，否则就当没找到
        self.score_threshold = score_threshold

        # In-memory storage
        #内存存储区（相当于数据库的内存表）
        self.entries: List[CacheEntry] = []    #存放所有的缓存条目（包括问题和答案）
        self.tokenized_corpus: List[List[str]] = []  #存放所有问题切分后的词（给BM25用）

        #4.BM25索引引擎核心
        #初始为空，等有了数据菜户建立索引
        self.bm25_index: Optional[BM25Okapi] = None

        # Load existing cache if provided
        #5.启动时加载
        #如果指定了文件且文件存在，就自动把硬盘里的缓存读到内存里
        if self.cache_file and self.cache_file.exists():
            self.load()

    #2、分词器
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25
        Simple tokenization supporting Chinese and English
        """
        # For Chinese, split by characters
        # For English, split by whitespace
        tokens = []              #用来存切分好的词
        current_word = ""        #用来暂存正在读的英文单词

        #逐字遍历文本
        for char in text:
            #1.如果中文字符（Unicode范围判断）
            if '\u4e00' <= char <= '\u9fff':
                #如果前面正在拼写英文单词，先把他存起来（英文单词结束）
                if current_word:
                    tokens.append(current_word.lower())  #转小写存入
                    current_word = ""
                    #中文单独成词（比如苹果，分成‘苹’，‘果’）
                tokens.append(char)

            #如果是英文字母或数字
            elif char.isalnum():
            #先攒着，因为英文单词是由多个字母组成的
                current_word += char

            #如果是空格，或者标点符号

            else:
            #说明一个单词结束了，存起来
                if current_word:
                    tokens.append(current_word.lower())
                    current_word = ""
        #循环结束后，如果还没存完的英文单词，存进去
        if current_word:
            tokens.append(current_word.lower())

        return tokens

    #3、添加缓存
    def add(self, query: str, answer: str, score: float = 1.0) -> None:
        """Add a query-answer pair to the cache"""
        #1.把问题变成词列表
        tokens = self._tokenize(query)

        #2.打包成对象
        entry = CacheEntry(
            query=query,      # 原文："怎么买"
            answer=answer,    # 答案："点击购买按钮"
            tokens=tokens,    # 分词：["怎", "么", "买"]
            score=score       # 初始分数
        )

        #3.存入内存列表
        self.entries.append(entry)
        self.tokenized_corpus.append(tokens)      #BM25  需要单独的词列表

        # Rebuild BM25 index
        #重建索引
        #只要有新数据加入，旧的搜索规则就失效了，必须重新计算BM25的统计参数
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """重建 BM25 索引"""
        if self.tokenized_corpus:
         # BM25Okapi 会分析所有词的频率，生成一个搜索用的数学模型
            self.bm25_index = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 1) -> Optional[CacheEntry]:
        """
        Search for similar queries in the cache

        Returns:
            Best matching CacheEntry if score exceeds threshold, None otherwise
        """
          # 1. 安全检查：如果没有数据或索引，直接返回 None
        if not self.entries or self.bm25_index is None:
            return None

        # 2. 把用户的问题也分词
        query_tokens = self._tokenize(query)
        # 3. 核心计算！
        # get_scores 会拿用户的问题词，去和库里所有问题的词进行比对
        # 返回一个数组，比如 [0.1, 0.9, 0.05]，代表和库里第0、1、2条记录的相似度
        scores = self.bm25_index.get_scores(query_tokens)

        if len(scores) == 0:
            return None

        # 4. 找最高分
        best_idx = scores.argmax()     # 找到分数最高的那个位置（比如 1）
        best_score = scores[best_idx]  # 拿到那个分数（比如 0.9）
        # 5. 判断是否及格
        if best_score >= self.score_threshold:
            return self.entries[best_idx]   # 返回那条缓存记录

        return None # 分数太低，不算命中


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
        #打分，返回分数列表
        scores = self.bm25_index.get_scores(query_tokens)

        results = []
        # 1. 排序：argsort()是从小到大排，[::-1]是反转成从大到小，[:top_k]是取前5个
        sorted_indices = scores.argsort()[::-1][:top_k]

          # 2. 打包结果
        for idx in sorted_indices:
            if scores[idx] >= self.score_threshold:
              # 把 (缓存条目, 分数) 作为一个元组放进去
                results.append((self.entries[idx], scores[idx]))

        return results

    def contains(self, query: str) -> bool:
        """检查库里有没有类似的问题（直接调用 search）"""
        return self.search(query) is not None

    def get(self, query: str) -> Optional[str]:
        """直接获取答案字符串，如果没找到就返回 None"""
        result = self.search(query)
        return result.answer if result else None

    #持久化缓存
    def save(self) -> None:
        """把内存里的缓存存到硬盘上"""
        if self.cache_file:
             # 1. 确保文件夹存在（比如 data/ 目录）
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            # 2. 用 pickle 把 entries 列表（包含所有对象）写入文件
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.entries, f)

    def load(self) -> None:
        """Load cache from disk"""
        """从硬盘读取缓存到内存"""
        if self.cache_file and self.cache_file.exists():
              # 1. 读取文件
            with open(self.cache_file, 'rb') as f:
                self.entries = pickle.load(f)
            # 2. 重新构建词列表（因为 pickle 只存了对象，没存 BM25 索引对象）
            self.tokenized_corpus = [entry.tokens for entry in self.entries]
             # 3. 重新激活搜索引擎（这一步很重要，否则读进来也不能搜）
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
    BM25 retrieval layer for the QA system
    Combines caching with direct BM25 retrieval from documents
    """

    def __init__(
        self,
        cache_file: Optional[str] = None,
        score_threshold: float = 0.5,
        use_cache: bool = True
    ):
        # 1. 创建“快捷问答本”（缓存对象）
        # 这里直接实例化了刚才讲过的 BM25Cache 类
        self.cache = BM25Cache(cache_file, score_threshold)
        # 2. 开关：是否允许查缓存？
        # 如果设为 False，它每次都会强制去查大仓库，忽略缓存
        self.use_cache = use_cache

        # Document corpus for direct BM25 retrieval
        # 3. 准备“大仓库”的数据容器
        self.documents: List[str] = []          # 存原始文档文本（比如“产品说明书全文”）
        self.doc_metadata: List[dict] = []      # 存文档的元数据（比如 {"来源": "说明书.pdf"}）
        self.tokenized_docs: List[List[str]] = [] # 存切好词的文档（给 BM25 算法用的）
        # 4. 大仓库的搜索引擎
        # 初始为空，等存入文档后才会建立索引
        self.bm25_index: Optional[BM25Okapi] = None

    def add_to_cache(self, query: str, answer: str) -> None:
        """Add a query-answer pair to the cache"""
        """把一问一答存进‘快捷本’"""
        # 直接调用内部 cache 对象的 add 方法
        self.cache.add(query, answer)

    def query_cache(self, query: str) -> Optional[str]:
        """Query the cache for an answer"""
        """去‘快捷本’里找答案"""
        # 1. 检查开关：如果 use_cache 是 False，直接放弃，返回 None
        if not self.use_cache:
            return None

        # 2. 查缓存：调用内部 cache 对象的 get 方法
        # 如果找到了相似问题，就返回答案；没找到返回 None
        return self.cache.get(query)

    def add_documents(self, documents: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """把一堆文档扔进‘大仓库’，并建立索引"""

        # 1. 遍历所有传入的文档
        for doc in documents:
            # 把原文存起来
            self.documents.append(doc)

            # 把原文切分成词（调用 cache 里的分词器），存起来备用
            self.tokenized_docs.append(self.cache._tokenize(doc))

        # 2. 处理元数据（比如文件名、作者等）
        if metadatas:
            # 如果传了元数据，就加进去
            self.doc_metadata.extend(metadatas)
        else:
            # 如果没传，就补上空字典，保证数量和文档一一对应
            self.doc_metadata.extend([{} for _ in documents])

        # 3. 重建索引（核心！）
        # 文档变了，搜索引擎必须重新计算词频统计
        self.bm25_index = BM25Okapi(self.tokenized_docs)

    def search_documents(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, dict]]:
        """
        在‘大仓库’里搜文档
        返回：[(文档内容, 分数, 元数据), ...]
        """
        # 1. 安全检查：没文档或没索引，直接返回空
        if not self.documents or self.bm25_index is None:
            return []

        # 2. 把用户的问题切词
        query_tokens = self.cache._tokenize(query)

        # 3. 算分！
        # 拿着问题的词，去跟仓库里所有文档比对，算出相似度
        scores = self.bm25_index.get_scores(query_tokens)

        results = []
        # 4. 排序：从高到低排，取前 top_k 个（比如前5个）
        sorted_indices = scores.argsort()[::-1][:top_k]

        # 5. 打包结果
        for idx in sorted_indices:
            # 只要分数大于 0（说明沾点边），就加进结果列表
            if scores[idx] > 0:
                results.append((
                    self.documents[idx],      # 文档原文
                    scores[idx],              # 匹配分数
                    self.doc_metadata[idx]    # 元数据（如来源）
                ))

        return results

    def is_hit(self, query: str) -> bool:
        """检查‘快捷本’里有没有答案（不返回内容，只返回 True/False）"""
        return self.cache.contains(query)

    def save_cache(self) -> None:
        """把‘快捷本’存到硬盘上（持久化）"""
        self.cache.save()

    def load_cache(self) -> None:
        """从硬盘读取‘快捷本’（恢复记忆）"""
        self.cache.load()


if __name__ == "__main__":
    # Example usage  上岗准备
    bm25_layer = BM25Layer()

    # Add some cached Q&A pairs  岗前培训（填鸭式教学）
    bm25_layer.add_to_cache("什么是 RAG?", "RAG 是检索增强生成 (Retrieval-Augmented Generation) 的缩写")
    bm25_layer.add_to_cache("如何安装依赖？", "运行 pip install -r requirements.txt")

    # Query cache   模拟提问（查缓存）
    result = bm25_layer.query_cache("什么是 RAG")
    print(f"Cache result: {result}")

    # Check for hit  确认命中
    print(f"Is hit: {bm25_layer.is_hit('什么是 RAG')}")
