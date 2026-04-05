"""
Document Processing and Text Chunking Module
Handles document cleaning, preprocessing, and splitting into chunks
"""
# 导入必要的库
from abc import ABC, abstractmethod          # 抽象基类，提供子类继承重写接口
from dataclasses import dataclass            # 用于创建简单的类，自动生成 __init__, __repr__ 等方法
from typing import List, Optional            # 类型标识，帮助IDE进行类型检查
import re                                    # 正则表达式库，用于清洗文本


@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    content: str                             # 文本块的实际内容
    chunk_id: str                            # 文本块的唯一ID
    parent_document: str                     # 原始文档的ID
    chunk_index: int                         # 在父文档中的索引位置
    metadata: dict                           # 额外的元数据（如起始位置、结束位置、长度等）


class BaseChunker(ABC):
    """Base class for text chunking strategies"""

    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> List[TextChunk]:
        """Split text into chunks"""
        # 抽象方法，强制子类必须实现此方法
        pass


class FixedSizeChunker(BaseChunker):
    """
    Split text into fixed-size chunks with overlap
    固定大小分块器：按照固定字符数切分文本，允许重叠
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        初始化固定大小分块器
        :param chunk_size: 每个块的大小（字符数）
        :param chunk_overlap: 块之间的重叠字符数
        """
        self.chunk_size = chunk_size         # 每个文本块的大小
        self.chunk_overlap = chunk_overlap   # 块之间的重叠部分

    def chunk(self, text: str, doc_id: str) -> List[TextChunk]:
        """
        将文本分割成固定大小的块
        :param text: 要分割的文本
        :param doc_id: 文档ID
        :return: TextChunk 对象列表
        """
        chunks = []                          # 存储所有文本块
        start = 0                            # 当前块的起始位置
        chunk_index = 0                      # 块的索引

        # 当起始位置小于文本总长度时，继续分割
        while start < len(text):
            # 计算当前块的结束位置
            end = start + self.chunk_size
            # 提取当前块的文本内容
            chunk_text = text[start:end]

            # 尝试在句子边界处分割，避免切断句子
            if end < len(text):  # 如果不是最后一个块
                # 查找最近的句号或换行符
                last_period = chunk_text.rfind(".")      # 查找最后的句号
                last_newline = chunk_text.rfind("\n")    # 查找最后的换行符
                # 选择较靠后的分割点
                break_point = max(last_period, last_newline)
                # 如果分割点在块的中间之后，才进行调整
                if break_point > self.chunk_size // 2:
                    chunk_text = chunk_text[:break_point + 1]  # 截取到句号或换行符
                    end = start + break_point + 1              # 更新结束位置

            # 创建文本块对象并添加到列表中
            chunks.append(TextChunk(
                content=chunk_text.strip(),  # 去除首尾空白字符
                chunk_id=f"{doc_id}_chunk_{chunk_index}",  # 生成唯一的块ID
                parent_document=doc_id,       # 父文档ID
                chunk_index=chunk_index,      # 块索引
                metadata={                    # 元数据信息
                    "start": start,          # 块在原文中的起始位置
                    "end": end,              # 块在原文中的结束位置
                    "length": len(chunk_text) # 块的字符长度
                }
            ))

            # 更新下一个块的起始位置（考虑重叠）
            start = end - self.chunk_overlap
            chunk_index += 1                 # 增加块索引

        return chunks


class RecursiveCharacterChunker(BaseChunker):
    """
    Split text recursively by character separators
    Preferred for maintaining semantic coherence
    递归字符分块器：按照特定分隔符递归分割，更适合保持语义连贯性
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        初始化递归字符分块器
        :param separators: 分隔符列表，按优先级排序
        :param chunk_size: 每个块的最大大小
        :param chunk_overlap: 块之间的重叠大小
        """
        # 设置默认的分隔符列表，按优先级排序（从粗粒度到细粒度）
        self.separators = separators or [
            "\n\n",      # 段落分隔符
            "\n",        # 行分隔符
            "。",        # 中文句号
            ".",         # 英文句号
            "!",         # 感叹号
            "?",         # 问号
            "；",        # 中文分号
            ";",         # 英文分号
            " ",         # 空格
            ""           # 字符级别（当其他分隔符都不匹配时）
        ]
        self.chunk_size = chunk_size         # 每个块的最大大小
        self.chunk_overlap = chunk_overlap   # 块之间的重叠大小

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        使用给定的分隔符分割文本
        :param text: 要分割的文本
        :param separators: 分隔符列表
        :return: 分割后的文本片段列表
        """
        # 如果没有分隔符了，直接返回原文本
        if not separators:
            return [text]

        # 取第一个分隔符（优先级最高的）
        separator = separators[0]
        # 剩余的分隔符列表
        rest_separators = separators[1:]

        if not separator:  # 如果是空字符串分隔符
            # 按字符计数分割（最细粒度）
            splits = []
            for i in range(0, len(text), self.chunk_size):
                splits.append(text[i:i + self.chunk_size])
            return splits

        # 使用当前分隔符分割文本
        splits = text.split(separator)

        # 如果分割后的片段仍然太大，递归使用下一级分隔符
        final_splits = []
        for split in splits:
            if len(split) > self.chunk_size:
                # 如果片段太大，用剩余的分隔符继续分割
                final_splits.extend(self._split_text(split, rest_separators))
            else:
                # 如果片段合适，直接添加
                final_splits.append(split)

        return final_splits

    def chunk(self, text: str, doc_id: str) -> List[TextChunk]:
        """
        将文本分割成块（使用递归字符分割）
        :param text: 要分割的文本
        :param doc_id: 文档ID
        :return: TextChunk 对象列表
        """
        # 首先使用递归方法分割文本
        raw_chunks = self._split_text(text, self.separators)

        # 合并小块并应用重叠
        chunks = []                          # 存储最终的文本块
        chunk_index = 0                      # 块索引
        current_chunk = ""                   # 当前正在构建的块

        # 遍历所有分割后的片段
        for raw_chunk in raw_chunks:
            raw_chunk = raw_chunk.strip()    # 去除首尾空白
            if not raw_chunk:                # 如果是空字符串，跳过
                continue

            # 如果当前块加上新片段不超过大小限制
            if len(current_chunk) + len(raw_chunk) <= self.chunk_size:
                current_chunk += raw_chunk   # 将新片段加入当前块
            else:
                # 如果超过大小限制
                if current_chunk:            # 如果当前块不为空
                    # 创建当前块的TextChunk对象
                    chunks.append(TextChunk(
                        content=current_chunk,
                        chunk_id=f"{doc_id}_chunk_{chunk_index}",
                        parent_document=doc_id,
                        chunk_index=chunk_index,
                        metadata={"length": len(current_chunk)}  # 记录长度
                    ))
                    chunk_index += 1         # 增加块索引
                    # 注意：这里没有应用重叠，与FixedSizeChunker不同
                current_chunk = raw_chunk    # 开始新的块

        # 处理最后一个块（循环结束后剩余的部分）
        if current_chunk:
            chunks.append(TextChunk(
                content=current_chunk,
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                parent_document=doc_id,
                chunk_index=chunk_index,
                metadata={"length": len(current_chunk)}  # 记录长度
            ))

        return chunks


class DocumentProcessor:
    """
    Main document processor that handles cleaning and chunking
    文档处理器主类：负责文本清洗和分块
    """

    def __init__(
        self,
        chunker: Optional[BaseChunker] = None,
        remove_extra_whitespace: bool = True,
        remove_special_characters: bool = False
    ):
        """
        初始化文档处理器
        :param chunker: 分块器实例，默认使用递归字符分块器
        :param remove_extra_whitespace: 是否移除多余空白字符
        :param remove_special_characters: 是否移除特殊字符
        """
        # 如果没有提供分块器，使用默认的递归字符分块器
        self.chunker = chunker or RecursiveCharacterChunker()
        # 是否移除多余空白字符
        self.remove_extra_whitespace = remove_extra_whitespace
        # 是否移除特殊字符
        self.remove_special_characters = remove_special_characters

    def clean(self, text: str) -> str:
        """
        清洗和预处理文本
        :param text: 原始文本
        :return: 清洗后的文本
        """
        if self.remove_extra_whitespace:
            # 使用正则表达式将多个空白字符替换为单个空格
            text = re.sub(r'\s+', ' ', text)
            # 按行分割，去除每行的首尾空白，再重新组合
            lines = [line.strip() for line in text.split('\n')]
            text = '\n'.join(lines)

        if self.remove_special_characters:
            # 移除特殊字符，但保留字母、数字、中文字符、常用标点
            text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()""\'\-]', '', text)

        # 去除首尾空白并返回
        return text.strip()

    def process(self, content: str, doc_id: str) -> List[TextChunk]:
        """
        处理文档内容并返回块
        :param content: 文档内容
        :param doc_id: 文档ID
        :return: TextChunk 对象列表
        """
        # 首先清洗文本
        cleaned_text = self.clean(content)
        # 然后使用分块器分割文本
        chunks = self.chunker.chunk(cleaned_text, doc_id)
        return chunks

    def process_batch(self, documents: List[tuple]) -> List[TextChunk]:
        """
        批量处理多个文档
        :param documents: (content, doc_id) 元组列表
        :return: 所有文档的 TextChunk 列表
        """
        all_chunks = []                      # 存储所有文档的所有块
        # 遍历每个文档
        for content, doc_id in documents:
            # 处理单个文档
            chunks = self.process(content, doc_id)
            # 将该文档的块添加到总列表中
            all_chunks.extend(chunks)
        return all_chunks


if __name__ == "__main__":
    # 示例用法
    # 创建文档处理器实例（使用默认设置）
    processor = DocumentProcessor()
    # 示例文本
    sample_text = "这是第一段。\n\n这是第二段。\n这是第二段的继续。"
    # 处理文本，获得文本块
    chunks = processor.process(sample_text, "doc_001")
    # 打印每个块的信息（只显示前50个字符）
    for chunk in chunks:
        print(f"Chunk {chunk.chunk_id}: {chunk.content[:50]}...")