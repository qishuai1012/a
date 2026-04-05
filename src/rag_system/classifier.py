"""
RAG System - Query Classification Module
Classifies queries as general knowledge or professional/domain-specific
"""

from enum import Enum      # 状态定义器，枚举
from typing import Optional, List, Tuple  # 类型提示工具箱
import re  # 正则表达式模块

# 问题分类器
class QueryType(Enum):
    """查询类型枚举"""
    GENERAL = "general"      # 通用知识，可以直接由大语言模型回答
    PROFESSIONAL = "professional"  # 特定领域问题，需要进行检索


class QueryClassifier:
    """
    Classifies user queries into general knowledge or professional/domain-specific

    Uses a combination of:
    - Keyword matching
    - Query pattern analysis
    - Optional LLM-based classification
    """

    def __init__(
        self,
        professional_keywords: Optional[List[str]] = None,
        general_patterns: Optional[List[str]] = None
    ):
        """
        Initialize the classifier

        Args:
            professional_keywords: Keywords that indicate professional queries
            general_patterns: Regex patterns that indicate general queries
        """
        # 默认专业关键词列表（可以根据领域定制）
        self.professional_keywords = professional_keywords or [
            # 公司/产品相关
            "公司", "产品", "服务", "客户", "项目",
            # 技术术语
            "架构", "部署", "配置", "实现", "源码",
            # 领域特定词汇
            "政策", "流程", "规范", "标准", "协议",
        ]

        # 通用查询模式列表
        self.general_patterns = general_patterns or [
            r"^什么 (是 | 意思).*$",      # 什么 是... / 什么 意思...
            r"^怎么 (样 | 做| 弄).*$",     # 怎么 样 / 怎么 做 / 怎么 弄...
            r"^为什么.*$",              # 为什么...
            r"^谁.*$",                 # 谁...
            r"^何时.*$",               # 何时...
            r"^哪里.*$",               # 哪里...
            r"^多少.*$",               # 多少...
            r"^请 (问 | 解释).*$",        # 请 问 / 请 解释...
            r"^介绍一下.*$",            # 介绍一下...
            r"^简述.*$",              # 简述...
        ]

        # 编译正则表达式模式，提高匹配效率
        self._compiled_patterns = [
            re.compile(p) for p in self.general_patterns
        ]

    def classify(self, query: str) -> QueryType:
        """
        Classify a query

        Args:
            query: The user's query text

        Returns:
            QueryType enum value
        """
        # 将查询转为小写，便于匹配
        query_lower = query.lower()

        # 首先检查专业关键词
        for keyword in self.professional_keywords:
            if keyword.lower() in query_lower:
                # 额外检查：如果查询非常短，可能仍然是通用的
                if len(query) > 10:  # 如果查询长度大于10个字符
                    return QueryType.PROFESSIONAL

        # 检查通用模式
        for pattern in self._compiled_patterns:
            if pattern.match(query):  # 使用编译后的模式进行匹配
                # 如果查询匹配通用模式但包含专业关键词，则仍为专业查询
                for keyword in self.professional_keywords:
                    if keyword.lower() in query_lower:
                        return QueryType.PROFESSIONAL
                return QueryType.GENERAL

        # 默认策略：对于较长、具体化的查询，默认为专业查询
        if len(query) > 20:  # 如果查询长度超过20个字符
            return QueryType.PROFESSIONAL

        # 默认为通用查询（适用于短查询）
        return QueryType.GENERAL

    def classify_with_confidence(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify query with confidence score

        Returns:
            Tuple of (QueryType, confidence between 0-1)
        """
        # 将查询转为小写
        query_lower = query.lower()

        professional_score = 0  # 专业查询得分
        general_score = 0       # 通用查询得分

        # 统计专业关键词出现次数
        for keyword in self.professional_keywords:
            if keyword.lower() in query_lower:
                professional_score += 1

        # 统计通用模式匹配次数
        for pattern in self._compiled_patterns:
            if pattern.match(query):
                general_score += 1

        # 计算置信度
        total = professional_score + general_score
        if total == 0:  # 如果没有任何明确信号
            # 使用长度启发式：较长的查询更可能是专业查询
            if len(query) > 20:
                return QueryType.PROFESSIONAL, 0.5  # 返回专业查询，置信度0.5
            return QueryType.GENERAL, 0.5  # 返回通用查询，置信度0.5

        # 比较专业和通用得分
        if professional_score > general_score:
            # 专业得分更高，计算置信度（基础0.5 + 差值*0.1，上限1.0）
            confidence = min(1.0, 0.5 + (professional_score - general_score) * 0.1)
            return QueryType.PROFESSIONAL, confidence
        elif general_score > professional_score:
            # 通用得分更高，计算置信度
            confidence = min(1.0, 0.5 + (general_score - professional_score) * 0.1)
            return QueryType.GENERAL, confidence
        else:
            # 得分相等，返回通用查询
            return QueryType.GENERAL, 0.5

    def add_professional_keyword(self, keyword: str) -> None:
        """添加专业关键词"""
        # 避免重复添加
        if keyword not in self.professional_keywords:
            self.professional_keywords.append(keyword)

    def add_general_pattern(self, pattern: str) -> None:
        """添加通用查询模式"""
        # 避免重复添加
        if pattern not in self.general_patterns:
            self.general_patterns.append(pattern)
            # 同时添加编译后的模式到缓存列表
            self._compiled_patterns.append(re.compile(pattern))


class SimpleClassifier(QueryClassifier):
    """
    Simplified classifier using only keyword matching
    Faster but less accurate
    简化版分类器，仅使用关键词匹配
    更快但准确性较低
    """

    def __init__(self, keywords: Optional[List[str]] = None):
        # 调用父类构造函数，只传入专业关键词
        super().__init__(professional_keywords=keywords)

    def classify(self, query: str) -> QueryType:
        # 将查询转为小写
        query_lower = query.lower()

        # 检查是否包含专业关键词
        for keyword in self.professional_keywords:
            if keyword.lower() in query_lower:
                return QueryType.PROFESSIONAL  # 包含关键词则为专业查询

        return QueryType.GENERAL  # 不包含关键词则为通用查询


if __name__ == "__main__":
    # 示例用法
    classifier = QueryClassifier()

    test_queries = [
        "什么是 RAG?",              # 通用知识问题
        "公司的产品架构是什么？",    # 包含专业关键词"架构"的查询
        "请介绍一下你们的服务",     # 通用介绍请求，但包含"服务"关键词
        "如何配置部署环境？",       # 包含专业关键词"配置"和"部署"的查询
        "今天天气怎么样",           # 纯通用查询
    ]

    # 测试每个查询
    for query in test_queries:
        result = classifier.classify(query)
        print(f"'{query}' -> {result.value}")  # 打印查询和分类结果