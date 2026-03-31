"""
RAG System - Query Classification Module
Classifies queries as general knowledge or professional/domain-specific
"""

from enum import Enum
from typing import Optional, List, Tuple
import re


class QueryType(Enum):
    """Types of queries"""
    GENERAL = "general"      # General knowledge, can be answered by LLM directly
    PROFESSIONAL = "professional"  # Domain-specific, requires retrieval


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
        # Default professional keywords (can be customized per domain)
        self.professional_keywords = professional_keywords or [
            # Company/Product specific
            "公司", "产品", "服务", "客户", "项目",
            # Technical terms
            "架构", "部署", "配置", "实现", "源码",
            # Domain specific
            "政策", "流程", "规范", "标准", "协议",
        ]

        # General query patterns
        self.general_patterns = general_patterns or [
            r"^什么 (是 | 意思).*$",      # What is...
            r"^怎么 (样 | 做| 弄).*$",     # How to...
            r"^为什么.*$",              # Why...
            r"^谁.*$",                 # Who...
            r"^何时.*$",               # When...
            r"^哪里.*$",               # Where...
            r"^多少.*$",               # How many/much...
            r"^请 (问 | 解释).*$",        # Please explain...
            r"^介绍一下.*$",            # Introduce...
            r"^简述.*$",              # Briefly describe...
        ]

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
        query_lower = query.lower()

        # Check for professional keywords first
        for keyword in self.professional_keywords:
            if keyword.lower() in query_lower:
                # Additional check: if it's a very short query, might still be general
                if len(query) > 10:
                    return QueryType.PROFESSIONAL

        # Check general patterns
        for pattern in self._compiled_patterns:
            if pattern.match(query):
                # If query contains professional keywords despite matching general pattern
                for keyword in self.professional_keywords:
                    if keyword.lower() in query_lower:
                        return QueryType.PROFESSIONAL
                return QueryType.GENERAL

        # Default to professional for longer, specific queries
        if len(query) > 20:
            return QueryType.PROFESSIONAL

        # Default to general for short queries
        return QueryType.GENERAL

    def classify_with_confidence(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify query with confidence score

        Returns:
            Tuple of (QueryType, confidence between 0-1)
        """
        query_lower = query.lower()

        professional_score = 0
        general_score = 0

        # Count professional keywords
        for keyword in self.professional_keywords:
            if keyword.lower() in query_lower:
                professional_score += 1

        # Count general pattern matches
        for pattern in self._compiled_patterns:
            if pattern.match(query):
                general_score += 1

        # Calculate confidence
        total = professional_score + general_score
        if total == 0:
            # No clear signals, use length heuristic
            if len(query) > 20:
                return QueryType.PROFESSIONAL, 0.5
            return QueryType.GENERAL, 0.5

        if professional_score > general_score:
            confidence = min(1.0, 0.5 + (professional_score - general_score) * 0.1)
            return QueryType.PROFESSIONAL, confidence
        elif general_score > professional_score:
            confidence = min(1.0, 0.5 + (general_score - professional_score) * 0.1)
            return QueryType.GENERAL, confidence
        else:
            return QueryType.GENERAL, 0.5

    def add_professional_keyword(self, keyword: str) -> None:
        """Add a professional keyword"""
        if keyword not in self.professional_keywords:
            self.professional_keywords.append(keyword)

    def add_general_pattern(self, pattern: str) -> None:
        """Add a general query pattern"""
        if pattern not in self.general_patterns:
            self.general_patterns.append(pattern)
            self._compiled_patterns.append(re.compile(pattern))


class SimpleClassifier(QueryClassifier):
    """
    Simplified classifier using only keyword matching
    Faster but less accurate
    """

    def __init__(self, keywords: Optional[List[str]] = None):
        super().__init__(professional_keywords=keywords)

    def classify(self, query: str) -> QueryType:
        query_lower = query.lower()

        for keyword in self.professional_keywords:
            if keyword.lower() in query_lower:
                return QueryType.PROFESSIONAL

        return QueryType.GENERAL


if __name__ == "__main__":
    # Example usage
    classifier = QueryClassifier()

    test_queries = [
        "什么是 RAG?",
        "公司的产品架构是什么？",
        "请介绍一下你们的服务",
        "如何配置部署环境？",
        "今天天气怎么样",
    ]

    for query in test_queries:
        result = classifier.classify(query)
        print(f"'{query}' -> {result.value}")
