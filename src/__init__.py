"""
Integrated QA System

A RAG-based question answering system with:
- Document loading (PDF, PPT, DOCX, Images)
- Text processing and chunking
- Vector search with BM25 caching
- Query classification
- Hybrid retrieval and re-ranking
- Multi-turn dialogue support
- Streaming responses
"""

from .integrated_qa import IntegratedQASystem, QAResponse

__all__ = [
    "IntegratedQASystem",
    "QAResponse"
]

# Package version
__version__ = "0.1.0"
