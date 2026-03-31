"""Document Loader Package"""

from .loader import (
    Document,
    BaseLoader,
    PDFLoader,
    PPTLoader,
    DOCXLoader,
    ImageLoader,
    DocumentLoader
)

__all__ = [
    "Document",
    "BaseLoader",
    "PDFLoader",
    "PPTLoader",
    "DOCXLoader",
    "ImageLoader",
    "DocumentLoader"
]
