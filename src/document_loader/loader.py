"""
Document Loader Module
Supports loading PDF, PPT, DOCX, and Image files
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a loaded document with metadata"""
    content: str
    file_path: str
    file_type: str
    metadata: dict


class BaseLoader(ABC):
    """Base class for document loaders"""

    @abstractmethod
    def load(self, file_path: str) -> Document:
        """Load a document from the given file path"""
        pass

    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """Check if this loader supports the given file extension"""
        pass


class PDFLoader(BaseLoader):
    """Loader for PDF files"""

    def load(self, file_path: str) -> Document:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("Please install pypdf: pip install pypdf")

        reader = PdfReader(file_path)
        content = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                content.append(text)

        return Document(
            content="\n".join(content),
            file_path=file_path,
            file_type="pdf",
            metadata={"num_pages": len(reader.pages)}
        )

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == ".pdf"


class PPTLoader(BaseLoader):
    """Loader for PPT/PPTX files"""

    def load(self, file_path: str) -> Document:
        try:
            from pptx import Presentation
        except ImportError:
            raise ImportError("Please install python-pptx: pip install python-pptx")

        prs = Presentation(file_path)
        content = []
        for i, slide in enumerate(prs.slides):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_text.append(shape.text)
            if slide_text:
                content.append(f"[Slide {i+1}]\n" + "\n".join(slide_text))

        return Document(
            content="\n".join(content),
            file_path=file_path,
            file_type="ppt",
            metadata={"num_slides": len(prs.slides)}
        )

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in [".ppt", ".pptx"]


class DOCXLoader(BaseLoader):
    """Loader for DOCX files"""

    def load(self, file_path: str) -> Document:
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError("Please install python-docx: pip install python-docx")

        doc = DocxDocument(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text]

        return Document(
            content="\n".join(paragraphs),
            file_path=file_path,
            file_type="docx",
            metadata={"num_paragraphs": len(paragraphs)}
        )

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == ".docx"


class ImageLoader(BaseLoader):
    """Loader for Image files with OCR support"""

    def __init__(self, lang: str = "chi_sim+eng"):
        self.lang = lang

    def load(self, file_path: str) -> Document:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError("Please install pytesseract and Pillow: pip install pytesseract Pillow")

        image = Image.open(file_path)
        text = pytesseract.image_to_string(image, lang=self.lang)

        return Document(
            content=text,
            file_path=file_path,
            file_type="image",
            metadata={"image_size": image.size, "mode": image.mode}
        )

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"]


class DocumentLoader:
    """
    Unified document loader that automatically selects the appropriate loader
    based on file extension
    """

    def __init__(self):
        self.loaders: List[BaseLoader] = [
            PDFLoader(),
            PPTLoader(),
            DOCXLoader(),
            ImageLoader()
        ]

    def load(self, file_path: str) -> Document:
        """Load a document from the given file path"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix
        for loader in self.loaders:
            if loader.supports(extension):
                return loader.load(file_path)

        raise ValueError(f"Unsupported file type: {extension}")

    def load_directory(self, directory_path: str, file_types: Optional[List[str]] = None) -> List[Document]:
        """Load all documents from a directory"""
        path = Path(directory_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        documents = []
        for file_path in path.iterdir():
            if file_types is None or file_path.suffix in file_types:
                try:
                    doc = self.load(str(file_path))
                    documents.append(doc)
                except (ValueError, ImportError) as e:
                    print(f"Skipping {file_path}: {e}")

        return documents


if __name__ == "__main__":
    # Example usage
    loader = DocumentLoader()
    doc = loader.load("example.pdf")
    print(f"Loaded {doc.file_type} with {len(doc.content)} characters")
