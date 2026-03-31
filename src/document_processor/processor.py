"""
Document Processing and Text Chunking Module
Handles document cleaning, preprocessing, and splitting into chunks
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    content: str
    chunk_id: str
    parent_document: str
    chunk_index: int
    metadata: dict


class BaseChunker(ABC):
    """Base class for text chunking strategies"""

    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> List[TextChunk]:
        """Split text into chunks"""
        pass


class FixedSizeChunker(BaseChunker):
    """
    Split text into fixed-size chunks with overlap
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, doc_id: str) -> List[TextChunk]:
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind(".")
                last_newline = chunk_text.rfind("\n")
                break_point = max(last_period, last_newline)
                if break_point > self.chunk_size // 2:
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1

            chunks.append(TextChunk(
                content=chunk_text.strip(),
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                parent_document=doc_id,
                chunk_index=chunk_index,
                metadata={
                    "start": start,
                    "end": end,
                    "length": len(chunk_text)
                }
            ))

            start = end - self.chunk_overlap
            chunk_index += 1

        return chunks


class RecursiveCharacterChunker(BaseChunker):
    """
    Split text recursively by character separators
    Preferred for maintaining semantic coherence
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.separators = separators or [
            "\n\n",      # Paragraph
            "\n",        # Line
            "。",        # Chinese sentence end
            ".",         # English sentence end
            "!",         # Exclamation
            "?",         # Question
            "；",        # Chinese semicolon
            ";",         # English semicolon
            " ",         # Space
            ""           # Character level
        ]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split text using the given separators"""
        if not separators:
            return [text]

        separator = separators[0]
        rest_separators = separators[1:]

        if not separator:
            # Split by character count
            splits = []
            for i in range(0, len(text), self.chunk_size):
                splits.append(text[i:i + self.chunk_size])
            return splits

        # Split by current separator
        splits = text.split(separator)

        # If splits are still too large, recursively split
        final_splits = []
        for split in splits:
            if len(split) > self.chunk_size:
                final_splits.extend(self._split_text(split, rest_separators))
            else:
                final_splits.append(split)

        return final_splits

    def chunk(self, text: str, doc_id: str) -> List[TextChunk]:
        raw_chunks = self._split_text(text, self.separators)

        # Merge small chunks and apply overlap
        chunks = []
        chunk_index = 0
        current_chunk = ""

        for raw_chunk in raw_chunks:
            raw_chunk = raw_chunk.strip()
            if not raw_chunk:
                continue

            if len(current_chunk) + len(raw_chunk) <= self.chunk_size:
                current_chunk += raw_chunk
            else:
                if current_chunk:
                    chunks.append(TextChunk(
                        content=current_chunk,
                        chunk_id=f"{doc_id}_chunk_{chunk_index}",
                        parent_document=doc_id,
                        chunk_index=chunk_index,
                        metadata={"length": len(current_chunk)}
                    ))
                    chunk_index += 1
                    # Keep overlap
                current_chunk = raw_chunk

        if current_chunk:
            chunks.append(TextChunk(
                content=current_chunk,
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                parent_document=doc_id,
                chunk_index=chunk_index,
                metadata={"length": len(current_chunk)}
            ))

        return chunks


class DocumentProcessor:
    """
    Main document processor that handles cleaning and chunking
    """

    def __init__(
        self,
        chunker: Optional[BaseChunker] = None,
        remove_extra_whitespace: bool = True,
        remove_special_characters: bool = False
    ):
        self.chunker = chunker or RecursiveCharacterChunker()
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_special_characters = remove_special_characters

    def clean(self, text: str) -> str:
        """Clean and preprocess text"""
        if self.remove_extra_whitespace:
            # Remove multiple whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove leading/trailing whitespace per line
            lines = [line.strip() for line in text.split('\n')]
            text = '\n'.join(lines)

        if self.remove_special_characters:
            # Remove special characters but keep Chinese/English characters
            text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()""\'\-]', '', text)

        return text.strip()

    def process(self, content: str, doc_id: str) -> List[TextChunk]:
        """
        Process document content and return chunks
        """
        cleaned_text = self.clean(content)
        chunks = self.chunker.chunk(cleaned_text, doc_id)
        return chunks

    def process_batch(self, documents: List[tuple]) -> List[TextChunk]:
        """
        Process multiple documents
        Args:
            documents: List of (content, doc_id) tuples
        Returns:
            List of all TextChunks
        """
        all_chunks = []
        for content, doc_id in documents:
            chunks = self.process(content, doc_id)
            all_chunks.extend(chunks)
        return all_chunks


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    sample_text = "这是第一段。\n\n这是第二段。\n这是第二段的继续。"
    chunks = processor.process(sample_text, "doc_001")
    for chunk in chunks:
        print(f"Chunk {chunk.chunk_id}: {chunk.content[:50]}...")
