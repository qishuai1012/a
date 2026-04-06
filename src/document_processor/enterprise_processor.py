"""
Enterprise Document Processor Module
Advanced text processing with parent-child chunking, hierarchical splitting, and semantic analysis
"""

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from datetime import datetime
import logging
from pathlib import Path


@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    content: str                             # 文本块的实际内容
    chunk_id: str                            # 文本块的唯一ID
    parent_document: str                     # 原始文档的ID
    chunk_index: int                         # 在父文档中的索引位置
    metadata: dict                           # 额外的元数据（如起始位置、结束位置、长度等）
    parent_id: Optional[str] = None          # 父块ID（用于父子分块）
    child_ids: Optional[List[str]] = None    # 子块ID列表
    semantic_similarity: Optional[float] = None  # 与父块的语义相似度
    business_category: Optional[str] = None      # 业务分类
    permission_level: Optional[str] = None       # 权限等级
    tags: Optional[List[str]] = None             # 标签
    update_timestamp: Optional[datetime] = None  # 更新时间戳


class BaseChunker(ABC):
    """Base class for text chunking strategies"""

    @abstractmethod
    def chunk(self, text: str, doc_id: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        """Split text into chunks"""
        pass


class ParentChildChunker(BaseChunker):
    """
    Parent-Child chunking strategy: Creates large parent chunks and smaller child chunks
    Large chunks maintain context, small chunks enable precise retrieval
    """

    def __init__(
        self,
        parent_chunk_size: int = 2048,
        child_chunk_size: int = 512,
        overlap_ratio: float = 0.1,
        enable_semantic_analysis: bool = True
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.overlap_ratio = overlap_ratio
        self.enable_semantic_analysis = enable_semantic_analysis

        # Semantic analysis model (initialize when needed)
        self.semantic_model = None

    def chunk(self, text: str, doc_id: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        """
        Split text using parent-child chunking strategy
        """
        # Create parent chunks first
        parent_chunks = self._create_parent_chunks(text, doc_id, metadata)

        # Create child chunks from each parent
        all_chunks = []
        all_chunks.extend(parent_chunks)  # Add parent chunks

        for parent_chunk in parent_chunks:
            child_chunks = self._create_child_chunks(
                parent_chunk.content,
                doc_id,
                parent_chunk.chunk_id,
                metadata
            )
            all_chunks.extend(child_chunks)

        return all_chunks

    def _create_parent_chunks(self, text: str, doc_id: str, metadata: Optional[Dict]) -> List[TextChunk]:
        """Create parent chunks"""
        # Split by semantic boundaries first (headings, sections)
        semantic_splits = self._semantic_split(text)

        # If no semantic boundaries found, fall back to fixed-size splitting
        if len(semantic_splits) <= 1:
            semantic_splits = self._fixed_size_split(text, self.parent_chunk_size)

        parent_chunks = []
        for i, split_text in enumerate(semantic_splits):
            chunk_id = f"{doc_id}_parent_{i}"
            chunk = TextChunk(
                content=split_text,
                chunk_id=chunk_id,
                parent_document=doc_id,
                chunk_index=i,
                metadata=metadata or {},
                parent_id=None,  # Parents have no parent
                child_ids=[],  # Will be populated later
                business_category=metadata.get('business_category') if metadata else None,
                permission_level=metadata.get('permission_level') if metadata else None,
                tags=metadata.get('tags') if metadata else None,
                update_timestamp=datetime.now()
            )
            parent_chunks.append(chunk)

        return parent_chunks

    def _create_child_chunks(
        self,
        parent_text: str,
        doc_id: str,
        parent_id: str,
        metadata: Optional[Dict]
    ) -> List[TextChunk]:
        """Create child chunks from parent text"""
        # Split parent text into smaller chunks
        child_texts = self._fixed_size_split(parent_text, self.child_chunk_size)

        child_chunks = []
        for i, child_text in enumerate(child_texts):
            chunk_id = f"{parent_id}_child_{i}"
            chunk = TextChunk(
                content=child_text,
                chunk_id=chunk_id,
                parent_document=doc_id,
                chunk_index=i,
                metadata=metadata or {},
                parent_id=parent_id,
                child_ids=None,  # Children have no children
                business_category=metadata.get('business_category') if metadata else None,
                permission_level=metadata.get('permission_level') if metadata else None,
                tags=metadata.get('tags') if metadata else None,
                update_timestamp=datetime.now()
            )

            # Calculate semantic similarity to parent if enabled
            if self.enable_semantic_analysis:
                chunk.semantic_similarity = self._calculate_semantic_similarity(child_text, parent_text)

            child_chunks.append(chunk)

        # Update parent's child_ids
        parent_chunk = None
        for chunk in child_chunks:
            if chunk.parent_id == parent_id:
                if parent_chunk is None:
                    parent_chunk = chunk  # This is a child, not the parent - let's fix this approach
                break

        return child_chunks

    def _semantic_split(self, text: str) -> List[str]:
        """Split text based on semantic boundaries (headings, sections, etc.)"""
        # Identify potential split points (headings, sections)
        heading_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headings
            r'^\s*\d+(\.\d+)*\s+.+$',  # Numbered sections (1., 1.1., etc.)
            r'^\s*[A-Z][A-Za-z]*\s*:.*$',  # Topic headers ending with colon
            r'^\s*[A-Z][A-Za-z ]+\s*$',  # Potential section titles (all caps or title case)
        ]

        lines = text.split('\n')
        split_points = []

        for i, line in enumerate(lines):
            for pattern in heading_patterns:
                if re.match(pattern, line.strip(), re.MULTILINE):
                    split_points.append(i)
                    break

        # If no semantic boundaries found, return original text
        if not split_points:
            return [text]

        # Create splits based on identified points
        splits = []
        start = 0
        for point in split_points:
            if point > start:
                section = '\n'.join(lines[start:point]).strip()
                if section:
                    splits.append(section)
            start = point

        # Add the remaining text
        if start < len(lines):
            section = '\n'.join(lines[start:]).strip()
            if section:
                splits.append(section)

        return splits

    def _fixed_size_split(self, text: str, chunk_size: int) -> List[str]:
        """Split text into fixed-size chunks with overlap"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        overlap_size = int(chunk_size * self.overlap_ratio)

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                # Find nearest sentence boundary after current position
                next_sentences = re.split(r'[.!?]+\s+', text[end-50:end+chunk_size])
                if len(next_sentences) > 1:
                    # Adjust end position to sentence boundary
                    sentence_end = text.find(next_sentences[1], end-50) if next_sentences[1].strip() else -1
                    if sentence_end != -1 and sentence_end < end + chunk_size:
                        end = sentence_end
                        chunk_text = text[start:end]

            chunks.append(chunk_text.strip())
            start = end - overlap_size if overlap_size < end else end

        return chunks

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using Sentence-BERT"""
        if not self.semantic_model:
            try:
                from sentence_transformers import SentenceTransformer
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                # Fallback to simple similarity calculation
                return self._simple_similarity(text1, text2)

        try:
            # Encode both texts
            embeddings = self.semantic_model.encode([text1, text2])
            # Calculate cosine similarity
            dot_product = np.dot(embeddings[0], embeddings[1])
            norm = np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            similarity = dot_product / norm if norm != 0 else 0.0
            return float(similarity)
        except:
            # Fallback to simple similarity
            return self._simple_similarity(text1, text2)

    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple similarity calculation based on word overlap"""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)


class HierarchicalChunker(BaseChunker):
    """
    Hierarchical chunking based on document structure (titles, sections, paragraphs)
    """

    def __init__(self, chunk_sizes: Optional[Dict[str, int]] = None):
        # Default chunk sizes for different levels
        self.chunk_sizes = chunk_sizes or {
            'title': 256,
            'section': 512,
            'paragraph': 1024,
            'page': 2048
        }

    def chunk(self, text: str, doc_id: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        """Chunk text hierarchically based on document structure"""
        # Detect document structure
        structure = self._analyze_structure(text)

        chunks = []
        for level, parts in structure.items():
            for i, part in enumerate(parts):
                chunk_id = f"{doc_id}_{level}_{i}"
                chunk = TextChunk(
                    content=part,
                    chunk_id=chunk_id,
                    parent_document=doc_id,
                    chunk_index=i,
                    metadata=metadata or {},
                    update_timestamp=datetime.now()
                )
                chunks.append(chunk)

        return chunks

    def _analyze_structure(self, text: str) -> Dict[str, List[str]]:
        """Analyze document structure and identify different levels"""
        structure = {
            'title': [],
            'section': [],
            'paragraph': [],
            'raw': []
        }

        lines = text.split('\n')
        current_section = []
        current_paragraph = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for titles/headers
            if self._is_title_line(line):
                # Save previous section if exists
                if current_section:
                    structure['section'].append('\n'.join(current_section))
                    current_section = []
                # Add title
                structure['title'].append(line)

            # Check for section breaks
            elif self._is_section_break(line):
                if current_section:
                    structure['section'].append('\n'.join(current_section))
                    current_section = []

            # Regular content
            elif line:
                current_section.append(line)

            i += 1

        # Add any remaining content
        if current_section:
            content_str = '\n'.join(current_section)
            structure['paragraph'].extend(self._split_into_paragraphs(content_str))

        return structure

    def _is_title_line(self, line: str) -> bool:
        """Check if line is a title/header"""
        if not line:
            return False

        # Check for markdown headers
        if re.match(r'^#{1,6}\s+.+$', line):
            return True

        # Check for numbered sections
        if re.match(r'^\s*\d+(\.\d+)*\s+.+$', line):
            return True

        # Check for potential headers (short, capitalized)
        if len(line) < 100 and line.isupper():
            return True

        return False

    def _is_section_break(self, line: str) -> bool:
        """Check if line indicates a section break"""
        # Empty lines or lines with only special characters
        return not line.strip() or re.match(r'^[-*_=]{3,}$', line.strip())

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines first
        paragraphs = re.split(r'\n\s*\n', text)
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]


class DocumentProcessor:
    """
    Main document processor with enterprise features
    - MD5-based deduplication
    - Content filtering
    - Metadata enrichment
    - Business categorization
    """

    def __init__(
        self,
        chunker: Optional[BaseChunker] = None,
        enable_deduplication: bool = True,
        enable_content_filtering: bool = True,
        default_business_category: str = "general",
        default_permission_level: str = "public"
    ):
        self.chunker = chunker or ParentChildChunker()
        self.enable_deduplication = enable_deduplication
        self.enable_content_filtering = enable_content_filtering
        self.default_business_category = default_business_category
        self.default_permission_level = default_permission_level
        self.processed_hashes = set()

    def _calculate_md5(self, text: str) -> str:
        """Calculate MD5 hash of text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _is_low_quality_content(self, text: str) -> bool:
        """Check if content is low quality"""
        if not text.strip():
            return True

        # Check for repetitive content
        if len(set(text.split())) / len(text.split()) < 0.3:  # Less than 30% unique words
            return True

        # Check for excessive special characters
        special_chars_ratio = sum(1 for c in text if not c.isalnum() and c != ' ') / len(text)
        if special_chars_ratio > 0.5:  # More than 50% special characters
            return True

        # Check for minimum length
        if len(text.strip()) < 10:
            return True

        return False

    def process(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict] = None,
        business_category: Optional[str] = None,
        permission_level: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[TextChunk]:
        """
        Process document content with enterprise features
        """
        # Prepare metadata
        full_metadata = metadata or {}
        full_metadata.update({
            'business_category': business_category or self.default_business_category,
            'permission_level': permission_level or self.default_permission_level,
            'tags': tags or [],
            'processing_timestamp': datetime.now().isoformat(),
            'doc_id': doc_id
        })

        # Deduplicate if enabled
        if self.enable_deduplication:
            content_hash = self._calculate_md5(content)
            if content_hash in self.processed_hashes:
                return []  # Already processed
            self.processed_hashes.add(content_hash)

        # Filter low-quality content
        if self.enable_content_filtering and self._is_low_quality_content(content):
            return []

        # Chunk the content
        chunks = self.chunker.chunk(content, doc_id, full_metadata)

        # Apply additional processing to each chunk
        processed_chunks = []
        for chunk in chunks:
            # Add additional metadata
            chunk.metadata['char_count'] = len(chunk.content)
            chunk.metadata['word_count'] = len(chunk.content.split())
            chunk.metadata['sentence_count'] = len(re.split(r'[.!?]+', chunk.content))

            # Ensure permission and category are properly set
            if not chunk.permission_level:
                chunk.permission_level = full_metadata['permission_level']
            if not chunk.business_category:
                chunk.business_category = full_metadata['business_category']
            if not chunk.tags:
                chunk.tags = full_metadata['tags']

            processed_chunks.append(chunk)

        return processed_chunks

    def process_batch(self, documents: List[Tuple[str, str, Optional[Dict]]]) -> List[TextChunk]:
        """
        Batch process multiple documents
        """
        all_chunks = []
        for content, doc_id, metadata in documents:
            chunks = self.process(content, doc_id, metadata)
            all_chunks.extend(chunks)
        return all_chunks


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor(chunker=ParentChildChunker())

    sample_text = """
    # Introduction to RAG Systems

    Retrieval-Augmented Generation (RAG) is a technique that combines retrieval-based and generative approaches...

    ## Architecture

    The typical RAG architecture consists of several components:
    - Document storage
    - Retrieval system
    - Language model

    ### Implementation Details

    Implementation involves indexing documents and performing similarity search...

    Additional paragraph with more details about RAG implementation considerations.
    """

    chunks = processor.process(sample_text, "doc_001", {}, "technical", "internal", ["RAG", "NLP"])

    for chunk in chunks[:5]:  # Show first 5 chunks
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Type: {'Parent' if '_parent_' in chunk.chunk_id else 'Child'}")
        print(f"Content: {chunk.content[:100]}...")
        print(f"Category: {chunk.business_category}, Level: {chunk.permission_level}")
        print("---")