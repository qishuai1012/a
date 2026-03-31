# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Integrated QA System - A RAG-based question answering system with BM25 caching and hybrid retrieval.

### Architecture

```
数据源 → 文档加载器 → 文档处理 → 分層切分 → 向量化 → 向量数据库
                                ↓
用户查询 → IntegratedQASystem → BM25 缓存层 → (命中→直接返回 | 未命中→RAG)
                                ↓
                        查询分类 → (通用知识→直接 LLM | 专业咨询→检索策略→混合检索→重排序→构建 Prompt→LLM 流式生成)
```

### Module Structure

- `src/document_loader/` - Document loading (PDF, PPT, DOCX, Images)
- `src/document_processor/` - Text processing and chunking strategies
- `src/vector_store/` - Vector embeddings and ChromaDB storage
- `src/bm25_cache/` - BM25-based query caching layer
- `src/dialogue_history/` - Multi-turn conversation management
- `src/rag_system/` - RAG components:
  - `classifier.py` - Query type classification
  - `retriever.py` - Hybrid retrieval and re-ranking
  - `prompt_builder.py` - Prompt construction
- `src/api/` - FastAPI REST API server

## Commands

### Installation
```bash
pip install -e .
```

### Run API Server
```bash
python main.py server --host 0.0.0.0 --port 8000
```

### Ingest Documents
```bash
# Single file
python main.py ingest path/to/document.pdf

# Directory
python main.py ingest path/to/docs/
```

### Query
```bash
python main.py query "什么是 RAG?"
```

### Run Tests
```bash
pytest tests/
```

## Key Design Patterns

1. **Two-stage retrieval**: BM25 cache first, then RAG for uncached queries
2. **Query routing**: General queries → direct LLM; Professional queries → full RAG pipeline
3. **Hybrid search**: Vector similarity + BM25 keyword matching with weighted combination
4. **Modular pipeline**: Each component (loader, processor, retriever) is independently testable
