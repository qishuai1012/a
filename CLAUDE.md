# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Enterprise-Grade Integrated QA System - A RAG-based question answering system with BM25 caching, hybrid retrieval, comprehensive security, and microservice architecture. Uses **Milvus** as the primary enterprise vector database.

### Architecture

```
数据源 → 文档加载器 → 文档处理 → 分层切分 → 向量化 → 向量数据库(Milvus)
                                ↓
用户查询 → API网关 → 认证授权 → IntegratedQASystem → BM25 缓存层 → (命中→直接返回 | 未命中→RAG)
                                ↓
                        查询分类 → (通用知识→直接 LLM | 专业咨询→检索策略→混合检索→重排序→构建 Prompt→LLM 流式生成)

微服务架构:
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   API Gateway   │───▶│ Authentication   │───▶│    QA Service    │
│   (Port 8000)   │    │   (Port 8001)    │    │   (Port 8002)    │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                             │                         │
                    ┌────────▼────────┐        ┌───────▼────────┐
            ┌──────▶│  BM25 Cache     │        │ Vector Store   │
            │       │   (Port 8005)   │        │  (Port 8004)   │
            │       └─────────────────┘        └────────────────┘
            │                                    (Milvus DB)
    ┌───────▼────────┐
    │Document Processor│
    │   (Port 8006)  │
    └────────────────┘
```

### Module Structure

- `src/document_loader/` - Document loading (PDF, PPT, DOCX, Images)
- `src/document_processor/` - Text processing and chunking strategies
- `src/vector_store/` - Vector embeddings and enterprise vector storage (Milvus/ChromaDB)
- `src/bm25_cache/` - Enhanced BM25-based query caching layer with multi-level caching
- `src/dialogue_history/` - Multi-turn conversation management
- `src/rag_system/` - RAG components:
  - `classifier.py` - Query type classification
  - `retriever.py` - Hybrid retrieval and re-ranking
  - `prompt_builder.py` - Prompt construction
- `src/security/` - Authentication, authorization, and security controls
  - `authentication.py` - JWT-based authentication and RBAC
  - `middleware.py` - Security middleware for API protection
  - `security_config.py` - Security configuration management
- `src/utils/` - Utility functions including error handling
  - `error_handling.py` - Circuit breaker, retry mechanisms, rate limiting
- `src/architecture/` - Microservice architecture components
  - `api_gateway.py` - API Gateway service
  - `microservice_config.py` - Microservice configuration
  - `service_orchestrator.py` - Service orchestration layer
- `src/api/` - FastAPI REST API server

## Commands

### Installation
```bash
pip install -e .
```

### Run Monolithic Server
```bash
python main.py server --host 0.0.0.0 --port 8000
```

### Run in Microservice Mode
```bash
# Start API Gateway
python main.py microservice gateway --port 8000

# Start QA Service  
python main.py microservice qa --port 8002

# Start Auth Service
python main.py microservice auth --port 8001

# Start Vector Store Service (uses Milvus)
python main.py microservice vector_store --port 8004

# Start BM25 Cache Service
python main.py microservice bm25_cache --port 8005
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
5. **Enterprise Security**: JWT authentication, RBAC, rate limiting, circuit breaker patterns
6. **Microservice Architecture**: Decoupled services with API gateway and service orchestration
7. **Multi-level Caching**: Memory → Redis → Persistent caching hierarchy
8. **Resilience Patterns**: Circuit breakers, retry mechanisms, health checks
9. **Enterprise Vector Store**: Uses Milvus as the primary vector database for scalability