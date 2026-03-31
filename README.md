# Integrated QA System

RAG-based Question Answering System with BM25 caching and hybrid retrieval.

## Features

- 📄 **Multi-format Document Support**: PDF, PPT, DOCX, Images (with OCR)
- 🔍 **Hybrid Retrieval**: Vector search + BM25 keyword matching
- 💾 **Query Caching**: Fast BM25-based cache for repeated queries
- 🧠 **Smart Query Routing**: General knowledge → LLM, Professional → RAG
- 💬 **Multi-turn Dialogue**: Conversation history management
- ⚡ **Streaming Support**: Real-time streaming responses

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Ingest Documents

```bash
python main.py ingest ./docs
```

### 2. Run API Server

```bash
python main.py server --port 8000
```

### 3. Query

```bash
python main.py query "什么是 RAG?"
```

## API Endpoints

- `POST /query` - Process a query
- `POST /ingest/file` - Upload and ingest a file
- `POST /ingest/directory` - Ingest all documents in a directory
- `GET /session/{session_id}` - Get conversation session
- `DELETE /session/{session_id}` - Clear conversation
- `GET /health` - Health check

## Project Structure

```
src/
├── document_loader/     # PDF, PPT, DOCX, Image loading
├── document_processor/  # Text cleaning and chunking
├── vector_store/        # Embeddings and vector database
├── bm25_cache/          # BM25 caching layer
├── dialogue_history/    # Conversation management
├── rag_system/
│   ├── classifier.py    # Query type classification
│   ├── retriever.py     # Hybrid retrieval
│   └── prompt_builder.py # Prompt construction
└── api/                 # FastAPI server
```

## Configuration

Set LLM API credentials via environment variables:

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="http://localhost:11434/api"
```

## License

MIT
