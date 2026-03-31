"""
API Layer - FastAPI based REST API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

from ..integrated_qa import IntegratedQASystem, QAResponse


class QueryRequest(BaseModel):
    """Query request model"""
    query: str
    session_id: Optional[str] = None
    top_k: int = 5


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    source_documents: List[str]
    confidence: float
    cache_hit: bool
    query_type: str
    conversation_id: str


class IngestResponse(BaseModel):
    """Document ingestion response"""
    file_path: str
    chunks_created: int


def create_app(qa_system: Optional[IntegratedQASystem] = None) -> FastAPI:
    """Create FastAPI application"""

    app = FastAPI(
        title="Integrated QA System API",
        description="RAG-based Question Answering System",
        version="0.1.0"
    )

    # Store QA system instance
    app.state.qa_system = qa_system or IntegratedQASystem()

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        """
        Process a user query

        - **query**: The question to ask
        - **session_id**: Optional conversation session ID
        - **top_k**: Number of documents to retrieve
        """
        try:
            response = app.state.qa_system.query(
                query_text=request.query,
                session_id=request.session_id,
                top_k=request.top_k
            )
            return QueryResponse(
                answer=response.answer,
                source_documents=response.source_documents,
                confidence=response.confidence,
                cache_hit=response.cache_hit,
                query_type=response.query_type,
                conversation_id=response.conversation_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest/file")
    async def ingest_file(file: UploadFile = File(...), doc_id: Optional[str] = None):
        """
        Ingest a document file

        - **file**: The document file to ingest
        - **doc_id**: Optional document ID
        """
        # Save uploaded file temporarily
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)

        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            chunks = app.state.qa_system.ingest_document(file_path, doc_id)
            return IngestResponse(file_path=file_path, chunks_created=chunks)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up temp file
            if os.path.exists(file_path):
                os.remove(file_path)

    @app.post("/ingest/directory")
    async def ingest_directory(directory_path: str, file_types: Optional[List[str]] = None):
        """
        Ingest all documents from a directory

        - **directory_path**: Path to the directory
        - **file_types**: Optional list of file extensions to process
        """
        try:
            chunks = app.state.qa_system.ingest_directory(directory_path, file_types)
            return {"directory": directory_path, "chunks_created": chunks}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/session/{session_id}")
    async def get_session(session_id: str):
        """Get conversation session info"""
        session = app.state.qa_system.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.get_summary()

    @app.delete("/session/{session_id}")
    async def clear_session(session_id: str):
        """Clear a conversation session"""
        if not app.state.qa_system.clear_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "ok"}

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy"}

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    qa_system: Optional[IntegratedQASystem] = None
):
    """Run the API server"""
    app = create_app(qa_system)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
