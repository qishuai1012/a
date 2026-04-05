"""
API Layer - FastAPI based REST API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

from ..integrated_qa import IntegratedQASystem, QAResponse

#请求体模型
#问题，会话id，召回（文档）
class QueryRequest(BaseModel):
    """Query request model"""
    query: str
    session_id: Optional[str] = None
    top_k: int = 5

#响应模型
#答案、来源文档，置信度，是否命中缓存，问题类型，会话id
class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    source_documents: List[str]
    confidence: float
    cache_hit: bool
    query_type: str
    conversation_id: str

#文档导入成功返回
class IngestResponse(BaseModel):
    """Document ingestion response"""
    file_path: str
    chunks_created: int

#创建FastAPI应用
def create_app(qa_system: Optional[IntegratedQASystem] = None) -> FastAPI:
    """Create FastAPI application"""

    app = FastAPI(
        title="Integrated QA System API",
        description="RAG-based Question Answering System",
        version="0.1.0"
    )

    # Store QA system instance
    app.state.qa_system = qa_system or IntegratedQASystem()

    #提问/query
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
    #传入文档
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
        #保存临时目录
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

    #导入整个目录
    @app.post("/ingest/directory")
    async def ingest_directory(directory_path: str, file_types: Optional[List[str]] = None):
        """
        Ingest all documents from a directory

        - **directory_path**: Path to the directory
        - **file_types**: Optional list of file extensions to process
        """
        try:
            #对传进来的文档也进行切块
            chunks = app.state.qa_system.ingest_directory(directory_path, file_types)
            return {"directory": directory_path, "chunks_created": chunks}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    通过会话id获取上下文
    #
    @app.get("/session/{session_id}")
    async def get_session(session_id: str):
        """Get conversation session info"""
        session = app.state.qa_system.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        #返回会话id的所有内容，通常是整理好的字典
        return session.get_summary()

    #根据会话id删除历史记忆
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
    #返回操作完成后的FastAPI接口
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    qa_system: Optional[IntegratedQASystem] = None
):
    """Run the API server"""
    app = create_app(qa_system)
    #uvicorn是一个ASGI服务器，（理解为超级司机）
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
