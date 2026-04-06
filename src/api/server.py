"""
API Layer - FastAPI based REST API with Security Features
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import tempfile
import os

# Import security modules
from ..security.authentication import AuthenticationService, RBACManager, UserRole, SecurityConfig
from ..security.security_config import get_security_manager
from ..security.middleware import (
    init_security_middleware,
    get_current_user,
    require_permission,
    check_rate_limit,
    verify_document_access
)

from ..integrated_qa import IntegratedQASystem, QAResponse

# Initialize security components globally
security_manager = get_security_manager()
auth_service = AuthenticationService(security_manager.security_config)
rbac_manager = RBACManager(auth_service)

# Initialize security middleware
init_security_middleware(auth_service, rbac_manager)

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

#用户认证请求模型
class LoginRequest(BaseModel):
    username: str
    password: str

#用户认证响应模型
class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

#创建FastAPI应用
def create_app(qa_system: Optional[IntegratedQASystem] = None) -> FastAPI:
    """Create FastAPI application with security features"""

    app = FastAPI(
        title="Integrated QA System API",
        description="RAG-based Question Answering System with Security Features",
        version="1.0.0",
        dependencies=[Depends(check_rate_limit)]  # Apply rate limiting to all endpoints
    )

    # Store QA system instance
    app.state.qa_system = qa_system or IntegratedQASystem(
        vector_store_provider="milvus"  # Use Milvus as default vector store
    )
    app.state.auth_service = auth_service
    app.state.rbac_manager = rbac_manager

    # Public endpoint for health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint (public)"""
        return {"status": "healthy"}

    # Authentication endpoints
    @app.post("/login", response_model=LoginResponse)
    async def login(request: LoginRequest):
        """User login endpoint"""
        token_payload = auth_service.authenticate_user(request.username, request.password)

        if not token_payload:
            raise HTTPException(status_code=401, detail="Incorrect username or password")

        # Create JWT token string
        import jwt
        token = jwt.encode({
            "user_id": token_payload.user_id,
            "username": token_payload.username,
            "role": token_payload.role,
            "exp": token_payload.exp,
            "iat": token_payload.iat,
            "jti": token_payload.jti
        }, security_manager.settings.jwt_secret_key, algorithm=security_manager.settings.jwt_algorithm)

        return LoginResponse(access_token=token, token_type="bearer")

    # Protected endpoints
    @app.post("/query", response_model=QueryResponse)
    async def query(
        request: QueryRequest,
        current_user: dict = Depends(get_current_user)
    ):
        """
        Process a user query (requires authentication)

        - **query**: The question to ask
        - **session_id**: Optional conversation session ID
        - **top_k**: Number of documents to retrieve
        """
        try:
            # Check if user is authorized to perform queries
            if not app.state.rbac_manager.authorize_query(current_user.user_id, request.query):
                raise HTTPException(status_code=403, detail="Query not authorized")

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
    async def ingest_file(
        file: UploadFile = File(...),
        doc_id: Optional[str] = None,
        current_user: dict = Depends(get_current_user)
    ):
        # Check permission within the function
        if not app.state.rbac_manager.has_permission(current_user.user_id, "write"):
            raise HTTPException(status_code=403, detail="Write permission required")
        """
        Ingest a document file (requires write permission)

        - **file**: The document file to ingest
        - **doc_id**: Optional document ID
        """
        # Save uploaded file temporarily
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
    async def ingest_directory(
        directory_path: str,
        file_types: Optional[List[str]] = None,
        current_user: dict = Depends(get_current_user)
    ):
        # Check permission within the function
        if not app.state.rbac_manager.has_permission(current_user.user_id, "write"):
            raise HTTPException(status_code=403, detail="Write permission required")
        """
        Ingest all documents from a directory (requires write permission)

        - **directory_path**: Path to the directory
        - **file_types**: Optional list of file extensions to process
        """
        try:
            #对传进来的文档也进行切块
            chunks = app.state.qa_system.ingest_directory(directory_path, file_types)
            return {"directory": directory_path, "chunks_created": chunks}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/session/{session_id}")
    async def get_session(session_id: str, current_user: dict = Depends(get_current_user)):
        """Get conversation session info (requires authentication)"""
        # Check if user has access to this session
        session = app.state.qa_system.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # In a real system, we'd check if the user owns this session
        # For now, assuming the session belongs to the authenticated user
        return session.get_summary()

    @app.delete("/session/{session_id}")
    async def clear_session(
        session_id: str,
        current_user: dict = Depends(get_current_user)
    ):
        # Check permission within the function
        if not app.state.rbac_manager.has_permission(current_user.user_id, "delete"):
            raise HTTPException(status_code=403, detail="Delete permission required")
        """Clear a conversation session (requires delete permission)"""
        if not app.state.qa_system.clear_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "ok"}

    # Admin endpoints
    @app.get("/admin/stats")
    async def get_admin_stats(current_user: dict = Depends(get_current_user)):
        # Check permission within the function
        if not app.state.rbac_manager.has_permission(current_user.user_id, "admin"):
            raise HTTPException(status_code=403, detail="Admin permission required")
        """Get system statistics (admin only)"""
        return {
            "message": "Admin statistics endpoint",
            "user": current_user.username,
            "role": current_user.role
        }

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
