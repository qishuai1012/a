"""
Enhanced API Layer - FastAPI based REST API with Advanced Security Features
Includes JWT authentication, RBAC, rate limiting, and streaming responses
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import tempfile
import os
import time
import asyncio
from collections import deque
import jwt
from datetime import datetime, timedelta
from functools import wraps

# Import security modules
from ..security.authentication import AuthenticationService, RBACManager, UserRole, SecurityConfig
from ..security.security_config import get_security_manager
from ..security.middleware import (
    init_security_middleware,
    get_current_user,
    require_permission,
    verify_document_access
)

from ..integrated_qa import IntegratedQASystem, QAResponse

# Rate limiting utilities
class TokenBucket:
    """Token bucket algorithm for rate limiting"""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = capacity
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket"""
        now = time.time()
        # Add tokens based on time passed
        tokens_to_add = (now - self.last_refill) * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

# Global rate limiter storage (in production, use Redis)
rate_limiters: Dict[str, TokenBucket] = {}

def get_client_ip(request: Request) -> str:
    """Get client IP address from request"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

def rate_limit(calls: int, per: int):
    """
    Rate limiting decorator
    Args:
        calls: Number of calls allowed
        per: Time period in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request') or args[0]  # FastAPI injects request as first param
            client_ip = get_client_ip(request)

            # Create rate limiter for this IP if it doesn't exist
            if client_ip not in rate_limiters:
                rate_limiters[client_ip] = TokenBucket(capacity=calls, refill_rate=calls/per)

            # Check if request is allowed
            if not rate_limiters[client_ip].consume(1):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Initialize security components globally
security_manager = get_security_manager()
auth_service = AuthenticationService(security_manager.security_config)
rbac_manager = RBACManager(auth_service)

# Initialize security middleware
init_security_middleware(auth_service, rbac_manager)

# Request/Response Models
class QueryRequest(BaseModel):
    """Query request model"""
    query: str
    session_id: Optional[str] = None
    top_k: int = 5
    business_category_filter: Optional[str] = None
    permission_level_filter: Optional[str] = None

class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    source_documents: List[str]
    confidence: float
    cache_hit: bool
    query_type: str
    conversation_id: str

class StreamQueryResponse(BaseModel):
    """Streaming query response model"""
    token: str
    done: bool
    conversation_id: str

class IngestResponse(BaseModel):
    """Document ingestion response"""
    file_path: str
    chunks_created: int

class LoginRequest(BaseModel):
    """User login request model"""
    username: str
    password: str

class LoginResponse(BaseModel):
    """User login response model"""
    access_token: str
    token_type: str = "bearer"

# Create FastAPI app with CORS support
def create_app(qa_system: Optional[IntegratedQASystem] = None) -> FastAPI:
    """Create FastAPI application with advanced security features"""

    app = FastAPI(
        title="Integrated QA System API - Enterprise Edition",
        description="RAG-based Question Answering System with Advanced Security Features",
        version="2.0.0"
    )

    # Store QA system instance
    app.state.qa_system = qa_system or IntegratedQASystem(
        vector_store_provider="milvus"  # Use Milvus as default vector store
    )
    app.state.auth_service = auth_service
    app.state.rbac_manager = rbac_manager

    # Health check endpoint (public)
    @app.get("/health")
    async def health_check():
        """Health check endpoint (public)"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0"
        }

    # Metrics endpoint (public)
    @app.get("/metrics")
    async def get_metrics():
        """Get system metrics (public)"""
        # This could integrate with Prometheus in the future
        return {
            "active_users": len(rate_limiters),
            "requests_served": getattr(app.state, 'request_count', 0),
            "uptime": getattr(app.state, 'start_time', datetime.utcnow()).isoformat()
        }

    # Authentication endpoints
    @app.post("/login", response_model=LoginResponse)
    @rate_limit(calls=5, per=60)  # Limit login attempts
    async def login(request: LoginRequest):
        """User login endpoint with rate limiting"""
        token_payload = auth_service.authenticate_user(request.username, request.password)

        if not token_payload:
            raise HTTPException(status_code=401, detail="Incorrect username or password")

        # Create JWT token string
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
    @rate_limit(calls=30, per=60)  # Standard user rate limit
    async def query(
        request: QueryRequest,
        current_user: dict = Depends(get_current_user)
    ):
        """
        Process a user query (requires authentication)

        - **query**: The question to ask
        - **session_id**: Optional conversation session ID
        - **top_k**: Number of documents to retrieve
        - **business_category_filter**: Optional filter for business category
        - **permission_level_filter**: Optional filter for permission level
        """
        try:
            # Check if user is authorized to perform queries
            if not app.state.rbac_manager.authorize_query(current_user.user_id, request.query):
                raise HTTPException(status_code=403, detail="Query not authorized")

            # Increment request counter
            app.state.request_count = getattr(app.state, 'request_count', 0) + 1

            response = app.state.qa_system.query(
                query_text=request.query,
                session_id=request.session_id,
                top_k=request.top_k,
                business_category_filter=request.business_category_filter,
                permission_level_filter=request.permission_level_filter
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

    # Streaming query endpoint
    @app.post("/query/stream")
    @rate_limit(calls=20, per=60)  # Slightly lower rate limit for streaming
    async def stream_query(
        request: QueryRequest,
        current_user: dict = Depends(get_current_user)
    ):
        """
        Process a query with streaming response (requires authentication)
        """
        # Check if user is authorized to perform queries
        if not app.state.rbac_manager.authorize_query(current_user.user_id, request.query):
            raise HTTPException(status_code=403, detail="Query not authorized")

        # Increment request counter
        app.state.request_count = getattr(app.state, 'request_count', 0) + 1

        async def generate_stream():
            try:
                async for token in app.state.qa_system.query_stream(
                    query_text=request.query,
                    session_id=request.session_id,
                    top_k=request.top_k
                ):
                    # Yield server-sent event
                    yield f"data: {token}\n\n"

                # Send end marker
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"event: error\ndata: {str(e)}\n\n"

        return StreamingResponse(generate_stream(), media_type="text/plain")

    # Document ingestion endpoints
    @app.post("/ingest/file")
    @rate_limit(calls=10, per=60)  # Lower rate limit for file uploads
    async def ingest_file(
        file: UploadFile = File(...),
        doc_id: Optional[str] = None,
        business_category: Optional[str] = "general",
        permission_level: Optional[str] = "public",
        tags: Optional[List[str]] = [],
        current_user: dict = Depends(get_current_user)
    ):
        """
        Ingest a document file (requires write permission)

        - **file**: The document file to ingest
        - **doc_id**: Optional document ID
        - **business_category**: Business category for the document
        - **permission_level**: Permission level for the document
        - **tags**: List of tags for the document
        """
        # Check permission within the function
        if not app.state.rbac_manager.has_permission(current_user.user_id, "write"):
            raise HTTPException(status_code=403, detail="Write permission required")

        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.pptx', '.txt', '.csv', '.xlsx', '.xls']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File type not supported: {file_ext}")

        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)

        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            chunks = app.state.qa_system.ingest_document(
                file_path=file_path,
                doc_id=doc_id,
                business_category=business_category,
                permission_level=permission_level,
                tags=tags
            )
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
        business_category: Optional[str] = "general",
        permission_level: Optional[str] = "public",
        tags: Optional[List[str]] = [],
        current_user: dict = Depends(get_current_user)
    ):
        """
        Ingest all documents from a directory (requires write permission)

        - **directory_path**: Path to the directory
        - **file_types**: Optional list of file extensions to process
        - **business_category**: Business category for the documents
        - **permission_level**: Permission level for the documents
        - **tags**: List of tags for the documents
        """
        # Check permission within the function
        if not app.state.rbac_manager.has_permission(current_user.user_id, "write"):
            raise HTTPException(status_code=403, detail="Write permission required")

        try:
            chunks = app.state.qa_system.ingest_directory(
                directory_path=directory_path,
                file_types=file_types,
                business_category=business_category,
                permission_level=permission_level,
                tags=tags
            )
            return {"directory": directory_path, "chunks_created": chunks}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Session management endpoints
    @app.get("/session/{session_id}")
    async def get_session(session_id: str, current_user: dict = Depends(get_current_user)):
        """Get conversation session info (requires authentication)"""
        session = app.state.qa_system.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # In a real system, we'd check if the user owns this session
        return session.get_summary()

    @app.delete("/session/{session_id}")
    async def clear_session(
        session_id: str,
        current_user: dict = Depends(get_current_user)
    ):
        """Clear a conversation session (requires delete permission)"""
        if not app.state.rbac_manager.has_permission(current_user.user_id, "delete"):
            raise HTTPException(status_code=403, detail="Delete permission required")

        if not app.state.qa_system.clear_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "ok"}

    # User profile endpoint
    @app.get("/profile")
    async def get_profile(current_user: dict = Depends(get_current_user)):
        """Get user profile information"""
        return {
            "user_id": current_user.user_id,
            "username": current_user.username,
            "role": current_user.role,
            "permissions": app.state.rbac_manager.get_user_permissions(current_user.user_id)
        }

    # Admin endpoints
    @app.get("/admin/stats")
    async def get_admin_stats(current_user: dict = Depends(get_current_user)):
        """Get system statistics (admin only)"""
        if not app.state.rbac_manager.has_permission(current_user.user_id, "admin"):
            raise HTTPException(status_code=403, detail="Admin permission required")

        return {
            "message": "Admin statistics endpoint",
            "user": current_user.username,
            "role": current_user.role,
            "active_sessions": len(app.state.qa_system.dialogue_manager.sessions),
            "rate_limited_ips": len(rate_limiters)
        }

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    qa_system: Optional[IntegratedQASystem] = None
):
    """Run the API server"""
    app = create_app(qa_system)
    # Set start time for metrics
    app.state.start_time = datetime.utcnow()
    app.state.request_count = 0
    # Run uvicorn server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()