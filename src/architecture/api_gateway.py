"""
API Gateway for Microservices Architecture
Handles routing, authentication, and request/response transformation
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
import structlog
from typing import Dict, Any, Optional
import time
from .service_orchestrator import (
    MicroserviceOrchestrator,
    ServiceRequest,
    ServiceType,
    AuthServiceClient,
    QAServiceClient
)
from pydantic import BaseModel


class GatewayConfig:
    """Configuration for the API gateway"""
    def __init__(self):
        self.rate_limit_requests = 100  # per minute
        self.rate_limit_window = 60  # seconds
        self.default_timeout = 30  # seconds
        self.enable_caching = True
        self.enable_compression = True


class QueryRequest(BaseModel):
    """Request model for QA queries"""
    query: str
    session_id: Optional[str] = None
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response model for QA queries"""
    answer: str
    source_documents: list
    confidence: float
    cache_hit: bool
    query_type: str
    conversation_id: str


class APIServiceGateway:
    """API Gateway implementing service orchestration"""

    def __init__(self, config: GatewayConfig = None):
        self.config = config or GatewayConfig()
        self.logger = structlog.get_logger(__name__).bind(component="api_gateway")
        self.orchestrator = MicroserviceOrchestrator()

        # Initialize service clients
        self.auth_client = AuthServiceClient(self.orchestrator)
        self.qa_client = QAServiceClient(self.orchestrator)

        # Rate limiting storage (in production, use Redis)
        self.request_counts: Dict[str, list] = {}

        # Setup FastAPI app
        self.app = FastAPI(
            title="Enterprise QA System API Gateway",
            description="Microservice-based QA System with Authentication and Caching",
            version="1.0.0"
        )

        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""
        # Health check
        @self.app.get("/health")
        async def health_check():
            try:
                health_results = await self.orchestrator.health_check_all_services()
                overall_healthy = all(health_results.values())

                return {
                    "status": "healthy" if overall_healthy else "degraded",
                    "timestamp": time.time(),
                    "services": {k.value: v for k, v in health_results.items()}
                }
            except Exception as e:
                self.logger.error("Health check failed", error=str(e))
                raise HTTPException(status_code=503, detail="Health check failed")

        # Auth endpoints
        @self.app.post("/login")
        async def login(username: str, password: str):
            try:
                response = await self.auth_client.login(username, password)
                if response.status == 200:
                    return response.data
                else:
                    raise HTTPException(status_code=response.status, detail=response.data)
            except Exception as e:
                self.logger.error("Login failed", error=str(e))
                raise HTTPException(status_code=401, detail="Invalid credentials")

        # Main QA endpoint
        @self.app.post("/query", response_model=QueryResponse)
        async def query_endpoint(request: QueryRequest):
            start_time = time.time()

            try:
                response = await self.qa_client.query(
                    query_text=request.query,
                    session_id=request.session_id,
                    top_k=request.top_k
                )

                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail=response.data)

                # Log request metrics
                duration = time.time() - start_time
                self.logger.info(
                    "Query processed",
                    query_len=len(request.query),
                    duration_ms=round(duration * 1000, 2),
                    status_code=response.status
                )

                return response.data

            except Exception as e:
                self.logger.error("Query failed", error=str(e))
                raise HTTPException(status_code=500, detail="Query processing failed")

        # Document ingestion
        @self.app.post("/ingest")
        async def ingest_document(file_path: str, doc_id: Optional[str] = None):
            try:
                response = await self.qa_client.ingest_document(file_path, doc_id)

                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail=response.data)

                return response.data
            except Exception as e:
                self.logger.error("Document ingestion failed", error=str(e))
                raise HTTPException(status_code=500, detail="Document ingestion failed")

        # Service metrics
        @self.app.get("/metrics")
        async def get_metrics():
            try:
                metrics = await self.orchestrator.get_system_metrics()
                return metrics
            except Exception as e:
                self.logger.error("Metrics retrieval failed", error=str(e))
                raise HTTPException(status_code=500, detail="Metrics retrieval failed")

    async def authenticate_request(self, request: Request) -> Optional[str]:
        """Authenticate incoming request"""
        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header.split(" ")[1]

        # Validate token with auth service
        response = await self.auth_client.validate_token(token)

        if response.status == 200 and response.data and response.data.get("valid"):
            return token

        return None

    def check_rate_limit(self, client_ip: str) -> bool:
        """Check if request exceeds rate limits"""
        if not hasattr(self, 'request_counts'):
            self.request_counts = {}

        current_time = time.time()
        window_start = current_time - self.config.rate_limit_window

        # Clean old requests
        if client_ip in self.request_counts:
            self.request_counts[client_ip] = [
                req_time for req_time in self.request_counts[client_ip]
                if req_time >= window_start
            ]
        else:
            self.request_counts[client_ip] = []

        # Check limit
        if len(self.request_counts[client_ip]) >= self.config.rate_limit_requests:
            return False

        # Add current request
        self.request_counts[client_ip].append(current_time)
        return True

    def get_app(self):
        """Get the FastAPI application instance"""
        return self.app


# Global gateway instance
_gateway = None


def get_api_gateway() -> APIServiceGateway:
    """Get the global API gateway instance"""
    global _gateway
    if _gateway is None:
        _gateway = APIServiceGateway()
    return _gateway


def get_fastapi_app():
    """Get the FastAPI app for running with uvicorn"""
    gateway = get_api_gateway()
    return gateway.get_app()


# For running directly with uvicorn
app = get_fastapi_app()


if __name__ == "__main__":
    import uvicorn

    # Run the API gateway
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)