"""
Service Orchestrator for Microservices Architecture
Coordinates communication between microservices in the QA system
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import structlog
from datetime import datetime
import time


class ServiceType(Enum):
    """Types of services in the QA system"""
    API_GATEWAY = "api_gateway"
    AUTH_SERVICE = "auth_service"
    QA_SERVICE = "qa_service"
    VECTOR_STORE_SERVICE = "vector_store_service"
    BM25_CACHE_SERVICE = "bm25_cache_service"
    DOCUMENT_PROCESSOR_SERVICE = "document_processor_service"
    STORAGE_SERVICE = "storage_service"
    LLM_SERVICE = "llm_service"


@dataclass
class ServiceEndpoint:
    """Represents a service endpoint"""
    service_type: ServiceType
    host: str
    port: int
    base_path: str = ""

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}{self.base_path}"


@dataclass
class ServiceRequest:
    """Represents a request to a microservice"""
    service_type: ServiceType
    endpoint: str
    method: str = "GET"
    data: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    timeout: int = 30


@dataclass
class ServiceResponse:
    """Represents a response from a microservice"""
    status: int
    data: Optional[Dict[str, Any]]
    headers: Optional[Dict[str, str]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class ServiceRegistry:
    """Registry of available services"""

    def __init__(self):
        self.endpoints: Dict[ServiceType, ServiceEndpoint] = {}
        self.health_status: Dict[ServiceType, bool] = {}
        self.response_times: Dict[ServiceType, List[float]] = {}

    def register_service(self, service_type: ServiceType, endpoint: ServiceEndpoint):
        """Register a service endpoint"""
        self.endpoints[service_type] = endpoint
        self.health_status[service_type] = False
        self.response_times[service_type] = []

    def get_endpoint(self, service_type: ServiceType) -> Optional[ServiceEndpoint]:
        """Get endpoint for a service"""
        return self.endpoints.get(service_type)

    def update_health(self, service_type: ServiceType, is_healthy: bool):
        """Update health status of a service"""
        self.health_status[service_type] = is_healthy

    def record_response_time(self, service_type: ServiceType, response_time: float):
        """Record response time for a service"""
        if service_type not in self.response_times:
            self.response_times[service_type] = []
        self.response_times[service_type].append(response_time)
        # Keep only last 100 measurements
        if len(self.response_times[service_type]) > 100:
            self.response_times[service_type] = self.response_times[service_type][-100:]


class MicroserviceOrchestrator:
    """Orchestrates communication between microservices"""

    def __init__(self):
        self.registry = ServiceRegistry()
        self.logger = structlog.get_logger(__name__).bind(component="orchestrator")
        self.session: Optional[aiohttp.ClientSession] = None

        # Initialize default endpoints
        self._init_default_endpoints()

    def _init_default_endpoints(self):
        """Initialize default service endpoints"""
        defaults = {
            ServiceType.API_GATEWAY: ServiceEndpoint(ServiceType.API_GATEWAY, "localhost", 8000),
            ServiceType.AUTH_SERVICE: ServiceEndpoint(ServiceType.AUTH_SERVICE, "localhost", 8001),
            ServiceType.QA_SERVICE: ServiceEndpoint(ServiceType.QA_SERVICE, "localhost", 8002),
            ServiceType.VECTOR_STORE_SERVICE: ServiceEndpoint(ServiceType.VECTOR_STORE_SERVICE, "localhost", 8004),
            ServiceType.BM25_CACHE_SERVICE: ServiceEndpoint(ServiceType.BM25_CACHE_SERVICE, "localhost", 8005),
            ServiceType.DOCUMENT_PROCESSOR_SERVICE: ServiceEndpoint(ServiceType.DOCUMENT_PROCESSOR_SERVICE, "localhost", 8006),
            ServiceType.STORAGE_SERVICE: ServiceEndpoint(ServiceType.STORAGE_SERVICE, "localhost", 8007),
            ServiceType.LLM_SERVICE: ServiceEndpoint(ServiceType.LLM_SERVICE, "localhost", 8008),
        }

        for service_type, endpoint in defaults.items():
            self.registry.register_service(service_type, endpoint)

    async def __aenter__(self):
        """Initialize the client session"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the client session"""
        if self.session:
            await self.session.close()

    async def call_service(self, request: ServiceRequest) -> ServiceResponse:
        """Make a call to a microservice"""
        if not self.session:
            raise RuntimeError("Orchestrator not properly initialized. Use 'async with' statement.")

        endpoint = self.registry.get_endpoint(request.service_type)
        if not endpoint:
            raise ValueError(f"No endpoint registered for service: {request.service_type}")

        url = f"{endpoint.url}{request.endpoint}"
        headers = request.headers or {}

        self.logger.info(
            "Calling service",
            service_type=request.service_type.value,
            url=url,
            method=request.method
        )

        start_time = time.time()

        try:
            async with self.session.request(
                method=request.method,
                url=url,
                json=request.data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                response_time = time.time() - start_time

                # Record response time and update health
                self.registry.record_response_time(request.service_type, response_time)
                self.registry.update_health(request.service_type, response.status < 500)

                response_data = await response.json() if response.content_length != 0 else None

                service_response = ServiceResponse(
                    status=response.status,
                    data=response_data,
                    headers=dict(response.headers)
                )

                self.logger.info(
                    "Service call completed",
                    service_type=request.service_type.value,
                    status=service_response.status,
                    response_time_ms=round(response_time * 1000, 2)
                )

                return service_response

        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            self.registry.record_response_time(request.service_type, response_time)
            self.registry.update_health(request.service_type, False)

            self.logger.error(
                "Service call timed out",
                service_type=request.service_type.value,
                timeout=request.timeout
            )

            return ServiceResponse(
                status=408,
                data={"error": "Service call timed out"},
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            response_time = time.time() - start_time
            self.registry.record_response_time(request.service_type, response_time)
            self.registry.update_health(request.service_type, False)

            self.logger.error(
                "Service call failed",
                service_type=request.service_type.value,
                error=str(e)
            )

            return ServiceResponse(
                status=500,
                data={"error": f"Service call failed: {str(e)}"},
                timestamp=datetime.utcnow()
            )

    async def health_check_all_services(self) -> Dict[ServiceType, bool]:
        """Perform health check on all registered services"""
        health_results = {}

        for service_type in self.registry.endpoints.keys():
            endpoint = self.registry.get_endpoint(service_type)
            health_request = ServiceRequest(
                service_type=service_type,
                endpoint="/health",
                method="GET",
                timeout=10
            )

            response = await self.call_service(health_request)
            is_healthy = response.status == 200

            health_results[service_type] = is_healthy
            self.registry.update_health(service_type, is_healthy)

            self.logger.info(
                "Health check completed",
                service_type=service_type.value,
                healthy=is_healthy,
                status_code=response.status
            )

        return health_results

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics for monitoring"""
        health_results = await self.health_check_all_services()

        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "services": {},
            "overall_health": all(health_results.values())
        }

        for service_type in self.registry.endpoints.keys():
            response_times = self.registry.response_times.get(service_type, [])
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            metrics["services"][service_type.value] = {
                "healthy": health_results[service_type],
                "response_time_avg_ms": round(avg_response_time * 1000, 2),
                "response_time_samples": len(response_times)
            }

        return metrics


# Service-specific clients
class AuthServiceClient:
    """Client for the authentication service"""

    def __init__(self, orchestrator: MicroserviceOrchestrator):
        self.orchestrator = orchestrator

    async def login(self, username: str, password: str) -> ServiceResponse:
        """Authenticate user"""
        request = ServiceRequest(
            service_type=ServiceType.AUTH_SERVICE,
            endpoint="/login",
            method="POST",
            data={"username": username, "password": password}
        )
        return await self.orchestrator.call_service(request)

    async def validate_token(self, token: str) -> ServiceResponse:
        """Validate JWT token"""
        request = ServiceRequest(
            service_type=ServiceType.AUTH_SERVICE,
            endpoint="/validate",
            method="POST",
            data={"token": token}
        )
        return await self.orchestrator.call_service(request)


class QAServiceClient:
    """Client for the QA service"""

    def __init__(self, orchestrator: MicroserviceOrchestrator):
        self.orchestrator = orchestrator

    async def query(self, query_text: str, session_id: Optional[str] = None, top_k: int = 5) -> ServiceResponse:
        """Process a query"""
        request = ServiceRequest(
            service_type=ServiceType.QA_SERVICE,
            endpoint="/query",
            method="POST",
            data={
                "query": query_text,
                "session_id": session_id,
                "top_k": top_k
            }
        )
        return await self.orchestrator.call_service(request)

    async def ingest_document(self, file_path: str, doc_id: Optional[str] = None) -> ServiceResponse:
        """Ingest a document"""
        request = ServiceRequest(
            service_type=ServiceType.QA_SERVICE,
            endpoint="/ingest",
            method="POST",
            data={
                "file_path": file_path,
                "doc_id": doc_id
            }
        )
        return await self.orchestrator.call_service(request)


class VectorStoreClient:
    """Client for the vector store service"""

    def __init__(self, orchestrator: MicroserviceOrchestrator):
        self.orchestrator = orchestrator

    async def search(self, query: str, top_k: int = 5) -> ServiceResponse:
        """Search in vector store"""
        request = ServiceRequest(
            service_type=ServiceType.VECTOR_STORE_SERVICE,
            endpoint="/search",
            method="POST",
            data={
                "query": query,
                "top_k": top_k
            }
        )
        return await self.orchestrator.call_service(request)

    async def add_document(self, content: str, doc_id: str) -> ServiceResponse:
        """Add document to vector store"""
        request = ServiceRequest(
            service_type=ServiceType.VECTOR_STORE_SERVICE,
            endpoint="/add",
            method="POST",
            data={
                "content": content,
                "doc_id": doc_id
            }
        )
        return await self.orchestrator.call_service(request)


class BM25CacheClient:
    """Client for the BM25 cache service"""

    def __init__(self, orchestrator: MicroserviceOrchestrator):
        self.orchestrator = orchestrator

    async def get_cached_response(self, query: str) -> ServiceResponse:
        """Get cached response for query"""
        request = ServiceRequest(
            service_type=ServiceType.BM25_CACHE_SERVICE,
            endpoint="/get",
            method="POST",
            data={"query": query}
        )
        return await self.orchestrator.call_service(request)

    async def cache_response(self, query: str, response: str) -> ServiceResponse:
        """Cache response for query"""
        request = ServiceRequest(
            service_type=ServiceType.BM25_CACHE_SERVICE,
            endpoint="/cache",
            method="POST",
            data={
                "query": query,
                "response": response
            }
        )
        return await self.orchestrator.call_service(request)


# Example usage
async def example_usage():
    """Example of how to use the orchestrator"""

    async with MicroserviceOrchestrator() as orchestrator:
        # Perform health check
        health_results = await orchestrator.health_check_all_services()
        print("Health Results:", health_results)

        # Get system metrics
        metrics = await orchestrator.get_system_metrics()
        print("System Metrics:", json.dumps(metrics, indent=2, default=str))

        # Use service-specific clients
        qa_client = QAServiceClient(orchestrator)
        response = await qa_client.query("What is RAG?", top_k=3)
        print("QA Response:", response.data)


if __name__ == "__main__":
    asyncio.run(example_usage())