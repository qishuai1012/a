"""
Architecture Package for Enterprise QA System
"""

from .microservice_config import (
    ServiceConfig,
    MicroserviceArchitecture,
    ServiceDiscovery,
    HealthChecker
)
from .service_orchestrator import (
    ServiceType,
    ServiceEndpoint,
    ServiceRequest,
    ServiceResponse,
    ServiceRegistry,
    MicroserviceOrchestrator,
    AuthServiceClient,
    QAServiceClient,
    VectorStoreClient,
    BM25CacheClient
)
from .api_gateway import (
    GatewayConfig,
    APIServiceGateway,
    get_api_gateway,
    get_fastapi_app
)

__all__ = [
    # Microservice Config
    'ServiceConfig',
    'MicroserviceArchitecture',
    'ServiceDiscovery',
    'HealthChecker',

    # Service Orchestrator
    'ServiceType',
    'ServiceEndpoint',
    'ServiceRequest',
    'ServiceResponse',
    'ServiceRegistry',
    'MicroserviceOrchestrator',
    'AuthServiceClient',
    'QAServiceClient',
    'VectorStoreClient',
    'BM25CacheClient',

    # API Gateway
    'GatewayConfig',
    'APIServiceGateway',
    'get_api_gateway',
    'get_fastapi_app'
]