"""
Microservices Architecture Configuration
Defines the service mesh for the enterprise QA system
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import yaml
import json


@dataclass
class ServiceConfig:
    """Configuration for a microservice"""
    name: str
    port: int
    host: str = "0.0.0.0"
    replicas: int = 1
    health_check_path: str = "/health"
    dependencies: List[str] = None
    environment: Dict[str, str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.environment is None:
            self.environment = {}


class MicroserviceArchitecture:
    """Defines the complete microservice architecture"""

    def __init__(self):
        self.services = {
            "api_gateway": ServiceConfig(
                name="api_gateway",
                port=8000,
                dependencies=["auth_service", "qa_service"],
                environment={
                    "API_TIMEOUT": "30",
                    "MAX_CONNECTIONS": "100"
                }
            ),
            "auth_service": ServiceConfig(
                name="auth_service",
                port=8001,
                dependencies=[],
                environment={
                    "JWT_SECRET_KEY": "secret-key-change-in-production",
                    "ACCESS_TOKEN_EXPIRE_MINUTES": "30"
                }
            ),
            "qa_service": ServiceConfig(
                name="qa_service",
                port=8002,
                dependencies=["vector_store_service", "bm25_cache_service"],
                environment={
                    "VECTOR_STORE_URL": "http://vector_store_service:8004",
                    "BM25_CACHE_URL": "http://bm25_cache_service:8005"
                }
            ),
            "vector_store_service": ServiceConfig(
                name="vector_store_service",
                port=8004,
                dependencies=[],
                environment={
                    "VECTOR_DB_PROVIDER": "chromadb",
                    "CHROMA_PERSIST_DIR": "/data/vector_store"
                }
            ),
            "bm25_cache_service": ServiceConfig(
                name="bm25_cache_service",
                port=8005,
                dependencies=[],
                environment={
                    "CACHE_TTL_SECONDS": "3600",
                    "CACHE_MAX_SIZE": "1000"
                }
            ),
            "document_processor_service": ServiceConfig(
                name="document_processor_service",
                port=8006,
                dependencies=["storage_service"],
                environment={
                    "MAX_FILE_SIZE_MB": "50"
                }
            ),
            "storage_service": ServiceConfig(
                name="storage_service",
                port=8007,
                dependencies=[],
                environment={
                    "STORAGE_BACKEND": "local",
                    "STORAGE_PATH": "/data/documents"
                }
            ),
            "llm_service": ServiceConfig(
                name="llm_service",
                port=8008,
                dependencies=["auth_service"],
                environment={
                    "LLM_PROVIDER": "openai",
                    "LLM_MODEL": "gpt-4"
                }
            )
        }

    def get_service(self, service_name: str) -> Optional[ServiceConfig]:
        """Get configuration for a specific service"""
        return self.services.get(service_name)

    def get_dependencies(self, service_name: str) -> List[str]:
        """Get all dependencies for a service (including transitive)"""
        service = self.get_service(service_name)
        if not service:
            return []

        all_deps = set(service.dependencies)
        visited = set()

        while True:
            new_deps = set(all_deps)
            for dep in list(all_deps):
                if dep not in visited and dep in self.services:
                    visited.add(dep)
                    deps = self.services[dep].dependencies
                    new_deps.update(deps)

            if new_deps == all_deps:
                break

            all_deps = new_deps

        return list(all_deps)

    def generate_docker_compose(self) -> str:
        """Generate Docker Compose configuration"""
        compose_config = {
            "version": "3.8",
            "services": {},
            "networks": {
                "qa_network": {
                    "driver": "bridge"
                }
            },
            "volumes": {
                "vector_store_data": {},
                "document_storage": {},
                "cache_data": {}
            }
        }

        for service_name, config in self.services.items():
            service_def = {
                "build": ".",
                "ports": [f"{config.port}:{config.port}"],
                "environment": config.environment,
                "depends_on": config.dependencies,
                "networks": ["qa_network"],
                "restart": "unless-stopped"
            }

            # Special volume mounts for persistent services
            if "vector_store" in service_name:
                service_def["volumes"] = ["vector_store_data:/data/vector_store"]
            elif "storage" in service_name:
                service_def["volumes"] = ["document_storage:/data/documents"]
            elif "cache" in service_name:
                service_def["volumes"] = ["cache_data:/data/cache"]

            compose_config["services"][service_name] = service_def

        return yaml.dump(compose_config, default_flow_style=False)

    def generate_kubernetes_deployment(self) -> Dict:
        """Generate Kubernetes deployment configuration"""
        k8s_config = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": "qa-system"
            }
        }

        deployments = []
        services = []

        for service_name, config in self.services.items():
            # Deployment
            deployment = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": service_name,
                    "namespace": "qa-system"
                },
                "spec": {
                    "replicas": config.replicas,
                    "selector": {
                        "matchLabels": {"app": service_name}
                    },
                    "template": {
                        "metadata": {
                            "labels": {"app": service_name}
                        },
                        "spec": {
                            "containers": [{
                                "name": service_name,
                                "image": f"qa-{service_name}:latest",
                                "ports": [{"containerPort": config.port}],
                                "env": [{"name": k, "value": v} for k, v in config.environment.items()],
                                "resources": {
                                    "requests": {
                                        "memory": "256Mi",
                                        "cpu": "100m"
                                    },
                                    "limits": {
                                        "memory": "512Mi",
                                        "cpu": "500m"
                                    }
                                }
                            }]
                        }
                    }
                }
            }

            # Service
            svc = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": service_name,
                    "namespace": "qa-system"
                },
                "spec": {
                    "selector": {"app": service_name},
                    "ports": [{"port": config.port, "targetPort": config.port}]
                }
            }

            deployments.append(deployment)
            services.append(svc)

        return {
            "namespace": k8s_config,
            "deployments": deployments,
            "services": services
        }

    def generate_load_balancer_config(self) -> str:
        """Generate load balancer configuration"""
        lb_config = """
# Load Balancer Configuration for QA System
upstream api_gateway {
    server api_gateway:8000;
}

server {
    listen 80;

    location /api/ {
        proxy_pass http://api_gateway;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        proxy_pass http://api_gateway/health;
    }
}
"""
        return lb_config


# Service Discovery Interface
class ServiceDiscovery:
    """Interface for service discovery"""

    def register_service(self, service_name: str, host: str, port: int):
        """Register a service"""
        raise NotImplementedError

    def discover_service(self, service_name: str) -> Optional[Dict[str, any]]:
        """Discover a service"""
        raise NotImplementedError

    def deregister_service(self, service_name: str):
        """Deregister a service"""
        raise NotImplementedError


# Health Check Interface
class HealthChecker:
    """Interface for service health checking"""

    def check_health(self, service_name: str) -> Dict[str, any]:
        """Check health of a service"""
        raise NotImplementedError

    def get_overall_health(self) -> Dict[str, any]:
        """Get overall system health"""
        raise NotImplementedError


if __name__ == "__main__":
    arch = MicroserviceArchitecture()

    print("=== Service Dependencies ===")
    for service_name in arch.services.keys():
        deps = arch.get_dependencies(service_name)
        print(f"{service_name}: {deps}")

    print("\n=== Docker Compose Configuration ===")
    print(arch.generate_docker_compose())

    print("\n=== Load Balancer Configuration ===")
    print(arch.generate_load_balancer_config())