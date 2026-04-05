"""
Security Middleware for API Protection
Implements authentication, authorization, and rate limiting for API endpoints
"""

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import time
from collections import defaultdict, deque

from .authentication import AuthenticationService, RBACManager, TokenPayload
from .security_config import get_security_manager


class SecurityMiddleware:
    """Security middleware for API endpoints"""

    def __init__(self, auth_service: AuthenticationService, rbac_manager: RBACManager):
        self.auth_service = auth_service
        self.rbac_manager = rbac_manager
        self.security_manager = get_security_manager()

        # For rate limiting
        self.request_counts = defaultdict(deque)
        self.blocked_ips = set()

    async def authenticate_request(self, request: Request) -> Optional[TokenPayload]:
        """Authenticate incoming request"""
        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header.split(" ")[1]
        return self.auth_service.decode_token(token)

    async def check_rate_limit(self, request: Request) -> bool:
        """Check if request exceeds rate limits"""
        if not self.security_manager.settings.rate_limit_enabled:
            return True

        client_ip = request.client.host if request.client else "unknown"

        # Block if IP is in blocked list
        if client_ip in self.blocked_ips:
            return False

        current_time = time.time()
        window_start = current_time - 60  # 1 minute window

        # Remove old requests outside the window
        while (self.request_counts[client_ip] and
               self.request_counts[client_ip][0] < window_start):
            self.request_counts[client_ip].popleft()

        # Check if limit exceeded
        if len(self.request_counts[client_ip]) >= self.security_manager.settings.max_requests_per_minute:
            # Block this IP for security
            self.blocked_ips.add(client_ip)
            return False

        # Add current request
        self.request_counts[client_ip].append(current_time)
        return True

    async def log_access_attempt(self, request: Request, success: bool, user_id: Optional[str] = None):
        """Log access attempts for audit trail"""
        if self.security_manager.settings.audit_logging_enabled:
            # In a real system, this would log to a proper audit system
            import logging
            logger = logging.getLogger(__name__)

            client_ip = request.client.host if request.client else "unknown"
            status = "SUCCESS" if success else "FAILED"

            logger.info(f"Access {status} - IP: {client_ip}, User: {user_id}, Endpoint: {request.url.path}")


# Initialize security components
security = HTTPBearer()
security_middleware = None


def init_security_middleware(auth_service: AuthenticationService, rbac_manager: RBACManager):
    """Initialize the security middleware"""
    global security_middleware
    security_middleware = SecurityMiddleware(auth_service, rbac_manager)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenPayload:
    """Dependency to get current user from JWT token"""
    if not security_middleware:
        raise HTTPException(status_code=500, detail="Security middleware not initialized")

    token = credentials.credentials
    payload = security_middleware.auth_service.decode_token(token)

    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


async def require_permission(permission: str):
    """Create a dependency to require specific permission"""
    async def permission_dependency(current_user: TokenPayload = Depends(get_current_user)):
        if not security_middleware.rbac_manager.has_permission(current_user.user_id, permission):
            raise HTTPException(
                status_code=403,
                detail="Not enough permissions"
            )
        return current_user
    return permission_dependency


async def check_rate_limit(request: Request):
    """Dependency to check rate limits"""
    if not security_middleware:
        raise HTTPException(status_code=500, detail="Security middleware not initialized")

    is_allowed = await security_middleware.check_rate_limit(request)
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )


def verify_document_access(user_id: str, document_owner: str, document_permissions: list = None):
    """Verify that a user can access a specific document"""
    if not security_middleware:
        raise HTTPException(status_code=500, detail="Security middleware not initialized")

    doc_perms = document_permissions or ["read"]
    can_access = security_middleware.rbac_manager.can_access_document(
        user_id, document_owner, doc_perms
    )

    if not can_access:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this document"
        )


# Example of securing an endpoint
def secured_endpoint(require_auth: bool = True, require_permission: Optional[str] = None, check_rate_limit: bool = True):
    """Decorator to secure API endpoints"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request') or (args[0] if args and hasattr(args[0], 'headers') else None)

            if not request:
                raise HTTPException(status_code=500, detail="Request object not available")

            # Check rate limit
            if check_rate_limit:
                await check_rate_limit(request)

            # Require authentication
            if require_auth:
                current_user = await get_current_user(request)
                kwargs['current_user'] = current_user

            # Require specific permission
            if require_permission:
                if 'current_user' not in kwargs:
                    current_user = await get_current_user(request)
                else:
                    current_user = kwargs['current_user']

                user_perm = await require_permission(require_permission)
                user_perm(current_user)

            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage in API
async def example_secured_route(request: Request, current_user: TokenPayload = Depends(get_current_user)):
    """Example of a secured route"""
    return {
        "message": f"Hello {current_user.username}",
        "user_id": current_user.user_id,
        "role": current_user.role
    }


if __name__ == "__main__":
    # This would typically be initialized when starting the API server
    print("Security middleware module loaded.")
    print("Use init_security_middleware() to initialize with auth service and RBAC manager.")