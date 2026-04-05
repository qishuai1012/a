"""
Security Package for Enterprise QA System
"""

from .authentication import (
    AuthenticationService,
    RBACManager,
    User,
    UserRole,
    SecurityConfig,
    TokenPayload,
    generate_secure_key,
    DataEncryptor
)
from .security_config import SecuritySettings, SecurityManager, get_security_manager
from .middleware import (
    SecurityMiddleware,
    init_security_middleware,
    get_current_user,
    require_permission,
    check_rate_limit,
    verify_document_access,
    secured_endpoint
)

__all__ = [
    # Authentication
    'AuthenticationService',
    'RBACManager',
    'User',
    'UserRole',
    'SecurityConfig',
    'TokenPayload',
    'generate_secure_key',
    'DataEncryptor',

    # Security Config
    'SecuritySettings',
    'SecurityManager',
    'get_security_manager',

    # Middleware
    'SecurityMiddleware',
    'init_security_middleware',
    'get_current_user',
    'require_permission',
    'check_rate_limit',
    'verify_document_access',
    'secured_endpoint'
]