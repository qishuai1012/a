"""
Security and Authentication Module
Implements JWT authentication, role-based access control, and data security measures
"""

import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import structlog
from functools import wraps
import asyncio
from enum import Enum
import bcrypt

# For token storage and blacklisting
from typing import Set
import time


class UserRole(Enum):
    """User roles for access control"""
    ADMIN = "admin"
    USER = "user"
    READ_ONLY = "read_only"
    GUEST = "guest"


@dataclass
class User:
    """User representation"""
    user_id: str
    username: str
    role: UserRole
    email: Optional[str] = None
    permissions: Optional[List[str]] = None
    created_at: datetime = None
    last_login: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class TokenPayload:
    """JWT token payload"""
    user_id: str
    username: str
    role: str
    exp: int
    iat: int
    jti: str  # JWT ID for blacklisting


class SecurityConfig:
    """Security configuration"""
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
        password_salt_rounds: int = 12,
        max_login_attempts: int = 5,
        login_lockout_duration: int = 300  # 5 minutes
    ):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.password_salt_rounds = password_salt_rounds
        self.max_login_attempts = max_login_attempts
        self.login_lockout_duration = login_lockout_duration


class TokenBlacklist:
    """Simple in-memory token blacklist (for production, use Redis)"""

    def __init__(self):
        self.blacklisted_tokens: Set[str] = set()
        self.token_expiry_times: Dict[str, float] = {}

    def add_token(self, jti: str, exp_time: float):
        """Add a token to the blacklist"""
        self.blacklisted_tokens.add(jti)
        self.token_expiry_times[jti] = exp_time

    def is_blacklisted(self, jti: str) -> bool:
        """Check if a token is blacklisted"""
        current_time = time.time()

        # Clean up expired tokens
        expired_tokens = [
            token for token, expiry in self.token_expiry_times.items()
            if expiry < current_time
        ]
        for token in expired_tokens:
            self.blacklisted_tokens.discard(token)
            self.token_expiry_times.pop(token, None)

        return jti in self.blacklisted_tokens


class AuthenticationService:
    """Authentication service for user management and token handling"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.users: Dict[str, User] = {}  # In production, use a database
        self.logger = structlog.get_logger(__name__).bind(component="auth_service")
        self.failed_login_attempts: Dict[str, int] = {}  # user_id -> attempts
        self.lockout_times: Dict[str, float] = {}  # user_id -> timestamp
        self.token_blacklist = TokenBlacklist()

    def register_user(
        self,
        user_id: str,
        username: str,
        password: str,
        role: UserRole = UserRole.USER,
        email: Optional[str] = None
    ) -> User:
        """Register a new user"""
        if user_id in self.users:
            raise ValueError(f"User with id {user_id} already exists")

        # Hash password
        hashed_password = self._hash_password(password)

        user = User(
            user_id=user_id,
            username=username,
            role=role,
            email=email,
            permissions=self._get_default_permissions(role)
        )

        # Store user (password would typically be stored separately in DB)
        self.users[user_id] = user
        self.logger.info("User registered", user_id=user_id, username=username)

        return user

    def authenticate_user(self, username: str, password: str) -> Optional[TokenPayload]:
        """Authenticate user and return JWT token"""
        # Find user by username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break

        if not user:
            self.logger.warning("Authentication failed", reason="user_not_found", username=username)
            return None

        # Check if user is locked out
        if self._is_user_locked_out(user.user_id):
            self.logger.warning("Authentication failed", reason="account_locked", user_id=user.user_id)
            return None

        # Verify password
        if not self._verify_password(password, self._get_hashed_password_for_user(user)):
            # Increment failed attempts
            self.failed_login_attempts[user.user_id] = \
                self.failed_login_attempts.get(user.user_id, 0) + 1

            if self.failed_login_attempts[user.user_id] >= self.config.max_login_attempts:
                self.lockout_times[user.user_id] = time.time()
                self.logger.warning("Account locked due to failed login attempts", user_id=user.user_id)

            self.logger.warning("Authentication failed", reason="invalid_credentials", user_id=user.user_id)
            return None

        # Reset failed attempts on successful login
        self.failed_login_attempts.pop(user.user_id, None)

        # Update last login time
        user.last_login = datetime.utcnow()

        # Generate access token
        return self._generate_access_token(user)

    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt(rounds=self.config.password_salt_rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def _get_hashed_password_for_user(self, user: User) -> str:
        """Get a fake hashed password for demo purposes"""
        # In a real system, passwords would be securely stored in DB
        return "$2b$12$LQv3c1yqJ9.nkQZ.yFm48OVi8.kg3R7vzLQv3c1yqJ9.nkQZ.yFm48O"  # Demo hash

    def _get_default_permissions(self, role: UserRole) -> List[str]:
        """Get default permissions based on role"""
        permissions_map = {
            UserRole.ADMIN: ["read", "write", "delete", "admin"],
            UserRole.USER: ["read", "write"],
            UserRole.READ_ONLY: ["read"],
            UserRole.GUEST: ["read"]
        }
        return permissions_map.get(role, [])

    def _is_user_locked_out(self, user_id: str) -> bool:
        """Check if a user account is locked out"""
        lockout_time = self.lockout_times.get(user_id)
        if lockout_time:
            elapsed = time.time() - lockout_time
            if elapsed < self.config.login_lockout_duration:
                return True
            else:
                # Remove lockout after expiration
                del self.lockout_times[user_id]
                return False
        return False

    def _generate_access_token(self, user: User) -> TokenPayload:
        """Generate JWT access token"""
        expire = datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)
        jti = secrets.token_urlsafe(32)  # Unique token identifier

        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "exp": int(expire.timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
            "jti": jti
        }

        token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)

        token_payload = TokenPayload(
            user_id=user.user_id,
            username=user.username,
            role=user.role.value,
            exp=int(expire.timestamp()),
            iat=int(datetime.utcnow().timestamp()),
            jti=jti
        )

        return token_payload

    def decode_token(self, token: str) -> Optional[TokenPayload]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, self.config.secret_key, algorithms=[self.config.algorithm])

            # Check if token is blacklisted
            jti = payload.get("jti")
            if jti and self.token_blacklist.is_blacklisted(jti):
                self.logger.warning("Token is blacklisted", jti=jti)
                return None

            return TokenPayload(
                user_id=payload["user_id"],
                username=payload["username"],
                role=payload["role"],
                exp=payload["exp"],
                iat=payload["iat"],
                jti=payload.get("jti", "")
            )
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning("Invalid token", error=str(e))
            return None

    def logout_token(self, token: str) -> bool:
        """Logout by adding token to blacklist"""
        try:
            payload = jwt.decode(token, self.config.secret_key, algorithms=[self.config.algorithm])
            jti = payload.get("jti")
            exp = payload.get("exp")

            if jti and exp:
                self.token_blacklist.add_token(jti, exp)
                self.logger.info("Token logged out", jti=jti)
                return True
        except jwt.InvalidTokenError:
            pass

        return False


class RBACManager:
    """Role-Based Access Control Manager"""

    def __init__(self, auth_service: AuthenticationService):
        self.auth_service = auth_service
        self.logger = structlog.get_logger(__name__).bind(component="rbac_manager")

    def has_permission(self, user_id: str, required_permission: str) -> bool:
        """Check if user has required permission"""
        user = self.auth_service.users.get(user_id)
        if not user:
            return False

        return required_permission in user.permissions

    def can_access_document(self, user_id: str, document_owner: str, document_permissions: List[str]) -> bool:
        """Check if user can access a specific document"""
        user = self.auth_service.users.get(user_id)
        if not user:
            return False

        # Admins can access everything
        if user.role == UserRole.ADMIN:
            return True

        # Document owners have full access
        if user_id == document_owner:
            return True

        # Check document-specific permissions
        for perm in document_permissions:
            if user.role.value in perm or user_id in perm:
                return True

        # Check user permissions
        for user_perm in user.permissions:
            if user_perm in document_permissions:
                return True

        return False

    def authorize_query(self, user_id: str, query: str) -> bool:
        """Authorize a query based on user permissions"""
        user = self.auth_service.users.get(user_id)
        if not user:
            return False

        # Prevent potentially dangerous queries (basic SQL injection prevention)
        dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'exec', 'execute']
        query_lower = query.lower()
        for keyword in dangerous_keywords:
            if keyword in query_lower:
                self.logger.warning("Potentially dangerous query blocked", user_id=user_id, query=query)
                return False

        # Basic check: users can make read queries
        return True


def require_auth(f):
    """Decorator to require authentication for API endpoints"""
    @wraps(f)
    async def decorated(*args, **kwargs):
        # This would integrate with the framework-specific way of getting headers
        # For example, in FastAPI, you'd get the token from request headers
        auth_header = kwargs.get('authorization') or kwargs.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            raise PermissionError("Authorization header missing or invalid")

        token = auth_header.split(' ')[1]
        auth_service = kwargs.get('auth_service')  # Pass auth_service in kwargs

        if not auth_service:
            raise ValueError("Auth service not provided")

        decoded_token = auth_service.decode_token(token)
        if not decoded_token:
            raise PermissionError("Invalid or expired token")

        # Add user info to the request context
        kwargs['current_user'] = decoded_token
        return await f(*args, **kwargs)
    return decorated


def require_permission(permission: str):
    """Decorator to require specific permission for API endpoints"""
    def decorator(f):
        @wraps(f)
        async def decorated(*args, **kwargs):
            current_user = kwargs.get('current_user')
            auth_service = kwargs.get('auth_service')
            rbac_manager = kwargs.get('rbac_manager')

            if not current_user or not auth_service or not rbac_manager:
                raise PermissionError("Missing authentication context")

            has_perm = rbac_manager.has_permission(current_user.user_id, permission)
            if not has_perm:
                raise PermissionError(f"Permission '{permission}' required")

            return await f(*args, **kwargs)
        return decorated
    return decorator


# Data encryption utilities
class DataEncryptor:
    """Simple data encryption utility for sensitive information"""

    def __init__(self, key: bytes):
        from cryptography.fernet import Fernet
        self.cipher_suite = Fernet(key)

    def encrypt(self, data: str) -> str:
        """Encrypt data"""
        encrypted_bytes = self.cipher_suite.encrypt(data.encode('utf-8'))
        return encrypted_bytes.decode('utf-8')

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_data.encode('utf-8'))
        return decrypted_bytes.decode('utf-8')


# Utility function to generate secure keys
def generate_secure_key() -> str:
    """Generate a secure key for encryption"""
    return Fernet.generate_key().decode()


if __name__ == "__main__":
    from cryptography.fernet import Fernet

    # Example usage
    config = SecurityConfig()
    auth_service = AuthenticationService(config)
    rbac_manager = RBACManager(auth_service)

    # Register a user
    user = auth_service.register_user(
        user_id="user123",
        username="john_doe",
        password="secure_password",
        role=UserRole.USER,
        email="john@example.com"
    )

    print(f"Registered user: {user.username} with role: {user.role}")

    # Authenticate user
    token_payload = auth_service.authenticate_user("john_doe", "secure_password")
    if token_payload:
        print(f"Authentication successful, token generated for: {token_payload.username}")

        # Check permissions
        can_read = rbac_manager.has_permission("user123", "read")
        print(f"User has read permission: {can_read}")

    # Example of checking document access
    can_access = rbac_manager.can_access_document(
        user_id="user123",
        document_owner="user123",
        document_permissions=["read", "write"]
    )
    print(f"User can access document: {can_access}")