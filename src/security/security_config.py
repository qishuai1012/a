"""
Security Configuration Module
Manages security settings and sensitive data handling
"""

import os
from typing import Optional
from dataclasses import dataclass
from .authentication import SecurityConfig


@dataclass
class SecuritySettings:
    """Security settings for the application"""

    # JWT settings
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Password policy
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special_chars: bool = False

    # Rate limiting
    rate_limit_enabled: bool = True
    max_requests_per_minute: int = 60

    # API security
    cors_origins: list = None
    ssl_required: bool = True

    # Logging & Monitoring
    audit_logging_enabled: bool = True
    suspicious_activity_alerts: bool = True

    def __post_init__(self):
        # Load from environment variables if not set
        if not self.jwt_secret_key:
            self.jwt_secret_key = os.getenv("JWT_SECRET_KEY", "")

        if self.jwt_secret_key == "":
            # Generate a secure key if none provided (only for dev)
            import secrets
            self.jwt_secret_key = secrets.token_urlsafe(32)
            print("WARNING: Generated temporary JWT secret key. Set JWT_SECRET_KEY env var for production.")

        if self.cors_origins is None:
            self.cors_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else ["*"]


class SecurityManager:
    """Centralized security manager for the application"""

    def __init__(self, settings: Optional[SecuritySettings] = None):
        self.settings = settings or SecuritySettings()
        self.security_config = self._create_security_config()

    def _create_security_config(self):
        """Create security configuration from settings"""
        return SecurityConfig(
            secret_key=self.settings.jwt_secret_key,
            algorithm=self.settings.jwt_algorithm,
            access_token_expire_minutes=self.settings.access_token_expire_minutes,
            refresh_token_expire_days=self.settings.refresh_token_expire_days
        )

    def is_valid_password(self, password: str) -> bool:
        """Validate password according to security policy"""
        if len(password) < self.settings.password_min_length:
            return False

        if self.settings.password_require_uppercase and not any(c.isupper() for c in password):
            return False

        if self.settings.password_require_lowercase and not any(c.islower() for c in password):
            return False

        if self.settings.password_require_numbers and not any(c.isdigit() for c in password):
            return False

        if self.settings.password_require_special_chars and not any(not c.isalnum() for c in password):
            return False

        return True

    def get_security_headers(self) -> dict:
        """Return security headers for API responses"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        }


# Global security manager instance
_security_manager = None


def get_security_manager() -> SecurityManager:
    """Get the global security manager instance"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


if __name__ == "__main__":
    # Example usage
    sec_manager = get_security_manager()
    print(f"CORS Origins: {sec_manager.settings.cors_origins}")
    print(f"Valid password 'MyPass123': {sec_manager.is_valid_password('MyPass123')}")
    print(f"Security headers: {sec_manager.get_security_headers()}")