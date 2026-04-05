"""
Enterprise Error Handling and Circuit Breaker Module
Implements robust error handling, circuit breakers, and retry mechanisms
"""

import asyncio
import logging
import time
import functools
from enum import Enum
from typing import Callable, Any, Optional, Type, Union
from dataclasses import dataclass
from contextlib import contextmanager
import traceback
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)
from limits import storage, strategies


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Tripped, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Number of failures to trip circuit
    recovery_timeout: int = 60          # Time in seconds before allowing test requests
    expected_exception: Type[Exception] = Exception  # Exceptions that count as failures
    timeout: int = 30                   # Timeout for individual requests


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascade failures
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self._lock = threading.Lock()
        self.logger = structlog.get_logger(__name__).bind(component="circuit_breaker")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitState.OPEN:
                # Check if enough time has passed to try again
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                    )

        try:
            result = func(*args, **kwargs)

            with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    # Success in half-open state resets the circuit
                    self._reset()
                    self.logger.info("Circuit breaker reset after successful call in HALF_OPEN state")

            return result

        except self.config.expected_exception as e:
            self._record_failure()
            raise

        except Exception as e:
            # Count all other exceptions as failures too
            self._record_failure()
            raise

    def _record_failure(self):
        """Record a failure and potentially trip the circuit"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.logger.warning(
                    "Circuit breaker TRIPPED",
                    failure_count=self.failure_count,
                    threshold=self.config.failure_threshold
                )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt resetting the circuit"""
        if self.last_failure_time is None:
            return False

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.config.recovery_timeout

    def _reset(self):
        """Reset the circuit breaker to normal operation"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.logger.info("Circuit breaker RESET")


class CircuitBreakerOpenError(Exception):
    """Raised when attempting to call a service while circuit breaker is open"""
    pass


class RetryHandler:
    """
    Retry mechanism with exponential backoff
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exceptions: tuple = (Exception,),
        multiplier: float = 2.0
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exceptions = exceptions
        self.multiplier = multiplier
        self.logger = structlog.get_logger(__name__).bind(component="retry_handler")

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(
                        "Operation succeeded after retry",
                        attempt=attempt + 1,
                        function=func.__name__
                    )
                return result

            except self.exceptions as e:
                last_exception = e

                if attempt == self.max_attempts - 1:
                    # Final attempt failed
                    self.logger.error(
                        "Operation failed after all retries",
                        attempts=self.max_attempts,
                        function=func.__name__,
                        error=str(e)
                    )
                    break

                delay = min(
                    self.base_delay * (self.multiplier ** attempt),
                    self.max_delay
                )

                self.logger.warning(
                    "Operation failed, retrying",
                    attempt=attempt + 1,
                    max_attempts=self.max_attempts,
                    delay=delay,
                    error=str(e)
                )

                time.sleep(delay)

        raise last_exception


class RateLimiter:
    """
    Rate limiting implementation
    """

class RateLimiter:
    """
    Rate limiting implementation
    """

    def __init__(self, max_calls: int, time_unit: str = "1 second"):
        """
        Initialize rate limiter

        Args:
            max_calls: Maximum number of calls allowed
            time_unit: Time unit (e.g., "1 second", "1 minute", "1 hour")
        """
        self.max_calls = max_calls
        self.time_unit = time_unit

        # Convert time_unit to seconds
        if "minute" in time_unit.lower():
            self.window_seconds = 60
        elif "hour" in time_unit.lower():
            self.window_seconds = 3600
        else:  # default to seconds
            self.window_seconds = 1

        self.requests = {}  # key -> [(timestamp, count), ...]
        self.logger = structlog.get_logger(__name__).bind(component="rate_limiter")

    def check_limit(self, key: str) -> bool:
        """Check if the rate limit has been exceeded for the given key"""
        import time
        current_time = time.time()

        # Clean old requests outside the time window
        if key in self.requests:
            self.requests[key] = [
                (req_time, count) for req_time, count in self.requests[key]
                if current_time - req_time < self.window_seconds
            ]
        else:
            self.requests[key] = []

        # Check if we're over the limit
        total_requests = sum(count for _, count in self.requests[key])

        if total_requests >= self.max_calls:
            self.logger.warning("Rate limit exceeded", key=key, limit=self.max_calls)
            return False

        # Add current request
        self.requests[key].append((current_time, 1))
        return True


class ErrorHandler:
    """
    Centralized error handler with logging and metrics
    """

    def __init__(self, enable_metrics: bool = True):
        self.logger = structlog.get_logger(__name__)
        self.enable_metrics = enable_metrics
        self.metrics = {
            'total_errors': 0,
            'error_by_type': {},
            'last_error_time': None
        }

    def handle_error(
        self,
        error: Exception,
        context: Optional[dict] = None,
        should_raise: bool = True
    ) -> Optional[Exception]:
        """Handle an error with logging and metrics"""
        error_type = type(error).__name__

        # Log the error
        log_context = {
            'error_type': error_type,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat()
        }

        if context:
            log_context.update(context)

        self.logger.error("Error occurred", **log_context)
        self.logger.error("Full traceback", traceback=traceback.format_exc())

        # Update metrics
        if self.enable_metrics:
            self.metrics['total_errors'] += 1
            self.metrics['error_by_type'][error_type] = \
                self.metrics['error_by_type'].get(error_type, 0) + 1
            self.metrics['last_error_time'] = datetime.now()

        if should_raise:
            return error
        return None

    def get_metrics(self) -> dict:
        """Get error handling metrics"""
        return self.metrics.copy()


# Decorators for easy integration

def circuit_breaker(config: CircuitBreakerConfig):
    """Decorator to apply circuit breaker to a function"""
    circuit_breaker_instance = CircuitBreaker(config)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return circuit_breaker_instance.call(func, *args, **kwargs)
        return wrapper
    return decorator


def retry_handler(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """Decorator to apply retry logic to a function"""
    retry_handler_instance = RetryHandler(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exceptions=exceptions
    )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return retry_handler_instance.execute(func, *args, **kwargs)
        return wrapper
    return decorator


def rate_limit(max_calls: int, time_unit: str = "1 second"):
    """Decorator to apply rate limiting to a function"""
    rate_limiter = RateLimiter(max_calls, time_unit)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__module__}.{func.__name__}"
            if not rate_limiter.check_limit(key):
                # Using a simple exception for rate limiting
                raise Exception(f"Rate limit exceeded for {key}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


class ComprehensiveErrorHandler:
    """
    Comprehensive error handling system that combines all error handling strategies
    """

    def __init__(self):
        self.error_handler = ErrorHandler()
        self.logger = structlog.get_logger(__name__).bind(component="comprehensive_error_handler")

    @contextmanager
    def error_context(self, operation: str, **context):
        """Context manager for error handling with automatic logging"""
        start_time = time.time()
        context['operation'] = operation

        try:
            yield
            duration = time.time() - start_time
            self.logger.info(
                "Operation completed successfully",
                operation=operation,
                duration=duration,
                **context
            )
        except Exception as e:
            duration = time.time() - start_time
            context['duration'] = duration
            self.error_handler.handle_error(e, context)
            raise

    def safe_execute(
        self,
        func: Callable,
        *args,
        error_fallback: Optional[Callable] = None,
        error_return_value: Any = None,
        **kwargs
    ) -> Any:
        """Safely execute a function with fallback options"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(
                "Function execution failed",
                function=func.__name__,
                error=str(e)
            )

            if error_fallback:
                try:
                    self.logger.info("Executing fallback function")
                    return error_fallback(*args, **kwargs)
                except Exception as fallback_error:
                    self.logger.error(
                        "Fallback function also failed",
                        error=str(fallback_error)
                    )

            if error_return_value is not None:
                self.logger.info("Returning error fallback value")
                return error_return_value

            raise


# Example usage configurations
DEFAULT_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=Exception,
    timeout=30
)

DEFAULT_RETRY_CONFIG = {
    'max_attempts': 3,
    'base_delay': 1.0,
    'max_delay': 60.0,
    'exceptions': (Exception,)
}


if __name__ == "__main__":
    import random

    # Example usage of circuit breaker
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=5  # 5 seconds
    )

    cb = CircuitBreaker(config)

    def unreliable_service():
        # Simulate 70% failure rate
        if random.random() < 0.7:
            raise ConnectionError("Service temporarily unavailable")
        return "Success!"

    # Test circuit breaker
    print("Testing circuit breaker...")
    for i in range(10):
        try:
            result = cb.call(unreliable_service)
            print(f"Call {i+1}: {result}")
        except CircuitBreakerOpenError as e:
            print(f"Call {i+1}: Circuit breaker OPEN - {e}")
        except Exception as e:
            print(f"Call {i+1}: Failed - {e}")

        time.sleep(0.5)

    # Example usage of retry handler
    print("\nTesting retry handler...")
    retry_hdl = RetryHandler(max_attempts=3, base_delay=0.5)

    def flaky_function():
        if random.random() < 0.8:  # 80% failure rate
            raise TimeoutError("Request timed out")
        return "Eventually successful!"

    try:
        result = retry_hdl.execute(flaky_function)
        print(f"Flaky function result: {result}")
    except Exception as e:
        print(f"Flaky function failed after retries: {e}")

    # Example usage of rate limiter
    print("\nTesting rate limiter...")
    rl = RateLimiter(max_calls=2, time_unit="5 seconds")

    for i in range(5):
        if rl.check_limit("test_user"):
            print(f"Request {i+1}: Allowed")
        else:
            print(f"Request {i+1}: Rate limited")
        time.sleep(1)