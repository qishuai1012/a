"""
Redis-based Multi-level Cache System for Enterprise RAG
Implements distributed caching with TTL and performance monitoring
"""

import redis
import pickle
import json
import hashlib
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta
import structlog
import time
from enum import Enum

class CacheLevel(Enum):
    """Cache level enumeration for multi-level caching"""
    L1_MEMORY = "l1_memory"      # Local memory cache (fastest)
    L2_REDIS = "l2_redis"        # Redis cache (distributed)
    L3_PERSISTENT = "l3_persistent"  # Persistent storage (slowest but reliable)

class RedisCacheManager:
    """
    Redis-based distributed cache manager
    Provides enterprise-level caching with TTL, serialization, and monitoring
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        default_ttl: int = 3600,  # 1 hour default TTL
        namespace: str = "qa_cache"
    ):
        """
        Initialize Redis cache manager

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis authentication password
            default_ttl: Default TTL in seconds
            namespace: Namespace for cache keys
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=False  # We'll handle decoding ourselves
        )

        self.default_ttl = default_ttl
        self.namespace = namespace
        self.logger = structlog.get_logger(__name__)

        # Test connection
        try:
            self.redis_client.ping()
            self.logger.info("Connected to Redis successfully")
        except redis.ConnectionError:
            self.logger.error("Failed to connect to Redis")
            raise

    def _make_key(self, key: str) -> str:
        """Create namespaced key"""
        return f"{self.namespace}:{key}"

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_level: CacheLevel = CacheLevel.L2_REDIS
    ) -> bool:
        """
        Set a value in Redis cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (overrides default)
            cache_level: Cache level (only L2_REDIS is implemented)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Serialize the value
            serialized_value = pickle.dumps(value)
            redis_key = self._make_key(key)
            ttl = ttl or self.default_ttl

            # Set with TTL
            result = self.redis_client.setex(redis_key, ttl, serialized_value)
            if result:
                self.logger.debug(f"Cached key: {key} with TTL: {ttl}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to cache key {key}: {e}")
            return False

    def get(self, key: str, cache_level: CacheLevel = CacheLevel.L2_REDIS) -> Optional[Any]:
        """
        Get a value from Redis cache

        Args:
            key: Cache key
            cache_level: Cache level (only L2_REDIS is implemented)

        Returns:
            Cached value or None if not found/expired
        """
        try:
            redis_key = self._make_key(key)
            serialized_value = self.redis_client.get(redis_key)

            if serialized_value is not None:
                value = pickle.loads(serialized_value)
                self.logger.debug(f"Cache hit for key: {key}")
                return value

            self.logger.debug(f"Cache miss for key: {key}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve key {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        try:
            redis_key = self._make_key(key)
            result = self.redis_client.delete(redis_key)
            return bool(result)
        except Exception as e:
            self.logger.error(f"Failed to delete key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            redis_key = self._make_key(key)
            return bool(self.redis_client.exists(redis_key))
        except Exception as e:
            self.logger.error(f"Failed to check existence of key {key}: {e}")
            return False

    def flush(self) -> bool:
        """Flush all keys in the namespace"""
        try:
            pattern = f"{self.namespace}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            self.logger.error(f"Failed to flush cache: {e}")
            return False

    def ttl(self, key: str) -> int:
        """Get TTL for a key"""
        try:
            redis_key = self._make_key(key)
            return self.redis_client.ttl(redis_key)
        except Exception as e:
            self.logger.error(f"Failed to get TTL for key {key}: {e}")
            return -1

    def get_info(self) -> Dict[str, Any]:
        """Get Redis server info"""
        try:
            info = self.redis_client.info()
            return {
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "total_commands_processed": info.get("total_commands_processed"),
                "uptime_in_seconds": info.get("uptime_in_seconds")
            }
        except Exception as e:
            self.logger.error(f"Failed to get Redis info: {e}")
            return {}


class MultiLevelCache:
    """
    Multi-level cache system combining memory, Redis, and persistent storage
    """

    def __init__(
        self,
        redis_config: Optional[Dict[str, Any]] = None,
        memory_maxsize: int = 1000,
        default_ttl: int = 3600
    ):
        """
        Initialize multi-level cache

        Args:
            redis_config: Redis configuration dict
            memory_maxsize: Maximum size of in-memory cache
            default_ttl: Default TTL for cached items
        """
        self.memory_cache = {}
        self.memory_maxsize = memory_maxsize
        self.default_ttl = default_ttl

        # Initialize Redis cache if config provided
        if redis_config:
            try:
                self.redis_cache = RedisCacheManager(**redis_config)
                self.use_redis = True
            except Exception as e:
                print(f"Warning: Could not initialize Redis cache: {e}")
                self.use_redis = False
        else:
            self.use_redis = False

    def _evict_memory_if_needed(self):
        """Evict oldest items from memory cache if needed"""
        if len(self.memory_cache) >= self.memory_maxsize:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_levels: Optional[list] = None
    ) -> Dict[CacheLevel, bool]:
        """
        Set value in multiple cache levels

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            cache_levels: Which cache levels to use (defaults to all)

        Returns:
            Dict mapping cache level to success status
        """
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]

        results = {}

        # Set in memory cache
        if CacheLevel.L1_MEMORY in cache_levels:
            self._evict_memory_if_needed()
            # Store with expiration time
            expiration_time = time.time() + (ttl or self.default_ttl)
            self.memory_cache[key] = {
                "value": value,
                "expires_at": expiration_time
            }
            results[CacheLevel.L1_MEMORY] = True

        # Set in Redis cache
        if self.use_redis and CacheLevel.L2_REDIS in cache_levels:
            results[CacheLevel.L2_REDIS] = self.redis_cache.set(
                key, value, ttl=ttl, cache_level=CacheLevel.L2_REDIS
            )

        return results

    def get(self, key: str, cache_levels: Optional[list] = None) -> tuple:
        """
        Get value from multiple cache levels (L1 -> L2 -> L3)

        Args:
            key: Cache key
            cache_levels: Which cache levels to check (defaults to all)

        Returns:
            Tuple of (value, source_cache_level) or (None, None)
        """
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]

        # Check memory cache first
        if CacheLevel.L1_MEMORY in cache_levels:
            if key in self.memory_cache:
                item = self.memory_cache[key]
                if time.time() < item["expires_at"]:
                    return item["value"], CacheLevel.L1_MEMORY
                else:
                    # Expired, remove from cache
                    del self.memory_cache[key]

        # Check Redis cache
        if self.use_redis and CacheLevel.L2_REDIS in cache_levels:
            value = self.redis_cache.get(key)
            if value is not None:
                # Also populate memory cache for faster subsequent access
                ttl = self.redis_cache.ttl(key)
                if ttl > 0:
                    self._evict_memory_if_needed()
                    self.memory_cache[key] = {
                        "value": value,
                        "expires_at": time.time() + ttl
                    }
                return value, CacheLevel.L2_REDIS

        return None, None

    def delete(self, key: str, cache_levels: Optional[list] = None) -> Dict[CacheLevel, bool]:
        """Delete from multiple cache levels"""
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]

        results = {}

        if CacheLevel.L1_MEMORY in cache_levels:
            if key in self.memory_cache:
                del self.memory_cache[key]
            results[CacheLevel.L1_MEMORY] = True

        if self.use_redis and CacheLevel.L2_REDIS in cache_levels:
            results[CacheLevel.L2_REDIS] = self.redis_cache.delete(key)

        return results


# Cache decorator for easy use
def cached(ttl: int = 3600, cache_param: str = "cache_manager", key_func=None):
    """
    Decorator to cache function results

    Args:
        ttl: Time-to-live in seconds
        cache_param: Name of the cache manager parameter in the function
        key_func: Function to generate cache key from function args
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get cache manager from kwargs or try to get it from instance
            cache_manager = kwargs.get(cache_param)
            if cache_manager is None and args:
                # Assume it's a method and cache manager is an attribute
                instance = args[0]
                cache_manager = getattr(instance, cache_param, None)

            if not cache_manager or not hasattr(cache_manager, 'get'):
                return func(*args, **kwargs)

            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Default key generation using function name and args
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args[1:])  # Skip 'self' if present
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            result, _ = cache_manager.get(key)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(key, result, ttl=ttl)
            return result

        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    redis_config = {
        "redis_host": "localhost",
        "redis_port": 6379,
        "default_ttl": 1800
    }

    try:
        cache_manager = MultiLevelCache(redis_config=redis_config)

        # Test caching
        cache_manager.set("test_key", {"data": "some_value"}, ttl=60)
        value, source = cache_manager.get("test_key")
        print(f"Retrieved: {value}, from: {source}")

        # Test cache info
        if hasattr(cache_manager, 'redis_cache'):
            info = cache_manager.redis_cache.get_info()
            print(f"Redis info: {info}")
    except Exception as e:
        print(f"Redis not available: {e}")
        # Fallback to memory-only cache
        cache_manager = MultiLevelCache()
        cache_manager.set("test_key", {"data": "some_value"}, ttl=60)
        value, source = cache_manager.get("test_key")
        print(f"Retrieved from memory: {value}, from: {source}")