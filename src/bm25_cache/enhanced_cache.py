"""
Enhanced Multi-level Cache Module
Implements LRU cache, Redis cache, and persistent cache for enterprise-level performance
"""

import pickle
import hashlib
import time
import json
import logging
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from functools import wraps
import asyncio
import redis.asyncio as redis
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)

try:
    import aioredis
    AIREDIS_AVAILABLE = True
except ImportError:
    AIREDIS_AVAILABLE = False


@dataclass
class CacheConfig:
    """Configuration for cache system"""
    # In-memory cache settings
    lru_maxsize: int = 1000
    ttl_seconds: int = 3600

    # Redis cache settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl: int = 7200  # 2 hours

    # Persistent cache settings
    persistent_path: Optional[str] = "./data/persistent_cache.pkl"
    persistent_ttl: int = 86400  # 24 hours

    # Cache hierarchy priority: memory -> redis -> persistent
    enable_memory_cache: bool = True
    enable_redis_cache: bool = False  # Disabled by default to avoid dependency
    enable_persistent_cache: bool = True


class LRUCache:
    """
    Thread-safe Least Recently Used cache implementation
    """

    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 3600):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry has expired"""
        return time.time() - timestamp > self.ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                return None

            value, timestamp = self._cache[key]

            if self._is_expired(timestamp):
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        with self._lock:
            # Remove expired entries if cache is full
            while len(self._cache) >= self.maxsize:
                oldest_key = next(iter(self._cache))
                oldest_value, oldest_timestamp = self._cache[oldest_key]

                if self._is_expired(oldest_timestamp):
                    del self._cache[oldest_key]
                else:
                    # Remove oldest item if cache still full
                    del self._cache[oldest_key]
                    break

            self._cache[key] = (value, time.time())

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            return self._cache.pop(key, None) is not None

    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()


class RedisCache:
    """
    Redis-based cache implementation
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, ttl: int = 7200):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.ttl = ttl
        self.lock = threading.Lock()

    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis cache"""
        try:
            value = await self.redis_client.get(key)
            return value.decode('utf-8') if value else None
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")
            return None

    async def set(self, key: str, value: str) -> bool:
        """Set value in Redis cache"""
        try:
            result = await self.redis_client.setex(key, self.ttl, value)
            return result
        except Exception as e:
            logger.warning(f"Redis SET error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.warning(f"Redis DELETE error: {e}")
            return False


class PersistentCache:
    """
    Persistent cache using file storage
    """

    def __init__(self, file_path: str, ttl: int = 86400):
        self.file_path = Path(file_path)
        self.ttl = ttl
        self.lock = threading.Lock()

        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_cache(self) -> Dict[str, Tuple[Any, float]]:
        """Load cache from file"""
        if not self.file_path.exists():
            return {}

        try:
            with open(self.file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading persistent cache: {e}")
            return {}

    def _save_cache(self, cache: Dict[str, Tuple[Any, float]]) -> None:
        """Save cache to file"""
        try:
            with open(self.file_path, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            logger.error(f"Error saving persistent cache: {e}")

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry has expired"""
        return time.time() - timestamp > self.ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache"""
        cache = self._load_cache()

        if key in cache:
            value, timestamp = cache[key]

            if self._is_expired(timestamp):
                # Remove expired entry
                del cache[key]
                self._save_cache(cache)
                return None

            return value

        return None

    def set(self, key: str, value: Any) -> bool:
        """Set value in persistent cache"""
        cache = self._load_cache()
        cache[key] = (value, time.time())
        self._save_cache(cache)
        return True

    def delete(self, key: str) -> bool:
        """Delete key from persistent cache"""
        cache = self._load_cache()
        result = cache.pop(key, None) is not None
        if result:
            self._save_cache(cache)
        return result

    def clear(self) -> None:
        """Clear all persistent cache entries"""
        if self.file_path.exists():
            self.file_path.unlink()


class MultiLevelCache:
    """
    Multi-level cache system with memory -> Redis -> persistent hierarchy
    """

    def __init__(self, config: CacheConfig):
        self.config = config

        # Initialize caches based on configuration
        if config.enable_memory_cache:
            self.memory_cache = LRUCache(config.lru_maxsize, config.ttl_seconds)
        else:
            self.memory_cache = None

        if config.enable_redis_cache and AIREDIS_AVAILABLE:
            self.redis_cache = RedisCache(config.redis_host, config.redis_port, config.redis_db, config.redis_ttl)
        else:
            self.redis_cache = None

        if config.enable_persistent_cache:
            self.persistent_cache = PersistentCache(config.persistent_path, config.persistent_ttl)
        else:
            self.persistent_cache = None

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache hierarchy"""
        # Try memory cache first
        if self.memory_cache:
            value = self.memory_cache.get(key)
            if value is not None:
                logger.debug(f"Cache HIT (memory): {key}")
                return value

        # Try Redis cache
        if self.redis_cache:
            import asyncio
            try:
                # Run async Redis operation in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                value = loop.run_until_complete(self.redis_cache.get(key))
                loop.close()

                if value is not None:
                    logger.debug(f"Cache HIT (Redis): {key}")

                    # Populate memory cache if available
                    if self.memory_cache:
                        self.memory_cache.set(key, value)

                    return json.loads(value)
            except RuntimeError:
                # Handle event loop issues
                pass

        # Try persistent cache
        if self.persistent_cache:
            value = self.persistent_cache.get(key)
            if value is not None:
                logger.debug(f"Cache HIT (persistent): {key}")

                # Populate higher-level caches if available
                if self.memory_cache:
                    self.memory_cache.set(key, value)

                return value

        logger.debug(f"Cache MISS: {key}")
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in all applicable cache levels"""
        # Set in memory cache
        if self.memory_cache:
            self.memory_cache.set(key, value)

        # Set in Redis cache
        if self.redis_cache:
            import asyncio
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.redis_cache.set(key, json.dumps(value)))
                loop.close()
            except RuntimeError:
                # Handle event loop issues
                pass

        # Set in persistent cache
        if self.persistent_cache:
            self.persistent_cache.set(key, value)

    def delete(self, key: str) -> bool:
        """Delete from all cache levels"""
        results = []

        if self.memory_cache:
            results.append(self.memory_cache.delete(key))

        if self.redis_cache:
            import asyncio
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.redis_cache.delete(key))
                results.append(result)
                loop.close()
            except RuntimeError:
                results.append(False)

        if self.persistent_cache:
            results.append(self.persistent_cache.delete(key))

        return any(results)  # Return True if any cache deletion succeeded

    def clear_all(self) -> None:
        """Clear all cache levels"""
        if self.memory_cache:
            self.memory_cache.clear()

        if self.persistent_cache:
            self.persistent_cache.clear()


class CacheDecorator:
    """
    Decorator for caching function results with multi-level cache
    """

    def __init__(self, cache: MultiLevelCache, ttl: Optional[int] = None):
        self.cache = cache
        self.ttl = ttl

    def __call__(self, func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            key = self.cache._generate_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            cached_result = self.cache.get(key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            self.cache.set(key, result)
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = self.cache._generate_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            cached_result = self.cache.get(key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            self.cache.set(key, result)
            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


# Enhanced BM25 Cache with multi-level caching
class EnhancedBM25Cache:
    """
    Enhanced BM25 cache with multi-level caching and better performance
    """

    def __init__(self, config: CacheConfig):
        self.multi_cache = MultiLevelCache(config)
        self.config = config
        self.lock = threading.Lock()

    def get_cached_response(self, query: str) -> Optional[str]:
        """Get cached response for query"""
        key = f"bm25_response:{hashlib.sha256(query.encode()).hexdigest()}"
        return self.multi_cache.get(key)

    def cache_response(self, query: str, response: str) -> None:
        """Cache response for query"""
        key = f"bm25_response:{hashlib.sha256(query.encode()).hexdigest()}"
        self.multi_cache.set(key, response)

    def get_similar_query(self, query: str) -> Optional[str]:
        """Find similar query in cache (simplified version)"""
        # This would normally involve checking for semantic similarity
        # For now, we'll use a simplified approach
        key = f"bm25_query:{hashlib.sha256(query.encode()).hexdigest()}"
        cached_query = self.multi_cache.get(key)
        if cached_query and self._queries_are_similar(cached_query, query):
            return cached_query
        return None

    def _queries_are_similar(self, query1: str, query2: str, threshold: float = 0.8) -> bool:
        """Simple similarity check (could be enhanced with more sophisticated algorithms)"""
        # Calculate Jaccard similarity of tokens
        tokens1 = set(query1.lower().split())
        tokens2 = set(query2.lower().split())

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        if len(union) == 0:
            return False

        similarity = len(intersection) / len(union)
        return similarity >= threshold

    def add_query_pair(self, original_query: str, cached_query: str) -> None:
        """Add query pair to cache"""
        key = f"bm25_query:{hashlib.sha256(original_query.encode()).hexdigest()}"
        self.multi_cache.set(key, cached_query)


# Global cache instance
global_cache = None


def get_global_cache(config: Optional[CacheConfig] = None) -> MultiLevelCache:
    """Get global cache instance"""
    global global_cache
    if global_cache is None:
        cache_config = config or CacheConfig()
        global_cache = MultiLevelCache(cache_config)
    return global_cache


if __name__ == "__main__":
    # Example usage
    config = CacheConfig(
        lru_maxsize=500,
        ttl_seconds=1800,
        persistent_path="./data/test_cache.pkl"
    )

    cache = MultiLevelCache(config)

    # Test caching
    cache.set("test_key", "test_value")
    value = cache.get("test_key")
    print(f"Cached value: {value}")

    # Test decorator
    @CacheDecorator(cache)
    def expensive_operation(x, y):
        time.sleep(0.1)  # Simulate expensive operation
        return x + y

    result = expensive_operation(1, 2)
    print(f"Result: {result}")

    # Second call should be cached
    result2 = expensive_operation(1, 2)
    print(f"Cached result: {result2}")