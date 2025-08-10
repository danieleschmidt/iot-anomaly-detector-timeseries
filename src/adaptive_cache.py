#!/usr/bin/env python3
"""
Adaptive Caching System with TTL Optimization for IoT Anomaly Detection Platform
Provides intelligent caching with dynamic TTL adjustment, LRU eviction, and performance analytics.
"""

import asyncio
import hashlib
import json
import pickle
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import weakref
import gc
import sys

import numpy as np
import redis
from concurrent.futures import ThreadPoolExecutor

from .logging_config import setup_logging


class CacheBackend(Enum):
    """Available cache backends"""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    computation_time: float = 0.0
    hit_rate: float = 0.0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        return time.time() > (self.timestamp + self.ttl)
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds"""
        return time.time() - self.timestamp
    
    @property  
    def access_frequency(self) -> float:
        """Calculate access frequency (accesses per second)"""
        age = self.age_seconds
        return self.access_count / max(age, 1.0)


@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    expired_entries: int = 0
    memory_usage_bytes: int = 0
    avg_response_time_ms: float = 0.0
    hit_rate: float = 0.0
    
    def update_hit_rate(self) -> None:
        """Update hit rate calculation"""
        if self.total_requests > 0:
            self.hit_rate = self.cache_hits / self.total_requests


class CacheBackendInterface(ABC):
    """Abstract interface for cache backends"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key"""
        pass
    
    @abstractmethod
    async def set(self, entry: CacheEntry) -> bool:
        """Set cache entry"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cache entry by key"""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get number of entries in cache"""
        pass


class InMemoryCache(CacheBackendInterface):
    """High-performance in-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 256):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = asyncio.Lock()
        self.logger = setup_logging(self.__class__.__name__)
        
        # Background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired entries"""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {e}")
    
    async def _cleanup_expired(self):
        """Remove expired entries"""
        async with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
    
    async def _evict_lru(self):
        """Evict least recently used entries if necessary"""
        current_memory = sum(entry.size_bytes for entry in self.cache.values())
        
        while (len(self.cache) >= self.max_size or 
               current_memory > self.max_memory_bytes) and self.cache:
            # Remove oldest (LRU) entry
            key, entry = self.cache.popitem(last=False)
            current_memory -= entry.size_bytes
            self.logger.debug(f"Evicted LRU entry: {key}")
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key"""
        async with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired:
                del self.cache[key]
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            
            return entry
    
    async def set(self, entry: CacheEntry) -> bool:
        """Set cache entry"""
        async with self.lock:
            # Calculate size if not set
            if entry.size_bytes == 0:
                entry.size_bytes = self._calculate_size(entry.value)
            
            # Evict if necessary
            await self._evict_lru()
            
            # Add entry
            self.cache[entry.key] = entry
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete cache entry by key"""
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self.lock:
            self.cache.clear()
    
    async def size(self) -> int:
        """Get number of entries in cache"""
        return len(self.cache)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float)):
                return sys.getsizeof(value)
            elif isinstance(value, (list, tuple, dict)):
                return sys.getsizeof(pickle.dumps(value))
            else:
                return sys.getsizeof(str(value))
        except Exception:
            return 100  # Default estimate


class RedisCache(CacheBackendInterface):
    """Redis-based cache backend"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, 
                 db: int = 0, password: str = None):
        self.redis_client = redis.Redis(
            host=host, port=port, db=db, password=password,
            decode_responses=False
        )
        self.logger = setup_logging(self.__class__.__name__)
    
    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry for Redis storage"""
        return pickle.dumps(entry)
    
    def _deserialize_entry(self, data: bytes) -> CacheEntry:
        """Deserialize cache entry from Redis"""
        return pickle.loads(data)
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key"""
        try:
            data = self.redis_client.get(key)
            if data is None:
                return None
            
            entry = self._deserialize_entry(data)
            
            if entry.is_expired:
                await self.delete(key)
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Update in Redis
            self.redis_client.set(key, self._serialize_entry(entry), 
                                ex=int(entry.ttl))
            
            return entry
            
        except Exception as e:
            self.logger.error(f"Error getting cache entry {key}: {e}")
            return None
    
    async def set(self, entry: CacheEntry) -> bool:
        """Set cache entry"""
        try:
            data = self._serialize_entry(entry)
            result = self.redis_client.set(
                entry.key, data, ex=int(entry.ttl)
            )
            return bool(result)
        except Exception as e:
            self.logger.error(f"Error setting cache entry {entry.key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete cache entry by key"""
        try:
            result = self.redis_client.delete(key)
            return bool(result)
        except Exception as e:
            self.logger.error(f"Error deleting cache entry {key}: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        try:
            self.redis_client.flushdb()
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    async def size(self) -> int:
        """Get number of entries in cache"""
        try:
            return self.redis_client.dbsize()
        except Exception as e:
            self.logger.error(f"Error getting cache size: {e}")
            return 0


class HybridCache(CacheBackendInterface):
    """Hybrid cache combining memory and Redis"""
    
    def __init__(self, memory_cache: InMemoryCache, redis_cache: RedisCache,
                 l1_ratio: float = 0.8):
        self.memory_cache = memory_cache
        self.redis_cache = redis_cache
        self.l1_ratio = l1_ratio  # Fraction of hot data in memory
        self.logger = setup_logging(self.__class__.__name__)
    
    def _should_cache_in_memory(self, entry: CacheEntry) -> bool:
        """Decide whether to cache entry in memory (L1)"""
        # Cache in memory if:
        # 1. High access frequency
        # 2. Recent access
        # 3. Small size
        frequency_threshold = 0.1  # accesses per second
        recency_threshold = 300    # 5 minutes
        size_threshold = 1024 * 10 # 10KB
        
        return (entry.access_frequency > frequency_threshold or
                (time.time() - entry.last_access) < recency_threshold or
                entry.size_bytes < size_threshold)
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry, trying memory first, then Redis"""
        # Try L1 (memory) cache first
        entry = await self.memory_cache.get(key)
        if entry is not None:
            return entry
        
        # Try L2 (Redis) cache
        entry = await self.redis_cache.get(key)
        if entry is not None:
            # Promote to L1 if frequently accessed
            if self._should_cache_in_memory(entry):
                await self.memory_cache.set(entry)
            return entry
        
        return None
    
    async def set(self, entry: CacheEntry) -> bool:
        """Set cache entry in appropriate tier"""
        # Always store in L2 (Redis)
        redis_success = await self.redis_cache.set(entry)
        
        # Store in L1 (memory) if criteria met
        if self._should_cache_in_memory(entry):
            memory_success = await self.memory_cache.set(entry)
            return redis_success and memory_success
        
        return redis_success
    
    async def delete(self, key: str) -> bool:
        """Delete from both tiers"""
        memory_result = await self.memory_cache.delete(key)
        redis_result = await self.redis_cache.delete(key)
        return memory_result or redis_result
    
    async def clear(self) -> None:
        """Clear both tiers"""
        await self.memory_cache.clear()
        await self.redis_cache.clear()
    
    async def size(self) -> int:
        """Get total entries across tiers"""
        memory_size = await self.memory_cache.size()
        redis_size = await self.redis_cache.size()
        return redis_size  # Redis is source of truth


class TTLOptimizer:
    """Intelligent TTL optimization based on access patterns"""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.optimal_ttls: Dict[str, float] = {}
        self.logger = setup_logging(self.__class__.__name__)
        
        # Default TTL values by data type/pattern
        self.default_ttls = {
            'model_prediction': 300,    # 5 minutes
            'preprocessed_data': 600,   # 10 minutes  
            'aggregated_metrics': 1800, # 30 minutes
            'configuration': 3600,      # 1 hour
            'static_reference': 86400   # 24 hours
        }
    
    def record_access(self, key: str, computation_time: float) -> None:
        """Record access pattern for TTL optimization"""
        self.access_patterns[key].append(time.time())
        
        # Keep only recent history (last 100 accesses)
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    def calculate_optimal_ttl(self, key: str, data_type: str = None) -> float:
        """Calculate optimal TTL based on access patterns"""
        if key not in self.access_patterns or len(self.access_patterns[key]) < 2:
            # Use default TTL for data type
            return self.default_ttls.get(data_type, 300)
        
        access_times = self.access_patterns[key]
        
        # Calculate access intervals
        intervals = [
            access_times[i] - access_times[i-1]
            for i in range(1, len(access_times))
        ]
        
        if not intervals:
            return self.default_ttls.get(data_type, 300)
        
        # Use average interval with some buffer
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # TTL should be longer than average interval but not excessive
        optimal_ttl = min(
            max(avg_interval + std_interval, 60),  # Minimum 1 minute
            86400  # Maximum 24 hours
        )
        
        # Apply exponential smoothing if we have previous optimal TTL
        if key in self.optimal_ttls:
            optimal_ttl = (self.learning_rate * optimal_ttl + 
                          (1 - self.learning_rate) * self.optimal_ttls[key])
        
        self.optimal_ttls[key] = optimal_ttl
        return optimal_ttl
    
    def get_ttl_recommendation(self, key: str, data_type: str = None, 
                              computation_cost: float = 1.0) -> float:
        """Get TTL recommendation considering computation cost"""
        base_ttl = self.calculate_optimal_ttl(key, data_type)
        
        # Adjust based on computation cost
        # Higher cost → longer TTL
        cost_multiplier = max(1.0, np.log(computation_cost + 1))
        recommended_ttl = base_ttl * cost_multiplier
        
        return min(recommended_ttl, 86400)  # Cap at 24 hours


class AdaptiveCache:
    """Adaptive caching system with intelligent TTL optimization"""
    
    def __init__(
        self,
        backend_type: CacheBackend = CacheBackend.MEMORY,
        max_size: int = 1000,
        max_memory_mb: int = 256,
        redis_config: Dict[str, Any] = None,
        enable_analytics: bool = True
    ):
        self.backend_type = backend_type
        self.enable_analytics = enable_analytics
        self.logger = setup_logging(self.__class__.__name__)
        
        # Initialize backend
        self.backend = self._create_backend(
            backend_type, max_size, max_memory_mb, redis_config or {}
        )
        
        # TTL optimizer
        self.ttl_optimizer = TTLOptimizer()
        
        # Statistics
        self.stats = CacheStats()
        self._stats_lock = threading.Lock()
        
        # Performance tracking
        self.request_times: List[float] = []
        
        # Background tasks
        self._analytics_task = None
        if enable_analytics:
            self._start_analytics_task()
    
    def _create_backend(
        self, 
        backend_type: CacheBackend, 
        max_size: int, 
        max_memory_mb: int,
        redis_config: Dict[str, Any]
    ) -> CacheBackendInterface:
        """Create appropriate cache backend"""
        if backend_type == CacheBackend.MEMORY:
            return InMemoryCache(max_size, max_memory_mb)
        elif backend_type == CacheBackend.REDIS:
            return RedisCache(**redis_config)
        elif backend_type == CacheBackend.HYBRID:
            memory_cache = InMemoryCache(max_size // 2, max_memory_mb // 2)
            redis_cache = RedisCache(**redis_config)
            return HybridCache(memory_cache, redis_cache)
        else:
            raise ValueError(f"Unsupported cache backend: {backend_type}")
    
    def _start_analytics_task(self):
        """Start background analytics task"""
        if self._analytics_task is None or self._analytics_task.done():
            self._analytics_task = asyncio.create_task(self._periodic_analytics())
    
    async def _periodic_analytics(self):
        """Periodic analytics and optimization"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._update_analytics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cache analytics: {e}")
    
    async def _update_analytics(self):
        """Update cache analytics and optimization"""
        with self._stats_lock:
            self.stats.update_hit_rate()
            
            # Calculate average response time
            if self.request_times:
                self.stats.avg_response_time_ms = np.mean(self.request_times)
                # Keep only recent measurements
                self.request_times = self.request_times[-1000:]
        
        self.logger.info(
            f"Cache Analytics - Hit Rate: {self.stats.hit_rate:.3f}, "
            f"Avg Response: {self.stats.avg_response_time_ms:.2f}ms"
        )
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments"""
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(
        self, 
        key: str, 
        func: Callable = None, 
        *args, 
        data_type: str = None,
        **kwargs
    ) -> Tuple[Any, bool]:
        """Get value from cache or compute it"""
        start_time = time.time()
        
        with self._stats_lock:
            self.stats.total_requests += 1
        
        # Try to get from cache
        entry = await self.backend.get(key)
        
        if entry is not None:
            # Cache hit
            with self._stats_lock:
                self.stats.cache_hits += 1
            
            response_time = (time.time() - start_time) * 1000
            self.request_times.append(response_time)
            
            self.logger.debug(f"Cache hit for key: {key[:20]}...")
            return entry.value, True
        
        # Cache miss - compute value if function provided
        if func is None:
            with self._stats_lock:
                self.stats.cache_misses += 1
            return None, False
        
        # Compute value
        compute_start = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                value = await func(*args, **kwargs)
            else:
                value = func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error computing value for key {key}: {e}")
            with self._stats_lock:
                self.stats.cache_misses += 1
            raise
        
        computation_time = time.time() - compute_start
        
        # Get optimal TTL
        optimal_ttl = self.ttl_optimizer.get_ttl_recommendation(
            key, data_type, computation_time
        )
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=optimal_ttl,
            computation_time=computation_time
        )
        
        # Store in cache
        await self.backend.set(entry)
        
        # Record access pattern
        self.ttl_optimizer.record_access(key, computation_time)
        
        with self._stats_lock:
            self.stats.cache_misses += 1
        
        response_time = (time.time() - start_time) * 1000
        self.request_times.append(response_time)
        
        self.logger.debug(f"Cache miss for key: {key[:20]}..., TTL: {optimal_ttl}s")
        return value, False
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        data_type: str = None
    ) -> bool:
        """Set value in cache with optional TTL override"""
        if ttl is None:
            ttl = self.ttl_optimizer.get_ttl_recommendation(key, data_type)
        
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=ttl
        )
        
        return await self.backend.set(entry)
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        return await self.backend.delete(key)
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        await self.backend.clear()
        with self._stats_lock:
            self.stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics"""
        with self._stats_lock:
            self.stats.update_hit_rate()
            return self.stats
    
    async def get_size(self) -> int:
        """Get current cache size"""
        return await self.backend.size()


# Decorator for caching function results
def cached(
    cache_instance: Optional[AdaptiveCache] = None,
    ttl: Optional[float] = None,
    data_type: str = None,
    key_generator: Optional[Callable] = None
):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        # Use default cache if none provided
        nonlocal cache_instance
        if cache_instance is None:
            cache_instance = AdaptiveCache()
        
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_generator:
                cache_key = key_generator(func.__name__, args, kwargs)
            else:
                key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Get from cache or compute
            result, hit = await cache_instance.get(
                cache_key, func, *args, data_type=data_type, **kwargs
            )
            return result
        
        def sync_wrapper(*args, **kwargs):
            # Convert to async for consistent handling
            async def async_func():
                return func(*args, **kwargs)
            
            # Generate cache key
            if key_generator:
                cache_key = key_generator(func.__name__, args, kwargs)
            else:
                key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Run in event loop
            loop = asyncio.get_event_loop()
            result, hit = loop.run_until_complete(
                cache_instance.get(
                    cache_key, async_func, data_type=data_type
                )
            )
            return result
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Example usage
if __name__ == "__main__":
    async def main():
        # Create adaptive cache
        cache = AdaptiveCache(
            backend_type=CacheBackend.MEMORY,
            max_size=1000,
            enable_analytics=True
        )
        
        @cached(cache_instance=cache, data_type="model_prediction")
        async def expensive_computation(x: float, y: float) -> float:
            """Simulate expensive computation"""
            await asyncio.sleep(1)  # Simulate computation time
            return x * y + np.random.random()
        
        # Test cache performance
        start_time = time.time()
        
        # First call (cache miss)
        result1 = await expensive_computation(5.0, 3.0)
        print(f"First call result: {result1}")
        
        # Second call (cache hit)
        result2 = await expensive_computation(5.0, 3.0)
        print(f"Second call result: {result2}")
        
        print(f"Total time: {time.time() - start_time:.3f}s")
        
        # Print cache stats
        stats = cache.get_stats()
        print(f"Cache Stats: Hit Rate: {stats.hit_rate:.3f}")
    
    asyncio.run(main())