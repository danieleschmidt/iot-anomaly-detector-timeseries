"""
Caching Strategy Implementation for IoT Anomaly Detection System

This module provides a comprehensive LRU-based caching system to improve
performance for repeated preprocessing and prediction operations.
"""

import hashlib
import time
import logging
from functools import wraps
from collections import OrderedDict
from typing import Any, Dict, Optional, Callable
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a deterministic cache key from function arguments.
    
    Handles numpy arrays, pandas DataFrames, and standard Python types.
    
    Parameters
    ----------
    *args : tuple
        Positional arguments to hash
    **kwargs : dict
        Keyword arguments to hash
        
    Returns
    -------
    str
        SHA-256 hash as cache key
    """
    def serialize_arg(arg):
        """Serialize a single argument for hashing."""
        if isinstance(arg, np.ndarray):
            return f"ndarray:{arg.shape}:{arg.dtype}:{hashlib.sha256(arg.tobytes(), usedforsecurity=False).hexdigest()}"
        elif isinstance(arg, pd.DataFrame):
            return f"dataframe:{arg.shape}:{hashlib.sha256(pd.util.hash_pandas_object(arg).values.tobytes(), usedforsecurity=False).hexdigest()}"
        elif isinstance(arg, pd.Series):
            return f"series:{len(arg)}:{hashlib.sha256(pd.util.hash_pandas_object(arg).values.tobytes(), usedforsecurity=False).hexdigest()}"
        elif isinstance(arg, (list, tuple)):
            return f"{type(arg).__name__}:{[serialize_arg(item) for item in arg]}"
        elif isinstance(arg, dict):
            return f"dict:{[(k, serialize_arg(v)) for k, v in sorted(arg.items())]}"
        else:
            return str(arg)
    
    # Serialize all arguments
    serialized_args = [serialize_arg(arg) for arg in args]
    serialized_kwargs = [(k, serialize_arg(v)) for k, v in sorted(kwargs.items())]
    
    # Combine into single string
    combined = f"args:{serialized_args}|kwargs:{serialized_kwargs}"
    
    # Generate SHA-256 hash
    return hashlib.sha256(combined.encode('utf-8'), usedforsecurity=False).hexdigest()


class CacheManager:
    """
    LRU Cache Manager for efficient storage and retrieval of computation results.
    
    Uses OrderedDict for O(1) access and LRU eviction policy.
    """
    
    def __init__(self, maxsize: int = 128):
        """
        Initialize cache manager.
        
        Parameters
        ----------
        maxsize : int, default 128
            Maximum number of items to store in cache
        """
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self._stats = {'hits': 0, 'misses': 0}
        
        logger.info(f"CacheManager initialized with maxsize={maxsize}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache using key.
        
        Parameters
        ----------
        key : str
            Cache key to retrieve
            
        Returns
        -------
        Any or None
            Cached value if found, None otherwise
        """
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._stats['hits'] += 1
            logger.debug(f"Cache hit for key: {key[:16]}...")
            return self._cache[key]
        else:
            self._stats['misses'] += 1
            logger.debug(f"Cache miss for key: {key[:16]}...")
            return None
    
    def put(self, key: str, value: Any) -> None:
        """
        Store item in cache with key.
        
        Parameters
        ----------
        key : str
            Cache key for storage
        value : Any
            Value to cache
        """
        if key in self._cache:
            # Update existing item and move to end
            self._cache[key] = value
            self._cache.move_to_end(key)
        else:
            # Add new item
            self._cache[key] = value
            
            # Evict oldest item if cache is full
            if len(self._cache) > self.maxsize:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Evicted oldest cache entry: {oldest_key[:16]}...")
        
        logger.debug(f"Cached item with key: {key[:16]}...")
    
    def clear(self) -> None:
        """Clear all items from cache."""
        self._cache.clear()
        self._stats = {'hits': 0, 'misses': 0}
        logger.info("Cache cleared")
    
    def size(self) -> int:
        """Return current number of items in cache."""
        return len(self._cache)
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get cache performance statistics.
        
        Returns
        -------
        dict
            Dictionary with 'hits', 'misses', and 'hit_rate' keys
        """
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'cache_size': self.size(),
            'max_size': self.maxsize
        }


# Global cache instances for different use cases
_preprocessing_cache = CacheManager(maxsize=50)
_prediction_cache = CacheManager(maxsize=100)
_model_cache = CacheManager(maxsize=20)


def cache_result(cache_manager: Optional[CacheManager] = None, 
                 cache_type: str = "default") -> Callable:
    """
    Decorator to cache function results based on input parameters.
    
    Parameters
    ----------
    cache_manager : CacheManager, optional
        Custom cache manager instance. If None, uses global cache based on cache_type.
    cache_type : str, default "default"
        Type of cache to use: "preprocessing", "prediction", "model", or "default"
        
    Returns
    -------
    Callable
        Decorated function with caching capability
    """
    def decorator(func: Callable) -> Callable:
        # Select cache manager
        if cache_manager is not None:
            manager = cache_manager
        elif cache_type == "preprocessing":
            manager = _preprocessing_cache
        elif cache_type == "prediction":
            manager = _prediction_cache
        elif cache_type == "model":
            manager = _model_cache
        else:
            manager = CacheManager(maxsize=128)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = generate_cache_key(*args, **kwargs)
            
            # Try to get from cache
            cached_result = manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for function {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for function {func.__name__}, executing...")
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Store in cache
            manager.put(cache_key, result)
            
            logger.debug(f"Function {func.__name__} executed in {execution_time:.4f}s and cached")
            return result
        
        # Add cache statistics method to wrapped function
        wrapper.get_cache_stats = lambda: manager.get_stats()
        wrapper.clear_cache = lambda: manager.clear()
        
        return wrapper
    
    return decorator


def get_cache_stats() -> Dict[str, Dict[str, float]]:
    """
    Get statistics for all global cache instances.
    
    Returns
    -------
    dict
        Dictionary with cache type keys and statistics values
    """
    return {
        'preprocessing': _preprocessing_cache.get_stats(),
        'prediction': _prediction_cache.get_stats(),
        'model': _model_cache.get_stats()
    }


def clear_all_caches() -> None:
    """Clear all global cache instances."""
    _preprocessing_cache.clear()
    _prediction_cache.clear()
    _model_cache.clear()
    logger.info("All global caches cleared")


# Convenience decorators for specific use cases
def cache_preprocessing(func: Callable) -> Callable:
    """Decorator for caching data preprocessing operations."""
    return cache_result(cache_type="preprocessing")(func)


def cache_prediction(func: Callable) -> Callable:
    """Decorator for caching model prediction operations."""
    return cache_result(cache_type="prediction")(func)


def cache_model_operation(func: Callable) -> Callable:
    """Decorator for caching model-related operations."""
    return cache_result(cache_type="model")(func)