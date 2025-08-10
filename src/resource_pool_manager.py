#!/usr/bin/env python3
"""
Advanced Resource Pool Manager for IoT Anomaly Detection Platform
Provides intelligent resource pooling, connection management, and memory optimization.
"""

import asyncio
import gc
import logging
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Generic, TypeVar
import psutil

import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

from .logging_config import setup_logging

T = TypeVar('T')

class PoolStrategy(Enum):
    """Resource pool management strategies"""
    FIFO = "fifo"
    LIFO = "lifo"
    LEAST_USED = "least_used"
    ROUND_ROBIN = "round_robin"


@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    created: int = 0
    destroyed: int = 0
    active: int = 0
    idle: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    avg_creation_time: float = 0.0
    avg_usage_time: float = 0.0
    memory_usage_mb: float = 0.0


class PoolableResource(ABC, Generic[T]):
    """Abstract base class for poolable resources"""
    
    def __init__(self):
        self.created_at = time.time()
        self.last_used = time.time()
        self.usage_count = 0
        self.is_healthy = True
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the resource"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup the resource"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if resource is healthy"""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        pass
    
    def mark_used(self) -> None:
        """Mark resource as used"""
        self.last_used = time.time()
        self.usage_count += 1


class MLModelResource(PoolableResource):
    """ML Model resource wrapper"""
    
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.model = None
        self._memory_usage = 0.0
    
    async def initialize(self) -> None:
        """Load ML model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self._calculate_memory_usage()
        except Exception as e:
            self.is_healthy = False
            raise e
    
    async def cleanup(self) -> None:
        """Cleanup model resources"""
        if self.model:
            del self.model
            self.model = None
            tf.keras.backend.clear_session()
            gc.collect()
    
    async def health_check(self) -> bool:
        """Check model health"""
        if not self.model:
            return False
        
        try:
            # Test with dummy input
            dummy_input = np.random.random((1, 30, 3))
            _ = self.model.predict(dummy_input, verbose=0)
            return True
        except Exception:
            return False
    
    def get_memory_usage(self) -> float:
        """Get model memory usage"""
        return self._memory_usage
    
    def _calculate_memory_usage(self) -> None:
        """Calculate model memory usage"""
        if self.model:
            # Estimate memory usage based on parameters
            total_params = self.model.count_params()
            # Approximate: 4 bytes per float32 parameter
            self._memory_usage = (total_params * 4) / (1024 * 1024)  # MB


class DatabaseConnectionResource(PoolableResource):
    """Database connection resource"""
    
    def __init__(self, connection_string: str):
        super().__init__()
        self.connection_string = connection_string
        self.connection = None
    
    async def initialize(self) -> None:
        """Initialize database connection"""
        # Placeholder for actual DB connection
        await asyncio.sleep(0.1)  # Simulate connection time
        self.connection = {"connected": True, "connection_string": self.connection_string}
    
    async def cleanup(self) -> None:
        """Close database connection"""
        if self.connection:
            # Placeholder for actual cleanup
            await asyncio.sleep(0.05)
            self.connection = None
    
    async def health_check(self) -> bool:
        """Check connection health"""
        return self.connection is not None and self.connection.get("connected", False)
    
    def get_memory_usage(self) -> float:
        """Get connection memory usage"""
        return 0.5  # Approximate MB per connection


class ResourcePool(Generic[T]):
    """Generic resource pool with intelligent management"""
    
    def __init__(
        self,
        resource_factory: Callable[[], PoolableResource[T]],
        min_size: int = 2,
        max_size: int = 10,
        strategy: PoolStrategy = PoolStrategy.LIFO,
        max_idle_time: float = 300.0,
        health_check_interval: float = 60.0
    ):
        self.resource_factory = resource_factory
        self.min_size = min_size
        self.max_size = max_size
        self.strategy = strategy
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        
        self.logger = setup_logging(f"ResourcePool.{resource_factory.__name__}")
        
        # Pool storage
        self.available_resources: deque = deque()
        self.active_resources: Dict[int, PoolableResource[T]] = {}
        self.all_resources: weakref.WeakSet = weakref.WeakSet()
        
        # Synchronization
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)
        
        # Metrics
        self.metrics = ResourceMetrics()
        
        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Round-robin counter
        self._rr_counter = 0
        
        # Initialize pool
        asyncio.create_task(self._initialize_pool())
    
    async def _initialize_pool(self) -> None:
        """Initialize pool with minimum resources"""
        async with self.lock:
            for _ in range(self.min_size):
                resource = await self._create_resource()
                if resource:
                    self.available_resources.append(resource)
        
        # Start maintenance tasks
        self._start_maintenance_tasks()
    
    def _start_maintenance_tasks(self) -> None:
        """Start background maintenance tasks"""
        if not self._maintenance_task or self._maintenance_task.done():
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        if not self._health_check_task or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _create_resource(self) -> Optional[PoolableResource[T]]:
        """Create new resource"""
        try:
            start_time = time.time()
            resource = self.resource_factory()
            await resource.initialize()
            
            creation_time = time.time() - start_time
            self.metrics.created += 1
            self.metrics.avg_creation_time = (
                (self.metrics.avg_creation_time * (self.metrics.created - 1) + creation_time) / 
                self.metrics.created
            )
            
            self.all_resources.add(resource)
            self.logger.debug(f"Created new resource (total: {self.metrics.created})")
            return resource
            
        except Exception as e:
            self.logger.error(f"Failed to create resource: {e}")
            return None
    
    async def _destroy_resource(self, resource: PoolableResource[T]) -> None:
        """Destroy resource"""
        try:
            await resource.cleanup()
            self.metrics.destroyed += 1
            self.logger.debug(f"Destroyed resource (total destroyed: {self.metrics.destroyed})")
        except Exception as e:
            self.logger.error(f"Error destroying resource: {e}")
    
    async def acquire(self, timeout: float = 30.0) -> Optional[PoolableResource[T]]:
        """Acquire resource from pool"""
        start_time = time.time()
        
        async with self.condition:
            # Wait for available resource
            while not self.available_resources and len(self.active_resources) >= self.max_size:
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout waiting for resource")
                    return None
            
            # Try to get existing resource
            resource = await self._get_available_resource()
            
            # Create new resource if needed and allowed
            if not resource and len(self.active_resources) < self.max_size:
                resource = await self._create_resource()
                if resource:
                    self.metrics.pool_misses += 1
            
            if resource:
                resource.mark_used()
                resource_id = id(resource)
                self.active_resources[resource_id] = resource
                self.metrics.active += 1
                self.metrics.pool_hits += 1
                
                usage_time = time.time() - start_time
                self.metrics.avg_usage_time = (
                    (self.metrics.avg_usage_time * (self.metrics.pool_hits - 1) + usage_time) / 
                    self.metrics.pool_hits
                )
                
                self.logger.debug(f"Acquired resource {resource_id}")
            
            return resource
    
    async def _get_available_resource(self) -> Optional[PoolableResource[T]]:
        """Get resource based on strategy"""
        if not self.available_resources:
            return None
        
        if self.strategy == PoolStrategy.FIFO:
            resource = self.available_resources.popleft()
        elif self.strategy == PoolStrategy.LIFO:
            resource = self.available_resources.pop()
        elif self.strategy == PoolStrategy.ROUND_ROBIN:
            if self.available_resources:
                index = self._rr_counter % len(self.available_resources)
                resources_list = list(self.available_resources)
                resource = resources_list[index]
                self.available_resources.remove(resource)
                self._rr_counter += 1
            else:
                return None
        elif self.strategy == PoolStrategy.LEAST_USED:
            resource = min(self.available_resources, key=lambda r: r.usage_count)
            self.available_resources.remove(resource)
        else:
            resource = self.available_resources.pop()
        
        self.metrics.idle -= 1
        return resource
    
    async def release(self, resource: PoolableResource[T]) -> None:
        """Release resource back to pool"""
        async with self.condition:
            resource_id = id(resource)
            
            if resource_id in self.active_resources:
                del self.active_resources[resource_id]
                self.metrics.active -= 1
                
                # Health check before returning to pool
                if resource.is_healthy and await resource.health_check():
                    self.available_resources.append(resource)
                    self.metrics.idle += 1
                    self.logger.debug(f"Released resource {resource_id} to pool")
                else:
                    await self._destroy_resource(resource)
                    self.logger.debug(f"Destroyed unhealthy resource {resource_id}")
                
                self.condition.notify()
    
    async def _maintenance_loop(self) -> None:
        """Background maintenance loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_idle_resources()
                await self._ensure_minimum_resources()
                self._update_memory_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in maintenance loop: {e}")
    
    async def _health_check_loop(self) -> None:
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._health_check_all_resources()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
    
    async def _cleanup_idle_resources(self) -> None:
        """Clean up idle resources that exceed max idle time"""
        async with self.lock:
            current_time = time.time()
            to_remove = []
            
            for resource in list(self.available_resources):
                if (current_time - resource.last_used > self.max_idle_time and 
                    len(self.available_resources) + len(self.active_resources) > self.min_size):
                    to_remove.append(resource)
            
            for resource in to_remove:
                self.available_resources.remove(resource)
                self.metrics.idle -= 1
                await self._destroy_resource(resource)
    
    async def _ensure_minimum_resources(self) -> None:
        """Ensure minimum number of resources"""
        async with self.lock:
            total_resources = len(self.available_resources) + len(self.active_resources)
            
            if total_resources < self.min_size:
                needed = self.min_size - total_resources
                for _ in range(needed):
                    resource = await self._create_resource()
                    if resource:
                        self.available_resources.append(resource)
                        self.metrics.idle += 1
    
    async def _health_check_all_resources(self) -> None:
        """Health check all available resources"""
        async with self.lock:
            unhealthy_resources = []
            
            for resource in list(self.available_resources):
                if not await resource.health_check():
                    unhealthy_resources.append(resource)
            
            for resource in unhealthy_resources:
                self.available_resources.remove(resource)
                self.metrics.idle -= 1
                await self._destroy_resource(resource)
    
    def _update_memory_metrics(self) -> None:
        """Update memory usage metrics"""
        total_memory = 0.0
        
        for resource in self.available_resources:
            total_memory += resource.get_memory_usage()
        
        for resource in self.active_resources.values():
            total_memory += resource.get_memory_usage()
        
        self.metrics.memory_usage_mb = total_memory
    
    def get_metrics(self) -> ResourceMetrics:
        """Get pool metrics"""
        self.metrics.active = len(self.active_resources)
        self.metrics.idle = len(self.available_resources)
        return self.metrics
    
    async def shutdown(self) -> None:
        """Shutdown resource pool"""
        # Cancel maintenance tasks
        if self._maintenance_task:
            self._maintenance_task.cancel()
        if self._health_check_task:
            self._health_check_task.cancel()
        
        # Clean up all resources
        async with self.lock:
            for resource in list(self.available_resources):
                await self._destroy_resource(resource)
            
            for resource in list(self.active_resources.values()):
                await self._destroy_resource(resource)
            
            self.available_resources.clear()
            self.active_resources.clear()
        
        self.logger.info("Resource pool shut down")


@asynccontextmanager
async def use_resource(pool: ResourcePool[T], timeout: float = 30.0):
    """Context manager for using pooled resources"""
    resource = await pool.acquire(timeout=timeout)
    if not resource:
        raise Exception("Failed to acquire resource from pool")
    
    try:
        yield resource
    finally:
        await pool.release(resource)


class ResourcePoolManager:
    """Central manager for multiple resource pools"""
    
    def __init__(self):
        self.pools: Dict[str, ResourcePool] = {}
        self.logger = setup_logging(self.__class__.__name__)
        
        # Global monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._start_monitoring()
    
    def _start_monitoring(self) -> None:
        """Start global monitoring"""
        if not self._monitoring_task or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self) -> None:
        """Global monitoring loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._log_global_metrics()
                await self._optimize_pools()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    async def _log_global_metrics(self) -> None:
        """Log global metrics across all pools"""
        total_memory = 0.0
        total_resources = 0
        
        for name, pool in self.pools.items():
            metrics = pool.get_metrics()
            total_memory += metrics.memory_usage_mb
            total_resources += metrics.active + metrics.idle
            
            self.logger.info(
                f"Pool {name}: {metrics.active} active, {metrics.idle} idle, "
                f"{metrics.memory_usage_mb:.1f}MB, hit rate: "
                f"{metrics.pool_hits / max(metrics.pool_hits + metrics.pool_misses, 1):.3f}"
            )
        
        # System memory info
        system_memory = psutil.virtual_memory()
        self.logger.info(
            f"Global: {total_resources} resources, {total_memory:.1f}MB pool memory, "
            f"{system_memory.percent:.1f}% system memory used"
        )
    
    async def _optimize_pools(self) -> None:
        """Optimize pool configurations based on usage patterns"""
        for name, pool in self.pools.items():
            metrics = pool.get_metrics()
            
            # Suggest optimizations based on metrics
            if metrics.pool_hits > 0:
                hit_rate = metrics.pool_hits / (metrics.pool_hits + metrics.pool_misses)
                
                if hit_rate < 0.8 and pool.max_size < 20:
                    self.logger.info(f"Consider increasing max_size for pool {name} (hit rate: {hit_rate:.3f})")
                elif hit_rate > 0.95 and pool.min_size > 1:
                    self.logger.info(f"Consider decreasing min_size for pool {name} (hit rate: {hit_rate:.3f})")
    
    def create_model_pool(
        self,
        name: str,
        model_path: str,
        min_size: int = 2,
        max_size: int = 8,
        strategy: PoolStrategy = PoolStrategy.LEAST_USED
    ) -> ResourcePool[MLModelResource]:
        """Create ML model resource pool"""
        factory = lambda: MLModelResource(model_path)
        
        pool = ResourcePool(
            resource_factory=factory,
            min_size=min_size,
            max_size=max_size,
            strategy=strategy
        )
        
        self.pools[name] = pool
        self.logger.info(f"Created model pool '{name}' with {min_size}-{max_size} instances")
        return pool
    
    def create_db_pool(
        self,
        name: str,
        connection_string: str,
        min_size: int = 5,
        max_size: int = 20,
        strategy: PoolStrategy = PoolStrategy.FIFO
    ) -> ResourcePool[DatabaseConnectionResource]:
        """Create database connection pool"""
        factory = lambda: DatabaseConnectionResource(connection_string)
        
        pool = ResourcePool(
            resource_factory=factory,
            min_size=min_size,
            max_size=max_size,
            strategy=strategy,
            max_idle_time=600.0  # 10 minutes for DB connections
        )
        
        self.pools[name] = pool
        self.logger.info(f"Created DB pool '{name}' with {min_size}-{max_size} connections")
        return pool
    
    def get_pool(self, name: str) -> Optional[ResourcePool]:
        """Get pool by name"""
        return self.pools.get(name)
    
    async def get_global_metrics(self) -> Dict[str, Any]:
        """Get global metrics across all pools"""
        metrics = {}
        
        for name, pool in self.pools.items():
            metrics[name] = {
                'active': pool.metrics.active,
                'idle': pool.metrics.idle,
                'created': pool.metrics.created,
                'destroyed': pool.metrics.destroyed,
                'pool_hits': pool.metrics.pool_hits,
                'pool_misses': pool.metrics.pool_misses,
                'memory_usage_mb': pool.metrics.memory_usage_mb,
                'avg_creation_time': pool.metrics.avg_creation_time,
                'hit_rate': pool.metrics.pool_hits / max(
                    pool.metrics.pool_hits + pool.metrics.pool_misses, 1
                )
            }
        
        return metrics
    
    async def shutdown_all(self) -> None:
        """Shutdown all resource pools"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        for name, pool in self.pools.items():
            self.logger.info(f"Shutting down pool: {name}")
            await pool.shutdown()
        
        self.pools.clear()
        self.logger.info("All resource pools shut down")


# Global pool manager instance
pool_manager = ResourcePoolManager()


# Example usage
async def main():
    """Example usage of resource pool manager"""
    logger = setup_logging("resource_pool_example")
    
    # Create model pool
    model_pool = pool_manager.create_model_pool(
        name="anomaly_models",
        model_path="saved_models/autoencoder.h5",
        min_size=2,
        max_size=6
    )
    
    # Create database pool
    db_pool = pool_manager.create_db_pool(
        name="main_db",
        connection_string="postgresql://user:pass@localhost/anomaly_db",
        min_size=3,
        max_size=10
    )
    
    # Simulate usage
    logger.info("Starting resource pool simulation...")
    
    # Test model pool
    async with use_resource(model_pool) as model_resource:
        logger.info(f"Using model resource: {id(model_resource)}")
        await asyncio.sleep(0.5)  # Simulate processing
    
    # Test database pool
    async with use_resource(db_pool) as db_resource:
        logger.info(f"Using DB resource: {id(db_resource)}")
        await asyncio.sleep(0.2)  # Simulate query
    
    # Get metrics
    await asyncio.sleep(2)  # Let metrics update
    metrics = await pool_manager.get_global_metrics()
    
    for pool_name, pool_metrics in metrics.items():
        logger.info(f"Pool {pool_name} metrics: {pool_metrics}")
    
    # Cleanup
    await pool_manager.shutdown_all()


if __name__ == "__main__":
    asyncio.run(main())