"""Scalable inference platform with distributed processing for Generation 3."""

import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import pickle
from pathlib import Path
import hashlib
import redis
from collections import defaultdict, deque
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes for different workloads."""
    SYNC = "sync"
    ASYNC = "async"
    BATCH = "batch"
    STREAM = "stream"
    DISTRIBUTED = "distributed"


class Priority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class InferenceRequest:
    """Inference request data structure."""
    id: str
    data: np.ndarray
    priority: Priority
    timestamp: float
    metadata: Dict[str, Any] = None
    callback: Optional[Callable] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class InferenceResult:
    """Inference result data structure."""
    request_id: str
    predictions: np.ndarray
    confidence: float
    processing_time: float
    model_version: str
    timestamp: float
    metadata: Dict[str, Any] = None


class AdaptiveLoadBalancer:
    """Adaptive load balancer with dynamic worker allocation."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.worker_stats = defaultdict(lambda: {
            'requests_processed': 0,
            'total_time': 0.0,
            'current_load': 0,
            'error_count': 0,
            'last_activity': time.time()
        })
        self.request_queue = deque()
        self._lock = threading.Lock()
        
    def get_optimal_worker(self) -> str:
        """Get optimal worker based on current load and performance."""
        with self._lock:
            if not self.worker_stats:
                return f"worker_0"
            
            best_worker = None
            best_score = float('inf')
            
            for worker_id, stats in self.worker_stats.items():
                # Calculate worker score (lower is better)
                avg_time = stats['total_time'] / max(1, stats['requests_processed'])
                load_factor = stats['current_load'] / self.max_workers
                error_factor = stats['error_count'] / max(1, stats['requests_processed'])
                
                score = avg_time * (1 + load_factor) * (1 + error_factor)
                
                if score < best_score:
                    best_score = score
                    best_worker = worker_id
            
            return best_worker or f"worker_0"
    
    def update_worker_stats(
        self,
        worker_id: str,
        processing_time: float,
        success: bool = True
    ):
        """Update worker statistics."""
        with self._lock:
            stats = self.worker_stats[worker_id]
            stats['requests_processed'] += 1
            stats['total_time'] += processing_time
            stats['last_activity'] = time.time()
            
            if not success:
                stats['error_count'] += 1
    
    def update_worker_load(self, worker_id: str, current_load: int):
        """Update current worker load."""
        with self._lock:
            self.worker_stats[worker_id]['current_load'] = current_load


class DistributedCache:
    """Distributed caching system with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # In-memory cache (in production, use Redis)
        self.cache = {}
        self.access_times = {}
        self.expiry_times = {}
        self._lock = threading.Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
    def _generate_key(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> str:
        """Generate cache key from input data."""
        # Create hash from data and metadata
        hasher = hashlib.md5()
        hasher.update(data.tobytes())
        
        if metadata:
            hasher.update(json.dumps(metadata, sort_keys=True).encode())
        
        return hasher.hexdigest()
    
    def get(self, key: str) -> Optional[InferenceResult]:
        """Get item from cache."""
        with self._lock:
            current_time = time.time()
            
            # Check if key exists and not expired
            if key in self.cache and current_time < self.expiry_times.get(key, 0):
                self.access_times[key] = current_time
                self.hits += 1
                return self.cache[key]
            
            # Remove expired item
            if key in self.cache:
                self._remove_key(key)
            
            self.misses += 1
            return None
    
    def put(
        self,
        key: str,
        value: InferenceResult,
        ttl: Optional[int] = None
    ):
        """Put item in cache."""
        with self._lock:
            current_time = time.time()
            ttl = ttl or self.default_ttl
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = current_time
            self.expiry_times[key] = current_time + ttl
    
    def _remove_key(self, key: str):
        """Remove key from all tracking structures."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.expiry_times.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self._remove_key(lru_key)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }


class AsyncInferenceEngine:
    """Asynchronous inference engine with batching and optimization."""
    
    def __init__(
        self,
        model_path: str,
        batch_size: int = 32,
        max_batch_wait_time: float = 0.1,
        max_workers: int = None
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_batch_wait_time = max_batch_wait_time
        self.max_workers = max_workers or mp.cpu_count()
        
        # Core components
        self.load_balancer = AdaptiveLoadBalancer(self.max_workers)
        self.cache = DistributedCache()
        
        # Request handling
        self.request_queue = asyncio.Queue()
        self.batch_queue = deque()
        self.active_batches = {}
        
        # Worker management
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.worker_pools = {}
        
        # Statistics
        self.stats = {
            'requests_processed': 0,
            'batches_processed': 0,
            'cache_hits': 0,
            'total_processing_time': 0.0,
            'start_time': time.time()
        }
        
        # Background tasks
        self._batch_processor_task = None
        self._stats_reporter_task = None
        
    async def start(self):
        """Start the inference engine."""
        logger.info("Starting scalable inference engine...")
        
        # Start background tasks
        self._batch_processor_task = asyncio.create_task(self._batch_processor())
        self._stats_reporter_task = asyncio.create_task(self._stats_reporter())
        
        logger.info(f"Inference engine started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the inference engine."""
        logger.info("Stopping inference engine...")
        
        # Cancel background tasks
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
        if self._stats_reporter_task:
            self._stats_reporter_task.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Inference engine stopped")
    
    async def predict_async(
        self,
        data: np.ndarray,
        priority: Priority = Priority.NORMAL,
        use_cache: bool = True,
        metadata: Dict[str, Any] = None
    ) -> InferenceResult:
        """Asynchronous prediction with caching and batching."""
        
        request_id = self._generate_request_id()
        request = InferenceRequest(
            id=request_id,
            data=data,
            priority=priority,
            timestamp=time.time(),
            metadata=metadata
        )
        
        # Check cache first
        if use_cache:
            cache_key = self.cache._generate_key(data, metadata)
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                logger.debug(f"Cache hit for request {request_id}")
                return cached_result
        
        # Add to request queue
        await self.request_queue.put(request)
        
        # Wait for result (in production, use proper async coordination)
        # This is simplified for demo purposes
        result = await self._wait_for_result(request_id)
        
        # Cache result if enabled
        if use_cache and result:
            cache_key = self.cache._generate_key(data, metadata)
            self.cache.put(cache_key, result)
        
        return result
    
    async def _batch_processor(self):
        """Process requests in batches for optimal throughput."""
        while True:
            try:
                batch_requests = []
                batch_start_time = time.time()
                
                # Collect requests for batch
                while (
                    len(batch_requests) < self.batch_size and
                    (time.time() - batch_start_time) < self.max_batch_wait_time
                ):
                    try:
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=self.max_batch_wait_time
                        )
                        batch_requests.append(request)
                    except asyncio.TimeoutError:
                        break
                
                if batch_requests:
                    await self._process_batch(batch_requests)
                
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self, requests: List[InferenceRequest]):
        """Process a batch of requests."""
        batch_id = self._generate_batch_id()
        logger.debug(f"Processing batch {batch_id} with {len(requests)} requests")
        
        start_time = time.time()
        
        try:
            # Sort by priority
            requests.sort(key=lambda r: r.priority.value, reverse=True)
            
            # Prepare batch data
            batch_data = np.array([req.data for req in requests])
            
            # Get optimal worker
            worker_id = self.load_balancer.get_optimal_worker()
            
            # Process batch
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                self.executor,
                self._run_inference,
                batch_data
            )
            
            processing_time = time.time() - start_time
            
            # Create results
            results = []
            for i, request in enumerate(requests):
                result = InferenceResult(
                    request_id=request.id,
                    predictions=predictions[i] if len(predictions) > i else np.array([]),
                    confidence=0.9,  # Placeholder
                    processing_time=processing_time,
                    model_version="v1.0",
                    timestamp=time.time(),
                    metadata={
                        'batch_id': batch_id,
                        'batch_size': len(requests),
                        'worker_id': worker_id
                    }
                )
                results.append(result)
            
            # Update statistics
            self.stats['requests_processed'] += len(requests)
            self.stats['batches_processed'] += 1
            self.stats['total_processing_time'] += processing_time
            
            # Update load balancer
            self.load_balancer.update_worker_stats(worker_id, processing_time, True)
            
            logger.debug(f"Batch {batch_id} completed in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")
            
            # Create error results
            for request in requests:
                error_result = InferenceResult(
                    request_id=request.id,
                    predictions=np.array([]),
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    model_version="v1.0",
                    timestamp=time.time(),
                    metadata={'error': str(e)}
                )
    
    def _run_inference(self, batch_data: np.ndarray) -> np.ndarray:
        """Run inference on batch data (simplified simulation)."""
        # Simulate model inference
        time.sleep(0.01 * len(batch_data))  # Simulate processing time
        
        # Return random predictions for demo
        return np.random.random((len(batch_data), 1))
    
    async def _wait_for_result(self, request_id: str) -> InferenceResult:
        """Wait for result (simplified implementation)."""
        # In production, use proper async coordination mechanisms
        # This is a simplified implementation for demo
        await asyncio.sleep(0.1)  # Simulate waiting
        
        return InferenceResult(
            request_id=request_id,
            predictions=np.random.random(1),
            confidence=0.9,
            processing_time=0.1,
            model_version="v1.0",
            timestamp=time.time()
        )
    
    async def _stats_reporter(self):
        """Report statistics periodically."""
        while True:
            try:
                await asyncio.sleep(30)  # Report every 30 seconds
                
                uptime = time.time() - self.stats['start_time']
                requests_per_second = self.stats['requests_processed'] / uptime
                avg_processing_time = (
                    self.stats['total_processing_time'] / 
                    max(1, self.stats['requests_processed'])
                )
                
                cache_stats = self.cache.get_cache_stats()
                
                logger.info(f"Performance Stats:")
                logger.info(f"  Uptime: {uptime:.1f}s")
                logger.info(f"  Requests/sec: {requests_per_second:.2f}")
                logger.info(f"  Avg processing time: {avg_processing_time:.3f}s")
                logger.info(f"  Cache hit rate: {cache_stats['hit_rate']:.2%}")
                logger.info(f"  Batches processed: {self.stats['batches_processed']}")
                
            except Exception as e:
                logger.error(f"Error in stats reporter: {e}")
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return f"req_{int(time.time() * 1000000)}"
    
    def _generate_batch_id(self) -> str:
        """Generate unique batch ID."""
        return f"batch_{int(time.time() * 1000000)}"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        uptime = time.time() - self.stats['start_time']
        
        return {
            'uptime_seconds': uptime,
            'requests_processed': self.stats['requests_processed'],
            'batches_processed': self.stats['batches_processed'],
            'requests_per_second': self.stats['requests_processed'] / uptime if uptime > 0 else 0,
            'average_processing_time': (
                self.stats['total_processing_time'] / 
                max(1, self.stats['requests_processed'])
            ),
            'cache_stats': self.cache.get_cache_stats(),
            'worker_stats': dict(self.load_balancer.worker_stats),
            'system_stats': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'active_threads': threading.active_count()
            }
        }


class AutoScaler:
    """Automatic scaling based on load metrics."""
    
    def __init__(
        self,
        inference_engine: AsyncInferenceEngine,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        min_workers: int = 1,
        max_workers: int = None
    ):
        self.inference_engine = inference_engine
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_workers = min_workers
        self.max_workers = max_workers or mp.cpu_count() * 2
        
        self.current_workers = inference_engine.max_workers
        self.scaling_history = deque(maxlen=100)
        
    async def monitor_and_scale(self, check_interval: int = 60):
        """Monitor system load and auto-scale."""
        logger.info("Starting auto-scaler...")
        
        while True:
            try:
                await asyncio.sleep(check_interval)
                
                metrics = self.inference_engine.get_performance_metrics()
                
                # Calculate load metrics
                cpu_usage = metrics['system_stats']['cpu_usage'] / 100
                queue_size = self.inference_engine.request_queue.qsize()
                requests_per_second = metrics['requests_per_second']
                
                # Combined load score
                load_score = max(cpu_usage, min(1.0, queue_size / 100))
                
                scaling_decision = self._make_scaling_decision(load_score)
                
                if scaling_decision != 0:
                    await self._execute_scaling(scaling_decision, load_score)
                
                self.scaling_history.append({
                    'timestamp': time.time(),
                    'load_score': load_score,
                    'workers': self.current_workers,
                    'scaling_decision': scaling_decision
                })
                
            except Exception as e:
                logger.error(f"Error in auto-scaler: {e}")
    
    def _make_scaling_decision(self, load_score: float) -> int:
        """Make scaling decision based on load score."""
        if load_score > self.scale_up_threshold and self.current_workers < self.max_workers:
            return 1  # Scale up
        elif load_score < self.scale_down_threshold and self.current_workers > self.min_workers:
            return -1  # Scale down
        else:
            return 0  # No scaling
    
    async def _execute_scaling(self, decision: int, load_score: float):
        """Execute scaling decision."""
        if decision > 0:
            new_workers = min(self.current_workers + 1, self.max_workers)
            action = "scale up"
        else:
            new_workers = max(self.current_workers - 1, self.min_workers)
            action = "scale down"
        
        if new_workers != self.current_workers:
            logger.info(f"Auto-scaling: {action} from {self.current_workers} to {new_workers} workers (load: {load_score:.2f})")
            
            # In production, this would actually modify worker pools
            self.current_workers = new_workers
            self.inference_engine.max_workers = new_workers


class GlobalOptimizer:
    """Global system optimizer with ML-based parameter tuning."""
    
    def __init__(self, inference_engine: AsyncInferenceEngine):
        self.inference_engine = inference_engine
        self.optimization_history = deque(maxlen=1000)
        self.current_config = {
            'batch_size': inference_engine.batch_size,
            'max_batch_wait_time': inference_engine.max_batch_wait_time,
            'cache_size': inference_engine.cache.max_size,
            'cache_ttl': inference_engine.cache.default_ttl
        }
        
    async def optimize_continuously(self, optimization_interval: int = 300):
        """Continuously optimize system parameters."""
        logger.info("Starting global optimizer...")
        
        while True:
            try:
                await asyncio.sleep(optimization_interval)
                
                # Collect current metrics
                metrics = self.inference_engine.get_performance_metrics()
                
                # Run optimization
                optimized_config = self._optimize_parameters(metrics)
                
                if self._should_apply_optimization(optimized_config):
                    await self._apply_optimization(optimized_config)
                
                # Record optimization attempt
                self.optimization_history.append({
                    'timestamp': time.time(),
                    'metrics': metrics,
                    'config': optimized_config.copy(),
                    'applied': self._should_apply_optimization(optimized_config)
                })
                
            except Exception as e:
                logger.error(f"Error in global optimizer: {e}")
    
    def _optimize_parameters(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize parameters based on current metrics."""
        optimized = self.current_config.copy()
        
        # Optimize batch size based on throughput
        current_rps = metrics['requests_per_second']
        if current_rps > 0:
            if metrics['average_processing_time'] > 0.2:  # High latency
                optimized['batch_size'] = max(8, self.current_config['batch_size'] - 4)
            elif metrics['cache_stats']['hit_rate'] < 0.5:  # Low cache efficiency
                optimized['batch_size'] = min(64, self.current_config['batch_size'] + 4)
        
        # Optimize cache parameters
        hit_rate = metrics['cache_stats']['hit_rate']
        if hit_rate < 0.3:  # Low hit rate
            optimized['cache_size'] = min(20000, int(self.current_config['cache_size'] * 1.2))
            optimized['cache_ttl'] = min(7200, int(self.current_config['cache_ttl'] * 1.5))
        elif hit_rate > 0.8:  # High hit rate, can reduce cache
            optimized['cache_size'] = max(5000, int(self.current_config['cache_size'] * 0.9))
        
        # Optimize batch wait time
        avg_batch_size = metrics.get('average_batch_size', self.current_config['batch_size'])
        if avg_batch_size < self.current_config['batch_size'] * 0.5:
            optimized['max_batch_wait_time'] = max(0.05, self.current_config['max_batch_wait_time'] * 0.8)
        
        return optimized
    
    def _should_apply_optimization(self, optimized_config: Dict[str, Any]) -> bool:
        """Determine if optimization should be applied."""
        # Calculate change magnitude
        total_change = 0
        for key in optimized_config:
            if key in self.current_config:
                current_val = self.current_config[key]
                new_val = optimized_config[key]
                if current_val != 0:
                    change_pct = abs(new_val - current_val) / current_val
                    total_change += change_pct
        
        # Apply if change is significant but not too drastic
        return 0.05 < total_change < 0.5
    
    async def _apply_optimization(self, optimized_config: Dict[str, Any]):
        """Apply optimized configuration."""
        logger.info(f"Applying optimization: {optimized_config}")
        
        # Update inference engine parameters
        # Note: In production, some changes might require restart
        if 'batch_size' in optimized_config:
            self.inference_engine.batch_size = optimized_config['batch_size']
        
        if 'max_batch_wait_time' in optimized_config:
            self.inference_engine.max_batch_wait_time = optimized_config['max_batch_wait_time']
        
        if 'cache_size' in optimized_config:
            self.inference_engine.cache.max_size = optimized_config['cache_size']
        
        if 'cache_ttl' in optimized_config:
            self.inference_engine.cache.default_ttl = optimized_config['cache_ttl']
        
        self.current_config = optimized_config.copy()


async def run_scalability_demo():
    """Demonstrate scalable inference platform."""
    logger.info("=== GENERATION 3: SCALABLE INFERENCE PLATFORM DEMO ===")
    
    # Initialize inference engine
    engine = AsyncInferenceEngine(
        model_path="demo_model.h5",
        batch_size=16,
        max_batch_wait_time=0.1,
        max_workers=4
    )
    
    # Initialize auto-scaler
    auto_scaler = AutoScaler(engine, min_workers=2, max_workers=8)
    
    # Initialize global optimizer
    optimizer = GlobalOptimizer(engine)
    
    try:
        # Start all components
        await engine.start()
        
        # Start background optimization tasks
        scaler_task = asyncio.create_task(auto_scaler.monitor_and_scale(check_interval=30))
        optimizer_task = asyncio.create_task(optimizer.optimize_continuously(optimization_interval=60))
        
        # Simulate load testing
        logger.info("Starting load simulation...")
        
        async def simulate_requests():
            for i in range(100):
                data = np.random.random((10, 5))  # Random time series data
                priority = Priority.NORMAL if i % 10 != 0 else Priority.HIGH
                
                try:
                    result = await engine.predict_async(
                        data,
                        priority=priority,
                        metadata={'request_index': i}
                    )
                    
                    if i % 20 == 0:
                        logger.info(f"Processed request {i}, confidence: {result.confidence:.3f}")
                
                except Exception as e:
                    logger.error(f"Request {i} failed: {e}")
                
                # Variable request rate
                await asyncio.sleep(0.1 if i % 5 == 0 else 0.05)
        
        # Run simulation
        await simulate_requests()
        
        # Wait a bit for processing to complete
        await asyncio.sleep(5)
        
        # Get final metrics
        final_metrics = engine.get_performance_metrics()
        
        logger.info("=== FINAL PERFORMANCE METRICS ===")
        logger.info(f"Uptime: {final_metrics['uptime_seconds']:.1f}s")
        logger.info(f"Requests processed: {final_metrics['requests_processed']}")
        logger.info(f"Requests/second: {final_metrics['requests_per_second']:.2f}")
        logger.info(f"Average processing time: {final_metrics['average_processing_time']:.3f}s")
        logger.info(f"Cache hit rate: {final_metrics['cache_stats']['hit_rate']:.2%}")
        logger.info(f"Current workers: {auto_scaler.current_workers}")
        
        # Cancel background tasks
        scaler_task.cancel()
        optimizer_task.cancel()
        
    finally:
        await engine.stop()
    
    logger.info("=== GENERATION 3 SCALABLE PLATFORM COMPLETE ===")


if __name__ == "__main__":
    # Run the scalability demonstration
    asyncio.run(run_scalability_demo())