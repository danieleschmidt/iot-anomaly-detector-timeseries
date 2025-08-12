"""Scalable High-Performance Inference Engine for Anomaly Detection.

This module implements a high-performance, horizontally scalable inference engine
optimized for real-time anomaly detection in large-scale IoT deployments with
advanced caching, load balancing, and distributed processing capabilities.
"""

import asyncio
import time
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import threading
import queue
import warnings
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available. Using in-memory caching fallback.")

try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

from .logging_config import get_logger
from .adaptive_multi_modal_detector import DetectionResult
from .resilient_anomaly_pipeline import ResilientAnomalyPipeline


class ProcessingStrategy(Enum):
    """Processing strategies for different workload patterns."""
    BATCH = "batch"
    STREAMING = "streaming"
    MICRO_BATCH = "micro_batch"
    ADAPTIVE = "adaptive"


class CacheStrategy(Enum):
    """Caching strategies for inference optimization."""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"
    DISTRIBUTED = "distributed"


@dataclass
class InferenceRequest:
    """Individual inference request."""
    request_id: str
    data: np.ndarray
    timestamp: float
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None


@dataclass
class InferenceResult:
    """Inference result with performance metrics."""
    request_id: str
    result: DetectionResult
    processing_time: float
    cache_hit: bool
    worker_id: str
    queue_time: float
    total_time: float


@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking."""
    requests_per_second: float
    avg_processing_time: float
    avg_queue_time: float
    cache_hit_rate: float
    active_workers: int
    queue_depth: int
    memory_usage_mb: float
    cpu_utilization: float
    throughput_mbps: float
    error_rate: float


class AdaptiveCache:
    """High-performance adaptive caching system."""
    
    def __init__(self, 
                 max_size: int = 10000,
                 ttl_seconds: int = 3600,
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 redis_url: Optional[str] = None):
        
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.strategy = strategy
        self.logger = get_logger(f"{__name__}.AdaptiveCache")
        
        # Initialize cache backend
        if redis_url and REDIS_AVAILABLE:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                self.redis_client.ping()
                self.distributed_cache = True
                self.logger.info("Connected to Redis for distributed caching")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {e}, falling back to local cache")
                self.distributed_cache = False
        else:
            self.distributed_cache = False
        
        # Local cache structures
        self.cache = {}
        self.access_count = defaultdict(int)
        self.access_time = {}
        self.cache_lock = threading.RLock()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _generate_key(self, data: np.ndarray) -> str:
        """Generate deterministic cache key from data."""
        # Use hash of array bytes for key generation
        data_bytes = data.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[DetectionResult]:
        """Retrieve item from cache."""
        if self.distributed_cache:
            return self._get_distributed(key)
        else:
            return self._get_local(key)
    
    def put(self, key: str, value: DetectionResult) -> None:
        """Store item in cache."""
        if self.distributed_cache:
            self._put_distributed(key, value)
        else:
            self._put_local(key, value)
    
    def _get_local(self, key: str) -> Optional[DetectionResult]:
        """Get from local cache."""
        with self.cache_lock:
            if key in self.cache:
                item, timestamp = self.cache[key]
                
                # Check TTL
                if time.time() - timestamp < self.ttl_seconds:
                    self.access_count[key] += 1
                    self.access_time[key] = time.time()
                    self.hits += 1
                    return item
                else:
                    # Expired
                    del self.cache[key]
                    if key in self.access_count:
                        del self.access_count[key]
                    if key in self.access_time:
                        del self.access_time[key]
            
            self.misses += 1
            return None
    
    def _put_local(self, key: str, value: DetectionResult) -> None:
        """Put to local cache with eviction."""
        with self.cache_lock:
            current_time = time.time()
            
            # Check if we need to evict
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_local()
            
            self.cache[key] = (value, current_time)
            self.access_count[key] = 1
            self.access_time[key] = current_time
    
    def _evict_local(self) -> None:
        """Evict items based on strategy."""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest_key = min(self.access_time.keys(), key=lambda k: self.access_time[k])
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            oldest_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
        else:  # ADAPTIVE
            # Combine recency and frequency
            scores = {}
            current_time = time.time()
            for key in self.cache.keys():
                recency_score = current_time - self.access_time.get(key, 0)
                frequency_score = 1.0 / (self.access_count.get(key, 1))
                scores[key] = recency_score * frequency_score
            oldest_key = max(scores.keys(), key=lambda k: scores[k])
        
        # Remove the selected key
        if oldest_key in self.cache:
            del self.cache[oldest_key]
            del self.access_count[oldest_key]
            del self.access_time[oldest_key]
            self.evictions += 1
    
    def _get_distributed(self, key: str) -> Optional[DetectionResult]:
        """Get from distributed cache."""
        try:
            serialized = self.redis_client.get(f"anomaly:{key}")
            if serialized:
                self.hits += 1
                # Deserialize result
                data = json.loads(serialized)
                return DetectionResult(
                    anomaly_scores=np.array(data["anomaly_scores"]),
                    anomaly_predictions=np.array(data["anomaly_predictions"]),
                    confidence_scores=np.array(data["confidence_scores"]),
                    detection_method=data["detection_method"],
                    metadata=data["metadata"]
                )
        except Exception as e:
            self.logger.warning(f"Distributed cache get error: {e}")
        
        self.misses += 1
        return None
    
    def _put_distributed(self, key: str, value: DetectionResult) -> None:
        """Put to distributed cache."""
        try:
            # Serialize result
            data = {
                "anomaly_scores": value.anomaly_scores.tolist(),
                "anomaly_predictions": value.anomaly_predictions.tolist(),
                "confidence_scores": value.confidence_scores.tolist(),
                "detection_method": value.detection_method,
                "metadata": value.metadata
            }
            serialized = json.dumps(data)
            
            self.redis_client.setex(f"anomaly:{key}", self.ttl_seconds, serialized)
        except Exception as e:
            self.logger.warning(f"Distributed cache put error: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "distributed": self.distributed_cache
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.cache_lock:
            self.cache.clear()
            self.access_count.clear()
            self.access_time.clear()
        
        if self.distributed_cache:
            try:
                # Clear Redis keys with anomaly prefix
                keys = self.redis_client.keys("anomaly:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                self.logger.warning(f"Error clearing distributed cache: {e}")


class LoadBalancer:
    """Intelligent load balancer for worker processes."""
    
    def __init__(self, balancing_strategy: str = "least_loaded"):
        self.balancing_strategy = balancing_strategy
        self.worker_stats = {}
        self.worker_queues = {}
        self.stats_lock = threading.Lock()
        self.logger = get_logger(f"{__name__}.LoadBalancer")
    
    def register_worker(self, worker_id: str, queue_obj: queue.Queue) -> None:
        """Register a new worker."""
        with self.stats_lock:
            self.worker_stats[worker_id] = {
                "active_requests": 0,
                "total_requests": 0,
                "total_processing_time": 0.0,
                "avg_processing_time": 0.0,
                "queue_depth": 0,
                "last_activity": time.time()
            }
            self.worker_queues[worker_id] = queue_obj
        
        self.logger.info(f"Registered worker: {worker_id}")
    
    def select_worker(self) -> Optional[str]:
        """Select optimal worker based on strategy."""
        with self.stats_lock:
            if not self.worker_stats:
                return None
            
            if self.balancing_strategy == "round_robin":
                # Simple round-robin selection
                workers = list(self.worker_stats.keys())
                return workers[0] if workers else None
            
            elif self.balancing_strategy == "least_loaded":
                # Select worker with least active requests
                return min(self.worker_stats.keys(), 
                          key=lambda w: self.worker_stats[w]["active_requests"])
            
            elif self.balancing_strategy == "fastest_average":
                # Select worker with best average processing time
                valid_workers = {w: stats for w, stats in self.worker_stats.items() 
                               if stats["total_requests"] > 0}
                if not valid_workers:
                    return min(self.worker_stats.keys(), 
                              key=lambda w: self.worker_stats[w]["active_requests"])
                
                return min(valid_workers.keys(),
                          key=lambda w: valid_workers[w]["avg_processing_time"])
            
            else:
                # Default to least loaded
                return min(self.worker_stats.keys(),
                          key=lambda w: self.worker_stats[w]["active_requests"])
    
    def update_worker_stats(self, worker_id: str, processing_time: float) -> None:
        """Update worker performance statistics."""
        with self.stats_lock:
            if worker_id in self.worker_stats:
                stats = self.worker_stats[worker_id]
                stats["active_requests"] = max(0, stats["active_requests"] - 1)
                stats["total_requests"] += 1
                stats["total_processing_time"] += processing_time
                stats["avg_processing_time"] = (
                    stats["total_processing_time"] / stats["total_requests"]
                )
                stats["last_activity"] = time.time()
    
    def increment_active_requests(self, worker_id: str) -> None:
        """Increment active request count for worker."""
        with self.stats_lock:
            if worker_id in self.worker_stats:
                self.worker_stats[worker_id]["active_requests"] += 1
    
    def get_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get current worker statistics."""
        with self.stats_lock:
            return {k: v.copy() for k, v in self.worker_stats.items()}


class InferenceWorker:
    """High-performance inference worker process."""
    
    def __init__(self, worker_id: str, pipeline: ResilientAnomalyPipeline):
        self.worker_id = worker_id
        self.pipeline = pipeline
        self.logger = get_logger(f"{__name__}.Worker.{worker_id}")
        self.is_running = False
        self.requests_processed = 0
    
    def start(self, request_queue: queue.Queue, result_queue: queue.Queue) -> None:
        """Start processing requests from queue."""
        self.is_running = True
        self.logger.info(f"Worker {self.worker_id} started")
        
        while self.is_running:
            try:
                # Get request with timeout
                request = request_queue.get(timeout=1.0)
                
                if request is None:  # Shutdown signal
                    break
                
                # Process request
                result = self._process_request(request)
                result_queue.put(result)
                
                self.requests_processed += 1
                
                # Mark task as done
                request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {self.worker_id} error: {e}")
                # Put error result
                if 'request' in locals():
                    error_result = InferenceResult(
                        request_id=request.request_id,
                        result=DetectionResult(
                            anomaly_scores=np.array([]),
                            anomaly_predictions=np.array([]),
                            confidence_scores=np.array([]),
                            detection_method="Error",
                            metadata={"error": str(e)}
                        ),
                        processing_time=0.0,
                        cache_hit=False,
                        worker_id=self.worker_id,
                        queue_time=0.0,
                        total_time=0.0
                    )
                    result_queue.put(error_result)
                    request_queue.task_done()
        
        self.logger.info(f"Worker {self.worker_id} stopped. Processed {self.requests_processed} requests")
    
    def _process_request(self, request: InferenceRequest) -> InferenceResult:
        """Process individual inference request."""
        start_time = time.time()
        queue_time = start_time - request.timestamp
        
        try:
            # Run inference
            processing_start = time.time()
            detection_result = self.pipeline.predict(request.data, method="ensemble")
            processing_time = time.time() - processing_start
            
            total_time = time.time() - start_time
            
            return InferenceResult(
                request_id=request.request_id,
                result=detection_result,
                processing_time=processing_time,
                cache_hit=False,  # Cache hits handled at engine level
                worker_id=self.worker_id,
                queue_time=queue_time,
                total_time=total_time
            )
            
        except Exception as e:
            self.logger.error(f"Processing error for request {request.request_id}: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the worker."""
        self.is_running = False


class ScalableInferenceEngine:
    """High-performance scalable inference engine."""
    
    def __init__(self,
                 num_workers: int = 4,
                 max_queue_size: int = 1000,
                 batch_size: int = 32,
                 batch_timeout: float = 0.1,
                 cache_config: Optional[Dict[str, Any]] = None,
                 enable_async: bool = True):
        
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.enable_async = enable_async
        self.logger = get_logger(__name__)
        
        # Initialize core pipeline
        self.pipeline = ResilientAnomalyPipeline()
        
        # Initialize caching
        cache_config = cache_config or {}
        self.cache = AdaptiveCache(**cache_config)
        
        # Initialize load balancer
        self.load_balancer = LoadBalancer()
        
        # Worker management
        self.workers = {}
        self.worker_threads = {}
        self.request_queues = {}
        self.result_queue = queue.Queue()
        
        # Processing state
        self.is_running = False
        self.processing_strategy = ProcessingStrategy.ADAPTIVE
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics(
            requests_per_second=0.0,
            avg_processing_time=0.0,
            avg_queue_time=0.0,
            cache_hit_rate=0.0,
            active_workers=0,
            queue_depth=0,
            memory_usage_mb=0.0,
            cpu_utilization=0.0,
            throughput_mbps=0.0,
            error_rate=0.0
        )
        
        # Request tracking
        self.pending_requests = {}
        self.request_metrics = deque(maxlen=10000)  # Keep last 10k requests
        
        # Async support
        if enable_async and UVLOOP_AVAILABLE:
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        
        self.logger.info(f"Scalable Inference Engine initialized with {num_workers} workers")
    
    def start(self) -> None:
        """Start the inference engine."""
        if self.is_running:
            self.logger.warning("Engine already running")
            return
        
        self.logger.info("Starting scalable inference engine")
        
        # Start workers
        for i in range(self.num_workers):
            self._start_worker(f"worker_{i}")
        
        # Start result processing thread
        self.result_processor = threading.Thread(target=self._process_results, daemon=True)
        self.result_processor.start()
        
        # Start metrics collection thread
        self.metrics_collector = threading.Thread(target=self._collect_metrics, daemon=True)
        self.metrics_collector.start()
        
        self.is_running = True
        self.logger.info("Scalable inference engine started")
    
    def stop(self) -> None:
        """Stop the inference engine."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping scalable inference engine")
        self.is_running = False
        
        # Stop workers
        for worker_id in self.workers:
            self.request_queues[worker_id].put(None)  # Shutdown signal
            self.workers[worker_id].stop()
        
        # Wait for workers to finish
        for thread in self.worker_threads.values():
            thread.join(timeout=5.0)
        
        self.logger.info("Scalable inference engine stopped")
    
    def _start_worker(self, worker_id: str) -> None:
        """Start an individual worker."""
        # Create request queue for this worker
        request_queue = queue.Queue(maxsize=self.max_queue_size // self.num_workers)
        
        # Create worker
        worker = InferenceWorker(worker_id, self.pipeline)
        
        # Create and start worker thread
        worker_thread = threading.Thread(
            target=worker.start,
            args=(request_queue, self.result_queue),
            daemon=True
        )
        
        # Store references
        self.workers[worker_id] = worker
        self.worker_threads[worker_id] = worker_thread
        self.request_queues[worker_id] = request_queue
        
        # Register with load balancer
        self.load_balancer.register_worker(worker_id, request_queue)
        
        # Start the worker
        worker_thread.start()
        
        self.logger.debug(f"Started worker: {worker_id}")
    
    def _process_results(self) -> None:
        """Process results from workers."""
        while self.is_running:
            try:
                result = self.result_queue.get(timeout=1.0)
                
                # Update load balancer stats
                self.load_balancer.update_worker_stats(
                    result.worker_id, 
                    result.processing_time
                )
                
                # Store result for pending request
                if result.request_id in self.pending_requests:
                    callback = self.pending_requests[result.request_id]
                    if callback:
                        try:
                            callback(result)
                        except Exception as e:
                            self.logger.error(f"Callback error for {result.request_id}: {e}")
                    del self.pending_requests[result.request_id]
                
                # Track metrics
                self.request_metrics.append({
                    "timestamp": time.time(),
                    "processing_time": result.processing_time,
                    "queue_time": result.queue_time,
                    "total_time": result.total_time,
                    "cache_hit": result.cache_hit,
                    "error": "error" in result.result.metadata
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Result processing error: {e}")
    
    def _collect_metrics(self) -> None:
        """Collect and update performance metrics."""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Get recent metrics (last 60 seconds)
                recent_metrics = [
                    m for m in self.request_metrics 
                    if current_time - m["timestamp"] <= 60.0
                ]
                
                if recent_metrics:
                    # Calculate RPS
                    self.performance_metrics.requests_per_second = len(recent_metrics) / 60.0
                    
                    # Calculate average times
                    self.performance_metrics.avg_processing_time = np.mean([
                        m["processing_time"] for m in recent_metrics
                    ])
                    self.performance_metrics.avg_queue_time = np.mean([
                        m["queue_time"] for m in recent_metrics
                    ])
                    
                    # Calculate cache hit rate
                    cache_hits = sum(1 for m in recent_metrics if m["cache_hit"])
                    self.performance_metrics.cache_hit_rate = cache_hits / len(recent_metrics)
                    
                    # Calculate error rate
                    errors = sum(1 for m in recent_metrics if m["error"])
                    self.performance_metrics.error_rate = errors / len(recent_metrics)
                
                # Update worker stats
                worker_stats = self.load_balancer.get_worker_stats()
                self.performance_metrics.active_workers = len([
                    w for w in worker_stats.values() 
                    if w["active_requests"] > 0
                ])
                
                self.performance_metrics.queue_depth = sum(
                    w["active_requests"] for w in worker_stats.values()
                )
                
                # Update cache stats
                cache_stats = self.cache.get_cache_stats()
                if cache_stats["hits"] + cache_stats["misses"] > 0:
                    self.performance_metrics.cache_hit_rate = cache_stats["hit_rate"]
                
                # System metrics (if available)
                try:
                    import psutil
                    process = psutil.Process()
                    self.performance_metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                    self.performance_metrics.cpu_utilization = process.cpu_percent() / 100.0
                except ImportError:
                    pass
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
            
            time.sleep(10.0)  # Collect metrics every 10 seconds
    
    def fit(self, data: np.ndarray) -> None:
        """Train the inference pipeline."""
        self.logger.info("Training inference pipeline")
        self.pipeline.fit(data, enable_quantum=True)
        self.logger.info("Inference pipeline training completed")
    
    def predict(self, 
                data: np.ndarray, 
                request_id: Optional[str] = None,
                priority: int = 1,
                callback: Optional[Callable] = None) -> Union[InferenceResult, str]:
        """Submit prediction request."""
        
        if not self.is_running:
            raise RuntimeError("Inference engine is not running")
        
        # Generate request ID if not provided
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}"
        
        # Check cache first
        cache_key = self.cache._generate_key(data)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            # Cache hit - return immediately
            result = InferenceResult(
                request_id=request_id,
                result=cached_result,
                processing_time=0.0,
                cache_hit=True,
                worker_id="cache",
                queue_time=0.0,
                total_time=0.0
            )
            
            if callback:
                callback(result)
                return request_id
            else:
                return result
        
        # Cache miss - submit to worker
        request = InferenceRequest(
            request_id=request_id,
            data=data,
            timestamp=time.time(),
            priority=priority,
            callback=callback
        )
        
        # Select worker
        worker_id = self.load_balancer.select_worker()
        if not worker_id:
            raise RuntimeError("No workers available")
        
        # Submit to worker queue
        try:
            self.request_queues[worker_id].put(request, timeout=1.0)
            self.load_balancer.increment_active_requests(worker_id)
            
            # Store callback for async processing
            if callback:
                self.pending_requests[request_id] = callback
                return request_id
            else:
                # Synchronous processing - wait for result
                self.pending_requests[request_id] = None
                
                # Wait for result (with timeout)
                timeout = 30.0
                start_time = time.time()
                
                while request_id in self.pending_requests:
                    if time.time() - start_time > timeout:
                        if request_id in self.pending_requests:
                            del self.pending_requests[request_id]
                        raise TimeoutError(f"Request {request_id} timed out")
                    time.sleep(0.01)
                
                # This would need to be implemented differently for sync returns
                # For now, return request ID
                return request_id
                
        except queue.Full:
            raise RuntimeError("Request queue is full")
    
    async def predict_async(self,
                           data: np.ndarray,
                           request_id: Optional[str] = None,
                           priority: int = 1) -> InferenceResult:
        """Async prediction interface."""
        if not self.enable_async:
            raise RuntimeError("Async mode not enabled")
        
        # Create future for result
        future = asyncio.Future()
        
        def callback(result: InferenceResult):
            if not future.done():
                future.set_result(result)
        
        # Submit request
        req_id = self.predict(data, request_id, priority, callback)
        
        # Wait for result
        return await future
    
    def predict_batch(self, 
                     data_batch: List[np.ndarray],
                     request_ids: Optional[List[str]] = None,
                     priority: int = 1) -> List[str]:
        """Submit batch of prediction requests."""
        request_ids = request_ids or [f"batch_{i}_{int(time.time() * 1000)}" for i in range(len(data_batch))]
        
        submitted_ids = []
        for i, data in enumerate(data_batch):
            try:
                req_id = self.predict(data, request_ids[i], priority)
                submitted_ids.append(req_id)
            except Exception as e:
                self.logger.error(f"Failed to submit batch request {i}: {e}")
        
        return submitted_ids
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.performance_metrics
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        worker_stats = self.load_balancer.get_worker_stats()
        cache_stats = self.cache.get_cache_stats()
        
        return {
            "is_running": self.is_running,
            "num_workers": self.num_workers,
            "active_workers": len([w for w in worker_stats.values() if w["active_requests"] > 0]),
            "total_queue_depth": sum(w["active_requests"] for w in worker_stats.values()),
            "pending_requests": len(self.pending_requests),
            "performance_metrics": {
                "requests_per_second": self.performance_metrics.requests_per_second,
                "avg_processing_time": self.performance_metrics.avg_processing_time,
                "avg_queue_time": self.performance_metrics.avg_queue_time,
                "cache_hit_rate": self.performance_metrics.cache_hit_rate,
                "error_rate": self.performance_metrics.error_rate,
                "memory_usage_mb": self.performance_metrics.memory_usage_mb,
                "cpu_utilization": self.performance_metrics.cpu_utilization
            },
            "worker_stats": worker_stats,
            "cache_stats": cache_stats,
            "pipeline_status": self.pipeline.get_system_status()
        }
    
    def scale_workers(self, target_workers: int) -> None:
        """Dynamically scale the number of workers."""
        current_workers = len(self.workers)
        
        if target_workers > current_workers:
            # Scale up
            for i in range(current_workers, target_workers):
                worker_id = f"worker_{i}"
                self._start_worker(worker_id)
            self.num_workers = target_workers
            self.logger.info(f"Scaled up to {target_workers} workers")
            
        elif target_workers < current_workers:
            # Scale down
            workers_to_stop = list(self.workers.keys())[target_workers:]
            for worker_id in workers_to_stop:
                # Signal shutdown
                self.request_queues[worker_id].put(None)
                self.workers[worker_id].stop()
                
                # Wait for worker to stop
                self.worker_threads[worker_id].join(timeout=5.0)
                
                # Clean up
                del self.workers[worker_id]
                del self.worker_threads[worker_id]
                del self.request_queues[worker_id]
            
            self.num_workers = target_workers
            self.logger.info(f"Scaled down to {target_workers} workers")
    
    def auto_scale(self) -> None:
        """Automatically scale based on current load."""
        current_rps = self.performance_metrics.requests_per_second
        avg_queue_time = self.performance_metrics.avg_queue_time
        current_workers = len(self.workers)
        
        # Scale up if queue time is high or RPS is high
        if avg_queue_time > 1.0 or current_rps > current_workers * 10:
            new_target = min(current_workers + 2, 16)  # Max 16 workers
            self.scale_workers(new_target)
            
        # Scale down if load is low
        elif avg_queue_time < 0.1 and current_rps < current_workers * 2 and current_workers > 2:
            new_target = max(current_workers - 1, 2)  # Min 2 workers
            self.scale_workers(new_target)
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        self.logger.info("Cache cleared")
    
    def save_state(self, path: Path) -> None:
        """Save engine state for persistence."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        self.pipeline.save(path / "pipeline")
        
        # Save engine state
        engine_state = {
            "num_workers": self.num_workers,
            "max_queue_size": self.max_queue_size,
            "batch_size": self.batch_size,
            "batch_timeout": self.batch_timeout,
            "processing_strategy": self.processing_strategy.value,
            "performance_metrics": {
                "requests_per_second": self.performance_metrics.requests_per_second,
                "avg_processing_time": self.performance_metrics.avg_processing_time,
                "cache_hit_rate": self.performance_metrics.cache_hit_rate
            }
        }
        
        import pickle
        with open(path / "engine_state.pkl", "wb") as f:
            pickle.dump(engine_state, f)
        
        self.logger.info(f"Engine state saved to {path}")
    
    def load_state(self, path: Path) -> None:
        """Load engine state from persistence."""
        # Load pipeline
        self.pipeline.load(path / "pipeline")
        
        # Load engine state
        engine_state_path = path / "engine_state.pkl"
        if engine_state_path.exists():
            import pickle
            with open(engine_state_path, "rb") as f:
                engine_state = pickle.load(f)
                
                # Apply loaded configuration
                self.processing_strategy = ProcessingStrategy(engine_state["processing_strategy"])
                # Other settings would typically require restart to apply
        
        self.logger.info(f"Engine state loaded from {path}")
    
    def __del__(self):
        """Cleanup on destruction."""
        if self.is_running:
            self.stop()