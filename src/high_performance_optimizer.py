"""High-performance optimization engine for IoT anomaly detection systems."""

import time
import logging
import threading
import multiprocessing as mp
import concurrent.futures
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum
from pathlib import Path
import queue
import gc
import contextlib

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class OptimizationLevel(Enum):
    """Optimization levels."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


class ResourceType(Enum):
    """System resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    GPU = "gpu"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_name: str
    execution_time_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    throughput_ops_per_sec: float
    latency_p95_ms: float
    error_rate_percent: float = 0.0
    optimization_applied: str = "none"
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    level: OptimizationLevel = OptimizationLevel.BASIC
    enable_caching: bool = True
    enable_batching: bool = True
    enable_parallel_processing: bool = True
    enable_memory_optimization: bool = True
    enable_vectorization: bool = True
    
    # Resource limits
    max_cpu_cores: Optional[int] = None
    max_memory_mb: Optional[int] = None
    max_batch_size: int = 1000
    cache_size_mb: int = 256
    
    # Performance targets
    target_latency_ms: float = 100.0
    target_throughput_ops_sec: float = 1000.0
    target_error_rate_percent: float = 1.0


class MemoryPool:
    """High-performance memory pool for object reuse."""
    
    def __init__(self, object_factory: Callable, initial_size: int = 100, max_size: int = 1000):
        self.object_factory = object_factory
        self.max_size = max_size
        self.pool = queue.Queue(maxsize=max_size)
        self._lock = threading.RLock()
        self.stats = {
            "created": 0,
            "reused": 0,
            "pool_hits": 0,
            "pool_misses": 0
        }
        
        # Pre-populate pool
        for _ in range(initial_size):
            try:
                obj = self.object_factory()
                self.pool.put_nowait(obj)
                self.stats["created"] += 1
            except queue.Full:
                break
    
    def acquire(self):
        """Acquire object from pool."""
        try:
            obj = self.pool.get_nowait()
            self.stats["pool_hits"] += 1
            self.stats["reused"] += 1
            return obj
        except queue.Empty:
            self.stats["pool_misses"] += 1
            obj = self.object_factory()
            self.stats["created"] += 1
            return obj
    
    def release(self, obj):
        """Return object to pool."""
        try:
            # Reset object state if it has a reset method
            if hasattr(obj, 'reset'):
                obj.reset()
            
            self.pool.put_nowait(obj)
        except queue.Full:
            # Pool is full, let object be garbage collected
            pass
    
    @contextlib.contextmanager
    def get_object(self):
        """Context manager for acquiring and releasing objects."""
        obj = self.acquire()
        try:
            yield obj
        finally:
            self.release(obj)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            **self.stats,
            "pool_size": self.pool.qsize(),
            "hit_rate": self.stats["pool_hits"] / (self.stats["pool_hits"] + self.stats["pool_misses"]) 
                       if (self.stats["pool_hits"] + self.stats["pool_misses"]) > 0 else 0.0
        }


class BatchProcessor:
    """High-performance batch processing engine."""
    
    def __init__(self, batch_size: int = 100, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.batch_handlers: Dict[str, Callable] = {}
        
        self.processing_thread = None
        self.is_processing = False
        self.stats = {
            "batches_processed": 0,
            "items_processed": 0,
            "average_batch_size": 0.0,
            "processing_time_ms": 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def register_handler(self, operation_type: str, handler: Callable[[List[Any]], List[Any]]) -> None:
        """Register batch processing handler."""
        self.batch_handlers[operation_type] = handler
    
    def start_processing(self) -> None:
        """Start batch processing thread."""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self) -> None:
        """Stop batch processing."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
    
    def submit_item(self, operation_type: str, item: Any, result_callback: Optional[Callable] = None) -> str:
        """Submit item for batch processing."""
        item_id = f"{operation_type}_{int(time.time() * 1000000)}"
        batch_item = {
            "id": item_id,
            "operation_type": operation_type,
            "data": item,
            "callback": result_callback,
            "submitted_at": time.time()
        }
        
        try:
            self.input_queue.put_nowait(batch_item)
            return item_id
        except queue.Full:
            self.logger.warning("Batch processing queue full")
            return ""
    
    def _processing_loop(self) -> None:
        """Main batch processing loop."""
        while self.is_processing:
            try:
                batch = self._collect_batch()
                if batch:
                    self._process_batch(batch)
                else:
                    time.sleep(0.01)  # Short sleep when no items
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
                time.sleep(0.1)
    
    def _collect_batch(self) -> Optional[List[Dict[str, Any]]]:
        """Collect batch of items for processing."""
        batch = []
        batch_start_time = time.time()
        
        # Collect items up to batch size or max wait time
        while (len(batch) < self.batch_size and 
               time.time() - batch_start_time < self.max_wait_time):
            try:
                item = self.input_queue.get(timeout=0.1)
                batch.append(item)
            except queue.Empty:
                if batch:  # If we have some items, process them
                    break
                continue
        
        return batch if batch else None
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of items."""
        start_time = time.time()
        
        # Group by operation type
        batches_by_type = {}
        for item in batch:
            op_type = item["operation_type"]
            if op_type not in batches_by_type:
                batches_by_type[op_type] = []
            batches_by_type[op_type].append(item)
        
        # Process each operation type
        for op_type, items in batches_by_type.items():
            if op_type in self.batch_handlers:
                try:
                    # Extract data for processing
                    data_items = [item["data"] for item in items]
                    
                    # Process batch
                    results = self.batch_handlers[op_type](data_items)
                    
                    # Handle results
                    for item, result in zip(items, results):
                        if item["callback"]:
                            try:
                                item["callback"](result)
                            except Exception as e:
                                self.logger.error(f"Callback error: {e}")
                        
                        # Store result for retrieval
                        self.output_queue.put({
                            "id": item["id"],
                            "result": result,
                            "processed_at": time.time()
                        })
                
                except Exception as e:
                    self.logger.error(f"Batch handler error for {op_type}: {e}")
        
        # Update statistics
        processing_time = (time.time() - start_time) * 1000
        self.stats["batches_processed"] += 1
        self.stats["items_processed"] += len(batch)
        self.stats["average_batch_size"] = self.stats["items_processed"] / self.stats["batches_processed"]
        self.stats["processing_time_ms"] += processing_time
    
    def get_result(self, item_id: str, timeout: float = 1.0) -> Optional[Any]:
        """Get result for processed item."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result_item = self.output_queue.get(timeout=0.1)
                if result_item["id"] == item_id:
                    return result_item["result"]
                else:
                    # Put back if not matching
                    self.output_queue.put(result_item)
            except queue.Empty:
                continue
        
        return None


class VectorizedOperations:
    """High-performance vectorized operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def vectorized_anomaly_score(data: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        """Vectorized anomaly score calculation."""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy not available for vectorized operations")
        
        # Ensure data is numpy array
        data = np.asarray(data, dtype=np.float32)
        baseline = np.asarray(baseline, dtype=np.float32)
        
        # Vectorized computation
        deviation = np.abs(data - baseline)
        normalized_deviation = deviation / (baseline + 1e-8)
        
        # Apply non-linear scoring
        scores = np.tanh(normalized_deviation * 2.0)  # Compress to [0, 1)
        
        return scores
    
    @staticmethod
    def batch_correlation_analysis(data_matrix: np.ndarray) -> np.ndarray:
        """Batch correlation analysis for multiple sensor streams."""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy not available for vectorized operations")
        
        # Ensure float32 for performance
        data_matrix = np.asarray(data_matrix, dtype=np.float32)
        
        # Compute correlation matrix efficiently
        correlation_matrix = np.corrcoef(data_matrix)
        
        # Handle NaN values
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        return correlation_matrix
    
    @staticmethod
    def sliding_window_stats(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Efficient sliding window statistics computation."""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy not available for vectorized operations")
        
        data = np.asarray(data, dtype=np.float32)
        n = len(data)
        
        if n < window_size:
            return np.array([]), np.array([])
        
        # Use numpy stride tricks for efficient sliding window
        from numpy.lib.stride_tricks import sliding_window_view
        
        windows = sliding_window_view(data, window_shape=window_size)
        
        # Compute statistics vectorized
        means = np.mean(windows, axis=1)
        stds = np.std(windows, axis=1)
        
        return means, stds


# Numba-optimized functions (if available)
if NUMBA_AVAILABLE:
    
    @njit(parallel=True)
    def numba_batch_anomaly_detection(data_matrix, baseline_vector, threshold):
        """Numba-optimized batch anomaly detection."""
        n_samples, n_features = data_matrix.shape
        anomaly_scores = np.zeros(n_samples, dtype=np.float32)
        
        for i in prange(n_samples):
            score = 0.0
            for j in range(n_features):
                deviation = abs(data_matrix[i, j] - baseline_vector[j])
                normalized = deviation / (baseline_vector[j] + 1e-8)
                score += normalized ** 2
            
            anomaly_scores[i] = score / n_features
        
        return anomaly_scores
    
    @njit
    def numba_sliding_window_mean(data, window_size):
        """Numba-optimized sliding window mean."""
        n = len(data)
        result = np.zeros(n - window_size + 1, dtype=np.float32)
        
        # Calculate first window
        window_sum = 0.0
        for i in range(window_size):
            window_sum += data[i]
        result[0] = window_sum / window_size
        
        # Slide window and update sum
        for i in range(1, n - window_size + 1):
            window_sum += data[i + window_size - 1] - data[i - 1]
            result[i] = window_sum / window_size
        
        return result


class ParallelProcessor:
    """Parallel processing engine for CPU-intensive operations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)  # Cap at 8 cores
        self.thread_pool = None
        self.process_pool = None
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "total_processing_time": 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        """Context manager entry."""
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
    
    def submit_thread_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task to thread pool."""
        if not self.thread_pool:
            raise RuntimeError("Parallel processor not initialized")
        
        self.stats["tasks_submitted"] += 1
        future = self.thread_pool.submit(func, *args, **kwargs)
        
        # Add callback to update stats
        def update_stats(fut):
            self.stats["tasks_completed"] += 1
        
        future.add_done_callback(update_stats)
        return future
    
    def submit_process_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task to process pool."""
        if not self.process_pool:
            raise RuntimeError("Parallel processor not initialized")
        
        self.stats["tasks_submitted"] += 1
        future = self.process_pool.submit(func, *args, **kwargs)
        
        # Add callback to update stats
        def update_stats(fut):
            self.stats["tasks_completed"] += 1
        
        future.add_done_callback(update_stats)
        return future
    
    def map_parallel(
        self, 
        func: Callable, 
        iterable, 
        use_processes: bool = False,
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """Parallel map operation."""
        start_time = time.time()
        
        if use_processes and self.process_pool:
            results = list(self.process_pool.map(func, iterable, chunksize=chunk_size))
        elif self.thread_pool:
            results = list(self.thread_pool.map(func, iterable))
        else:
            results = list(map(func, iterable))  # Fallback to sequential
        
        processing_time = time.time() - start_time
        self.stats["total_processing_time"] += processing_time
        
        return results


class HighPerformanceOptimizer:
    """Main high-performance optimization engine."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.metrics: List[PerformanceMetrics] = []
        self.baseline_metrics: Dict[str, PerformanceMetrics] = {}
        
        # Optimization components
        self.memory_pools: Dict[str, MemoryPool] = {}
        self.batch_processor = BatchProcessor(
            batch_size=self.config.max_batch_size,
            max_wait_time=1.0
        ) if self.config.enable_batching else None
        
        self.vectorized_ops = VectorizedOperations()
        
        # Resource monitoring
        self.resource_monitor = None
        if PSUTIL_AVAILABLE:
            self._start_resource_monitoring()
        
        # Start batch processing if enabled
        if self.batch_processor:
            self.batch_processor.start_processing()
        
        self.logger.info(f"High-performance optimizer initialized with level: {self.config.level.value}")
    
    def _start_resource_monitoring(self) -> None:
        """Start resource monitoring thread."""
        def monitor_resources():
            while True:
                try:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_info = psutil.virtual_memory()
                    
                    # Log if resources are high
                    if cpu_percent > 80:
                        self.logger.warning(f"High CPU usage: {cpu_percent}%")
                    if memory_info.percent > 80:
                        self.logger.warning(f"High memory usage: {memory_info.percent}%")
                    
                    time.sleep(10)  # Check every 10 seconds
                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {e}")
                    time.sleep(30)
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    def create_memory_pool(
        self, 
        pool_name: str, 
        object_factory: Callable,
        initial_size: int = 100,
        max_size: int = 1000
    ) -> MemoryPool:
        """Create memory pool for object reuse."""
        pool = MemoryPool(object_factory, initial_size, max_size)
        self.memory_pools[pool_name] = pool
        return pool
    
    def get_memory_pool(self, pool_name: str) -> Optional[MemoryPool]:
        """Get existing memory pool."""
        return self.memory_pools.get(pool_name)
    
    @contextlib.contextmanager
    def performance_context(self, operation_name: str):
        """Context manager for performance measurement."""
        start_time = time.time()
        start_cpu = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0
        start_memory = psutil.virtual_memory().used / (1024**2) if PSUTIL_AVAILABLE else 0.0
        
        try:
            yield
        finally:
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            end_cpu = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0
            end_memory = psutil.virtual_memory().used / (1024**2) if PSUTIL_AVAILABLE else 0.0
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time_ms=execution_time,
                cpu_usage_percent=(start_cpu + end_cpu) / 2,
                memory_usage_mb=end_memory - start_memory,
                throughput_ops_per_sec=1000.0 / execution_time if execution_time > 0 else 0.0,
                latency_p95_ms=execution_time,  # Simplified for single operation
                optimization_applied=self.config.level.value
            )
            
            self.metrics.append(metrics)
            
            # Keep only recent metrics
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-500:]
    
    def optimize_anomaly_detection(
        self, 
        sensor_data: np.ndarray, 
        baseline_data: np.ndarray
    ) -> np.ndarray:
        """Optimized anomaly detection processing."""
        with self.performance_context("anomaly_detection"):
            if self.config.level == OptimizationLevel.NONE:
                return self._basic_anomaly_detection(sensor_data, baseline_data)
            
            elif self.config.level == OptimizationLevel.BASIC:
                return self._optimized_anomaly_detection(sensor_data, baseline_data)
            
            elif self.config.level == OptimizationLevel.AGGRESSIVE:
                return self._aggressive_optimized_detection(sensor_data, baseline_data)
            
            elif self.config.level == OptimizationLevel.EXTREME:
                return self._extreme_optimized_detection(sensor_data, baseline_data)
    
    def _basic_anomaly_detection(self, sensor_data: np.ndarray, baseline_data: np.ndarray) -> np.ndarray:
        """Basic unoptimized anomaly detection."""
        scores = []
        for i in range(len(sensor_data)):
            deviation = abs(sensor_data[i] - baseline_data[i % len(baseline_data)])
            normalized = deviation / (baseline_data[i % len(baseline_data)] + 1e-8)
            scores.append(normalized)
        return np.array(scores)
    
    def _optimized_anomaly_detection(self, sensor_data: np.ndarray, baseline_data: np.ndarray) -> np.ndarray:
        """Basic optimized anomaly detection using vectorization."""
        if NUMPY_AVAILABLE:
            return self.vectorized_ops.vectorized_anomaly_score(sensor_data, baseline_data)
        else:
            return self._basic_anomaly_detection(sensor_data, baseline_data)
    
    def _aggressive_optimized_detection(self, sensor_data: np.ndarray, baseline_data: np.ndarray) -> np.ndarray:
        """Aggressively optimized detection with batching and parallelization."""
        # Use vectorized operations
        scores = self._optimized_anomaly_detection(sensor_data, baseline_data)
        
        # Apply additional optimizations like memory pool usage
        if "anomaly_arrays" in self.memory_pools:
            pool = self.memory_pools["anomaly_arrays"]
            with pool.get_object() as temp_array:
                # Use pre-allocated arrays for intermediate computations
                temp_array[:len(scores)] = scores
                return temp_array[:len(scores)].copy()
        
        return scores
    
    def _extreme_optimized_detection(self, sensor_data: np.ndarray, baseline_data: np.ndarray) -> np.ndarray:
        """Extremely optimized detection using all available techniques."""
        # Use Numba if available
        if NUMBA_AVAILABLE and sensor_data.ndim == 2:
            return numba_batch_anomaly_detection(
                sensor_data.astype(np.float32),
                baseline_data.astype(np.float32),
                0.5  # threshold
            )
        else:
            return self._aggressive_optimized_detection(sensor_data, baseline_data)
    
    def optimize_batch_processing(
        self, 
        data_items: List[Any], 
        processing_func: Callable,
        use_parallel: bool = True
    ) -> List[Any]:
        """Optimize batch processing with parallelization."""
        with self.performance_context("batch_processing"):
            if not use_parallel or len(data_items) < 100:
                # Process sequentially for small batches
                return [processing_func(item) for item in data_items]
            
            # Use parallel processing for large batches
            with ParallelProcessor(max_workers=self.config.max_cpu_cores) as processor:
                results = processor.map_parallel(
                    processing_func,
                    data_items,
                    use_processes=False  # Use threads for I/O bound tasks
                )
                return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance optimization summary."""
        if not self.metrics:
            return {"status": "no_metrics_available"}
        
        # Calculate aggregated metrics
        total_operations = len(self.metrics)
        avg_execution_time = sum(m.execution_time_ms for m in self.metrics) / total_operations
        avg_cpu_usage = sum(m.cpu_usage_percent for m in self.metrics) / total_operations
        avg_throughput = sum(m.throughput_ops_per_sec for m in self.metrics) / total_operations
        
        # Get memory pool stats
        pool_stats = {
            name: pool.get_stats()
            for name, pool in self.memory_pools.items()
        }
        
        # Get batch processor stats
        batch_stats = {}
        if self.batch_processor:
            batch_stats = self.batch_processor.stats
        
        return {
            "optimization_level": self.config.level.value,
            "total_operations": total_operations,
            "performance_metrics": {
                "average_execution_time_ms": avg_execution_time,
                "average_cpu_usage_percent": avg_cpu_usage,
                "average_throughput_ops_per_sec": avg_throughput
            },
            "memory_pools": pool_stats,
            "batch_processing": batch_stats,
            "recent_operations": [
                {
                    "operation": m.operation_name,
                    "execution_time_ms": m.execution_time_ms,
                    "throughput": m.throughput_ops_per_sec,
                    "optimization": m.optimization_applied
                }
                for m in self.metrics[-10:]  # Last 10 operations
            ]
        }
    
    def export_performance_report(self, output_path: str) -> None:
        """Export detailed performance report."""
        report_data = {
            "timestamp": time.time(),
            "configuration": {
                "optimization_level": self.config.level.value,
                "enable_caching": self.config.enable_caching,
                "enable_batching": self.config.enable_batching,
                "enable_parallel_processing": self.config.enable_parallel_processing,
                "max_cpu_cores": self.config.max_cpu_cores,
                "max_memory_mb": self.config.max_memory_mb
            },
            "summary": self.get_performance_summary(),
            "detailed_metrics": [
                {
                    "operation": m.operation_name,
                    "execution_time_ms": m.execution_time_ms,
                    "cpu_usage_percent": m.cpu_usage_percent,
                    "memory_usage_mb": m.memory_usage_mb,
                    "throughput_ops_per_sec": m.throughput_ops_per_sec,
                    "timestamp": m.timestamp
                }
                for m in self.metrics
            ]
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Performance report exported to {output_path}")
    
    def shutdown(self) -> None:
        """Shutdown optimizer and cleanup resources."""
        if self.batch_processor:
            self.batch_processor.stop_processing()
        
        # Clear memory pools
        for pool in self.memory_pools.values():
            while not pool.pool.empty():
                try:
                    pool.pool.get_nowait()
                except queue.Empty:
                    break
        
        # Force garbage collection
        gc.collect()
        
        self.logger.info("High-performance optimizer shutdown complete")


# Factory functions
def create_optimizer(level: OptimizationLevel = OptimizationLevel.BASIC, **kwargs) -> HighPerformanceOptimizer:
    """Create performance optimizer with specified level."""
    config = OptimizationConfig(level=level, **kwargs)
    return HighPerformanceOptimizer(config)


# Example usage and benchmarking
if __name__ == "__main__":
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="High-Performance Optimizer")
    parser.add_argument("--level", choices=["none", "basic", "aggressive", "extreme"],
                       default="basic", help="Optimization level")
    parser.add_argument("--data-size", type=int, default=10000, help="Test data size")
    parser.add_argument("--iterations", type=int, default=100, help="Test iterations")
    parser.add_argument("--export-report", help="Export performance report")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = create_optimizer(
        level=OptimizationLevel(args.level),
        enable_batching=True,
        enable_parallel_processing=True,
        max_cpu_cores=4
    )
    
    print(f"Testing optimizer with level: {args.level}")
    print(f"Data size: {args.data_size}, Iterations: {args.iterations}")
    
    # Create memory pool for test arrays
    if NUMPY_AVAILABLE:
        optimizer.create_memory_pool(
            "anomaly_arrays",
            lambda: np.zeros(args.data_size, dtype=np.float32),
            initial_size=10,
            max_size=50
        )
        
        # Generate test data
        sensor_data = np.random.randn(args.data_size).astype(np.float32)
        baseline_data = np.random.randn(args.data_size).astype(np.float32)
        
        print("Running anomaly detection benchmark...")
        start_time = time.time()
        
        # Run benchmark
        for i in range(args.iterations):
            scores = optimizer.optimize_anomaly_detection(sensor_data, baseline_data)
            if i % 10 == 0:
                print(f"Completed iteration {i}")
        
        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per iteration: {total_time / args.iterations * 1000:.2f} ms")
        
        # Test batch processing
        print("\nTesting batch processing...")
        
        def dummy_processing_func(x):
            return x * 2 + 1
        
        test_items = list(range(1000))
        batch_results = optimizer.optimize_batch_processing(
            test_items,
            dummy_processing_func,
            use_parallel=True
        )
        
        print(f"Batch processing completed: {len(batch_results)} items")
    
    else:
        print("NumPy not available, running basic tests...")
        
        # Simple performance test without NumPy
        def simple_computation():
            result = 0
            for i in range(10000):
                result += i * 2
            return result
        
        start_time = time.time()
        for _ in range(args.iterations):
            simple_computation()
        
        total_time = time.time() - start_time
        print(f"Simple computation total time: {total_time:.2f} seconds")
    
    # Get performance summary
    summary = optimizer.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"Optimization Level: {summary.get('optimization_level', 'unknown')}")
    print(f"Total Operations: {summary.get('total_operations', 0)}")
    
    if 'performance_metrics' in summary:
        metrics = summary['performance_metrics']
        print(f"Average Execution Time: {metrics.get('average_execution_time_ms', 0):.2f} ms")
        print(f"Average Throughput: {metrics.get('average_throughput_ops_per_sec', 0):.2f} ops/sec")
    
    # Export report if requested
    if args.export_report:
        optimizer.export_performance_report(args.export_report)
        print(f"Performance report exported to {args.export_report}")
    
    optimizer.shutdown()
    print("Optimizer shutdown complete")