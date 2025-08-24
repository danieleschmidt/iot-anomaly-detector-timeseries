"""
Scalable IoT Anomaly Detection Pipeline
Generation 3: High-performance, optimized implementation with auto-scaling
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union, List, Iterator
import joblib
import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import psutil
import gc
from functools import lru_cache
import pickle
from collections import deque
import queue

from .robust_pipeline import RobustAnomalyPipeline
from .adaptive_cache import AdaptiveCache
from .resource_pool_manager import ResourcePoolManager
from .performance_monitor_cli import PerformanceMonitor
from .autoscaling_manager import AutoScalingManager
from .high_performance_optimizer import HighPerformanceOptimizer
from .streaming_processor import StreamingProcessor

logger = logging.getLogger(__name__)


class ScalablePipeline(RobustAnomalyPipeline):
    """High-performance, auto-scaling anomaly detection pipeline for production workloads."""
    
    def __init__(
        self,
        window_size: int = 30,
        latent_dim: int = 16,
        lstm_units: int = 32,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        max_memory_usage_gb: float = 8.0,
        enable_caching: bool = True,
        enable_parallel_processing: bool = True,
        max_workers: Optional[int] = None,
        enable_streaming: bool = False,
        streaming_buffer_size: int = 1000,
        enable_auto_scaling: bool = True,
        performance_optimization_level: str = "aggressive",
        **kwargs
    ):
        super().__init__(
            window_size=window_size,
            latent_dim=latent_dim,
            lstm_units=lstm_units,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            max_memory_usage_gb=max_memory_usage_gb,
            **kwargs
        )
        
        # Performance configuration
        self.enable_caching = enable_caching
        self.enable_parallel_processing = enable_parallel_processing
        self.max_workers = max_workers or min(cpu_count(), 8)
        self.enable_streaming = enable_streaming
        self.streaming_buffer_size = streaming_buffer_size
        self.enable_auto_scaling = enable_auto_scaling
        
        # Performance components
        self.cache = AdaptiveCache(max_size_gb=2.0) if enable_caching else None
        self.resource_pool = ResourcePoolManager(max_workers=self.max_workers)
        self.performance_monitor = PerformanceMonitor()
        self.optimizer = HighPerformanceOptimizer(optimization_level=performance_optimization_level)
        
        # Auto-scaling components
        if enable_auto_scaling:
            self.autoscaler = AutoScalingManager(
                target_cpu_utilization=70.0,
                target_memory_utilization=80.0,
                scale_up_threshold=5.0,
                scale_down_threshold=60.0
            )
        else:
            self.autoscaler = None
        
        # Streaming components
        if enable_streaming:
            self.streaming_processor = StreamingProcessor(
                buffer_size=streaming_buffer_size,
                batch_size=batch_size
            )
        else:
            self.streaming_processor = None
        
        # Performance metrics
        self.performance_metrics = {
            'throughput_samples_per_sec': 0,
            'latency_ms': {},
            'cache_hit_rate': 0,
            'memory_efficiency': 0,
            'cpu_utilization': 0,
            'parallel_speedup': 0,
            'batch_processing_stats': {}
        }
        
        logger.info(f"ScalablePipeline initialized with {self.max_workers} workers")
    
    @lru_cache(maxsize=128)
    def _cached_data_transform(self, data_hash: str, transform_params: tuple) -> np.ndarray:
        """Cached data transformation for repeated operations."""
        # This would contain the actual transformation logic
        # Cache key is based on data hash and parameters
        pass
    
    def _parallel_window_creation(self, data: np.ndarray, chunk_size: int = 1000) -> np.ndarray:
        """Create windows in parallel for large datasets."""
        
        def create_windows_chunk(start_idx: int, end_idx: int) -> List[np.ndarray]:
            """Create windows for a specific chunk of data."""
            chunk_windows = []
            for i in range(start_idx, min(end_idx, len(data) - self.window_size + 1)):
                chunk_windows.append(data[i:i + self.window_size])
            return chunk_windows
        
        if not self.enable_parallel_processing or len(data) < chunk_size:
            # Use single-threaded approach for small datasets
            return super().prepare_data(pd.DataFrame(data))[0]
        
        # Calculate optimal chunk size based on data size and worker count
        total_windows = len(data) - self.window_size + 1
        optimal_chunk_size = max(chunk_size, total_windows // self.max_workers)
        
        # Create chunks
        chunks = [(i, min(i + optimal_chunk_size, total_windows)) 
                 for i in range(0, total_windows, optimal_chunk_size)]
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            start_time = time.time()
            
            future_to_chunk = {
                executor.submit(create_windows_chunk, start, end): (start, end)
                for start, end in chunks
            }
            
            all_windows = []
            for future in future_to_chunk:
                chunk_windows = future.result()
                all_windows.extend(chunk_windows)
            
            processing_time = time.time() - start_time
            speedup = (total_windows / optimal_chunk_size) / processing_time if processing_time > 0 else 0
            self.performance_metrics['parallel_speedup'] = speedup
            
            logger.info(f"Parallel window creation: {len(all_windows)} windows in {processing_time:.2f}s (speedup: {speedup:.1f}x)")
        
        return np.array(all_windows)
    
    def _batch_predict_optimized(self, sequences: np.ndarray, batch_size: int = None) -> np.ndarray:
        """Optimized batch prediction with memory management."""
        if batch_size is None:
            batch_size = self.batch_size
        
        # Adaptive batch sizing based on memory
        available_memory_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
        if available_memory_gb < 2.0:
            batch_size = max(1, batch_size // 2)
            logger.warning(f"Low memory detected, reducing batch size to {batch_size}")
        
        predictions = []
        total_batches = (len(sequences) + batch_size - 1) // batch_size
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            
            # Cache key for batch predictions
            if self.cache:
                batch_hash = hash(batch.tobytes())
                cached_pred = self.cache.get(f"pred_{batch_hash}")
                if cached_pred is not None:
                    predictions.extend(cached_pred)
                    continue
            
            # Predict batch
            batch_pred = self.model.predict(batch, verbose=0)
            predictions.extend(batch_pred)
            
            # Cache result
            if self.cache:
                self.cache.put(f"pred_{batch_hash}", batch_pred)
            
            # Progress logging
            if i % (batch_size * 10) == 0:
                progress = (i // batch_size + 1) / total_batches * 100
                logger.debug(f"Prediction progress: {progress:.1f}%")
            
            # Memory management
            if i % (batch_size * 50) == 0:
                gc.collect()
        
        return np.array(predictions)
    
    def stream_process_data(self, data_stream: Iterator[pd.DataFrame]) -> Iterator[Dict[str, Any]]:
        """Process streaming data with real-time anomaly detection."""
        if not self.streaming_processor:
            raise RuntimeError("Streaming not enabled. Set enable_streaming=True")
        
        logger.info("Starting streaming processing")
        
        for data_chunk in data_stream:
            start_time = time.time()
            
            try:
                # Process chunk
                scaled_data = self.preprocessor.transform(data_chunk)
                
                # Create windows if possible
                if len(scaled_data) >= self.window_size:
                    windows = []
                    for i in range(len(scaled_data) - self.window_size + 1):
                        windows.append(scaled_data[i:i + self.window_size])
                    
                    if windows:
                        sequences = np.array(windows)
                        
                        # Detect anomalies
                        predictions = self._batch_predict_optimized(sequences)
                        errors = np.mean((sequences - predictions) ** 2, axis=(1, 2))
                        threshold = np.percentile(errors, 95)
                        anomalies = (errors > threshold).astype(int)
                        
                        # Update performance metrics
                        processing_time = time.time() - start_time
                        samples_per_sec = len(data_chunk) / processing_time if processing_time > 0 else 0
                        self.performance_metrics['throughput_samples_per_sec'] = samples_per_sec
                        
                        yield {
                            'timestamp': time.time(),
                            'chunk_size': len(data_chunk),
                            'anomaly_count': np.sum(anomalies),
                            'processing_time_ms': processing_time * 1000,
                            'throughput_samples_per_sec': samples_per_sec,
                            'anomalies': anomalies.tolist(),
                            'errors': errors.tolist()
                        }
            
            except Exception as e:
                logger.error(f"Error processing stream chunk: {e}")
                yield {
                    'timestamp': time.time(),
                    'error': str(e),
                    'chunk_size': len(data_chunk) if data_chunk is not None else 0
                }
    
    def auto_scale_resources(self, current_load: Dict[str, float]) -> Dict[str, Any]:
        """Automatically scale resources based on current system load."""
        if not self.autoscaler:
            return {'scaling_action': 'disabled'}
        
        cpu_percent = current_load.get('cpu_percent', psutil.cpu_percent())
        memory_percent = current_load.get('memory_percent', psutil.virtual_memory().percent)
        
        scaling_action = self.autoscaler.should_scale(
            cpu_utilization=cpu_percent,
            memory_utilization=memory_percent,
            queue_length=current_load.get('queue_length', 0)
        )
        
        if scaling_action['scale_up']:
            # Scale up resources
            new_workers = min(self.max_workers * 2, cpu_count())
            self.resource_pool.scale_up(new_workers)
            logger.info(f"Scaling up to {new_workers} workers")
            
        elif scaling_action['scale_down']:
            # Scale down resources
            new_workers = max(1, self.max_workers // 2)
            self.resource_pool.scale_down(new_workers)
            logger.info(f"Scaling down to {new_workers} workers")
        
        return scaling_action
    
    def optimize_performance(self, profile_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply performance optimizations based on profiling data."""
        optimization_results = {}
        
        # Memory optimization
        if self.cache:
            cache_stats = self.cache.get_stats()
            if cache_stats['hit_rate'] < 0.5:
                # Increase cache size
                self.cache.resize(self.cache.max_size_gb * 1.5)
                optimization_results['cache_resized'] = True
            
            self.performance_metrics['cache_hit_rate'] = cache_stats['hit_rate']
        
        # Batch size optimization
        if profile_data and 'avg_batch_time' in profile_data:
            avg_batch_time = profile_data['avg_batch_time']
            if avg_batch_time > 1.0:  # Too slow
                self.batch_size = max(1, self.batch_size // 2)
                optimization_results['batch_size_reduced'] = self.batch_size
            elif avg_batch_time < 0.1:  # Too fast, can increase
                self.batch_size = min(512, self.batch_size * 2)
                optimization_results['batch_size_increased'] = self.batch_size
        
        # Memory cleanup
        if psutil.virtual_memory().percent > 80:
            gc.collect()
            optimization_results['memory_cleanup'] = True
        
        # Apply optimizer-specific optimizations
        optimizer_results = self.optimizer.optimize_pipeline(
            current_metrics=self.performance_metrics,
            profile_data=profile_data
        )
        optimization_results.update(optimizer_results)
        
        return optimization_results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        # Update current metrics
        self.performance_metrics['cpu_utilization'] = psutil.cpu_percent(interval=1)
        self.performance_metrics['memory_efficiency'] = (
            psutil.virtual_memory().available / psutil.virtual_memory().total * 100
        )
        
        # Cache performance
        if self.cache:
            cache_stats = self.cache.get_stats()
            self.performance_metrics['cache_hit_rate'] = cache_stats['hit_rate']
        
        # Resource pool stats
        pool_stats = self.resource_pool.get_statistics()
        
        # Performance monitor stats
        monitor_stats = self.performance_monitor.get_current_metrics()
        
        return {
            'pipeline_metrics': self.performance_metrics.copy(),
            'resource_pool': pool_stats,
            'monitor_stats': monitor_stats,
            'system_stats': {
                'cpu_count': cpu_count(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                'memory_percent': psutil.virtual_memory().percent
            },
            'cache_stats': self.cache.get_stats() if self.cache else {},
            'autoscaler_stats': self.autoscaler.get_stats() if self.autoscaler else {}
        }
    
    def benchmark_performance(self, test_data_size: int = 10000, iterations: int = 3) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks."""
        logger.info(f"Starting performance benchmark with {test_data_size} samples, {iterations} iterations")
        
        # Generate test data
        test_data = np.random.randn(test_data_size, 3)  # 3 features
        
        benchmark_results = {
            'data_preparation': [],
            'model_prediction': [],
            'anomaly_detection': [],
            'memory_usage': []
        }
        
        for iteration in range(iterations):
            logger.info(f"Benchmark iteration {iteration + 1}/{iterations}")
            
            # Measure data preparation
            start_time = time.time()
            sequences = self._parallel_window_creation(test_data)
            prep_time = time.time() - start_time
            benchmark_results['data_preparation'].append(prep_time)
            
            # Measure model prediction
            if self.model:
                start_time = time.time()
                predictions = self._batch_predict_optimized(sequences)
                pred_time = time.time() - start_time
                benchmark_results['model_prediction'].append(pred_time)
                
                # Measure anomaly detection
                start_time = time.time()
                errors = np.mean((sequences - predictions) ** 2, axis=(1, 2))
                threshold = np.percentile(errors, 95)
                anomalies = (errors > threshold).astype(int)
                anomaly_time = time.time() - start_time
                benchmark_results['anomaly_detection'].append(anomaly_time)
            
            # Measure memory usage
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            benchmark_results['memory_usage'].append(memory_usage)
            
            # Cleanup between iterations
            gc.collect()
        
        # Calculate statistics
        stats = {}
        for metric, values in benchmark_results.items():
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Calculate throughput
        if benchmark_results['data_preparation']:
            avg_prep_time = np.mean(benchmark_results['data_preparation'])
            throughput = test_data_size / avg_prep_time if avg_prep_time > 0 else 0
            stats['throughput_samples_per_sec'] = throughput
        
        logger.info(f"Benchmark completed. Average throughput: {stats.get('throughput_samples_per_sec', 0):.2f} samples/sec")
        
        return stats


def main():
    """Example usage of the scalable pipeline with benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scalable IoT Anomaly Detection Pipeline")
    parser.add_argument("--data-path", required=True, help="Path to CSV data file")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")
    parser.add_argument("--workers", type=int, help="Number of worker threads")
    parser.add_argument("--optimization-level", choices=['conservative', 'balanced', 'aggressive'],
                       default='balanced', help="Performance optimization level")
    
    args = parser.parse_args()
    
    # Initialize scalable pipeline
    pipeline = ScalablePipeline(
        enable_parallel_processing=True,
        enable_caching=True,
        enable_streaming=args.streaming,
        max_workers=args.workers,
        performance_optimization_level=args.optimization_level
    )
    
    try:
        if args.benchmark:
            # Run benchmarks
            benchmark_results = pipeline.benchmark_performance()
            
            print("\n=== PERFORMANCE BENCHMARK RESULTS ===")
            for metric, stats in benchmark_results.items():
                if isinstance(stats, dict):
                    print(f"{metric}:")
                    print(f"  Mean: {stats['mean']:.4f}")
                    print(f"  Std:  {stats['std']:.4f}")
                else:
                    print(f"{metric}: {stats:.4f}")
        
        # Load and process data
        df = pipeline.load_data(args.data_path)
        X, y = pipeline.prepare_data(df)
        
        # Auto-optimize
        optimization_results = pipeline.optimize_performance()
        logger.info(f"Applied optimizations: {optimization_results}")
        
        # Generate performance report
        performance_report = pipeline.get_performance_report()
        
        print(f"\n=== PERFORMANCE REPORT ===")
        print(f"CPU Utilization: {performance_report['system_stats']['cpu_percent']:.1f}%")
        print(f"Memory Usage: {performance_report['system_stats']['memory_percent']:.1f}%")
        print(f"Cache Hit Rate: {performance_report['pipeline_metrics']['cache_hit_rate']:.2f}")
        print(f"Parallel Speedup: {performance_report['pipeline_metrics']['parallel_speedup']:.1f}x")
        
    except Exception as e:
        logger.error(f"Scalable pipeline execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())