"""Adaptive Real-Time Stream Processor for IoT Anomaly Detection.

This module provides advanced real-time processing capabilities with adaptive
algorithms, dynamic optimization, and intelligent resource management.
"""

import asyncio
import json
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from .adaptive_cache import AdaptiveCache
from .anomaly_detector import AnomalyDetector
from .circuit_breaker import CircuitBreaker
from .logging_config import get_logger


@dataclass
class ProcessingMetrics:
    """Metrics for real-time processing performance."""
    latency_ms: float
    throughput_ops_sec: float
    memory_usage_mb: float
    cpu_utilization: float
    queue_depth: int
    error_rate: float
    cache_hit_rate: float


@dataclass
class StreamConfig:
    """Configuration for stream processing."""
    batch_size: int = 32
    buffer_size: int = 1000
    max_latency_ms: float = 100.0
    auto_scale: bool = True
    enable_compression: bool = True
    enable_monitoring: bool = True


class AdaptiveRealTimeProcessor:
    """High-performance adaptive real-time processor for IoT streams."""

    def __init__(
        self,
        model_path: str,
        scaler_path: Optional[str] = None,
        config: Optional[StreamConfig] = None,
        max_workers: int = 4
    ):
        """Initialize the adaptive real-time processor.
        
        Args:
            model_path: Path to trained autoencoder model
            scaler_path: Path to fitted scaler
            config: Stream processing configuration
            max_workers: Maximum number of worker threads
        """
        self.logger = get_logger(__name__)
        self.config = config or StreamConfig()
        self.max_workers = max_workers

        # Initialize components
        self.detector = AnomalyDetector(model_path, scaler_path)
        self.cache = AdaptiveCache(max_size=10000)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout_duration=30.0
        )

        # Processing queues and locks
        self.input_queue = queue.Queue(maxsize=self.config.buffer_size)
        self.result_queue = queue.Queue()
        self.metrics_lock = Lock()

        # Performance tracking
        self.metrics = ProcessingMetrics(
            latency_ms=0.0,
            throughput_ops_sec=0.0,
            memory_usage_mb=0.0,
            cpu_utilization=0.0,
            queue_depth=0,
            error_rate=0.0,
            cache_hit_rate=0.0
        )

        # Adaptive parameters
        self.adaptive_batch_size = self.config.batch_size
        self.latency_history = []
        self.throughput_history = []

        # Worker pool
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = False

        self.logger.info(f"Initialized AdaptiveRealTimeProcessor with {max_workers} workers")

    async def start(self) -> None:
        """Start the real-time processing pipeline."""
        self.is_running = True
        self.logger.info("Starting adaptive real-time processor")

        # Start background tasks
        tasks = [
            asyncio.create_task(self._processing_loop()),
            asyncio.create_task(self._adaptive_tuning_loop()),
            asyncio.create_task(self._metrics_collection_loop())
        ]

        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        """Stop the real-time processing pipeline."""
        self.is_running = False
        self.executor.shutdown(wait=True)
        self.logger.info("Stopped adaptive real-time processor")

    def process_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data sample.
        
        Args:
            data: Input data sample
            
        Returns:
            Processing result with anomaly score and metadata
        """
        start_time = time.time()

        try:
            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                return {
                    "anomaly_score": 0.0,
                    "is_anomaly": False,
                    "error": "Circuit breaker open",
                    "timestamp": time.time()
                }

            # Cache check
            cache_key = self._generate_cache_key(data)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self._update_cache_metrics(hit=True)
                return cached_result

            # Convert to DataFrame
            df = pd.DataFrame([data])

            # Preprocess and predict
            sequences = self.detector.preprocessor.create_sequences(df)
            if len(sequences) > 0:
                scores = self.detector.score(sequences)
                anomaly_score = float(np.mean(scores))
                is_anomaly = anomaly_score > 0.5  # Default threshold
            else:
                anomaly_score = 0.0
                is_anomaly = False

            result = {
                "anomaly_score": anomaly_score,
                "is_anomaly": is_anomaly,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time()
            }

            # Cache result
            self.cache.put(cache_key, result)
            self._update_cache_metrics(hit=False)

            # Update circuit breaker on success
            self.circuit_breaker.record_success()

            return result

        except Exception as e:
            self.circuit_breaker.record_failure()
            self.logger.error(f"Processing error: {e}")
            return {
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of data samples efficiently.
        
        Args:
            batch_data: List of input data samples
            
        Returns:
            List of processing results
        """
        start_time = time.time()
        results = []

        try:
            # Convert batch to DataFrame
            df = pd.DataFrame(batch_data)

            # Batch preprocessing
            sequences = self.detector.preprocessor.create_sequences(df)

            if len(sequences) > 0:
                # Batch prediction
                scores = self.detector.score(sequences)

                # Process results
                for i, (sample, score) in enumerate(zip(batch_data, scores)):
                    result = {
                        "anomaly_score": float(score),
                        "is_anomaly": score > 0.5,
                        "batch_index": i,
                        "timestamp": time.time()
                    }
                    results.append(result)
            else:
                # No valid sequences
                for i, sample in enumerate(batch_data):
                    results.append({
                        "anomaly_score": 0.0,
                        "is_anomaly": False,
                        "batch_index": i,
                        "timestamp": time.time()
                    })

            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_processing_metrics(len(batch_data), processing_time)

            self.logger.debug(f"Processed batch of {len(batch_data)} samples in {processing_time:.2f}ms")

        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            # Return error results for all samples
            for i in range(len(batch_data)):
                results.append({
                    "anomaly_score": 0.0,
                    "is_anomaly": False,
                    "error": str(e),
                    "batch_index": i,
                    "timestamp": time.time()
                })

        return results

    async def _processing_loop(self) -> None:
        """Main processing loop for handling queued data."""
        batch_buffer = []
        last_process_time = time.time()

        while self.is_running:
            try:
                # Collect samples for batch processing
                while (len(batch_buffer) < self.adaptive_batch_size and
                       time.time() - last_process_time < 0.1):  # 100ms timeout
                    try:
                        sample = self.input_queue.get(timeout=0.01)
                        batch_buffer.append(sample)
                    except queue.Empty:
                        break

                # Process batch if we have samples
                if batch_buffer:
                    results = await asyncio.get_event_loop().run_in_executor(
                        self.executor, self.process_batch, batch_buffer.copy()
                    )

                    # Enqueue results
                    for result in results:
                        try:
                            self.result_queue.put_nowait(result)
                        except queue.Full:
                            self.logger.warning("Result queue full, dropping result")

                    batch_buffer.clear()
                    last_process_time = time.time()

                await asyncio.sleep(0.001)  # Small delay to prevent busy waiting

            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(0.1)

    async def _adaptive_tuning_loop(self) -> None:
        """Adaptive tuning loop for optimizing performance parameters."""
        while self.is_running:
            try:
                await asyncio.sleep(5.0)  # Tune every 5 seconds

                # Analyze recent performance
                if len(self.latency_history) >= 10:
                    avg_latency = np.mean(self.latency_history[-10:])
                    avg_throughput = np.mean(self.throughput_history[-10:])

                    # Adaptive batch size tuning
                    if avg_latency > self.config.max_latency_ms:
                        # Latency too high, reduce batch size
                        self.adaptive_batch_size = max(1, self.adaptive_batch_size - 1)
                        self.logger.debug(f"Reduced batch size to {self.adaptive_batch_size}")
                    elif avg_latency < self.config.max_latency_ms * 0.5 and avg_throughput > 0:
                        # Latency low, can increase batch size
                        self.adaptive_batch_size = min(128, self.adaptive_batch_size + 1)
                        self.logger.debug(f"Increased batch size to {self.adaptive_batch_size}")

                # Trim history to prevent memory growth
                if len(self.latency_history) > 100:
                    self.latency_history = self.latency_history[-50:]
                    self.throughput_history = self.throughput_history[-50:]

            except Exception as e:
                self.logger.error(f"Adaptive tuning error: {e}")

    async def _metrics_collection_loop(self) -> None:
        """Metrics collection loop for monitoring performance."""
        while self.is_running:
            try:
                await asyncio.sleep(1.0)  # Collect every second

                with self.metrics_lock:
                    # Update queue depth
                    self.metrics.queue_depth = self.input_queue.qsize()

                    # Update cache hit rate
                    cache_stats = self.cache.get_stats()
                    if cache_stats['total_requests'] > 0:
                        self.metrics.cache_hit_rate = (
                            cache_stats['hits'] / cache_stats['total_requests']
                        )

                self.logger.debug(f"Current metrics: {self.get_metrics()}")

            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")

    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key for data sample."""
        # Simple hash of sorted key-value pairs
        sorted_items = sorted(data.items())
        return str(hash(str(sorted_items)))

    def _update_cache_metrics(self, hit: bool) -> None:
        """Update cache-related metrics."""
        # This is handled by the cache internally
        pass

    def _update_processing_metrics(self, batch_size: int, processing_time_ms: float) -> None:
        """Update processing performance metrics."""
        with self.metrics_lock:
            self.metrics.latency_ms = processing_time_ms / batch_size
            self.metrics.throughput_ops_sec = batch_size / (processing_time_ms / 1000)

            # Add to history for adaptive tuning
            self.latency_history.append(self.metrics.latency_ms)
            self.throughput_history.append(self.metrics.throughput_ops_sec)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics."""
        with self.metrics_lock:
            return {
                "latency_ms": self.metrics.latency_ms,
                "throughput_ops_sec": self.metrics.throughput_ops_sec,
                "queue_depth": self.metrics.queue_depth,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "adaptive_batch_size": self.adaptive_batch_size,
                "circuit_breaker_state": self.circuit_breaker.state.value,
                "cache_stats": self.cache.get_stats()
            }

    def submit_data(self, data: Dict[str, Any]) -> bool:
        """Submit data for processing.
        
        Args:
            data: Input data sample
            
        Returns:
            True if successfully queued, False if queue is full
        """
        try:
            self.input_queue.put_nowait(data)
            return True
        except queue.Full:
            self.logger.warning("Input queue full, dropping sample")
            return False

    def get_result(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get processing result.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Processing result or None if timeout
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# CLI Interface for real-time processing
def main() -> None:
    """CLI entry point for adaptive real-time processor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Adaptive Real-Time IoT Anomaly Detection Processor"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained autoencoder model"
    )
    parser.add_argument(
        "--scaler-path",
        help="Path to fitted scaler"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Initial batch size for processing"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of worker threads"
    )
    parser.add_argument(
        "--input-file",
        help="Input CSV file for batch processing"
    )
    parser.add_argument(
        "--output-file",
        default="real_time_results.json",
        help="Output file for results"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = get_logger(__name__)

    # Create processor
    config = StreamConfig(batch_size=args.batch_size)
    processor = AdaptiveRealTimeProcessor(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        config=config,
        max_workers=args.max_workers
    )

    async def run_processor():
        if args.input_file:
            # Batch processing mode
            logger.info(f"Processing file: {args.input_file}")
            df = pd.read_csv(args.input_file)

            results = []
            for _, row in df.iterrows():
                data = row.to_dict()
                result = processor.process_sample(data)
                results.append(result)

            # Save results
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"Results saved to {args.output_file}")
            logger.info(f"Final metrics: {processor.get_metrics()}")
        else:
            # Interactive mode
            logger.info("Starting interactive mode. Enter 'quit' to exit.")
            logger.info("Submit JSON data samples for real-time processing.")

            await processor.start()

    asyncio.run(run_processor())


if __name__ == "__main__":
    main()
