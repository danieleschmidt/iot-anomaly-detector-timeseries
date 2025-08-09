"""Edge computing anomaly detector for resource-constrained IoT devices."""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue
import os

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    
try:
    from tensorflow.keras.models import load_model
    import tensorflow.lite as tflite
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


@dataclass
class EdgeConfig:
    """Configuration for edge anomaly detection."""
    model_path: str = "model_lite.tflite"
    buffer_size: int = 1000
    batch_size: int = 32
    max_memory_mb: int = 128
    inference_timeout_ms: int = 100
    enable_compression: bool = True
    quantization_bits: int = 8
    cache_size: int = 100


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    timestamp: float
    is_anomaly: bool
    confidence: float
    reconstruction_error: float
    sensor_values: List[float]
    processing_time_ms: float


class EdgeAnomalyDetector:
    """Lightweight anomaly detector optimized for edge devices."""
    
    def __init__(self, config: Optional[EdgeConfig] = None):
        """Initialize edge anomaly detector."""
        self.config = config or EdgeConfig()
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        self.data_buffer = queue.Queue(maxsize=self.config.buffer_size)
        self.result_cache = {}
        self.processing_thread = None
        self.is_running = False
        
        self._load_model()
        self._initialize_threading()
    
    def _load_model(self) -> None:
        """Load TensorFlow Lite model for edge inference."""
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow not available for edge inference")
            
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            self.interpreter = tflite.Interpreter(
                model_path=str(model_path),
                num_threads=1  # Single thread for edge devices
            )
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.logger.info(f"Loaded TFLite model: {model_path}")
            self.logger.info(f"Input shape: {self.input_details[0]['shape']}")
            self.logger.info(f"Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            self.logger.error(f"Failed to load TFLite model: {e}")
            raise
    
    def _initialize_threading(self) -> None:
        """Initialize background processing thread."""
        self.processing_thread = threading.Thread(
            target=self._process_data_continuously,
            daemon=True
        )
    
    def start_continuous_monitoring(self) -> None:
        """Start continuous anomaly detection."""
        if self.is_running:
            self.logger.warning("Already running continuous monitoring")
            return
            
        self.is_running = True
        self.processing_thread.start()
        self.logger.info("Started continuous anomaly monitoring")
    
    def stop_continuous_monitoring(self) -> None:
        """Stop continuous anomaly detection."""
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        self.logger.info("Stopped continuous anomaly monitoring")
    
    def _process_data_continuously(self) -> None:
        """Background processing loop."""
        while self.is_running:
            try:
                if not self.data_buffer.empty():
                    batch_data = self._collect_batch()
                    if batch_data:
                        results = self._process_batch(batch_data)
                        self._store_results(results)
                else:
                    time.sleep(0.01)  # Small delay to prevent CPU spinning
                    
            except Exception as e:
                self.logger.error(f"Error in continuous processing: {e}")
                time.sleep(0.1)  # Longer delay on error
    
    def _collect_batch(self) -> List[Dict[str, Any]]:
        """Collect batch of data from buffer."""
        batch = []
        batch_size = min(self.config.batch_size, self.data_buffer.qsize())
        
        for _ in range(batch_size):
            try:
                data = self.data_buffer.get_nowait()
                batch.append(data)
            except queue.Empty:
                break
        
        return batch
    
    def _process_batch(self, batch_data: List[Dict[str, Any]]) -> List[AnomalyResult]:
        """Process batch of sensor data."""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy not available for processing")
            
        results = []
        start_time = time.time()
        
        for data in batch_data:
            try:
                sensor_values = data["values"]
                timestamp = data.get("timestamp", time.time())
                
                # Prepare input for TFLite model
                input_data = np.array(sensor_values, dtype=np.float32)
                input_data = np.expand_dims(input_data, axis=0)
                
                # Run inference
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                
                # Get output
                reconstruction = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                # Calculate reconstruction error
                error = np.mean(np.square(input_data - reconstruction))
                
                # Simple threshold-based anomaly detection
                threshold = self._get_adaptive_threshold()
                is_anomaly = error > threshold
                confidence = min(error / threshold, 2.0) if threshold > 0 else 0.0
                
                result = AnomalyResult(
                    timestamp=timestamp,
                    is_anomaly=is_anomaly,
                    confidence=confidence,
                    reconstruction_error=float(error),
                    sensor_values=sensor_values,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing data point: {e}")
        
        return results
    
    def _get_adaptive_threshold(self) -> float:
        """Calculate adaptive threshold based on recent results."""
        if len(self.result_cache) < 10:
            return 0.1  # Default threshold
        
        recent_errors = [r.reconstruction_error for r in list(self.result_cache.values())[-100:]]
        if not recent_errors:
            return 0.1
            
        if NUMPY_AVAILABLE:
            mean_error = np.mean(recent_errors)
            std_error = np.std(recent_errors)
            return mean_error + 2.0 * std_error
        else:
            mean_error = sum(recent_errors) / len(recent_errors)
            return mean_error * 1.5
    
    def _store_results(self, results: List[AnomalyResult]) -> None:
        """Store results in cache with size management."""
        for result in results:
            key = str(result.timestamp)
            self.result_cache[key] = result
            
            # Maintain cache size limit
            if len(self.result_cache) > self.config.cache_size:
                oldest_key = min(self.result_cache.keys())
                del self.result_cache[oldest_key]
    
    def add_sensor_data(self, sensor_values: List[float], timestamp: Optional[float] = None) -> bool:
        """Add sensor data for processing."""
        if timestamp is None:
            timestamp = time.time()
        
        data = {
            "values": sensor_values,
            "timestamp": timestamp
        }
        
        try:
            self.data_buffer.put_nowait(data)
            return True
        except queue.Full:
            self.logger.warning("Data buffer full, dropping oldest data")
            try:
                self.data_buffer.get_nowait()  # Remove oldest
                self.data_buffer.put_nowait(data)  # Add new
                return True
            except queue.Empty:
                return False
    
    def get_recent_results(self, count: int = 10) -> List[AnomalyResult]:
        """Get recent anomaly detection results."""
        all_results = list(self.result_cache.values())
        return sorted(all_results, key=lambda x: x.timestamp)[-count:]
    
    def get_anomaly_count(self, time_window_seconds: int = 3600) -> int:
        """Count anomalies in recent time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window_seconds
        
        count = 0
        for result in self.result_cache.values():
            if result.timestamp >= cutoff_time and result.is_anomaly:
                count += 1
        
        return count
    
    def export_results(self, output_path: str) -> None:
        """Export results to JSON file."""
        results_data = [asdict(result) for result in self.result_cache.values()]
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Exported {len(results_data)} results to {output_path}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        return {
            "buffer_size": self.data_buffer.qsize(),
            "cache_size": len(self.result_cache),
            "is_running": self.is_running,
            "memory_usage_mb": self._get_memory_usage(),
            "recent_anomaly_count": self.get_anomaly_count(3600),
            "processing_thread_alive": self.processing_thread.is_alive() if self.processing_thread else False
        }
    
    def _get_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0  # Unknown if psutil not available
    
    def optimize_for_edge(self) -> None:
        """Apply edge-specific optimizations."""
        # Reduce buffer size if memory constrained
        if self._get_memory_usage() > self.config.max_memory_mb * 0.8:
            new_buffer_size = max(self.config.buffer_size // 2, 10)
            self.logger.info(f"Reducing buffer size to {new_buffer_size} due to memory constraints")
            
            # Create new buffer with reduced size
            old_buffer = self.data_buffer
            self.data_buffer = queue.Queue(maxsize=new_buffer_size)
            
            # Transfer existing data
            transfer_count = 0
            while not old_buffer.empty() and transfer_count < new_buffer_size:
                try:
                    data = old_buffer.get_nowait()
                    self.data_buffer.put_nowait(data)
                    transfer_count += 1
                except (queue.Empty, queue.Full):
                    break
        
        # Trim cache if needed
        if len(self.result_cache) > self.config.cache_size // 2:
            sorted_keys = sorted(self.result_cache.keys())
            keep_count = self.config.cache_size // 2
            keys_to_remove = sorted_keys[:-keep_count]
            
            for key in keys_to_remove:
                del self.result_cache[key]
            
            self.logger.info(f"Trimmed cache to {len(self.result_cache)} entries")


def create_edge_detector(model_path: str, **kwargs) -> EdgeAnomalyDetector:
    """Factory function to create edge anomaly detector."""
    config = EdgeConfig(model_path=model_path, **kwargs)
    return EdgeAnomalyDetector(config)


# CLI interface for edge deployment
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Edge Anomaly Detector")
    parser.add_argument("--model-path", required=True, help="Path to TFLite model")
    parser.add_argument("--buffer-size", type=int, default=1000, help="Data buffer size")
    parser.add_argument("--batch-size", type=int, default=32, help="Processing batch size")
    parser.add_argument("--max-memory", type=int, default=128, help="Max memory usage (MB)")
    parser.add_argument("--export-path", help="Export results to JSON file")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration (seconds)")
    
    args = parser.parse_args()
    
    config = EdgeConfig(
        model_path=args.model_path,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        max_memory_mb=args.max_memory
    )
    
    detector = EdgeAnomalyDetector(config)
    
    print(f"Starting edge anomaly detector for {args.duration} seconds...")
    detector.start_continuous_monitoring()
    
    # Simulate sensor data
    import random
    for i in range(args.duration * 10):  # 10 Hz data rate
        sensor_values = [random.gauss(0, 1) for _ in range(3)]  # 3 sensors
        detector.add_sensor_data(sensor_values)
        time.sleep(0.1)
    
    detector.stop_continuous_monitoring()
    
    stats = detector.get_system_stats()
    print(f"Processing complete. Stats: {stats}")
    
    if args.export_path:
        detector.export_results(args.export_path)
        print(f"Results exported to {args.export_path}")