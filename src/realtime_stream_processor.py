"""Real-time streaming processor for high-frequency IoT sensor data."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any, Union
from pathlib import Path
from collections import deque
import threading
import queue

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


@dataclass
class StreamConfig:
    """Configuration for real-time stream processing."""
    window_size: int = 30
    step_size: int = 1
    buffer_size: int = 10000
    processing_frequency_hz: int = 10
    max_latency_ms: int = 50
    enable_compression: bool = True
    batch_processing: bool = True
    websocket_port: int = 8765
    mqtt_broker: Optional[str] = None
    kafka_broker: Optional[str] = None


@dataclass 
class StreamData:
    """Real-time streaming data point."""
    timestamp: float
    sensor_id: str
    values: List[float]
    metadata: Dict[str, Any]


@dataclass
class ProcessingResult:
    """Result from real-time processing."""
    timestamp: float
    sensor_id: str
    is_anomaly: bool
    confidence: float
    processing_latency_ms: float
    window_data: List[List[float]]


class RealtimeStreamProcessor:
    """High-performance real-time stream processor for IoT anomaly detection."""
    
    def __init__(self, config: Optional[StreamConfig] = None, anomaly_detector=None):
        """Initialize real-time stream processor."""
        self.config = config or StreamConfig()
        self.detector = anomaly_detector
        self.logger = logging.getLogger(__name__)
        
        # Data structures for streaming
        self.sensor_buffers: Dict[str, deque] = {}
        self.processing_queue = queue.Queue(maxsize=self.config.buffer_size)
        self.result_callbacks: List[Callable[[ProcessingResult], None]] = []
        
        # Threading and async components
        self.processing_thread: Optional[threading.Thread] = None
        self.websocket_server = None
        self.is_running = False
        
        # Performance tracking
        self.processing_stats = {
            "total_processed": 0,
            "total_anomalies": 0,
            "avg_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "dropped_messages": 0,
            "processing_rate_hz": 0.0
        }
        
        self._last_stats_update = time.time()
        
    def add_result_callback(self, callback: Callable[[ProcessingResult], None]) -> None:
        """Add callback for processing results."""
        self.result_callbacks.append(callback)
        
    def start_processing(self) -> None:
        """Start real-time stream processing."""
        if self.is_running:
            self.logger.warning("Stream processor already running")
            return
            
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        # Start WebSocket server if enabled
        if WEBSOCKETS_AVAILABLE:
            asyncio.create_task(self._start_websocket_server())
        
        self.logger.info("Started real-time stream processing")
        
    async def _start_websocket_server(self) -> None:
        """Start WebSocket server for real-time data ingestion."""
        try:
            self.websocket_server = await websockets.serve(
                self._websocket_handler,
                "localhost",
                self.config.websocket_port
            )
            self.logger.info(f"WebSocket server started on port {self.config.websocket_port}")
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
    
    async def _websocket_handler(self, websocket, path) -> None:
        """Handle WebSocket connections and data."""
        self.logger.info(f"New WebSocket connection from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    stream_data = StreamData(
                        timestamp=data.get("timestamp", time.time()),
                        sensor_id=data.get("sensor_id", "unknown"),
                        values=data.get("values", []),
                        metadata=data.get("metadata", {})
                    )
                    self.ingest_data(stream_data)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON received: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"WebSocket connection closed: {websocket.remote_address}")
        except Exception as e:
            self.logger.error(f"WebSocket handler error: {e}")
    
    def stop_processing(self) -> None:
        """Stop real-time stream processing."""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            
        if self.websocket_server:
            self.websocket_server.close()
            
        self.logger.info("Stopped real-time stream processing")
    
    def ingest_data(self, stream_data: StreamData) -> bool:
        """Ingest real-time streaming data."""
        try:
            # Add to sensor-specific buffer
            sensor_id = stream_data.sensor_id
            if sensor_id not in self.sensor_buffers:
                self.sensor_buffers[sensor_id] = deque(maxsize=self.config.window_size * 2)
            
            self.sensor_buffers[sensor_id].append(stream_data)
            
            # Check if we have enough data for processing
            if len(self.sensor_buffers[sensor_id]) >= self.config.window_size:
                self._queue_for_processing(sensor_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error ingesting data: {e}")
            return False
    
    def _queue_for_processing(self, sensor_id: str) -> None:
        """Queue sensor data for processing."""
        try:
            # Extract window of data
            buffer = self.sensor_buffers[sensor_id]
            window_data = list(buffer)[-self.config.window_size:]
            
            processing_item = {
                "sensor_id": sensor_id,
                "window_data": window_data,
                "queued_time": time.time()
            }
            
            self.processing_queue.put_nowait(processing_item)
            
        except queue.Full:
            self.processing_stats["dropped_messages"] += 1
            self.logger.warning(f"Processing queue full, dropping data for sensor {sensor_id}")
        except Exception as e:
            self.logger.error(f"Error queuing data for processing: {e}")
    
    def _processing_loop(self) -> None:
        """Main processing loop running in background thread."""
        self.logger.info("Started processing loop")
        
        while self.is_running:
            try:
                # Get next item to process
                try:
                    item = self.processing_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the data
                result = self._process_sensor_window(item)
                if result:
                    self._handle_processing_result(result)
                    
                # Update statistics
                self._update_processing_stats()
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def _process_sensor_window(self, item: Dict[str, Any]) -> Optional[ProcessingResult]:
        """Process a window of sensor data."""
        start_time = time.time()
        
        try:
            sensor_id = item["sensor_id"]
            window_data = item["window_data"]
            queued_time = item["queued_time"]
            
            # Calculate queue latency
            queue_latency = (start_time - queued_time) * 1000
            
            # Prepare data for anomaly detection
            if not NUMPY_AVAILABLE or not self.detector:
                # Simple threshold-based detection if ML not available
                is_anomaly = self._simple_anomaly_detection(window_data)
                confidence = 0.8 if is_anomaly else 0.2
            else:
                # Use ML-based detection
                sensor_values = np.array([[point.values for point in window_data]])
                scores = self.detector.score(sensor_values)
                
                # Simple threshold
                threshold = np.mean(scores) + 2 * np.std(scores) if len(scores) > 1 else 0.5
                is_anomaly = scores[0] > threshold
                confidence = min(scores[0] / threshold, 2.0) if threshold > 0 else 0.0
            
            processing_time = time.time() - start_time
            total_latency = queue_latency + (processing_time * 1000)
            
            # Check latency requirement
            if total_latency > self.config.max_latency_ms:
                self.logger.warning(f"Processing latency {total_latency:.1f}ms exceeds limit {self.config.max_latency_ms}ms")
            
            result = ProcessingResult(
                timestamp=time.time(),
                sensor_id=sensor_id,
                is_anomaly=is_anomaly,
                confidence=float(confidence),
                processing_latency_ms=total_latency,
                window_data=[[point.values for point in window_data]]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing sensor window: {e}")
            return None
    
    def _simple_anomaly_detection(self, window_data: List[StreamData]) -> bool:
        """Simple anomaly detection without ML dependencies."""
        if len(window_data) < 3:
            return False
        
        try:
            # Calculate moving average and detect significant deviations
            recent_values = [point.values[0] if point.values else 0.0 for point in window_data[-5:]]
            historical_values = [point.values[0] if point.values else 0.0 for point in window_data[:-5]]
            
            if not recent_values or not historical_values:
                return False
            
            recent_avg = sum(recent_values) / len(recent_values)
            historical_avg = sum(historical_values) / len(historical_values)
            
            # Calculate standard deviation
            hist_variance = sum((x - historical_avg) ** 2 for x in historical_values) / len(historical_values)
            hist_std = hist_variance ** 0.5
            
            # Anomaly if recent average deviates by more than 2 standard deviations
            deviation = abs(recent_avg - historical_avg)
            return deviation > (2.0 * hist_std) if hist_std > 0 else False
            
        except Exception:
            return False
    
    def _handle_processing_result(self, result: ProcessingResult) -> None:
        """Handle processing result and trigger callbacks."""
        # Update statistics
        self.processing_stats["total_processed"] += 1
        if result.is_anomaly:
            self.processing_stats["total_anomalies"] += 1
        
        # Update latency stats
        current_latency = result.processing_latency_ms
        self.processing_stats["max_latency_ms"] = max(
            self.processing_stats["max_latency_ms"],
            current_latency
        )
        
        # Trigger callbacks
        for callback in self.result_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"Error in result callback: {e}")
    
    def _update_processing_stats(self) -> None:
        """Update processing rate statistics."""
        current_time = time.time()
        time_diff = current_time - self._last_stats_update
        
        if time_diff >= 1.0:  # Update every second
            processed_count = self.processing_stats["total_processed"]
            self.processing_stats["processing_rate_hz"] = processed_count / time_diff if time_diff > 0 else 0.0
            self._last_stats_update = current_time
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = self.processing_stats.copy()
        stats["queue_size"] = self.processing_queue.qsize()
        stats["active_sensors"] = len(self.sensor_buffers)
        stats["buffer_utilization"] = {
            sensor_id: len(buffer) for sensor_id, buffer in self.sensor_buffers.items()
        }
        return stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            "total_processed": 0,
            "total_anomalies": 0,
            "avg_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "dropped_messages": 0,
            "processing_rate_hz": 0.0
        }
        self._last_stats_update = time.time()
    
    def simulate_sensor_data(self, sensor_id: str, duration_seconds: int = 60, frequency_hz: int = 10) -> None:
        """Simulate sensor data for testing."""
        import random
        import math
        
        self.logger.info(f"Starting simulation for sensor {sensor_id} at {frequency_hz} Hz for {duration_seconds}s")
        
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < duration_seconds and self.is_running:
            # Generate realistic sensor data with patterns
            t = sample_count / frequency_hz
            
            # Base signal with noise and occasional anomalies
            base_value = math.sin(t * 0.5) + random.gauss(0, 0.1)
            
            # Inject anomalies occasionally
            if random.random() < 0.05:  # 5% anomaly rate
                base_value += random.gauss(0, 2.0)  # Significant deviation
            
            values = [
                base_value,
                base_value * 0.8 + random.gauss(0, 0.05),  # Correlated sensor
                random.gauss(1.0, 0.2)  # Independent sensor
            ]
            
            stream_data = StreamData(
                timestamp=time.time(),
                sensor_id=sensor_id,
                values=values,
                metadata={"simulation": True, "sample": sample_count}
            )
            
            self.ingest_data(stream_data)
            
            sample_count += 1
            sleep_time = 1.0 / frequency_hz
            time.sleep(sleep_time)
        
        self.logger.info(f"Simulation complete for sensor {sensor_id}. Generated {sample_count} samples.")


# Factory functions
def create_realtime_processor(anomaly_detector=None, **config_kwargs) -> RealtimeStreamProcessor:
    """Create real-time stream processor with configuration."""
    config = StreamConfig(**config_kwargs)
    return RealtimeStreamProcessor(config, anomaly_detector)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Stream Processor")
    parser.add_argument("--window-size", type=int, default=30, help="Processing window size")
    parser.add_argument("--buffer-size", type=int, default=10000, help="Buffer size")
    parser.add_argument("--websocket-port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--max-latency", type=int, default=50, help="Max latency (ms)")
    parser.add_argument("--duration", type=int, default=60, help="Run duration (seconds)")
    parser.add_argument("--simulate", action="store_true", help="Run with simulated data")
    
    args = parser.parse_args()
    
    config = StreamConfig(
        window_size=args.window_size,
        buffer_size=args.buffer_size,
        websocket_port=args.websocket_port,
        max_latency_ms=args.max_latency
    )
    
    processor = RealtimeStreamProcessor(config)
    
    # Add result callback for logging
    def log_result(result: ProcessingResult) -> None:
        if result.is_anomaly:
            print(f"ANOMALY detected for sensor {result.sensor_id}: confidence={result.confidence:.2f}, latency={result.processing_latency_ms:.1f}ms")
    
    processor.add_result_callback(log_result)
    
    print(f"Starting real-time processor for {args.duration} seconds...")
    processor.start_processing()
    
    if args.simulate:
        # Start simulation in background
        import threading
        sim_thread = threading.Thread(
            target=processor.simulate_sensor_data,
            args=("sensor_001", args.duration, 10),
            daemon=True
        )
        sim_thread.start()
    
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    processor.stop_processing()
    
    # Print final statistics
    stats = processor.get_processing_stats()
    print(f"\nFinal Statistics:")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Total anomalies: {stats['total_anomalies']}")
    print(f"Max latency: {stats['max_latency_ms']:.1f}ms")
    print(f"Processing rate: {stats['processing_rate_hz']:.1f} Hz")
    print(f"Dropped messages: {stats['dropped_messages']}")