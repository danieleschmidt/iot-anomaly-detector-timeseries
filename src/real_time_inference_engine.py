"""
Real-time Inference Engine for Generation 1 Core System
High-throughput, low-latency anomaly detection for IoT streams
"""

import asyncio
import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import AsyncGenerator, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, validator

from .generation_1_autonomous_core import AutonomousAnomalyCore, AnomalyResult
from .logging_config import setup_logging


class SensorReading(BaseModel):
    """Structured sensor reading with validation."""
    timestamp: float
    sensor_id: str
    values: Dict[str, float]
    metadata: Optional[Dict] = {}
    
    @validator('timestamp')
    def timestamp_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Timestamp must be positive')
        return v
    
    @validator('values')
    def values_must_be_numeric(cls, v):
        for key, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f'Value for {key} must be numeric')
        return v


class InferenceMetrics(BaseModel):
    """Real-time inference performance metrics."""
    total_processed: int = 0
    anomalies_detected: int = 0
    avg_latency_ms: float = 0.0
    throughput_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    queue_size: int = 0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0


class RealTimeInferenceEngine:
    """
    High-performance real-time inference engine for IoT anomaly detection.
    
    Features:
    - Asynchronous stream processing
    - Adaptive batching for throughput optimization
    - Circuit breaker for reliability
    - Real-time metrics and monitoring
    - Backpressure handling
    - Auto-scaling thread pool
    """
    
    def __init__(
        self,
        core_model: AutonomousAnomalyCore,
        batch_size: int = 32,
        max_queue_size: int = 10000,
        processing_timeout: float = 5.0,
        metrics_interval: float = 10.0,
        max_workers: int = 4
    ):
        self.core_model = core_model
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.processing_timeout = processing_timeout
        self.metrics_interval = metrics_interval
        self.max_workers = max_workers
        
        # Processing queues and buffers
        self.input_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.result_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.batch_buffer: List[SensorReading] = []
        
        # Performance tracking
        self.metrics = InferenceMetrics()
        self.processing_times: deque = deque(maxlen=1000)
        self.error_count = 0
        self.start_time = time.time()
        
        # Control flags
        self.is_running = False
        self.circuit_breaker_open = False
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        
        # Threading and async
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_tasks: List[asyncio.Task] = []
        
        self.logger = setup_logging(__name__)
        self.logger.info(f"RealTimeInferenceEngine initialized with batch_size={batch_size}")
    
    async def start(self) -> None:
        """Start the real-time inference engine."""
        if self.is_running:
            self.logger.warning("Inference engine already running")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start processing tasks
        self.processing_tasks = [
            asyncio.create_task(self._batch_processor()),
            asyncio.create_task(self._metrics_updater()),
            asyncio.create_task(self._circuit_breaker_monitor())
        ]
        
        self.logger.info("Real-time inference engine started")
    
    async def stop(self) -> None:
        """Stop the inference engine gracefully."""
        self.is_running = False
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Real-time inference engine stopped")
    
    async def submit_reading(self, reading: Union[SensorReading, Dict]) -> bool:
        """Submit sensor reading for processing."""
        if not self.is_running:
            raise RuntimeError("Inference engine not running")
        
        if self.circuit_breaker_open:
            self.logger.warning("Circuit breaker open, rejecting reading")
            return False
        
        try:
            # Convert dict to SensorReading if needed
            if isinstance(reading, dict):
                reading = SensorReading(**reading)
            
            # Check queue capacity
            if self.input_queue.qsize() >= self.max_queue_size:
                self.logger.warning("Input queue full, dropping reading")
                return False
            
            await self.input_queue.put(reading)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit reading: {str(e)}")
            self.error_count += 1
            return False
    
    async def get_result(self, timeout: Optional[float] = None) -> Optional[AnomalyResult]:
        """Get next anomaly detection result."""
        try:
            return await asyncio.wait_for(self.result_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.error(f"Failed to get result: {str(e)}")
            return None
    
    async def process_stream(
        self,
        readings: AsyncGenerator[SensorReading, None]
    ) -> AsyncGenerator[AnomalyResult, None]:
        """Process stream of sensor readings and yield results."""
        async for reading in readings:
            await self.submit_reading(reading)
            
            # Yield available results
            while True:
                result = await self.get_result(timeout=0.001)  # Non-blocking
                if result is None:
                    break
                yield result
    
    async def _batch_processor(self) -> None:
        """Main batch processing loop."""
        while self.is_running:
            try:
                # Collect readings for batch
                readings = await self._collect_batch()
                
                if not readings:
                    await asyncio.sleep(0.01)  # Short pause if no data
                    continue
                
                # Process batch
                start_time = time.time()
                results = await self._process_batch(readings)
                processing_time = time.time() - start_time
                
                # Store results
                for result in results:
                    try:
                        await self.result_queue.put(result)
                    except asyncio.QueueFull:
                        self.logger.warning("Result queue full, dropping result")
                
                # Update metrics
                self.processing_times.append(processing_time)
                self.metrics.total_processed += len(readings)
                self.metrics.anomalies_detected += sum(1 for r in results if r.is_anomaly)
                
                # Reset circuit breaker on success
                self.circuit_breaker_failures = 0
                
            except Exception as e:
                self.logger.error(f"Batch processing error: {str(e)}")
                self.error_count += 1
                self.circuit_breaker_failures += 1
                
                # Brief pause on error
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self) -> List[SensorReading]:
        """Collect readings into a batch for processing."""
        batch = []
        timeout = 0.1  # Max wait time for first reading
        
        try:
            # Get first reading with timeout
            first_reading = await asyncio.wait_for(self.input_queue.get(), timeout=timeout)
            batch.append(first_reading)
            
            # Collect additional readings without waiting
            while len(batch) < self.batch_size:
                try:
                    reading = self.input_queue.get_nowait()
                    batch.append(reading)
                except asyncio.QueueEmpty:
                    break
            
        except asyncio.TimeoutError:
            # No readings available
            pass
        
        return batch
    
    async def _process_batch(self, readings: List[SensorReading]) -> List[AnomalyResult]:
        """Process batch of readings through the core model."""
        if self.circuit_breaker_open:
            raise RuntimeError("Circuit breaker open")
        
        # Convert readings to DataFrame
        data_rows = []
        for reading in readings:
            row = reading.values.copy()
            row['timestamp'] = reading.timestamp
            row['sensor_id'] = reading.sensor_id
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self.executor,
            self._run_inference_sync,
            df
        )
        
        return results
    
    def _run_inference_sync(self, data: pd.DataFrame) -> List[AnomalyResult]:
        """Synchronous inference wrapper for thread pool execution."""
        try:
            # Remove non-sensor columns for model input
            sensor_columns = [col for col in data.columns if col not in ['timestamp', 'sensor_id']]
            model_input = data[sensor_columns]
            
            # Run inference synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                results = loop.run_until_complete(
                    self.core_model.predict_anomaly(model_input)
                )
                return results
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"Synchronous inference failed: {str(e)}")
            raise
    
    async def _metrics_updater(self) -> None:
        """Update real-time metrics periodically."""
        while self.is_running:
            try:
                # Calculate metrics
                current_time = time.time()
                self.metrics.uptime_seconds = current_time - self.start_time
                
                if self.processing_times:
                    self.metrics.avg_latency_ms = np.mean(list(self.processing_times)) * 1000
                    
                    # Calculate throughput
                    recent_window = 60  # 1 minute window
                    recent_count = sum(1 for t in self.processing_times 
                                     if current_time - t < recent_window)
                    self.metrics.throughput_per_sec = recent_count / min(recent_window, self.metrics.uptime_seconds)
                
                self.metrics.queue_size = self.input_queue.qsize()
                
                # Error rate
                if self.metrics.total_processed > 0:
                    self.metrics.error_rate = self.error_count / self.metrics.total_processed
                
                # Memory usage (approximate)
                try:
                    import psutil
                    process = psutil.Process()
                    self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                except ImportError:
                    pass
                
                self.logger.debug(f"Metrics: {self.metrics}")
                
            except Exception as e:
                self.logger.error(f"Metrics update error: {str(e)}")
            
            await asyncio.sleep(self.metrics_interval)
    
    async def _circuit_breaker_monitor(self) -> None:
        """Monitor circuit breaker state."""
        circuit_breaker_reset_time = 30.0  # Reset after 30 seconds
        
        while self.is_running:
            try:
                # Open circuit breaker if too many failures
                if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
                    if not self.circuit_breaker_open:
                        self.circuit_breaker_open = True
                        self.logger.error("Circuit breaker opened due to failures")
                    
                    # Reset after timeout
                    await asyncio.sleep(circuit_breaker_reset_time)
                    self.circuit_breaker_open = False
                    self.circuit_breaker_failures = 0
                    self.logger.info("Circuit breaker reset")
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Circuit breaker monitor error: {str(e)}")
    
    def get_metrics(self) -> InferenceMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    def get_health_status(self) -> Dict[str, Union[str, bool, float]]:
        """Get health status of the inference engine."""
        return {
            "status": "healthy" if self.is_running and not self.circuit_breaker_open else "unhealthy",
            "is_running": self.is_running,
            "circuit_breaker_open": self.circuit_breaker_open,
            "uptime_seconds": self.metrics.uptime_seconds,
            "error_rate": self.metrics.error_rate,
            "queue_utilization": self.input_queue.qsize() / self.max_queue_size,
            "throughput_per_sec": self.metrics.throughput_per_sec,
            "avg_latency_ms": self.metrics.avg_latency_ms
        }


# Example usage and testing utilities

class SimulatedSensorStream:
    """Simulated sensor data stream for testing."""
    
    def __init__(
        self,
        num_sensors: int = 5,
        sample_rate: float = 10.0,  # Hz
        anomaly_probability: float = 0.05
    ):
        self.num_sensors = num_sensors
        self.sample_rate = sample_rate
        self.anomaly_probability = anomaly_probability
        self.current_time = time.time()
        
    async def generate_readings(self) -> AsyncGenerator[SensorReading, None]:
        """Generate simulated sensor readings."""
        while True:
            # Generate normal sensor values
            values = {}
            for i in range(self.num_sensors):
                base_value = 20.0 + 5.0 * np.sin(self.current_time * 0.1 + i)
                noise = np.random.normal(0, 0.5)
                
                # Inject anomalies
                if np.random.random() < self.anomaly_probability:
                    anomaly_factor = np.random.uniform(3, 5)
                    noise *= anomaly_factor
                
                values[f"sensor_{i}"] = base_value + noise
            
            reading = SensorReading(
                timestamp=self.current_time,
                sensor_id=f"device_{np.random.randint(1, 10)}",
                values=values,
                metadata={"simulation": True}
            )
            
            yield reading
            
            # Update time and wait
            self.current_time += 1.0 / self.sample_rate
            await asyncio.sleep(1.0 / self.sample_rate)


async def demo_real_time_inference():
    """Demonstration of real-time inference capabilities."""
    from pathlib import Path
    
    # Create a mock core model (in practice, this would be loaded)
    core_model = AutonomousAnomalyCore(
        window_size=10,
        ensemble_size=2
    )
    
    # For demo, create minimal training data
    demo_data = pd.DataFrame({
        f'sensor_{i}': np.random.normal(20, 2, 1000) 
        for i in range(5)
    })
    
    print("Training demo model...")
    await core_model.train_ensemble(demo_data, epochs=5)
    
    # Create inference engine
    engine = RealTimeInferenceEngine(
        core_model=core_model,
        batch_size=16,
        max_queue_size=1000
    )
    
    # Create simulated sensor stream
    sensor_stream = SimulatedSensorStream(
        num_sensors=5,
        sample_rate=5.0,
        anomaly_probability=0.1
    )
    
    print("Starting real-time inference engine...")
    await engine.start()
    
    try:
        # Process stream for demo duration
        demo_duration = 30.0  # seconds
        start_time = time.time()
        result_count = 0
        
        async for result in engine.process_stream(sensor_stream.generate_readings()):
            if result.is_anomaly:
                print(f"ANOMALY DETECTED: Score={result.anomaly_score:.4f}, "
                      f"Confidence={result.confidence:.3f}, Time={result.timestamp:.1f}")
            
            result_count += 1
            
            if time.time() - start_time > demo_duration:
                break
        
        # Print final metrics
        metrics = engine.get_metrics()
        health = engine.get_health_status()
        
        print(f"\nDemo Results:")
        print(f"Total processed: {metrics.total_processed}")
        print(f"Anomalies detected: {metrics.anomalies_detected}")
        print(f"Average latency: {metrics.avg_latency_ms:.2f} ms")
        print(f"Throughput: {metrics.throughput_per_sec:.2f} samples/sec")
        print(f"Health status: {health['status']}")
        
    finally:
        await engine.stop()
        print("Demo completed.")


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demo_real_time_inference())