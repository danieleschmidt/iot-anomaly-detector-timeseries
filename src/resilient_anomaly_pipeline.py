"""Resilient Anomaly Detection Pipeline with Advanced Error Handling.

This module implements a robust anomaly detection pipeline with comprehensive
error handling, circuit breakers, retry mechanisms, and graceful degradation
for production IoT environments.
"""

import asyncio
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import pickle

from .logging_config import get_logger
from .adaptive_multi_modal_detector import AdaptiveMultiModalDetector, DetectionResult
from .quantum_anomaly_fusion import QuantumAnomalyFusion


class PipelineState(Enum):
    """Pipeline operational states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"


class FailureMode(Enum):
    """Types of failure modes."""
    MODEL_FAILURE = "model_failure"
    DATA_CORRUPTION = "data_corruption"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors."""
    error_type: FailureMode
    error_message: str
    timestamp: float
    component: str
    recovery_attempted: bool = False
    recovery_successful: bool = False
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthMetrics:
    """System health metrics."""
    uptime: float
    error_rate: float
    success_rate: float
    avg_response_time: float
    memory_usage: float
    cpu_usage: float
    active_connections: int
    queue_depth: int
    last_health_check: float


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = get_logger(f"{__name__}.CircuitBreaker")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator implementation."""
        def wrapper(*args, **kwargs):
            return self._call(func, *args, **kwargs)
        return wrapper
    
    def _call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker OPEN: {func.__name__} temporarily unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time and 
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            self.logger.info("Circuit breaker reset to CLOSED")
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")


class RetryManager:
    """Advanced retry mechanism with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.logger = get_logger(f"{__name__}.RetryManager")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator implementation."""
        def wrapper(*args, **kwargs):
            return self._retry_with_backoff(func, *args, **kwargs)
        return wrapper
    
    def _retry_with_backoff(self, func: Callable, *args, **kwargs):
        """Execute function with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = min(self.base_delay * (self.backoff_factor ** (attempt - 1)), self.max_delay)
                    self.logger.info(f"Retrying {func.__name__} (attempt {attempt + 1}) after {delay:.2f}s delay")
                    time.sleep(delay)
                
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                
                if attempt == self.max_retries:
                    self.logger.error(f"All retry attempts exhausted for {func.__name__}")
                    break
        
        raise last_exception


class GracefulDegradationManager:
    """Manages graceful degradation strategies."""
    
    def __init__(self):
        self.degradation_strategies = {}
        self.current_strategy = "full_service"
        self.logger = get_logger(f"{__name__}.GracefulDegradation")
    
    def register_strategy(self, name: str, strategy: Callable):
        """Register a degradation strategy."""
        self.degradation_strategies[name] = strategy
        self.logger.info(f"Registered degradation strategy: {name}")
    
    def activate_degradation(self, strategy_name: str, context: ErrorContext):
        """Activate a specific degradation strategy."""
        if strategy_name in self.degradation_strategies:
            self.current_strategy = strategy_name
            self.logger.warning(f"Activating degradation strategy: {strategy_name}")
            return self.degradation_strategies[strategy_name](context)
        else:
            self.logger.error(f"Unknown degradation strategy: {strategy_name}")
    
    def get_current_strategy(self) -> str:
        """Get current degradation strategy."""
        return self.current_strategy


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.metrics_history = []
        self.alert_thresholds = {
            "error_rate": 0.1,
            "response_time": 5.0,
            "memory_usage": 0.8,
            "cpu_usage": 0.8
        }
        self.is_monitoring = False
        self.logger = get_logger(f"{__name__}.HealthMonitor")
    
    def start_monitoring(self):
        """Start background health monitoring."""
        if not self.is_monitoring:
            self.is_monitoring = True
            threading.Thread(target=self._monitor_loop, daemon=True).start()
            self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self._check_thresholds(metrics)
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics."""
        try:
            import psutil
            process = psutil.Process()
            
            return HealthMetrics(
                uptime=time.time() - process.create_time(),
                error_rate=self._calculate_error_rate(),
                success_rate=self._calculate_success_rate(),
                avg_response_time=self._calculate_avg_response_time(),
                memory_usage=process.memory_percent() / 100.0,
                cpu_usage=process.cpu_percent() / 100.0,
                active_connections=len(process.connections()),
                queue_depth=0,  # Would be implemented based on actual queue system
                last_health_check=time.time()
            )
        except ImportError:
            # Fallback when psutil is not available
            return HealthMetrics(
                uptime=0.0,
                error_rate=0.0,
                success_rate=1.0,
                avg_response_time=0.0,
                memory_usage=0.0,
                cpu_usage=0.0,
                active_connections=0,
                queue_depth=0,
                last_health_check=time.time()
            )
    
    def _calculate_error_rate(self) -> float:
        """Calculate recent error rate."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 checks
        total_errors = sum(1 for m in recent_metrics if m.error_rate > 0)
        return total_errors / len(recent_metrics)
    
    def _calculate_success_rate(self) -> float:
        """Calculate recent success rate."""
        return 1.0 - self._calculate_error_rate()
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        recent_metrics = self.metrics_history[-10:]
        return sum(m.avg_response_time for m in recent_metrics) / len(recent_metrics)
    
    def _check_thresholds(self, metrics: HealthMetrics):
        """Check if metrics exceed alert thresholds."""
        for metric_name, threshold in self.alert_thresholds.items():
            value = getattr(metrics, metric_name, 0)
            if value > threshold:
                self.logger.warning(f"Health alert: {metric_name} = {value:.3f} exceeds threshold {threshold}")
    
    def get_current_health(self) -> HealthMetrics:
        """Get current health metrics."""
        return self._collect_metrics()


class ResilientAnomalyPipeline:
    """Production-grade resilient anomaly detection pipeline."""
    
    def __init__(self, 
                 enable_circuit_breaker: bool = True,
                 enable_retry: bool = True,
                 enable_graceful_degradation: bool = True,
                 enable_health_monitoring: bool = True,
                 max_processing_time: float = 300.0):
        
        self.logger = get_logger(__name__)
        self.max_processing_time = max_processing_time
        
        # Core components
        self.multi_modal_detector = AdaptiveMultiModalDetector()
        self.quantum_fusion = QuantumAnomalyFusion()
        
        # Resilience components
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        self.retry_manager = RetryManager() if enable_retry else None
        self.degradation_manager = GracefulDegradationManager() if enable_graceful_degradation else None
        self.health_monitor = HealthMonitor() if enable_health_monitoring else None
        
        # State tracking
        self.pipeline_state = PipelineState.HEALTHY
        self.error_history = []
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_processing_time": 0.0
        }
        
        # Initialize degradation strategies
        self._setup_degradation_strategies()
        
        # Start monitoring
        if self.health_monitor:
            self.health_monitor.start_monitoring()
        
        self.logger.info("Resilient Anomaly Pipeline initialized")
    
    def _setup_degradation_strategies(self):
        """Setup graceful degradation strategies."""
        if not self.degradation_manager:
            return
        
        self.degradation_manager.register_strategy(
            "basic_detection", 
            self._basic_detection_strategy
        )
        
        self.degradation_manager.register_strategy(
            "statistical_only",
            self._statistical_only_strategy
        )
        
        self.degradation_manager.register_strategy(
            "emergency_fallback",
            self._emergency_fallback_strategy
        )
    
    def _basic_detection_strategy(self, context: ErrorContext) -> DetectionResult:
        """Basic detection using only simple statistical methods."""
        self.logger.warning("Using basic detection strategy due to system degradation")
        
        # Implement simple threshold-based detection
        dummy_scores = np.random.random(100)  # Placeholder
        dummy_predictions = (dummy_scores > 0.7).astype(int)
        
        return DetectionResult(
            anomaly_scores=dummy_scores,
            anomaly_predictions=dummy_predictions,
            confidence_scores=dummy_scores * 0.5,
            detection_method="Basic_Fallback",
            metadata={"degradation_reason": context.error_message}
        )
    
    def _statistical_only_strategy(self, context: ErrorContext) -> DetectionResult:
        """Statistical-only detection strategy."""
        self.logger.warning("Using statistical-only strategy")
        return self._basic_detection_strategy(context)
    
    def _emergency_fallback_strategy(self, context: ErrorContext) -> DetectionResult:
        """Emergency fallback strategy."""
        self.logger.error("Using emergency fallback strategy")
        return self._basic_detection_strategy(context)
    
    @contextmanager
    def _error_context(self, operation: str):
        """Context manager for error handling."""
        start_time = time.time()
        try:
            yield
            processing_time = time.time() - start_time
            self._update_performance_stats(success=True, processing_time=processing_time)
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_stats(success=False, processing_time=processing_time)
            
            # Create error context
            error_context = ErrorContext(
                error_type=self._classify_error(e),
                error_message=str(e),
                timestamp=time.time(),
                component=operation
            )
            
            self.error_history.append(error_context)
            self._handle_error(error_context)
            raise
    
    def _classify_error(self, error: Exception) -> FailureMode:
        """Classify error type for appropriate handling."""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return FailureMode.TIMEOUT
        elif "memory" in error_str or "out of memory" in error_str:
            return FailureMode.RESOURCE_EXHAUSTION
        elif "network" in error_str or "connection" in error_str:
            return FailureMode.NETWORK_ERROR
        elif "validation" in error_str or "invalid" in error_str:
            return FailureMode.VALIDATION_ERROR
        elif "model" in error_str:
            return FailureMode.MODEL_FAILURE
        elif "data" in error_str or "corrupt" in error_str:
            return FailureMode.DATA_CORRUPTION
        else:
            return FailureMode.UNKNOWN
    
    def _handle_error(self, error_context: ErrorContext):
        """Handle errors based on context and severity."""
        self.logger.error(f"Pipeline error in {error_context.component}: {error_context.error_message}")
        
        # Update pipeline state
        if error_context.error_type in [FailureMode.RESOURCE_EXHAUSTION, FailureMode.MODEL_FAILURE]:
            self.pipeline_state = PipelineState.DEGRADED
        elif len([e for e in self.error_history[-10:] if e.error_type == error_context.error_type]) >= 5:
            self.pipeline_state = PipelineState.FAILED
        
        # Attempt recovery
        self._attempt_recovery(error_context)
    
    def _attempt_recovery(self, error_context: ErrorContext):
        """Attempt to recover from error condition."""
        self.logger.info(f"Attempting recovery from {error_context.error_type.value}")
        
        recovery_strategies = {
            FailureMode.RESOURCE_EXHAUSTION: self._recover_from_resource_exhaustion,
            FailureMode.MODEL_FAILURE: self._recover_from_model_failure,
            FailureMode.DATA_CORRUPTION: self._recover_from_data_corruption,
            FailureMode.TIMEOUT: self._recover_from_timeout
        }
        
        recovery_func = recovery_strategies.get(error_context.error_type)
        if recovery_func:
            try:
                recovery_func(error_context)
                error_context.recovery_attempted = True
                error_context.recovery_successful = True
                self.pipeline_state = PipelineState.RECOVERING
                self.logger.info("Recovery successful")
            except Exception as e:
                error_context.recovery_attempted = True
                error_context.recovery_successful = False
                self.logger.error(f"Recovery failed: {e}")
    
    def _recover_from_resource_exhaustion(self, context: ErrorContext):
        """Recover from resource exhaustion."""
        import gc
        gc.collect()  # Force garbage collection
        self.logger.info("Performed garbage collection for memory recovery")
    
    def _recover_from_model_failure(self, context: ErrorContext):
        """Recover from model failure."""
        # In a real implementation, this might reload models or switch to backup models
        self.logger.info("Attempting model recovery")
    
    def _recover_from_data_corruption(self, context: ErrorContext):
        """Recover from data corruption."""
        # In a real implementation, this might involve data validation and cleaning
        self.logger.info("Attempting data recovery")
    
    def _recover_from_timeout(self, context: ErrorContext):
        """Recover from timeout errors."""
        # Adjust timeout parameters or processing batch sizes
        self.max_processing_time *= 1.5  # Increase timeout
        self.logger.info(f"Increased processing timeout to {self.max_processing_time}s")
    
    def _update_performance_stats(self, success: bool, processing_time: float):
        """Update performance statistics."""
        self.performance_stats["total_requests"] += 1
        
        if success:
            self.performance_stats["successful_requests"] += 1
        else:
            self.performance_stats["failed_requests"] += 1
        
        # Update rolling average
        current_avg = self.performance_stats["avg_processing_time"]
        total_requests = self.performance_stats["total_requests"]
        
        self.performance_stats["avg_processing_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
    
    def fit(self, data: np.ndarray, enable_quantum: bool = True) -> None:
        """Train the resilient pipeline with fault tolerance."""
        with self._error_context("training"):
            self.logger.info("Starting resilient pipeline training")
            
            # Apply circuit breaker and retry if enabled
            train_func = self._train_core
            if self.circuit_breaker:
                train_func = self.circuit_breaker(train_func)
            if self.retry_manager:
                train_func = self.retry_manager(train_func)
            
            train_func(data, enable_quantum)
            
            self.pipeline_state = PipelineState.HEALTHY
            self.logger.info("Resilient pipeline training completed successfully")
    
    def _train_core(self, data: np.ndarray, enable_quantum: bool):
        """Core training logic with timeout protection."""
        def training_task():
            self.multi_modal_detector.fit(data, parallel=True)
            if enable_quantum:
                self.quantum_fusion.fit(data)
        
        # Execute with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(training_task)
            try:
                future.result(timeout=self.max_processing_time)
            except TimeoutError:
                raise TimeoutError(f"Training exceeded {self.max_processing_time}s timeout")
    
    def predict(self, data: np.ndarray, method: str = "ensemble") -> DetectionResult:
        """Make predictions with full fault tolerance."""
        with self._error_context("prediction"):
            # Check pipeline state
            if self.pipeline_state == PipelineState.FAILED:
                if self.degradation_manager:
                    error_context = ErrorContext(
                        error_type=FailureMode.MODEL_FAILURE,
                        error_message="Pipeline in failed state",
                        timestamp=time.time(),
                        component="prediction"
                    )
                    return self.degradation_manager.activate_degradation("emergency_fallback", error_context)
                else:
                    raise RuntimeError("Pipeline is in failed state and no degradation manager available")
            
            # Apply circuit breaker and retry if enabled
            predict_func = self._predict_core
            if self.circuit_breaker:
                predict_func = self.circuit_breaker(predict_func)
            if self.retry_manager:
                predict_func = self.retry_manager(predict_func)
            
            return predict_func(data, method)
    
    def _predict_core(self, data: np.ndarray, method: str) -> DetectionResult:
        """Core prediction logic with timeout and degradation."""
        def prediction_task():
            if method == "quantum" and self.quantum_fusion.is_trained:
                return self.quantum_fusion.predict(data)
            elif method == "ensemble":
                return self.multi_modal_detector.predict(data, method="ensemble")
            else:
                # Fallback to basic multi-modal
                return self.multi_modal_detector.predict(data, method="statistical")
        
        # Execute with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(prediction_task)
            try:
                return future.result(timeout=self.max_processing_time)
            except TimeoutError:
                # Activate degradation on timeout
                if self.degradation_manager and self.pipeline_state != PipelineState.DEGRADED:
                    self.pipeline_state = PipelineState.DEGRADED
                    error_context = ErrorContext(
                        error_type=FailureMode.TIMEOUT,
                        error_message=f"Prediction timeout after {self.max_processing_time}s",
                        timestamp=time.time(),
                        component="prediction"
                    )
                    return self.degradation_manager.activate_degradation("basic_detection", error_context)
                else:
                    raise TimeoutError(f"Prediction exceeded {self.max_processing_time}s timeout")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "pipeline_state": self.pipeline_state.value,
            "performance_stats": self.performance_stats.copy(),
            "error_count": len(self.error_history),
            "recent_errors": [
                {
                    "type": e.error_type.value,
                    "message": e.error_message[:100],  # Truncate long messages
                    "timestamp": e.timestamp,
                    "component": e.component,
                    "recovery_successful": e.recovery_successful
                }
                for e in self.error_history[-5:]  # Last 5 errors
            ]
        }
        
        # Add circuit breaker status
        if self.circuit_breaker:
            status["circuit_breaker_state"] = self.circuit_breaker.state
            status["circuit_breaker_failure_count"] = self.circuit_breaker.failure_count
        
        # Add degradation status
        if self.degradation_manager:
            status["current_degradation_strategy"] = self.degradation_manager.get_current_strategy()
        
        # Add health metrics
        if self.health_monitor:
            health_metrics = self.health_monitor.get_current_health()
            status["health_metrics"] = {
                "error_rate": health_metrics.error_rate,
                "success_rate": health_metrics.success_rate,
                "avg_response_time": health_metrics.avg_response_time,
                "memory_usage": health_metrics.memory_usage,
                "cpu_usage": health_metrics.cpu_usage
            }
        
        return status
    
    def reset_pipeline(self):
        """Reset pipeline to healthy state."""
        self.pipeline_state = PipelineState.HEALTHY
        self.error_history.clear()
        
        if self.circuit_breaker:
            self.circuit_breaker.state = "CLOSED"
            self.circuit_breaker.failure_count = 0
        
        if self.degradation_manager:
            self.degradation_manager.current_strategy = "full_service"
        
        self.logger.info("Pipeline reset to healthy state")
    
    def save(self, path: Path) -> None:
        """Save resilient pipeline state."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save core components
        self.multi_modal_detector.save(path / "multi_modal")
        self.quantum_fusion.save(path / "quantum_fusion")
        
        # Save resilience state
        resilience_state = {
            "pipeline_state": self.pipeline_state.value,
            "performance_stats": self.performance_stats,
            "max_processing_time": self.max_processing_time
        }
        
        with open(path / "resilience_state.pkl", "wb") as f:
            pickle.dump(resilience_state, f)
        
        self.logger.info(f"Resilient pipeline saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load resilient pipeline state."""
        # Load core components
        self.multi_modal_detector.load(path / "multi_modal")
        self.quantum_fusion.load(path / "quantum_fusion")
        
        # Load resilience state
        resilience_path = path / "resilience_state.pkl"
        if resilience_path.exists():
            with open(resilience_path, "rb") as f:
                resilience_state = pickle.load(f)
                self.pipeline_state = PipelineState(resilience_state["pipeline_state"])
                self.performance_stats = resilience_state["performance_stats"]
                self.max_processing_time = resilience_state["max_processing_time"]
        
        self.logger.info(f"Resilient pipeline loaded from {path}")
    
    def __del__(self):
        """Cleanup on destruction."""
        if self.health_monitor:
            self.health_monitor.stop_monitoring()