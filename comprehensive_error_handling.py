"""Comprehensive error handling and recovery system for Generation 2."""

import logging
import traceback
import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import functools
import pickle
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    RESTART = "restart"


@dataclass
class ErrorContext:
    """Context information for errors."""
    timestamp: datetime
    function_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    stack_trace: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'function_name': self.function_name,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'recovery_strategy': self.recovery_strategy.value,
            'stack_trace': self.stack_trace,
            'metadata': self.metadata or {}
        }


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.base_delay
        
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            delay *= (0.5 + np.random.random() * 0.5)
        
        return delay


class RobustErrorHandler:
    """Comprehensive error handling system."""
    
    def __init__(self, error_log_path: str = "error_log.json"):
        self.error_log_path = error_log_path
        self.error_history: List[ErrorContext] = []
        self.recovery_handlers: Dict[str, Callable] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        
        self._lock = threading.Lock()
        self._load_error_history()
    
    def register_recovery_handler(self, error_type: str, handler: Callable):
        """Register recovery handler for specific error type."""
        self.recovery_handlers[error_type] = handler
        logger.info(f"Registered recovery handler for {error_type}")
    
    def register_fallback_handler(self, function_name: str, handler: Callable):
        """Register fallback handler for specific function."""
        self.fallback_handlers[function_name] = handler
        logger.info(f"Registered fallback handler for {function_name}")
    
    def handle_error(
        self,
        error: Exception,
        function_name: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        metadata: Dict[str, Any] = None
    ) -> ErrorContext:
        """Handle error with comprehensive logging and recovery."""
        
        error_context = ErrorContext(
            timestamp=datetime.now(),
            function_name=function_name,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            recovery_strategy=recovery_strategy,
            stack_trace=traceback.format_exc(),
            metadata=metadata
        )
        
        with self._lock:
            self.error_history.append(error_context)
        
        # Log error
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        logger.log(
            log_level,
            f"Error in {function_name}: {error_context.error_type} - {error_context.error_message}"
        )
        
        # Execute recovery strategy
        self._execute_recovery_strategy(error_context)
        
        # Save error log
        self._save_error_history()
        
        return error_context
    
    def _execute_recovery_strategy(self, error_context: ErrorContext):
        """Execute appropriate recovery strategy."""
        error_type = error_context.error_type
        function_name = error_context.function_name
        
        if error_type in self.recovery_handlers:
            try:
                self.recovery_handlers[error_type](error_context)
                logger.info(f"Recovery handler executed for {error_type}")
            except Exception as e:
                logger.error(f"Recovery handler failed: {e}")
        
        elif error_context.recovery_strategy == RecoveryStrategy.FALLBACK:
            if function_name in self.fallback_handlers:
                try:
                    self.fallback_handlers[function_name](error_context)
                    logger.info(f"Fallback handler executed for {function_name}")
                except Exception as e:
                    logger.error(f"Fallback handler failed: {e}")
    
    def _load_error_history(self):
        """Load error history from file."""
        if Path(self.error_log_path).exists():
            try:
                with open(self.error_log_path, 'r') as f:
                    error_data = json.load(f)
                
                self.error_history = []
                for item in error_data:
                    # Convert back to ErrorContext objects
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    item['severity'] = ErrorSeverity(item['severity'])
                    item['recovery_strategy'] = RecoveryStrategy(item['recovery_strategy'])
                    
                    error_context = ErrorContext(**item)
                    self.error_history.append(error_context)
                
                logger.info(f"Loaded {len(self.error_history)} error records")
                
            except Exception as e:
                logger.warning(f"Failed to load error history: {e}")
    
    def _save_error_history(self):
        """Save error history to file."""
        try:
            error_data = [error.to_dict() for error in self.error_history]
            
            with open(self.error_log_path, 'w') as f:
                json.dump(error_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save error history: {e}")
    
    def get_error_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for recent period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [
            error for error in self.error_history
            if error.timestamp > cutoff_time
        ]
        
        if not recent_errors:
            return {"total_errors": 0, "period_hours": hours}
        
        # Count by type
        error_types = {}
        severity_counts = {severity.value: 0 for severity in ErrorSeverity}
        function_errors = {}
        
        for error in recent_errors:
            # Count by error type
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            
            # Count by severity
            severity_counts[error.severity.value] += 1
            
            # Count by function
            function_errors[error.function_name] = function_errors.get(error.function_name, 0) + 1
        
        return {
            "total_errors": len(recent_errors),
            "period_hours": hours,
            "error_types": error_types,
            "severity_counts": severity_counts,
            "function_errors": function_errors,
            "error_rate_per_hour": len(recent_errors) / hours if hours > 0 else 0
        }


def robust_retry(
    retry_config: RetryConfig = None,
    error_handler: RobustErrorHandler = None,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
):
    """Decorator for robust retry with comprehensive error handling."""
    
    if retry_config is None:
        retry_config = RetryConfig()
    
    if error_handler is None:
        error_handler = RobustErrorHandler()
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, retry_config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    
                    # Determine severity based on attempt
                    if attempt == 1:
                        severity = ErrorSeverity.LOW
                    elif attempt < retry_config.max_attempts:
                        severity = ErrorSeverity.MEDIUM
                    else:
                        severity = ErrorSeverity.HIGH
                    
                    # Handle error
                    error_context = error_handler.handle_error(
                        e,
                        func.__name__,
                        severity=severity,
                        recovery_strategy=recovery_strategy,
                        metadata={
                            'attempt': attempt,
                            'max_attempts': retry_config.max_attempts,
                            'args': str(args)[:100],  # Truncate for logging
                            'kwargs': str(kwargs)[:100]
                        }
                    )
                    
                    # Don't retry on final attempt
                    if attempt == retry_config.max_attempts:
                        break
                    
                    # Calculate delay and wait
                    delay = retry_config.get_delay(attempt)
                    logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt}/{retry_config.max_attempts})")
                    time.sleep(delay)
            
            # All retries failed
            error_handler.handle_error(
                last_exception,
                func.__name__,
                severity=ErrorSeverity.CRITICAL,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                metadata={'final_failure': True}
            )
            
            raise last_exception
        
        return wrapper
    return decorator


def graceful_degradation(
    fallback_value: Any = None,
    error_handler: RobustErrorHandler = None
):
    """Decorator for graceful degradation on errors."""
    
    if error_handler is None:
        error_handler = RobustErrorHandler()
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                error_handler.handle_error(
                    e,
                    func.__name__,
                    severity=ErrorSeverity.MEDIUM,
                    recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                    metadata={'fallback_value': str(fallback_value)}
                )
                
                logger.warning(f"Function {func.__name__} failed, returning fallback value: {fallback_value}")
                return fallback_value
        
        return wrapper
    return decorator


class HealthCheckManager:
    """Manage system health checks with error handling."""
    
    def __init__(self, error_handler: RobustErrorHandler = None):
        self.error_handler = error_handler or RobustErrorHandler()
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def run_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all health checks with error handling."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                status = check_func()
                duration = time.time() - start_time
                
                results[name] = {
                    'status': 'healthy' if status else 'unhealthy',
                    'timestamp': datetime.now().isoformat(),
                    'duration_ms': duration * 1000,
                    'details': status if isinstance(status, dict) else {'result': status}
                }
                
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    f"health_check_{name}",
                    severity=ErrorSeverity.MEDIUM,
                    recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION
                )
                
                results[name] = {
                    'status': 'error',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'details': {'error_type': type(e).__name__}
                }
        
        self.health_status = results
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.health_status:
            self.run_health_checks()
        
        healthy_count = sum(1 for status in self.health_status.values() if status['status'] == 'healthy')
        total_count = len(self.health_status)
        
        if total_count == 0:
            overall_status = 'unknown'
        elif healthy_count == total_count:
            overall_status = 'healthy'
        elif healthy_count > total_count * 0.5:
            overall_status = 'degraded'
        else:
            overall_status = 'unhealthy'
        
        return {
            'overall_status': overall_status,
            'healthy_checks': healthy_count,
            'total_checks': total_count,
            'health_percentage': (healthy_count / total_count * 100) if total_count > 0 else 0,
            'timestamp': datetime.now().isoformat(),
            'individual_checks': self.health_status
        }


# Example usage decorators and classes
class RobustModelManager:
    """Model manager with comprehensive error handling."""
    
    def __init__(self):
        self.error_handler = RobustErrorHandler()
        self.model = None
        self.fallback_model = None
        
        # Register recovery handlers
        self.error_handler.register_recovery_handler(
            'MemoryError',
            self._handle_memory_error
        )
        self.error_handler.register_fallback_handler(
            'predict',
            self._fallback_prediction
        )
    
    @robust_retry(
        retry_config=RetryConfig(max_attempts=3, base_delay=1.0),
        recovery_strategy=RecoveryStrategy.RETRY
    )
    def load_model(self, model_path: str):
        """Load model with retry logic."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Simulate model loading
        if np.random.random() < 0.2:  # 20% chance of failure for demo
            raise MemoryError("Insufficient memory to load model")
        
        self.model = f"Model loaded from {model_path}"
        logger.info("Model loaded successfully")
    
    @graceful_degradation(fallback_value=np.array([0.5]))
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make prediction with graceful degradation."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Simulate prediction
        if np.random.random() < 0.1:  # 10% chance of failure for demo
            raise RuntimeError("Prediction failed due to internal error")
        
        return np.random.random(len(data))
    
    def _handle_memory_error(self, error_context: ErrorContext):
        """Handle memory errors by cleaning up resources."""
        logger.info("Handling memory error: cleaning up resources")
        # Simulate cleanup
        import gc
        gc.collect()
    
    def _fallback_prediction(self, error_context: ErrorContext):
        """Fallback prediction handler."""
        logger.info("Using fallback prediction method")
        return np.array([0.5])  # Simple fallback


if __name__ == "__main__":
    # Generation 2 error handling demonstration
    logger.info("=== GENERATION 2: ROBUST ERROR HANDLING DEMO ===")
    
    # Initialize error handler
    error_handler = RobustErrorHandler()
    
    # Initialize model manager
    model_manager = RobustModelManager()
    
    # Test model loading with retries
    try:
        model_manager.load_model("nonexistent_model.h5")
    except Exception as e:
        logger.info(f"Model loading ultimately failed: {e}")
    
    # Test predictions with graceful degradation
    test_data = np.random.random(10)
    for i in range(5):
        try:
            predictions = model_manager.predict(test_data)
            logger.info(f"Prediction {i+1}: {predictions[:3]}...")  # Show first 3 values
        except Exception as e:
            logger.info(f"Prediction {i+1} failed: {e}")
    
    # Initialize health check manager
    health_manager = HealthCheckManager(error_handler)
    
    # Register sample health checks
    def check_memory_usage():
        import psutil
        return psutil.virtual_memory().percent < 85
    
    def check_disk_space():
        import psutil
        return psutil.disk_usage('/').percent < 90
    
    def check_model_availability():
        return model_manager.model is not None
    
    health_manager.register_health_check('memory', check_memory_usage)
    health_manager.register_health_check('disk', check_disk_space)
    health_manager.register_health_check('model', check_model_availability)
    
    # Run health checks
    health_results = health_manager.run_health_checks()
    overall_health = health_manager.get_overall_health()
    
    logger.info("Health Check Results:")
    logger.info(f"  Overall Status: {overall_health['overall_status']}")
    logger.info(f"  Health Percentage: {overall_health['health_percentage']:.1f}%")
    
    # Get error statistics
    error_stats = error_handler.get_error_stats(hours=1)
    logger.info("Error Statistics:")
    logger.info(f"  Total Errors: {error_stats['total_errors']}")
    logger.info(f"  Error Rate: {error_stats['error_rate_per_hour']:.1f}/hour")
    
    if error_stats['error_types']:
        logger.info("  Error Types:")
        for error_type, count in error_stats['error_types'].items():
            logger.info(f"    {error_type}: {count}")
    
    logger.info("=== GENERATION 2 ERROR HANDLING COMPLETE ===")