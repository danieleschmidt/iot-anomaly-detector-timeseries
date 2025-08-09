"""Comprehensive error handling and recovery system for IoT anomaly detection."""

import logging
import time
import traceback
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Type, Union
from enum import Enum
from pathlib import Path
import json
import queue
from contextlib import contextmanager


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA_VALIDATION = "data_validation"
    MODEL_INFERENCE = "model_inference"
    NETWORK_COMMUNICATION = "network_communication"
    RESOURCE_MANAGEMENT = "resource_management"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    HARDWARE = "hardware"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class ErrorContext:
    """Context information for error analysis."""
    timestamp: float
    component: str
    operation: str
    input_data: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None


@dataclass
class ErrorRecord:
    """Comprehensive error record."""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    operation: str
    error_type: str
    error_message: str
    stack_trace: str
    context: ErrorContext
    recovery_actions: List[str] = field(default_factory=list)
    resolution_status: str = "open"
    occurrence_count: int = 1
    first_occurrence: Optional[float] = None
    last_occurrence: Optional[float] = None


class RecoveryStrategy:
    """Base class for recovery strategies."""
    
    def __init__(self, name: str, max_attempts: int = 3, backoff_factor: float = 2.0):
        self.name = name
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.attempt_count = 0
        self.last_attempt_time = 0.0
    
    def can_attempt(self) -> bool:
        """Check if recovery can be attempted."""
        return self.attempt_count < self.max_attempts
    
    def calculate_delay(self) -> float:
        """Calculate delay before next attempt."""
        if self.attempt_count == 0:
            return 0.0
        return min(self.backoff_factor ** self.attempt_count, 60.0)  # Cap at 60 seconds
    
    def execute(self, error_record: ErrorRecord, context: Dict[str, Any]) -> bool:
        """Execute recovery strategy. Return True if successful."""
        if not self.can_attempt():
            return False
        
        delay = self.calculate_delay()
        if delay > 0:
            time.sleep(delay)
        
        self.attempt_count += 1
        self.last_attempt_time = time.time()
        
        try:
            return self._execute_recovery(error_record, context)
        except Exception as e:
            logging.error(f"Recovery strategy {self.name} failed: {e}")
            return False
    
    def _execute_recovery(self, error_record: ErrorRecord, context: Dict[str, Any]) -> bool:
        """Override this method to implement specific recovery logic."""
        raise NotImplementedError
    
    def reset(self) -> None:
        """Reset recovery strategy state."""
        self.attempt_count = 0
        self.last_attempt_time = 0.0


class RetryStrategy(RecoveryStrategy):
    """Simple retry recovery strategy."""
    
    def __init__(self, max_attempts: int = 3, backoff_factor: float = 2.0):
        super().__init__("retry", max_attempts, backoff_factor)
    
    def _execute_recovery(self, error_record: ErrorRecord, context: Dict[str, Any]) -> bool:
        """Retry the failed operation."""
        retry_function = context.get("retry_function")
        if retry_function and callable(retry_function):
            try:
                result = retry_function()
                return result is not None
            except Exception:
                return False
        return False


class FallbackStrategy(RecoveryStrategy):
    """Fallback to alternative implementation."""
    
    def __init__(self, fallback_function: Callable, max_attempts: int = 1):
        super().__init__("fallback", max_attempts, 1.0)
        self.fallback_function = fallback_function
    
    def _execute_recovery(self, error_record: ErrorRecord, context: Dict[str, Any]) -> bool:
        """Execute fallback function."""
        try:
            result = self.fallback_function(error_record, context)
            return result is not None
        except Exception:
            return False


class CircuitBreakerStrategy(RecoveryStrategy):
    """Circuit breaker pattern for failing services."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        super().__init__("circuit_breaker", 1, 1.0)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half_open
    
    def can_attempt(self) -> bool:
        """Check circuit breaker state."""
        current_time = time.time()
        
        if self.state == "open":
            if current_time - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                return True
            return False
        
        return True
    
    def record_success(self) -> None:
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def _execute_recovery(self, error_record: ErrorRecord, context: Dict[str, Any]) -> bool:
        """Execute operation with circuit breaker logic."""
        if not self.can_attempt():
            return False
        
        retry_function = context.get("retry_function")
        if retry_function and callable(retry_function):
            try:
                result = retry_function()
                if result is not None:
                    self.record_success()
                    return True
                else:
                    self.record_failure()
                    return False
            except Exception:
                self.record_failure()
                return False
        
        return False


class RobustErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(
        self,
        error_log_path: Optional[str] = None,
        max_error_history: int = 10000,
        enable_auto_recovery: bool = True
    ):
        self.error_log_path = error_log_path
        self.max_error_history = max_error_history
        self.enable_auto_recovery = enable_auto_recovery
        
        # Error tracking
        self.error_history: List[ErrorRecord] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, List[RecoveryStrategy]] = {}
        self.error_handlers: Dict[ErrorCategory, List[Callable]] = {}
        
        # Threading for async error processing
        self.error_queue = queue.Queue()
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False
        
        # Statistics
        self.stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "critical_errors": 0,
            "recovery_success_rate": 0.0,
            "most_common_errors": {}
        }
        
        self.logger = logging.getLogger(__name__)
        self._initialize_default_strategies()
        self._start_error_processing()
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default recovery strategies for different error categories."""
        # Network communication errors
        self.add_recovery_strategy(
            ErrorCategory.NETWORK_COMMUNICATION,
            RetryStrategy(max_attempts=3, backoff_factor=2.0)
        )
        self.add_recovery_strategy(
            ErrorCategory.NETWORK_COMMUNICATION,
            CircuitBreakerStrategy(failure_threshold=5, recovery_timeout=30.0)
        )
        
        # Model inference errors
        self.add_recovery_strategy(
            ErrorCategory.MODEL_INFERENCE,
            RetryStrategy(max_attempts=2, backoff_factor=1.5)
        )
        
        # External service errors
        self.add_recovery_strategy(
            ErrorCategory.EXTERNAL_SERVICE,
            CircuitBreakerStrategy(failure_threshold=3, recovery_timeout=60.0)
        )
    
    def _start_error_processing(self) -> None:
        """Start background error processing thread."""
        self.is_processing = True
        self.processing_thread = threading.Thread(
            target=self._error_processing_loop,
            daemon=True
        )
        self.processing_thread.start()
    
    def _error_processing_loop(self) -> None:
        """Background processing loop for errors."""
        while self.is_processing:
            try:
                try:
                    error_record = self.error_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                self._process_error_record(error_record)
                
            except Exception as e:
                logging.error(f"Error in error processing loop: {e}")
    
    def add_recovery_strategy(self, category: ErrorCategory, strategy: RecoveryStrategy) -> None:
        """Add recovery strategy for error category."""
        if category not in self.recovery_strategies:
            self.recovery_strategies[category] = []
        self.recovery_strategies[category].append(strategy)
    
    def add_error_handler(self, category: ErrorCategory, handler: Callable) -> None:
        """Add custom error handler for category."""
        if category not in self.error_handlers:
            self.error_handlers[category] = []
        self.error_handlers[category].append(handler)
    
    @contextmanager
    def error_context(
        self,
        component: str,
        operation: str,
        category: ErrorCategory = ErrorCategory.DATA_VALIDATION,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        retry_function: Optional[Callable] = None,
        **context_data
    ):
        """Context manager for error handling."""
        start_time = time.time()
        context = ErrorContext(
            timestamp=start_time,
            component=component,
            operation=operation,
            input_data=context_data.get("input_data"),
            system_state=context_data.get("system_state"),
            environment=context_data.get("environment")
        )
        
        try:
            yield context
        except Exception as e:
            error_record = self._create_error_record(
                e, category, severity, context, retry_function
            )
            self._handle_error(error_record, {"retry_function": retry_function})
            raise
    
    def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        category: ErrorCategory = ErrorCategory.DATA_VALIDATION,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context_data: Optional[Dict[str, Any]] = None,
        retry_function: Optional[Callable] = None
    ) -> bool:
        """Handle error with automatic recovery attempts."""
        context = ErrorContext(
            timestamp=time.time(),
            component=component,
            operation=operation,
            input_data=context_data.get("input_data") if context_data else None,
            system_state=context_data.get("system_state") if context_data else None,
            environment=context_data.get("environment") if context_data else None
        )
        
        error_record = self._create_error_record(
            error, category, severity, context, retry_function
        )
        
        return self._handle_error(error_record, {"retry_function": retry_function})
    
    def _create_error_record(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: ErrorContext,
        retry_function: Optional[Callable] = None
    ) -> ErrorRecord:
        """Create comprehensive error record."""
        error_id = f"{category.value}_{context.component}_{int(time.time() * 1000)}"
        
        return ErrorRecord(
            error_id=error_id,
            timestamp=context.timestamp,
            severity=severity,
            category=category,
            component=context.component,
            operation=context.operation,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context,
            first_occurrence=context.timestamp,
            last_occurrence=context.timestamp
        )
    
    def _handle_error(self, error_record: ErrorRecord, recovery_context: Dict[str, Any]) -> bool:
        """Handle error with recovery attempts."""
        # Add to queue for async processing
        try:
            self.error_queue.put_nowait(error_record)
        except queue.Full:
            self.logger.warning("Error queue full, processing error synchronously")
            self._process_error_record(error_record)
        
        # Attempt immediate recovery if enabled
        if self.enable_auto_recovery:
            return self._attempt_recovery(error_record, recovery_context)
        
        return False
    
    def _process_error_record(self, error_record: ErrorRecord) -> None:
        """Process error record and update statistics."""
        # Check for duplicate errors
        existing_record = self._find_similar_error(error_record)
        if existing_record:
            existing_record.occurrence_count += 1
            existing_record.last_occurrence = error_record.timestamp
            error_record = existing_record
        else:
            self.error_history.append(error_record)
        
        # Update statistics
        self.stats["total_errors"] += 1
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.stats["critical_errors"] += 1
        
        # Update error counts
        error_key = f"{error_record.category.value}_{error_record.error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Maintain history size
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
        
        # Log error
        self._log_error(error_record)
        
        # Call custom handlers
        self._call_error_handlers(error_record)
    
    def _find_similar_error(self, error_record: ErrorRecord) -> Optional[ErrorRecord]:
        """Find similar error in recent history."""
        for existing in reversed(self.error_history[-100:]):  # Check last 100 errors
            if (existing.category == error_record.category and
                existing.component == error_record.component and
                existing.error_type == error_record.error_type and
                existing.operation == error_record.operation):
                return existing
        return None
    
    def _attempt_recovery(self, error_record: ErrorRecord, context: Dict[str, Any]) -> bool:
        """Attempt recovery using available strategies."""
        strategies = self.recovery_strategies.get(error_record.category, [])
        
        for strategy in strategies:
            if strategy.can_attempt():
                self.logger.info(f"Attempting recovery with strategy: {strategy.name}")
                try:
                    if strategy.execute(error_record, context):
                        error_record.recovery_actions.append(f"Recovered with {strategy.name}")
                        error_record.resolution_status = "recovered"
                        self.stats["recovered_errors"] += 1
                        self._update_recovery_success_rate()
                        return True
                    else:
                        error_record.recovery_actions.append(f"Failed recovery with {strategy.name}")
                except Exception as e:
                    self.logger.error(f"Recovery strategy {strategy.name} raised exception: {e}")
                    error_record.recovery_actions.append(f"Recovery exception with {strategy.name}: {e}")
        
        error_record.resolution_status = "unrecovered"
        self._update_recovery_success_rate()
        return False
    
    def _update_recovery_success_rate(self) -> None:
        """Update recovery success rate statistics."""
        if self.stats["total_errors"] > 0:
            self.stats["recovery_success_rate"] = (
                self.stats["recovered_errors"] / self.stats["total_errors"]
            )
    
    def _log_error(self, error_record: ErrorRecord) -> None:
        """Log error to file and console."""
        log_message = (
            f"ERROR [{error_record.severity.value.upper()}] "
            f"{error_record.component}.{error_record.operation}: "
            f"{error_record.error_message}"
        )
        
        if error_record.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Write to error log file if configured
        if self.error_log_path:
            self._write_error_to_file(error_record)
    
    def _write_error_to_file(self, error_record: ErrorRecord) -> None:
        """Write error record to file."""
        try:
            error_data = {
                "error_id": error_record.error_id,
                "timestamp": error_record.timestamp,
                "severity": error_record.severity.value,
                "category": error_record.category.value,
                "component": error_record.component,
                "operation": error_record.operation,
                "error_type": error_record.error_type,
                "error_message": error_record.error_message,
                "occurrence_count": error_record.occurrence_count,
                "recovery_actions": error_record.recovery_actions,
                "resolution_status": error_record.resolution_status
            }
            
            log_path = Path(self.error_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_path, 'a') as f:
                f.write(json.dumps(error_data) + "\n")
                
        except Exception as e:
            self.logger.error(f"Failed to write error to file: {e}")
    
    def _call_error_handlers(self, error_record: ErrorRecord) -> None:
        """Call custom error handlers."""
        handlers = self.error_handlers.get(error_record.category, [])
        
        for handler in handlers:
            try:
                handler(error_record)
            except Exception as e:
                self.logger.error(f"Error handler failed: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        # Update most common errors
        most_common = sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        self.stats["most_common_errors"] = dict(most_common)
        
        return {
            **self.stats,
            "error_queue_size": self.error_queue.qsize(),
            "error_history_size": len(self.error_history),
            "recovery_strategies_count": sum(
                len(strategies) for strategies in self.recovery_strategies.values()
            )
        }
    
    def get_recent_errors(self, count: int = 50, severity: Optional[ErrorSeverity] = None) -> List[ErrorRecord]:
        """Get recent errors, optionally filtered by severity."""
        recent_errors = self.error_history[-count:] if count > 0 else self.error_history
        
        if severity:
            recent_errors = [e for e in recent_errors if e.severity == severity]
        
        return sorted(recent_errors, key=lambda e: e.timestamp, reverse=True)
    
    def export_error_report(self, output_path: str, include_stack_traces: bool = False) -> None:
        """Export comprehensive error report."""
        report_data = {
            "report_timestamp": time.time(),
            "statistics": self.get_error_statistics(),
            "recent_errors": []
        }
        
        for error in self.get_recent_errors(100):
            error_data = {
                "error_id": error.error_id,
                "timestamp": error.timestamp,
                "severity": error.severity.value,
                "category": error.category.value,
                "component": error.component,
                "operation": error.operation,
                "error_type": error.error_type,
                "error_message": error.error_message,
                "occurrence_count": error.occurrence_count,
                "recovery_actions": error.recovery_actions,
                "resolution_status": error.resolution_status
            }
            
            if include_stack_traces:
                error_data["stack_trace"] = error.stack_trace
            
            report_data["recent_errors"].append(error_data)
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Error report exported to {output_path}")
    
    def reset_statistics(self) -> None:
        """Reset error statistics."""
        self.stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "critical_errors": 0,
            "recovery_success_rate": 0.0,
            "most_common_errors": {}
        }
        self.error_counts.clear()
        
        # Reset recovery strategies
        for strategies in self.recovery_strategies.values():
            for strategy in strategies:
                strategy.reset()
    
    def shutdown(self) -> None:
        """Shutdown error handler."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)


# Utility functions and decorators
def with_error_handling(
    component: str,
    operation: str,
    category: ErrorCategory = ErrorCategory.DATA_VALIDATION,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    error_handler: Optional[RobustErrorHandler] = None
):
    """Decorator for automatic error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            handler = error_handler or get_global_error_handler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handled = handler.handle_error(
                    e, component, operation, category, severity,
                    context_data={"args": args, "kwargs": kwargs}
                )
                if not handled:
                    raise
                return None
        
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler: Optional[RobustErrorHandler] = None


def initialize_global_error_handler(**kwargs) -> RobustErrorHandler:
    """Initialize global error handler."""
    global _global_error_handler
    _global_error_handler = RobustErrorHandler(**kwargs)
    return _global_error_handler


def get_global_error_handler() -> RobustErrorHandler:
    """Get global error handler, creating if necessary."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = RobustErrorHandler()
    return _global_error_handler


# Example usage
if __name__ == "__main__":
    import random
    
    # Initialize error handler
    error_handler = RobustErrorHandler(
        error_log_path="error_log.jsonl",
        enable_auto_recovery=True
    )
    
    # Add custom recovery strategy
    def custom_fallback(error_record: ErrorRecord, context: Dict[str, Any]) -> Any:
        print(f"Custom fallback for {error_record.component}")
        return "fallback_result"
    
    error_handler.add_recovery_strategy(
        ErrorCategory.MODEL_INFERENCE,
        FallbackStrategy(custom_fallback)
    )
    
    # Simulate various errors
    for i in range(10):
        try:
            with error_handler.error_context(
                component="test_component",
                operation="test_operation",
                category=ErrorCategory.MODEL_INFERENCE,
                severity=ErrorSeverity.MEDIUM
            ):
                if random.random() < 0.5:
                    raise ValueError(f"Simulated error {i}")
                print(f"Operation {i} succeeded")
        except Exception:
            print(f"Error {i} was handled")
    
    # Print statistics
    stats = error_handler.get_error_statistics()
    print(f"Error Statistics: {stats}")
    
    # Export error report
    error_handler.export_error_report("error_report.json")
    
    # Shutdown
    error_handler.shutdown()