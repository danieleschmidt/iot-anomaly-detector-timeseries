#!/usr/bin/env python3
"""
Circuit Breaker Pattern Implementation for IoT Anomaly Detection System
Provides fault tolerance and system stability through intelligent failure handling.
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, Future
import functools

from .logging_config import setup_logging


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failure state, requests fail fast
    HALF_OPEN = "half_open" # Testing state, limited requests allowed


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5          # Failures to trigger open state
    success_threshold: int = 3          # Successes to close from half-open
    timeout_seconds: float = 60.0       # Time to wait before half-open
    max_retry_delay: float = 300.0      # Maximum delay between retries
    exponential_backoff_multiplier: float = 2.0
    jitter_max_seconds: float = 1.0     # Random jitter for retry timing


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics and statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    current_state: CircuitState = CircuitState.CLOSED
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return 1.0 - self.failure_rate


class CircuitBreakerError(Exception):
    """Circuit breaker specific exception"""
    pass


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open"""
    def __init__(self, service_name: str, next_attempt_time: float):
        self.service_name = service_name
        self.next_attempt_time = next_attempt_time
        super().__init__(
            f"Circuit breaker for '{service_name}' is OPEN. "
            f"Next attempt at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(next_attempt_time))}"
        )


class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies"""
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute fallback logic"""
        pass


class CacheFallback(FallbackStrategy):
    """Fallback strategy using cached responses"""
    
    def __init__(self, cache: Dict[str, Any]):
        self.cache = cache
        self.logger = setup_logging(self.__class__.__name__)
    
    def execute(self, cache_key: str, *args, **kwargs) -> Any:
        """Return cached response if available"""
        if cache_key in self.cache:
            self.logger.info(f"Using cached fallback for key: {cache_key}")
            return self.cache[cache_key]
        else:
            raise CircuitBreakerError(f"No cached fallback available for key: {cache_key}")


class DefaultValueFallback(FallbackStrategy):
    """Fallback strategy returning default value"""
    
    def __init__(self, default_value: Any):
        self.default_value = default_value
        self.logger = setup_logging(self.__class__.__name__)
    
    def execute(self, *args, **kwargs) -> Any:
        """Return default value"""
        self.logger.info(f"Using default value fallback: {self.default_value}")
        return self.default_value


class DegradedServiceFallback(FallbackStrategy):
    """Fallback strategy using simplified/degraded service"""
    
    def __init__(self, fallback_function: Callable):
        self.fallback_function = fallback_function
        self.logger = setup_logging(self.__class__.__name__)
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute degraded service function"""
        self.logger.info("Using degraded service fallback")
        return self.fallback_function(*args, **kwargs)


class CircuitBreaker:
    """
    Circuit breaker implementation with multiple failure detection strategies
    and fallback mechanisms.
    """
    
    def __init__(
        self, 
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[FallbackStrategy] = None,
        allowed_exceptions: Optional[List[type]] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback
        self.allowed_exceptions = allowed_exceptions or []
        
        self.metrics = CircuitBreakerMetrics()
        self.state = CircuitState.CLOSED
        self.next_attempt_time = 0.0
        
        self.logger = setup_logging(f"CircuitBreaker.{name}")
        self._lock = threading.Lock()
        
        # For async operations
        self._executor = ThreadPoolExecutor(max_workers=10)
    
    def _should_attempt_request(self) -> bool:
        """Check if request should be attempted based on current state"""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if current_time >= self.next_attempt_time:
                self._transition_to_half_open()
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _record_success(self) -> None:
        """Record successful request"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    self._transition_to_closed()
    
    def _record_failure(self, exception: Exception) -> None:
        """Record failed request"""
        with self._lock:
            # Check if exception should be ignored
            if any(isinstance(exception, exc_type) for exc_type in self.allowed_exceptions):
                return
                
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = time.time()
            
            if (self.state == CircuitState.CLOSED and 
                self.metrics.consecutive_failures >= self.config.failure_threshold):
                self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
    
    def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state"""
        if self.state != CircuitState.OPEN:
            previous_state = self.state
            self.state = CircuitState.OPEN
            self.metrics.state_changes += 1
            
            # Calculate next attempt time with exponential backoff and jitter
            backoff_delay = min(
                self.config.timeout_seconds * (
                    self.config.exponential_backoff_multiplier ** 
                    min(self.metrics.consecutive_failures - self.config.failure_threshold, 10)
                ),
                self.config.max_retry_delay
            )
            
            # Add jitter to prevent thundering herd
            import random
            jitter = random.uniform(0, self.config.jitter_max_seconds)
            self.next_attempt_time = time.time() + backoff_delay + jitter
            
            self.logger.warning(
                f"Circuit breaker '{self.name}' transitioned from {previous_state.value} to OPEN. "
                f"Next attempt at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.next_attempt_time))}"
            )
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state"""
        if self.state != CircuitState.HALF_OPEN:
            previous_state = self.state
            self.state = CircuitState.HALF_OPEN
            self.metrics.state_changes += 1
            self.metrics.consecutive_successes = 0
            
            self.logger.info(
                f"Circuit breaker '{self.name}' transitioned from {previous_state.value} to HALF_OPEN"
            )
    
    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state"""
        if self.state != CircuitState.CLOSED:
            previous_state = self.state
            self.state = CircuitState.CLOSED
            self.metrics.state_changes += 1
            self.metrics.consecutive_failures = 0
            
            self.logger.info(
                f"Circuit breaker '{self.name}' transitioned from {previous_state.value} to CLOSED"
            )
    
    @contextmanager
    def _execute_with_circuit_breaker(self):
        """Context manager for circuit breaker execution"""
        if not self._should_attempt_request():
            if self.fallback:
                try:
                    yield self.fallback
                    return
                except Exception as fallback_error:
                    self.logger.error(f"Fallback failed: {fallback_error}")
                    
            raise CircuitBreakerOpenError(self.name, self.next_attempt_time)
        
        try:
            yield None
        except Exception as e:
            self._record_failure(e)
            raise
        else:
            self._record_success()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for wrapping functions with circuit breaker"""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self.call_async(func, *args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._execute_with_circuit_breaker() as fallback:
            if fallback:
                return fallback.execute(*args, **kwargs)
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                self.logger.error(f"Function {func.__name__} failed: {e}")
                raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        if not self._should_attempt_request():
            if self.fallback:
                try:
                    return await self._run_fallback_async(*args, **kwargs)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback failed: {fallback_error}")
                    
            raise CircuitBreakerOpenError(self.name, self.next_attempt_time)
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self._executor, func, *args, **kwargs)
            
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure(e)
            self.logger.error(f"Async function {func.__name__} failed: {e}")
            raise
    
    async def _run_fallback_async(self, *args, **kwargs) -> Any:
        """Run fallback strategy asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            self.fallback.execute, 
            *args, 
            **kwargs
        )
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current circuit breaker metrics"""
        with self._lock:
            metrics = CircuitBreakerMetrics(
                total_requests=self.metrics.total_requests,
                successful_requests=self.metrics.successful_requests,
                failed_requests=self.metrics.failed_requests,
                consecutive_failures=self.metrics.consecutive_failures,
                consecutive_successes=self.metrics.consecutive_successes,
                state_changes=self.metrics.state_changes,
                last_failure_time=self.metrics.last_failure_time,
                last_success_time=self.metrics.last_success_time,
                current_state=self.state
            )
            return metrics
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.metrics = CircuitBreakerMetrics()
            self.next_attempt_time = 0.0
            
            self.logger.info(f"Circuit breaker '{self.name}' reset to initial state")


class CircuitBreakerManager:
    """Centralized management of circuit breakers"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = setup_logging(self.__class__.__name__)
    
    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[FallbackStrategy] = None,
        allowed_exceptions: Optional[List[type]] = None
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                config=config,
                fallback=fallback,
                allowed_exceptions=allowed_exceptions
            )
            self.logger.info(f"Created new circuit breaker: {name}")
        
        return self.circuit_breakers[name]
    
    def get_all_metrics(self) -> Dict[str, CircuitBreakerMetrics]:
        """Get metrics for all circuit breakers"""
        return {
            name: cb.get_metrics() 
            for name, cb in self.circuit_breakers.items()
        }
    
    def reset_all(self) -> None:
        """Reset all circuit breakers"""
        for cb in self.circuit_breakers.values():
            cb.reset()
        self.logger.info("Reset all circuit breakers")


# Global circuit breaker manager instance
circuit_breaker_manager = CircuitBreakerManager()


# Convenience decorators
def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 3,
    timeout_seconds: float = 60.0,
    fallback: Optional[FallbackStrategy] = None,
    allowed_exceptions: Optional[List[type]] = None
):
    """Decorator for adding circuit breaker to functions"""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        timeout_seconds=timeout_seconds
    )
    
    cb = circuit_breaker_manager.get_or_create(
        name=name,
        config=config,
        fallback=fallback,
        allowed_exceptions=allowed_exceptions
    )
    
    return cb


# Example usage functions
async def example_usage():
    """Example of circuit breaker usage"""
    
    # Create cache fallback
    cache = {"test_key": "cached_response"}
    cache_fallback = CacheFallback(cache)
    
    # Create circuit breaker with fallback
    @circuit_breaker(
        name="example_service",
        failure_threshold=3,
        timeout_seconds=30.0,
        fallback=cache_fallback
    )
    def unreliable_service(cache_key: str) -> str:
        import random
        if random.random() < 0.7:  # 70% failure rate for testing
            raise Exception("Service temporarily unavailable")
        return f"Success response for {cache_key}"
    
    # Test the circuit breaker
    logger = setup_logging("example")
    
    for i in range(10):
        try:
            result = unreliable_service("test_key")
            logger.info(f"Request {i}: {result}")
        except Exception as e:
            logger.error(f"Request {i} failed: {e}")
        
        await asyncio.sleep(1)
    
    # Check metrics
    metrics = circuit_breaker_manager.get_all_metrics()
    for name, metric in metrics.items():
        logger.info(f"Circuit breaker '{name}' metrics: {metric}")


if __name__ == "__main__":
    asyncio.run(example_usage())