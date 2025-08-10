#!/usr/bin/env python3
"""
Intelligent Retry Management System for IoT Anomaly Detection Platform
Provides sophisticated retry mechanisms with exponential backoff, jitter, and adaptive strategies.
"""

import asyncio
import functools
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import threading

from .logging_config import setup_logging


class RetryStrategy(Enum):
    """Available retry strategies"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    ADAPTIVE = "adaptive"


@dataclass
class RetryConfig:
    """Retry configuration parameters"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    backoff_multiplier: float = 2.0
    jitter_max: float = 1.0
    jitter_enabled: bool = True
    retry_on_exceptions: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    stop_on_exceptions: List[Type[Exception]] = field(default_factory=list)
    circuit_breaker_name: Optional[str] = None


@dataclass
class RetryAttempt:
    """Information about a retry attempt"""
    attempt_number: int
    delay: float
    exception: Optional[Exception]
    timestamp: float
    total_elapsed: float


@dataclass
class RetryResult:
    """Result of retry operation"""
    success: bool
    result: Any
    attempts: List[RetryAttempt]
    total_attempts: int
    total_time: float
    final_exception: Optional[Exception]


class BackoffStrategy(ABC):
    """Abstract base class for backoff strategies"""
    
    @abstractmethod
    def calculate_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        """Calculate delay for given attempt"""
        pass


class FixedDelayStrategy(BackoffStrategy):
    """Fixed delay between retries"""
    
    def calculate_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        return min(base_delay, max_delay)


class ExponentialBackoffStrategy(BackoffStrategy):
    """Exponential backoff strategy"""
    
    def __init__(self, multiplier: float = 2.0):
        self.multiplier = multiplier
    
    def calculate_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        delay = base_delay * (self.multiplier ** (attempt - 1))
        return min(delay, max_delay)


class LinearBackoffStrategy(BackoffStrategy):
    """Linear backoff strategy"""
    
    def calculate_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        delay = base_delay * attempt
        return min(delay, max_delay)


class FibonacciBackoffStrategy(BackoffStrategy):
    """Fibonacci sequence backoff strategy"""
    
    def __init__(self):
        self._fib_cache = {0: 0, 1: 1}
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number with caching"""
        if n not in self._fib_cache:
            self._fib_cache[n] = self._fibonacci(n-1) + self._fibonacci(n-2)
        return self._fib_cache[n]
    
    def calculate_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        fib_multiplier = self._fibonacci(attempt)
        delay = base_delay * fib_multiplier
        return min(delay, max_delay)


class AdaptiveBackoffStrategy(BackoffStrategy):
    """Adaptive backoff that learns from success/failure patterns"""
    
    def __init__(self):
        self.success_history: List[float] = []
        self.failure_history: List[float] = []
        self._lock = threading.Lock()
    
    def record_success(self, delay: float) -> None:
        """Record successful retry with delay"""
        with self._lock:
            self.success_history.append(delay)
            # Keep only recent history
            if len(self.success_history) > 100:
                self.success_history = self.success_history[-50:]
    
    def record_failure(self, delay: float) -> None:
        """Record failed retry with delay"""
        with self._lock:
            self.failure_history.append(delay)
            # Keep only recent history
            if len(self.failure_history) > 100:
                self.failure_history = self.failure_history[-50:]
    
    def calculate_delay(self, attempt: int, base_delay: float, max_delay: float) -> float:
        with self._lock:
            if not self.success_history and not self.failure_history:
                # No history, use exponential backoff
                delay = base_delay * (2.0 ** (attempt - 1))
            else:
                # Calculate optimal delay based on success history
                if self.success_history:
                    avg_success_delay = sum(self.success_history) / len(self.success_history)
                    # Slightly increase delay based on attempt number
                    delay = avg_success_delay * (1.0 + 0.5 * (attempt - 1))
                else:
                    # Only failures, increase delay more aggressively
                    delay = base_delay * (3.0 ** (attempt - 1))
            
            return min(delay, max_delay)


class RetryManager:
    """Intelligent retry manager with multiple strategies and adaptive behavior"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.logger = setup_logging(self.__class__.__name__)
        
        # Initialize backoff strategy
        self.backoff_strategy = self._create_backoff_strategy()
        
        # Metrics tracking
        self.retry_stats: Dict[str, Dict[str, Any]] = {}
        self._stats_lock = threading.Lock()
    
    def _create_backoff_strategy(self) -> BackoffStrategy:
        """Create appropriate backoff strategy based on config"""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            return FixedDelayStrategy()
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return ExponentialBackoffStrategy(self.config.backoff_multiplier)
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            return LinearBackoffStrategy()
        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            return FibonacciBackoffStrategy()
        elif self.config.strategy == RetryStrategy.ADAPTIVE:
            return AdaptiveBackoffStrategy()
        else:
            return ExponentialBackoffStrategy()
    
    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if retry should be attempted"""
        # Check max attempts
        if attempt >= self.config.max_attempts:
            return False
        
        # Check stop conditions
        if any(isinstance(exception, exc_type) for exc_type in self.config.stop_on_exceptions):
            return False
        
        # Check retry conditions
        if not any(isinstance(exception, exc_type) for exc_type in self.config.retry_on_exceptions):
            return False
        
        return True
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with jitter"""
        base_delay = self.backoff_strategy.calculate_delay(
            attempt, self.config.base_delay, self.config.max_delay
        )
        
        # Add jitter if enabled
        if self.config.jitter_enabled and self.config.jitter_max > 0:
            jitter = random.uniform(0, self.config.jitter_max)
            delay = base_delay + jitter
        else:
            delay = base_delay
        
        return min(delay, self.config.max_delay)
    
    def _record_attempt(
        self, 
        function_name: str, 
        attempt: int, 
        success: bool, 
        delay: float, 
        exception: Optional[Exception] = None
    ) -> None:
        """Record retry attempt for statistics"""
        with self._stats_lock:
            if function_name not in self.retry_stats:
                self.retry_stats[function_name] = {
                    'total_calls': 0,
                    'total_retries': 0,
                    'success_rate': 0.0,
                    'avg_attempts': 0.0,
                    'max_attempts': 0,
                    'total_delay': 0.0,
                    'last_updated': time.time()
                }
            
            stats = self.retry_stats[function_name]
            stats['total_calls'] += 1 if attempt == 1 else 0
            stats['total_retries'] += 1 if attempt > 1 else 0
            stats['max_attempts'] = max(stats['max_attempts'], attempt)
            stats['total_delay'] += delay
            stats['last_updated'] = time.time()
            
            # Update success rate (simplified calculation)
            if success:
                current_success = stats.get('successful_calls', 0)
                stats['successful_calls'] = current_success + 1
                stats['success_rate'] = stats['successful_calls'] / stats['total_calls']
    
    def retry(self, func: Callable, *args, **kwargs) -> RetryResult:
        """Execute function with retry logic"""
        function_name = getattr(func, '__name__', 'unknown')
        attempts: List[RetryAttempt] = []
        start_time = time.time()
        
        for attempt_num in range(1, self.config.max_attempts + 1):
            attempt_start = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record successful attempt
                elapsed = time.time() - start_time
                attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    delay=0.0,
                    exception=None,
                    timestamp=attempt_start,
                    total_elapsed=elapsed
                )
                attempts.append(attempt)
                
                # Update adaptive strategy if applicable
                if isinstance(self.backoff_strategy, AdaptiveBackoffStrategy) and attempt_num > 1:
                    # Record success with the delay that worked
                    prev_delay = attempts[-2].delay if len(attempts) > 1 else 0
                    self.backoff_strategy.record_success(prev_delay)
                
                self._record_attempt(function_name, attempt_num, True, 0.0)
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_attempts=attempt_num,
                    total_time=elapsed,
                    final_exception=None
                )
                
            except Exception as e:
                elapsed = time.time() - start_time
                
                # Check if we should retry
                if not self._should_retry(e, attempt_num):
                    attempt = RetryAttempt(
                        attempt_number=attempt_num,
                        delay=0.0,
                        exception=e,
                        timestamp=attempt_start,
                        total_elapsed=elapsed
                    )
                    attempts.append(attempt)
                    
                    self._record_attempt(function_name, attempt_num, False, 0.0, e)
                    
                    return RetryResult(
                        success=False,
                        result=None,
                        attempts=attempts,
                        total_attempts=attempt_num,
                        total_time=elapsed,
                        final_exception=e
                    )
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt_num) if attempt_num < self.config.max_attempts else 0.0
                
                attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    delay=delay,
                    exception=e,
                    timestamp=attempt_start,
                    total_elapsed=elapsed
                )
                attempts.append(attempt)
                
                # Log retry attempt
                self.logger.warning(
                    f"Attempt {attempt_num} failed for {function_name}: {e}. "
                    f"{'Retrying in ' + str(delay) + 's' if delay > 0 else 'No more retries'}"
                )
                
                # Update adaptive strategy
                if isinstance(self.backoff_strategy, AdaptiveBackoffStrategy):
                    self.backoff_strategy.record_failure(delay)
                
                self._record_attempt(function_name, attempt_num, False, delay, e)
                
                # Sleep before next attempt
                if delay > 0:
                    time.sleep(delay)
        
        # All attempts failed
        final_elapsed = time.time() - start_time
        return RetryResult(
            success=False,
            result=None,
            attempts=attempts,
            total_attempts=len(attempts),
            total_time=final_elapsed,
            final_exception=attempts[-1].exception if attempts else None
        )
    
    async def retry_async(self, func: Callable, *args, **kwargs) -> RetryResult:
        """Execute async function with retry logic"""
        function_name = getattr(func, '__name__', 'unknown')
        attempts: List[RetryAttempt] = []
        start_time = time.time()
        
        for attempt_num in range(1, self.config.max_attempts + 1):
            attempt_start = time.time()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record successful attempt
                elapsed = time.time() - start_time
                attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    delay=0.0,
                    exception=None,
                    timestamp=attempt_start,
                    total_elapsed=elapsed
                )
                attempts.append(attempt)
                
                self._record_attempt(function_name, attempt_num, True, 0.0)
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_attempts=attempt_num,
                    total_time=elapsed,
                    final_exception=None
                )
                
            except Exception as e:
                elapsed = time.time() - start_time
                
                # Check if we should retry
                if not self._should_retry(e, attempt_num):
                    attempt = RetryAttempt(
                        attempt_number=attempt_num,
                        delay=0.0,
                        exception=e,
                        timestamp=attempt_start,
                        total_elapsed=elapsed
                    )
                    attempts.append(attempt)
                    
                    self._record_attempt(function_name, attempt_num, False, 0.0, e)
                    
                    return RetryResult(
                        success=False,
                        result=None,
                        attempts=attempts,
                        total_attempts=attempt_num,
                        total_time=elapsed,
                        final_exception=e
                    )
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt_num) if attempt_num < self.config.max_attempts else 0.0
                
                attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    delay=delay,
                    exception=e,
                    timestamp=attempt_start,
                    total_elapsed=elapsed
                )
                attempts.append(attempt)
                
                self.logger.warning(
                    f"Async attempt {attempt_num} failed for {function_name}: {e}. "
                    f"{'Retrying in ' + str(delay) + 's' if delay > 0 else 'No more retries'}"
                )
                
                self._record_attempt(function_name, attempt_num, False, delay, e)
                
                # Sleep before next attempt
                if delay > 0:
                    await asyncio.sleep(delay)
        
        # All attempts failed
        final_elapsed = time.time() - start_time
        return RetryResult(
            success=False,
            result=None,
            attempts=attempts,
            total_attempts=len(attempts),
            total_time=final_elapsed,
            final_exception=attempts[-1].exception if attempts else None
        )
    
    def get_stats(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Get retry statistics"""
        with self._stats_lock:
            if function_name:
                return self.retry_stats.get(function_name, {})
            return self.retry_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset retry statistics"""
        with self._stats_lock:
            self.retry_stats.clear()


# Decorator for adding retry logic
def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    backoff_multiplier: float = 2.0,
    jitter_enabled: bool = True,
    retry_on_exceptions: Optional[List[Type[Exception]]] = None,
    stop_on_exceptions: Optional[List[Type[Exception]]] = None
):
    """Decorator for adding intelligent retry logic to functions"""
    
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=strategy,
        backoff_multiplier=backoff_multiplier,
        jitter_enabled=jitter_enabled,
        retry_on_exceptions=retry_on_exceptions or [Exception],
        stop_on_exceptions=stop_on_exceptions or []
    )
    
    retry_manager = RetryManager(config)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = retry_manager.retry(func, *args, **kwargs)
            if result.success:
                return result.result
            else:
                raise result.final_exception
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await retry_manager.retry_async(func, *args, **kwargs)
            if result.success:
                return result.result
            else:
                raise result.final_exception
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Example usage
@retry(
    max_attempts=5,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    backoff_multiplier=2.0,
    retry_on_exceptions=[ConnectionError, TimeoutError],
    stop_on_exceptions=[ValueError]
)
def unreliable_api_call(data: str) -> str:
    """Example function that might fail"""
    import random
    if random.random() < 0.7:  # 70% failure rate
        raise ConnectionError("Network error")
    return f"Success: {data}"


@retry(max_attempts=3, strategy=RetryStrategy.ADAPTIVE)
async def unreliable_async_call(data: str) -> str:
    """Example async function that might fail"""
    import random
    if random.random() < 0.6:  # 60% failure rate
        raise TimeoutError("Request timeout")
    return f"Async success: {data}"


async def main():
    """Example usage of retry mechanisms"""
    logger = setup_logging("retry_example")
    
    # Test sync retry
    try:
        result = unreliable_api_call("test_data")
        logger.info(f"Sync result: {result}")
    except Exception as e:
        logger.error(f"Sync call failed: {e}")
    
    # Test async retry
    try:
        result = await unreliable_async_call("async_test_data")
        logger.info(f"Async result: {result}")
    except Exception as e:
        logger.error(f"Async call failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())