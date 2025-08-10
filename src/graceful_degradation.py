#!/usr/bin/env python3
"""
Graceful Degradation System for IoT Anomaly Detection Platform
Provides fallback mechanisms and service degradation strategies to maintain system availability.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
import threading
import functools

import numpy as np
import pandas as pd

from .logging_config import setup_logging
from .circuit_breaker import CircuitState, circuit_breaker_manager
from .health_monitoring import HealthStatus


class ServiceLevel(Enum):
    """Service degradation levels"""
    FULL = "full"                    # Full functionality
    HIGH = "high"                    # Minor degradation
    MEDIUM = "medium"                # Moderate degradation  
    LOW = "low"                      # Significant degradation
    MINIMAL = "minimal"              # Basic functionality only
    EMERGENCY = "emergency"          # Emergency mode


class DegradationTrigger(Enum):
    """Triggers for service degradation"""
    SYSTEM_LOAD = "system_load"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    CIRCUIT_BREAKER = "circuit_breaker"
    HEALTH_CHECK = "health_check"
    MANUAL = "manual"
    RESOURCE_USAGE = "resource_usage"


@dataclass
class DegradationRule:
    """Rule defining when and how to degrade service"""
    trigger: DegradationTrigger
    threshold: float
    target_level: ServiceLevel
    priority: int = 1  # Lower number = higher priority
    enabled: bool = True
    description: str = ""
    
    def __post_init__(self):
        if not self.description:
            self.description = f"Degrade to {self.target_level.value} when {self.trigger.value} > {self.threshold}"


@dataclass
class ServiceState:
    """Current state of service degradation"""
    current_level: ServiceLevel = ServiceLevel.FULL
    active_triggers: Set[DegradationTrigger] = field(default_factory=set)
    degraded_since: Optional[float] = None
    last_change_time: float = field(default_factory=time.time)
    change_count: int = 0
    forced_level: Optional[ServiceLevel] = None  # Manual override


class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies"""
    
    @abstractmethod
    async def execute(self, original_function: Callable, *args, **kwargs) -> Any:
        """Execute fallback logic"""
        pass
    
    @abstractmethod
    def get_service_level(self) -> ServiceLevel:
        """Get the service level this fallback provides"""
        pass


class CachedResponseFallback(FallbackStrategy):
    """Fallback using cached responses"""
    
    def __init__(self, cache: Dict[str, Any], service_level: ServiceLevel = ServiceLevel.MEDIUM):
        self.cache = cache
        self.service_level = service_level
        self.logger = setup_logging(self.__class__.__name__)
    
    async def execute(self, original_function: Callable, *args, **kwargs) -> Any:
        cache_key = self._generate_cache_key(original_function.__name__, args, kwargs)
        
        if cache_key in self.cache:
            self.logger.info(f"Using cached response for {original_function.__name__}")
            return self.cache[cache_key]
        else:
            raise Exception(f"No cached response available for {original_function.__name__}")
    
    def get_service_level(self) -> ServiceLevel:
        return self.service_level
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments"""
        import hashlib
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()


class SimplifiedModelFallback(FallbackStrategy):
    """Fallback using simplified ML model"""
    
    def __init__(self, service_level: ServiceLevel = ServiceLevel.LOW):
        self.service_level = service_level
        self.logger = setup_logging(self.__class__.__name__)
        self._simple_threshold = 3.0  # Simple threshold-based detection
    
    async def execute(self, original_function: Callable, *args, **kwargs) -> Any:
        """Use simple statistical anomaly detection"""
        self.logger.info("Using simplified anomaly detection fallback")
        
        # Extract data from args/kwargs
        data = None
        if args and isinstance(args[0], (pd.DataFrame, np.ndarray, list)):
            data = args[0]
        elif 'data' in kwargs:
            data = kwargs['data']
        
        if data is None:
            raise Exception("No data provided for simplified anomaly detection")
        
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        elif isinstance(data, list):
            data_array = np.array(data)
        else:
            data_array = data
        
        # Simple statistical anomaly detection using Z-score
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(-1, 1)
        
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)
        z_scores = np.abs((data_array - mean) / (std + 1e-8))
        
        # Flag as anomaly if any feature exceeds threshold
        anomalies = np.any(z_scores > self._simple_threshold, axis=1).astype(int)
        
        return {
            'anomalies': anomalies.tolist(),
            'method': 'simplified_statistical',
            'service_level': self.service_level.value,
            'fallback': True
        }
    
    def get_service_level(self) -> ServiceLevel:
        return self.service_level


class MockResponseFallback(FallbackStrategy):
    """Fallback providing mock/default responses"""
    
    def __init__(self, default_response: Any, service_level: ServiceLevel = ServiceLevel.MINIMAL):
        self.default_response = default_response
        self.service_level = service_level
        self.logger = setup_logging(self.__class__.__name__)
    
    async def execute(self, original_function: Callable, *args, **kwargs) -> Any:
        self.logger.info(f"Using default response for {original_function.__name__}")
        
        if callable(self.default_response):
            return self.default_response(*args, **kwargs)
        else:
            return self.default_response
    
    def get_service_level(self) -> ServiceLevel:
        return self.service_level


class GracefulDegradationManager:
    """Central manager for graceful degradation strategies"""
    
    def __init__(self):
        self.logger = setup_logging(self.__class__.__name__)
        
        # Service state
        self.service_state = ServiceState()
        self._state_lock = threading.Lock()
        
        # Degradation rules
        self.degradation_rules: List[DegradationRule] = []
        self._rules_lock = threading.Lock()
        
        # Fallback strategies by service level
        self.fallback_strategies: Dict[ServiceLevel, List[FallbackStrategy]] = {
            ServiceLevel.FULL: [],
            ServiceLevel.HIGH: [],
            ServiceLevel.MEDIUM: [],
            ServiceLevel.LOW: [],
            ServiceLevel.MINIMAL: [],
            ServiceLevel.EMERGENCY: []
        }
        
        # Metrics collection
        self.metrics: Dict[DegradationTrigger, float] = {}
        self._metrics_lock = threading.Lock()
        
        # Initialize default rules
        self._initialize_default_rules()
        
        # Initialize default fallbacks
        self._initialize_default_fallbacks()
    
    def _initialize_default_rules(self):
        """Initialize default degradation rules"""
        default_rules = [
            DegradationRule(
                trigger=DegradationTrigger.ERROR_RATE,
                threshold=0.20,  # 20% error rate
                target_level=ServiceLevel.HIGH,
                priority=1,
                description="High error rate detected"
            ),
            DegradationRule(
                trigger=DegradationTrigger.ERROR_RATE,
                threshold=0.50,  # 50% error rate  
                target_level=ServiceLevel.LOW,
                priority=2,
                description="Very high error rate detected"
            ),
            DegradationRule(
                trigger=DegradationTrigger.RESPONSE_TIME,
                threshold=5000.0,  # 5 seconds
                target_level=ServiceLevel.MEDIUM,
                priority=3,
                description="High response time detected"
            ),
            DegradationRule(
                trigger=DegradationTrigger.SYSTEM_LOAD,
                threshold=90.0,  # 90% system load
                target_level=ServiceLevel.LOW,
                priority=1,
                description="High system load detected"
            ),
            DegradationRule(
                trigger=DegradationTrigger.RESOURCE_USAGE,
                threshold=95.0,  # 95% resource usage
                target_level=ServiceLevel.EMERGENCY,
                priority=0,
                description="Critical resource usage"
            )
        ]
        
        with self._rules_lock:
            self.degradation_rules.extend(default_rules)
    
    def _initialize_default_fallbacks(self):
        """Initialize default fallback strategies"""
        # Cache-based fallback for medium degradation
        cache_fallback = CachedResponseFallback({}, ServiceLevel.MEDIUM)
        self.fallback_strategies[ServiceLevel.MEDIUM].append(cache_fallback)
        
        # Simplified model for low degradation
        simple_fallback = SimplifiedModelFallback(ServiceLevel.LOW)
        self.fallback_strategies[ServiceLevel.LOW].append(simple_fallback)
        
        # Mock response for minimal/emergency
        mock_fallback = MockResponseFallback(
            default_response={'status': 'degraded', 'anomalies': []},
            service_level=ServiceLevel.MINIMAL
        )
        self.fallback_strategies[ServiceLevel.MINIMAL].append(mock_fallback)
        self.fallback_strategies[ServiceLevel.EMERGENCY].append(mock_fallback)
    
    def add_degradation_rule(self, rule: DegradationRule) -> None:
        """Add new degradation rule"""
        with self._rules_lock:
            self.degradation_rules.append(rule)
            # Sort by priority (lower number = higher priority)
            self.degradation_rules.sort(key=lambda r: r.priority)
        
        self.logger.info(f"Added degradation rule: {rule.description}")
    
    def remove_degradation_rule(self, trigger: DegradationTrigger, threshold: float) -> bool:
        """Remove degradation rule"""
        with self._rules_lock:
            for i, rule in enumerate(self.degradation_rules):
                if rule.trigger == trigger and rule.threshold == threshold:
                    removed_rule = self.degradation_rules.pop(i)
                    self.logger.info(f"Removed degradation rule: {removed_rule.description}")
                    return True
        return False
    
    def add_fallback_strategy(self, level: ServiceLevel, strategy: FallbackStrategy) -> None:
        """Add fallback strategy for service level"""
        self.fallback_strategies[level].append(strategy)
        self.logger.info(f"Added fallback strategy for level {level.value}")
    
    def update_metric(self, trigger: DegradationTrigger, value: float) -> None:
        """Update metric value"""
        with self._metrics_lock:
            self.metrics[trigger] = value
        
        # Check if degradation should be triggered
        self._evaluate_degradation_rules()
    
    def _evaluate_degradation_rules(self) -> None:
        """Evaluate degradation rules against current metrics"""
        with self._rules_lock, self._metrics_lock:
            triggered_rules = []
            
            for rule in self.degradation_rules:
                if not rule.enabled:
                    continue
                    
                metric_value = self.metrics.get(rule.trigger)
                if metric_value is not None and metric_value > rule.threshold:
                    triggered_rules.append(rule)
            
            # Apply highest priority (lowest number) rule
            if triggered_rules:
                highest_priority_rule = min(triggered_rules, key=lambda r: r.priority)
                self._apply_degradation(highest_priority_rule)
            else:
                # No rules triggered, try to restore service
                self._restore_service()
    
    def _apply_degradation(self, rule: DegradationRule) -> None:
        """Apply service degradation based on rule"""
        with self._state_lock:
            # Don't downgrade if manually forced to higher level
            if (self.service_state.forced_level and 
                self.service_state.forced_level.value < rule.target_level.value):
                return
            
            if self.service_state.current_level != rule.target_level:
                previous_level = self.service_state.current_level
                self.service_state.current_level = rule.target_level
                self.service_state.active_triggers.add(rule.trigger)
                self.service_state.last_change_time = time.time()
                self.service_state.change_count += 1
                
                if previous_level == ServiceLevel.FULL:
                    self.service_state.degraded_since = time.time()
                
                self.logger.warning(
                    f"Service degraded from {previous_level.value} to {rule.target_level.value} "
                    f"due to {rule.trigger.value} ({rule.description})"
                )
    
    def _restore_service(self) -> None:
        """Attempt to restore service to full level"""
        with self._state_lock:
            if self.service_state.forced_level:
                # Don't auto-restore if manually overridden
                return
                
            if self.service_state.current_level != ServiceLevel.FULL:
                previous_level = self.service_state.current_level
                self.service_state.current_level = ServiceLevel.FULL
                self.service_state.active_triggers.clear()
                self.service_state.last_change_time = time.time()
                self.service_state.change_count += 1
                self.service_state.degraded_since = None
                
                self.logger.info(f"Service restored from {previous_level.value} to FULL")
    
    def force_service_level(self, level: Optional[ServiceLevel]) -> None:
        """Manually force service to specific level"""
        with self._state_lock:
            previous_forced = self.service_state.forced_level
            previous_level = self.service_state.current_level
            
            self.service_state.forced_level = level
            
            if level:
                self.service_state.current_level = level
                self.service_state.last_change_time = time.time()
                self.service_state.change_count += 1
                
                self.logger.warning(f"Service manually forced to {level.value}")
            else:
                self.logger.info("Manual service level override removed")
                # Re-evaluate rules
                self._evaluate_degradation_rules()
    
    def get_current_service_level(self) -> ServiceLevel:
        """Get current service level"""
        with self._state_lock:
            return self.service_state.current_level
    
    def get_service_state(self) -> ServiceState:
        """Get complete service state"""
        with self._state_lock:
            return ServiceState(
                current_level=self.service_state.current_level,
                active_triggers=self.service_state.active_triggers.copy(),
                degraded_since=self.service_state.degraded_since,
                last_change_time=self.service_state.last_change_time,
                change_count=self.service_state.change_count,
                forced_level=self.service_state.forced_level
            )
    
    async def execute_with_degradation(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> tuple[Any, ServiceLevel]:
        """Execute function with degradation handling"""
        current_level = self.get_current_service_level()
        
        # Try to execute at current service level
        if current_level == ServiceLevel.FULL:
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result, current_level
            except Exception as e:
                # Function failed, try fallback
                self.logger.warning(f"Function {func.__name__} failed: {e}, trying fallback")
                current_level = ServiceLevel.MEDIUM
        
        # Try fallback strategies for current service level
        fallbacks = self.fallback_strategies.get(current_level, [])
        
        for fallback in fallbacks:
            try:
                result = await fallback.execute(func, *args, **kwargs)
                return result, fallback.get_service_level()
            except Exception as e:
                self.logger.warning(f"Fallback {fallback.__class__.__name__} failed: {e}")
                continue
        
        # Try lower service levels
        service_levels = [ServiceLevel.MEDIUM, ServiceLevel.LOW, ServiceLevel.MINIMAL, ServiceLevel.EMERGENCY]
        for level in service_levels:
            if level.value >= current_level.value:
                continue
                
            fallbacks = self.fallback_strategies.get(level, [])
            for fallback in fallbacks:
                try:
                    result = await fallback.execute(func, *args, **kwargs)
                    return result, fallback.get_service_level()
                except Exception as e:
                    continue
        
        # All fallbacks failed
        raise Exception(f"All degradation strategies failed for {func.__name__}")


# Global degradation manager
degradation_manager = GracefulDegradationManager()


def with_graceful_degradation(func: Callable) -> Callable:
    """Decorator to add graceful degradation to functions"""
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            result, service_level = await degradation_manager.execute_with_degradation(
                func, *args, **kwargs
            )
            return result
        except Exception as e:
            degradation_manager.logger.error(f"Function {func.__name__} failed completely: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            # Convert sync to async for consistent handling
            async def async_func(*args, **kwargs):
                return func(*args, **kwargs)
            
            loop = asyncio.get_event_loop()
            result, service_level = loop.run_until_complete(
                degradation_manager.execute_with_degradation(async_func, *args, **kwargs)
            )
            return result
        except Exception as e:
            degradation_manager.logger.error(f"Function {func.__name__} failed completely: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Example usage
@with_graceful_degradation
async def anomaly_detection_service(data: pd.DataFrame) -> dict:
    """Example anomaly detection service with graceful degradation"""
    # Simulate complex ML processing
    import random
    if random.random() < 0.3:  # 30% failure rate
        raise Exception("ML model temporarily unavailable")
    
    # Normal processing
    return {
        'anomalies': [0, 1, 0, 1, 0],
        'confidence': 0.95,
        'model': 'advanced_lstm_autoencoder'
    }


async def main():
    """Example usage of graceful degradation system"""
    logger = setup_logging("degradation_example")
    
    # Update some metrics to trigger degradation
    degradation_manager.update_metric(DegradationTrigger.ERROR_RATE, 0.25)
    degradation_manager.update_metric(DegradationTrigger.RESPONSE_TIME, 6000)
    
    # Test service with degradation
    test_data = pd.DataFrame({'sensor1': [1, 2, 3, 4, 5], 'sensor2': [2, 3, 4, 5, 6]})
    
    try:
        result = await anomaly_detection_service(test_data)
        logger.info(f"Service result: {result}")
    except Exception as e:
        logger.error(f"Service failed: {e}")
    
    # Check service state
    state = degradation_manager.get_service_state()
    logger.info(f"Current service level: {state.current_level.value}")
    logger.info(f"Active triggers: {[t.value for t in state.active_triggers]}")


if __name__ == "__main__":
    asyncio.run(main())