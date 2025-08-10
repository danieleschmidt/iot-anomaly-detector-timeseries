#!/usr/bin/env python3
"""
Intelligent Auto-Scaling System for IoT Anomaly Detection Platform
Provides dynamic resource scaling based on load, performance metrics, and predictive analytics.
"""

import asyncio
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import statistics
import subprocess
import sys
import os
import math

import numpy as np
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .logging_config import setup_logging
from .health_monitoring import HealthMonitor, HealthStatus


class ScalingDirection(Enum):
    """Scaling directions"""
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


class ResourceType(Enum):
    """Types of resources that can be scaled"""
    WORKER_PROCESSES = "worker_processes"
    THREAD_POOL = "thread_pool"
    MODEL_INSTANCES = "model_instances"
    CACHE_SIZE = "cache_size"
    BATCH_SIZE = "batch_size"
    CONNECTION_POOL = "connection_pool"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_rate: float = 0.0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    queue_length: int = 0
    active_connections: int = 0
    throughput_rps: float = 0.0
    prediction_latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingRule:
    """Rule for automatic scaling"""
    resource_type: ResourceType
    metric_name: str
    threshold_up: float
    threshold_down: float
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    min_instances: int = 1
    max_instances: int = 10
    cooldown_seconds: int = 300
    evaluation_period_seconds: int = 60
    enabled: bool = True
    priority: int = 1
    
    def __post_init__(self):
        if self.threshold_up <= self.threshold_down:
            raise ValueError("threshold_up must be greater than threshold_down")


@dataclass
class ScalingAction:
    """Scaling action to be executed"""
    resource_type: ResourceType
    direction: ScalingDirection
    current_instances: int
    target_instances: int
    reason: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def scale_factor(self) -> float:
        """Calculate scaling factor"""
        if self.current_instances == 0:
            return float('inf')
        return self.target_instances / self.current_instances


class ResourceManager(ABC):
    """Abstract base class for resource managers"""
    
    @abstractmethod
    async def get_current_instances(self) -> int:
        """Get current number of instances"""
        pass
    
    @abstractmethod
    async def scale_to(self, target_instances: int) -> bool:
        """Scale to target number of instances"""
        pass
    
    @abstractmethod
    async def get_resource_metrics(self) -> Dict[str, float]:
        """Get current resource metrics"""
        pass


class ProcessPoolManager(ResourceManager):
    """Manager for process pool scaling"""
    
    def __init__(self, initial_workers: int = 4, max_workers: int = 16):
        self.max_workers = max_workers
        self.executor: Optional[ProcessPoolExecutor] = None
        self.current_workers = initial_workers
        self.lock = asyncio.Lock()
        self.logger = setup_logging(self.__class__.__name__)
        
        # Initialize executor
        self._initialize_executor()
    
    def _initialize_executor(self):
        """Initialize process pool executor"""
        if self.executor:
            self.executor.shutdown(wait=False)
        
        self.executor = ProcessPoolExecutor(
            max_workers=self.current_workers,
            mp_context=None
        )
        
        self.logger.info(f"Initialized process pool with {self.current_workers} workers")
    
    async def get_current_instances(self) -> int:
        """Get current number of worker processes"""
        return self.current_workers
    
    async def scale_to(self, target_instances: int) -> bool:
        """Scale process pool to target number of workers"""
        async with self.lock:
            target_instances = max(1, min(target_instances, self.max_workers))
            
            if target_instances == self.current_workers:
                return True
            
            try:
                # Shutdown current executor
                if self.executor:
                    self.executor.shutdown(wait=True)
                
                # Create new executor with target workers
                self.current_workers = target_instances
                self.executor = ProcessPoolExecutor(
                    max_workers=self.current_workers,
                    mp_context=None
                )
                
                self.logger.info(f"Scaled process pool to {target_instances} workers")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to scale process pool: {e}")
                return False
    
    async def get_resource_metrics(self) -> Dict[str, float]:
        """Get process pool resource metrics"""
        return {
            'worker_count': self.current_workers,
            'max_workers': self.max_workers,
            'utilization': self.current_workers / self.max_workers
        }


class ThreadPoolManager(ResourceManager):
    """Manager for thread pool scaling"""
    
    def __init__(self, initial_workers: int = 8, max_workers: int = 32):
        self.max_workers = max_workers
        self.executor: Optional[ThreadPoolExecutor] = None
        self.current_workers = initial_workers
        self.lock = asyncio.Lock()
        self.logger = setup_logging(self.__class__.__name__)
        
        # Initialize executor
        self._initialize_executor()
    
    def _initialize_executor(self):
        """Initialize thread pool executor"""
        if self.executor:
            self.executor.shutdown(wait=False)
        
        self.executor = ThreadPoolExecutor(
            max_workers=self.current_workers,
            thread_name_prefix='worker'
        )
        
        self.logger.info(f"Initialized thread pool with {self.current_workers} workers")
    
    async def get_current_instances(self) -> int:
        """Get current number of worker threads"""
        return self.current_workers
    
    async def scale_to(self, target_instances: int) -> bool:
        """Scale thread pool to target number of workers"""
        async with self.lock:
            target_instances = max(1, min(target_instances, self.max_workers))
            
            if target_instances == self.current_workers:
                return True
            
            try:
                # Shutdown current executor
                if self.executor:
                    self.executor.shutdown(wait=True)
                
                # Create new executor with target workers
                self.current_workers = target_instances
                self.executor = ThreadPoolExecutor(
                    max_workers=self.current_workers,
                    thread_name_prefix='worker'
                )
                
                self.logger.info(f"Scaled thread pool to {target_instances} workers")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to scale thread pool: {e}")
                return False
    
    async def get_resource_metrics(self) -> Dict[str, float]:
        """Get thread pool resource metrics"""
        return {
            'worker_count': self.current_workers,
            'max_workers': self.max_workers,
            'utilization': self.current_workers / self.max_workers
        }


class ModelInstanceManager(ResourceManager):
    """Manager for ML model instance scaling"""
    
    def __init__(self, model_loader: Callable, initial_instances: int = 2, 
                 max_instances: int = 8):
        self.model_loader = model_loader
        self.max_instances = max_instances
        self.model_instances = []
        self.current_instances = 0
        self.lock = asyncio.Lock()
        self.logger = setup_logging(self.__class__.__name__)
        
        # Initialize instances
        self._initialize_instances(initial_instances)
    
    def _initialize_instances(self, count: int):
        """Initialize model instances"""
        for _ in range(count):
            try:
                model = self.model_loader()
                self.model_instances.append(model)
                self.current_instances += 1
            except Exception as e:
                self.logger.error(f"Failed to load model instance: {e}")
        
        self.logger.info(f"Initialized {self.current_instances} model instances")
    
    async def get_current_instances(self) -> int:
        """Get current number of model instances"""
        return self.current_instances
    
    async def scale_to(self, target_instances: int) -> bool:
        """Scale model instances to target count"""
        async with self.lock:
            target_instances = max(1, min(target_instances, self.max_instances))
            
            if target_instances == self.current_instances:
                return True
            
            try:
                if target_instances > self.current_instances:
                    # Scale up - add instances
                    instances_to_add = target_instances - self.current_instances
                    for _ in range(instances_to_add):
                        model = self.model_loader()
                        self.model_instances.append(model)
                        self.current_instances += 1
                    
                elif target_instances < self.current_instances:
                    # Scale down - remove instances
                    instances_to_remove = self.current_instances - target_instances
                    for _ in range(instances_to_remove):
                        if self.model_instances:
                            self.model_instances.pop()
                            self.current_instances -= 1
                
                self.logger.info(f"Scaled model instances to {target_instances}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to scale model instances: {e}")
                return False
    
    def get_available_model(self):
        """Get an available model instance"""
        if self.model_instances:
            return self.model_instances[0]  # Simple round-robin
        return None
    
    async def get_resource_metrics(self) -> Dict[str, float]:
        """Get model instance metrics"""
        return {
            'instance_count': self.current_instances,
            'max_instances': self.max_instances,
            'utilization': self.current_instances / self.max_instances
        }


class PredictiveScaler:
    """Predictive scaling based on historical patterns"""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.metric_history: Dict[str, List[Tuple[float, float]]] = {}  # timestamp, value
        self.prediction_horizon = 300  # 5 minutes
        self.logger = setup_logging(self.__class__.__name__)
    
    def record_metric(self, metric_name: str, value: float, timestamp: float = None):
        """Record metric value for prediction"""
        if timestamp is None:
            timestamp = time.time()
        
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        self.metric_history[metric_name].append((timestamp, value))
        
        # Keep only recent history
        if len(self.metric_history[metric_name]) > self.history_length:
            self.metric_history[metric_name] = self.metric_history[metric_name][-self.history_length:]
    
    def predict_metric(self, metric_name: str, horizon_seconds: int = None) -> Optional[float]:
        """Predict future metric value using simple linear trend"""
        if horizon_seconds is None:
            horizon_seconds = self.prediction_horizon
        
        if (metric_name not in self.metric_history or 
            len(self.metric_history[metric_name]) < 10):
            return None
        
        history = self.metric_history[metric_name]
        
        # Extract timestamps and values
        timestamps = np.array([t for t, v in history])
        values = np.array([v for t, v in history])
        
        # Simple linear regression for trend
        try:
            # Calculate trend
            time_diffs = timestamps - timestamps[0]
            slope, intercept = np.polyfit(time_diffs, values, 1)
            
            # Predict future value
            current_time = time.time()
            future_time_diff = (current_time + horizon_seconds) - timestamps[0]
            predicted_value = slope * future_time_diff + intercept
            
            # Apply bounds (metric values should be reasonable)
            predicted_value = max(0, min(predicted_value, values.max() * 2))
            
            return predicted_value
            
        except Exception as e:
            self.logger.error(f"Error predicting {metric_name}: {e}")
            return None
    
    def get_scaling_recommendation(
        self, 
        current_metrics: ScalingMetrics, 
        rules: List[ScalingRule]
    ) -> Optional[ScalingDirection]:
        """Get scaling recommendation based on predictions"""
        for rule in rules:
            if not rule.enabled:
                continue
            
            # Get current metric value
            current_value = getattr(current_metrics, rule.metric_name, 0)
            
            # Predict future value
            predicted_value = self.predict_metric(rule.metric_name)
            
            if predicted_value is not None:
                # Use predicted value for scaling decision
                if predicted_value > rule.threshold_up:
                    return ScalingDirection.UP
                elif predicted_value < rule.threshold_down:
                    return ScalingDirection.DOWN
        
        return ScalingDirection.MAINTAIN


class AutoScalingManager:
    """Intelligent auto-scaling manager"""
    
    def __init__(self):
        self.logger = setup_logging(self.__class__.__name__)
        
        # Resource managers
        self.resource_managers: Dict[ResourceType, ResourceManager] = {}
        
        # Scaling rules
        self.scaling_rules: List[ScalingRule] = []
        
        # Predictive scaler
        self.predictive_scaler = PredictiveScaler()
        
        # Health monitor for metrics
        self.health_monitor = HealthMonitor()
        
        # Scaling history and cooldowns
        self.last_scaling_time: Dict[ResourceType, float] = {}
        self.scaling_history: List[ScalingAction] = []
        
        # Metrics collection
        self.current_metrics = ScalingMetrics()
        self._metrics_lock = threading.Lock()
        
        # Background tasks
        self._monitoring_task = None
        self._scaling_task = None
        
        # Initialize default rules
        self._initialize_default_rules()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_default_rules(self):
        """Initialize default scaling rules"""
        default_rules = [
            ScalingRule(
                resource_type=ResourceType.WORKER_PROCESSES,
                metric_name="cpu_usage",
                threshold_up=80.0,
                threshold_down=30.0,
                scale_up_factor=1.5,
                scale_down_factor=0.8,
                min_instances=2,
                max_instances=16,
                cooldown_seconds=300
            ),
            ScalingRule(
                resource_type=ResourceType.THREAD_POOL,
                metric_name="response_time_ms",
                threshold_up=1000.0,
                threshold_down=200.0,
                scale_up_factor=1.3,
                scale_down_factor=0.9,
                min_instances=4,
                max_instances=32,
                cooldown_seconds=180
            ),
            ScalingRule(
                resource_type=ResourceType.MODEL_INSTANCES,
                metric_name="prediction_latency_ms",
                threshold_up=500.0,
                threshold_down=100.0,
                scale_up_factor=1.5,
                scale_down_factor=0.75,
                min_instances=1,
                max_instances=8,
                cooldown_seconds=600
            )
        ]
        
        self.scaling_rules.extend(default_rules)
    
    def _start_background_tasks(self):
        """Start background monitoring and scaling tasks"""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        if self._scaling_task is None or self._scaling_task.done():
            self._scaling_task = asyncio.create_task(self._scaling_loop())
    
    async def _monitoring_loop(self):
        """Background loop for metrics collection"""
        while True:
            try:
                await self._collect_metrics()
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _scaling_loop(self):
        """Background loop for scaling decisions"""
        while True:
            try:
                await self._evaluate_scaling()
                await asyncio.sleep(60)  # Evaluate scaling every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(120)
    
    async def _collect_metrics(self):
        """Collect system and application metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Application metrics (placeholder - would come from your app)
            request_rate = self._get_request_rate()
            response_time = self._get_avg_response_time()
            error_rate = self._get_error_rate()
            
            # Update current metrics
            with self._metrics_lock:
                self.current_metrics = ScalingMetrics(
                    cpu_usage=cpu_percent,
                    memory_usage=memory.percent,
                    request_rate=request_rate,
                    response_time_ms=response_time,
                    error_rate=error_rate,
                    queue_length=self._get_queue_length(),
                    active_connections=self._get_active_connections(),
                    throughput_rps=self._get_throughput(),
                    prediction_latency_ms=self._get_prediction_latency()
                )
            
            # Record metrics for prediction
            current_time = time.time()
            self.predictive_scaler.record_metric("cpu_usage", cpu_percent, current_time)
            self.predictive_scaler.record_metric("memory_usage", memory.percent, current_time)
            self.predictive_scaler.record_metric("response_time_ms", response_time, current_time)
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
    
    def _get_request_rate(self) -> float:
        """Get current request rate (placeholder)"""
        return 10.0  # Placeholder implementation
    
    def _get_avg_response_time(self) -> float:
        """Get average response time (placeholder)"""
        return 150.0  # Placeholder implementation
    
    def _get_error_rate(self) -> float:
        """Get current error rate (placeholder)"""
        return 0.02  # Placeholder implementation
    
    def _get_queue_length(self) -> int:
        """Get current queue length (placeholder)"""
        return 5  # Placeholder implementation
    
    def _get_active_connections(self) -> int:
        """Get active connections count (placeholder)"""
        return 25  # Placeholder implementation
    
    def _get_throughput(self) -> float:
        """Get current throughput (placeholder)"""
        return 50.0  # Placeholder implementation
    
    def _get_prediction_latency(self) -> float:
        """Get model prediction latency (placeholder)"""
        return 200.0  # Placeholder implementation
    
    async def _evaluate_scaling(self):
        """Evaluate if scaling is needed"""
        with self._metrics_lock:
            current_metrics = self.current_metrics
        
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            last_scaling = self.last_scaling_time.get(rule.resource_type, 0)
            if time.time() - last_scaling < rule.cooldown_seconds:
                continue
            
            # Check if resource manager exists
            if rule.resource_type not in self.resource_managers:
                continue
            
            # Get current metric value
            metric_value = getattr(current_metrics, rule.metric_name, 0)
            
            # Determine scaling direction
            scaling_direction = None
            if metric_value > rule.threshold_up:
                scaling_direction = ScalingDirection.UP
            elif metric_value < rule.threshold_down:
                scaling_direction = ScalingDirection.DOWN
            
            # Also consider predictive scaling
            predictive_direction = self.predictive_scaler.get_scaling_recommendation(
                current_metrics, [rule]
            )
            
            # Use predictive direction if it's more aggressive than reactive
            if (predictive_direction == ScalingDirection.UP and 
                scaling_direction != ScalingDirection.UP):
                scaling_direction = predictive_direction
            
            if scaling_direction in [ScalingDirection.UP, ScalingDirection.DOWN]:
                await self._execute_scaling(rule, scaling_direction, metric_value)
    
    async def _execute_scaling(
        self, 
        rule: ScalingRule, 
        direction: ScalingDirection, 
        trigger_value: float
    ):
        """Execute scaling action"""
        resource_manager = self.resource_managers[rule.resource_type]
        current_instances = await resource_manager.get_current_instances()
        
        # Calculate target instances
        if direction == ScalingDirection.UP:
            target_instances = max(
                rule.min_instances,
                min(rule.max_instances, int(current_instances * rule.scale_up_factor))
            )
        else:  # ScalingDirection.DOWN
            target_instances = max(
                rule.min_instances,
                min(rule.max_instances, int(current_instances * rule.scale_down_factor))
            )
        
        if target_instances == current_instances:
            return
        
        # Create scaling action
        action = ScalingAction(
            resource_type=rule.resource_type,
            direction=direction,
            current_instances=current_instances,
            target_instances=target_instances,
            reason=f"{rule.metric_name}={trigger_value} crossed threshold",
            confidence=0.8  # Placeholder confidence calculation
        )
        
        # Execute scaling
        try:
            success = await resource_manager.scale_to(target_instances)
            
            if success:
                self.last_scaling_time[rule.resource_type] = time.time()
                self.scaling_history.append(action)
                
                self.logger.info(
                    f"Scaled {rule.resource_type.value} {direction.value} from "
                    f"{current_instances} to {target_instances} instances "
                    f"(reason: {action.reason})"
                )
            else:
                self.logger.error(f"Failed to scale {rule.resource_type.value}")
                
        except Exception as e:
            self.logger.error(f"Error executing scaling action: {e}")
    
    def add_resource_manager(
        self, 
        resource_type: ResourceType, 
        manager: ResourceManager
    ):
        """Add resource manager"""
        self.resource_managers[resource_type] = manager
        self.logger.info(f"Added resource manager for {resource_type.value}")
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add scaling rule"""
        self.scaling_rules.append(rule)
        self.logger.info(f"Added scaling rule for {rule.resource_type.value}")
    
    def get_current_metrics(self) -> ScalingMetrics:
        """Get current metrics"""
        with self._metrics_lock:
            return self.current_metrics
    
    def get_scaling_history(self, limit: int = 100) -> List[ScalingAction]:
        """Get scaling history"""
        return self.scaling_history[-limit:]
    
    async def get_resource_status(self) -> Dict[ResourceType, Dict[str, Any]]:
        """Get status of all managed resources"""
        status = {}
        
        for resource_type, manager in self.resource_managers.items():
            try:
                instances = await manager.get_current_instances()
                metrics = await manager.get_resource_metrics()
                
                status[resource_type] = {
                    'instances': instances,
                    'metrics': metrics,
                    'last_scaling': self.last_scaling_time.get(resource_type)
                }
            except Exception as e:
                status[resource_type] = {'error': str(e)}
        
        return status


# Example usage and testing
async def main():
    """Example usage of auto-scaling manager"""
    logger = setup_logging("autoscaling_example")
    
    # Create auto-scaling manager
    autoscaler = AutoScalingManager()
    
    # Add resource managers
    process_manager = ProcessPoolManager(initial_workers=4, max_workers=16)
    thread_manager = ThreadPoolManager(initial_workers=8, max_workers=32)
    
    autoscaler.add_resource_manager(ResourceType.WORKER_PROCESSES, process_manager)
    autoscaler.add_resource_manager(ResourceType.THREAD_POOL, thread_manager)
    
    # Add custom scaling rule
    custom_rule = ScalingRule(
        resource_type=ResourceType.WORKER_PROCESSES,
        metric_name="memory_usage",
        threshold_up=85.0,
        threshold_down=40.0,
        scale_up_factor=1.4,
        scale_down_factor=0.8,
        min_instances=2,
        max_instances=12,
        cooldown_seconds=240
    )
    autoscaler.add_scaling_rule(custom_rule)
    
    # Monitor for a while
    logger.info("Starting auto-scaling monitoring...")
    
    try:
        for i in range(10):
            await asyncio.sleep(60)
            
            # Get current status
            metrics = autoscaler.get_current_metrics()
            status = await autoscaler.get_resource_status()
            
            logger.info(f"Iteration {i+1}:")
            logger.info(f"  CPU: {metrics.cpu_usage:.1f}%, Memory: {metrics.memory_usage:.1f}%")
            
            for resource_type, resource_status in status.items():
                if 'instances' in resource_status:
                    logger.info(f"  {resource_type.value}: {resource_status['instances']} instances")
    
    except KeyboardInterrupt:
        logger.info("Shutting down auto-scaling manager...")


if __name__ == "__main__":
    asyncio.run(main())