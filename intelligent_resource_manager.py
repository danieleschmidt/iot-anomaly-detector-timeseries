"""Intelligent resource management with predictive scaling for Generation 3."""

import asyncio
import threading
import time
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import psutil
import json
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


class AllocationStrategy(Enum):
    """Resource allocation strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    PREDICTIVE = "predictive"


@dataclass
class ResourceMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    load_average: List[float]
    temperature: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ResourcePrediction:
    """Resource usage prediction."""
    resource_type: ResourceType
    predicted_usage: float
    confidence: float
    time_horizon: int  # seconds
    factors: List[str]  # Contributing factors


class PredictiveResourceManager:
    """Intelligent resource manager with ML-based predictions."""
    
    def __init__(
        self,
        prediction_horizon: int = 300,  # 5 minutes
        history_size: int = 1000,
        allocation_strategy: AllocationStrategy = AllocationStrategy.BALANCED
    ):
        self.prediction_horizon = prediction_horizon
        self.history_size = history_size
        self.allocation_strategy = allocation_strategy
        
        # Resource tracking
        self.resource_history = deque(maxlen=history_size)
        self.allocation_history = deque(maxlen=history_size)
        self.prediction_accuracy = defaultdict(list)
        
        # Predictive models (simplified for demo)
        self.resource_models = {}
        self.seasonal_patterns = defaultdict(list)
        self.trend_analyzers = defaultdict(list)
        
        # Resource limits and thresholds
        self.resource_limits = {
            ResourceType.CPU: {'warning': 70, 'critical': 85, 'emergency': 95},
            ResourceType.MEMORY: {'warning': 75, 'critical': 90, 'emergency': 98},
            ResourceType.DISK: {'warning': 80, 'critical': 90, 'emergency': 95}
        }
        
        # Active resource reservations
        self.reservations = {}
        self.allocation_locks = defaultdict(threading.Lock)
        
        # Monitoring state
        self._monitoring_active = False
        self._monitor_thread = None
        
    def start_monitoring(self, interval_seconds: int = 30):
        """Start resource monitoring."""
        if self._monitoring_active:
            logger.warning("Resource monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Resource monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join()
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                metrics = self._collect_resource_metrics()
                self.resource_history.append(metrics)
                
                # Update predictive models
                self._update_models(metrics)
                
                # Check for resource pressure
                self._check_resource_pressure(metrics)
                
                # Optimize resource allocation
                self._optimize_allocation()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # CPU metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        load_avg = psutil.getloadavg()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Network metrics
        network_io = psutil.net_io_counters()
        network_metrics = {
            'bytes_sent': network_io.bytes_sent,
            'bytes_recv': network_io.bytes_recv,
            'packets_sent': network_io.packets_sent,
            'packets_recv': network_io.packets_recv
        }
        
        # Process count
        process_count = len(psutil.pids())
        
        return ResourceMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_metrics,
            process_count=process_count,
            load_average=list(load_avg),
            temperature=self._get_system_temperature()
        )
    
    def _get_system_temperature(self) -> Optional[float]:
        """Get system temperature if available."""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get CPU temperature if available
                for name, entries in temps.items():
                    if 'cpu' in name.lower() or 'core' in name.lower():
                        return entries[0].current if entries else None
        except:
            pass
        return None
    
    def _update_models(self, metrics: ResourceMetrics):
        """Update predictive models with new metrics."""
        if len(self.resource_history) < 10:
            return  # Need minimum history
        
        # Simple trend analysis
        recent_metrics = list(self.resource_history)[-10:]
        
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.DISK]:
            values = []
            timestamps = []
            
            for m in recent_metrics:
                timestamps.append(m.timestamp)
                if resource_type == ResourceType.CPU:
                    values.append(m.cpu_usage)
                elif resource_type == ResourceType.MEMORY:
                    values.append(m.memory_usage)
                elif resource_type == ResourceType.DISK:
                    values.append(m.disk_usage)
            
            # Calculate trend
            if len(values) >= 3:
                trend = self._calculate_trend(timestamps, values)
                self.trend_analyzers[resource_type].append(trend)
                
                # Keep only recent trends
                if len(self.trend_analyzers[resource_type]) > 100:
                    self.trend_analyzers[resource_type].pop(0)
        
        # Detect seasonal patterns (simplified)
        self._detect_seasonal_patterns(metrics)
    
    def _calculate_trend(self, timestamps: List[float], values: List[float]) -> float:
        """Calculate trend slope using simple linear regression."""
        n = len(timestamps)
        if n < 2:
            return 0.0
        
        # Normalize timestamps
        base_time = timestamps[0]
        x = [(t - base_time) for t in timestamps]
        y = values
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _detect_seasonal_patterns(self, metrics: ResourceMetrics):
        """Detect seasonal patterns in resource usage."""
        if len(self.resource_history) < 100:
            return
        
        # Analyze hourly patterns
        hour = time.localtime(metrics.timestamp).tm_hour
        
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY]:
            if resource_type == ResourceType.CPU:
                value = metrics.cpu_usage
            else:
                value = metrics.memory_usage
            
            # Store hourly patterns
            if hour not in self.seasonal_patterns:
                self.seasonal_patterns[hour] = []
            
            self.seasonal_patterns[hour].append(value)
            
            # Keep only recent patterns
            if len(self.seasonal_patterns[hour]) > 30:
                self.seasonal_patterns[hour].pop(0)
    
    def predict_resource_usage(
        self,
        resource_type: ResourceType,
        time_horizon: int = None
    ) -> ResourcePrediction:
        """Predict future resource usage."""
        time_horizon = time_horizon or self.prediction_horizon
        
        if len(self.resource_history) < 5:
            # Not enough history, return current usage
            current_metrics = self.resource_history[-1] if self.resource_history else None
            if current_metrics:
                if resource_type == ResourceType.CPU:
                    current_usage = current_metrics.cpu_usage
                elif resource_type == ResourceType.MEMORY:
                    current_usage = current_metrics.memory_usage
                elif resource_type == ResourceType.DISK:
                    current_usage = current_metrics.disk_usage
                else:
                    current_usage = 50.0
            else:
                current_usage = 50.0
            
            return ResourcePrediction(
                resource_type=resource_type,
                predicted_usage=current_usage,
                confidence=0.5,
                time_horizon=time_horizon,
                factors=['insufficient_history']
            )
        
        # Get recent trend
        recent_trend = 0.0
        if resource_type in self.trend_analyzers and self.trend_analyzers[resource_type]:
            recent_trend = np.mean(self.trend_analyzers[resource_type][-5:])
        
        # Get current usage
        current_metrics = self.resource_history[-1]
        if resource_type == ResourceType.CPU:
            current_usage = current_metrics.cpu_usage
        elif resource_type == ResourceType.MEMORY:
            current_usage = current_metrics.memory_usage
        elif resource_type == ResourceType.DISK:
            current_usage = current_metrics.disk_usage
        else:
            current_usage = 50.0
        
        # Get seasonal pattern
        future_time = time.time() + time_horizon
        future_hour = time.localtime(future_time).tm_hour
        seasonal_factor = 1.0
        
        if future_hour in self.seasonal_patterns and self.seasonal_patterns[future_hour]:
            current_hour = time.localtime().tm_hour
            if current_hour in self.seasonal_patterns and self.seasonal_patterns[current_hour]:
                current_seasonal = np.mean(self.seasonal_patterns[current_hour])
                future_seasonal = np.mean(self.seasonal_patterns[future_hour])
                if current_seasonal > 0:
                    seasonal_factor = future_seasonal / current_seasonal
        
        # Predict usage
        trend_impact = recent_trend * (time_horizon / 60)  # trend per minute
        predicted_usage = current_usage * seasonal_factor + trend_impact
        
        # Clamp to reasonable bounds
        predicted_usage = max(0, min(100, predicted_usage))
        
        # Calculate confidence
        trend_stability = self._calculate_trend_stability(resource_type)
        history_length = min(len(self.resource_history) / 100, 1.0)
        confidence = (trend_stability + history_length) / 2
        
        factors = []
        if abs(recent_trend) > 1.0:
            factors.append('strong_trend')
        if seasonal_factor != 1.0:
            factors.append('seasonal_pattern')
        if history_length < 0.5:
            factors.append('limited_history')
        
        return ResourcePrediction(
            resource_type=resource_type,
            predicted_usage=predicted_usage,
            confidence=confidence,
            time_horizon=time_horizon,
            factors=factors
        )
    
    def _calculate_trend_stability(self, resource_type: ResourceType) -> float:
        """Calculate stability of trend predictions."""
        if resource_type not in self.trend_analyzers or len(self.trend_analyzers[resource_type]) < 5:
            return 0.5
        
        recent_trends = self.trend_analyzers[resource_type][-10:]
        trend_variance = np.var(recent_trends)
        
        # Lower variance = higher stability
        stability = max(0, min(1, 1 - trend_variance / 10))
        return stability
    
    def _check_resource_pressure(self, metrics: ResourceMetrics):
        """Check for resource pressure and take action."""
        pressures = []
        
        # Check CPU pressure
        cpu_thresholds = self.resource_limits[ResourceType.CPU]
        if metrics.cpu_usage > cpu_thresholds['emergency']:
            pressures.append((ResourceType.CPU, 'emergency', metrics.cpu_usage))
        elif metrics.cpu_usage > cpu_thresholds['critical']:
            pressures.append((ResourceType.CPU, 'critical', metrics.cpu_usage))
        elif metrics.cpu_usage > cpu_thresholds['warning']:
            pressures.append((ResourceType.CPU, 'warning', metrics.cpu_usage))
        
        # Check memory pressure
        memory_thresholds = self.resource_limits[ResourceType.MEMORY]
        if metrics.memory_usage > memory_thresholds['emergency']:
            pressures.append((ResourceType.MEMORY, 'emergency', metrics.memory_usage))
        elif metrics.memory_usage > memory_thresholds['critical']:
            pressures.append((ResourceType.MEMORY, 'critical', metrics.memory_usage))
        elif metrics.memory_usage > memory_thresholds['warning']:
            pressures.append((ResourceType.MEMORY, 'warning', metrics.memory_usage))
        
        # Check disk pressure
        disk_thresholds = self.resource_limits[ResourceType.DISK]
        if metrics.disk_usage > disk_thresholds['emergency']:
            pressures.append((ResourceType.DISK, 'emergency', metrics.disk_usage))
        elif metrics.disk_usage > disk_thresholds['critical']:
            pressures.append((ResourceType.DISK, 'critical', metrics.disk_usage))
        elif metrics.disk_usage > disk_thresholds['warning']:
            pressures.append((ResourceType.DISK, 'warning', metrics.disk_usage))
        
        # Handle pressures
        for resource_type, level, usage in pressures:
            self._handle_resource_pressure(resource_type, level, usage)
    
    def _handle_resource_pressure(self, resource_type: ResourceType, level: str, usage: float):
        """Handle resource pressure based on severity."""
        logger.warning(f"Resource pressure detected: {resource_type.value} {level} ({usage:.1f}%)")
        
        if level == 'emergency':
            # Emergency actions
            if resource_type == ResourceType.MEMORY:
                self._trigger_memory_cleanup()
            elif resource_type == ResourceType.CPU:
                self._trigger_cpu_throttling()
            elif resource_type == ResourceType.DISK:
                self._trigger_disk_cleanup()
        
        elif level == 'critical':
            # Critical actions
            self._trigger_resource_optimization(resource_type)
        
        elif level == 'warning':
            # Warning actions
            self._trigger_predictive_scaling(resource_type)
    
    def _trigger_memory_cleanup(self):
        """Trigger memory cleanup procedures."""
        logger.info("Triggering memory cleanup")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear caches (simplified)
        # In production, this would clear application caches
        
    def _trigger_cpu_throttling(self):
        """Trigger CPU throttling for non-critical processes."""
        logger.info("Triggering CPU throttling")
        
        # In production, this would adjust process priorities
        # and implement CPU throttling
        
    def _trigger_disk_cleanup(self):
        """Trigger disk cleanup procedures."""
        logger.info("Triggering disk cleanup")
        
        # In production, this would clean temporary files,
        # rotate logs, and free up disk space
        
    def _trigger_resource_optimization(self, resource_type: ResourceType):
        """Trigger resource optimization for specific type."""
        logger.info(f"Triggering optimization for {resource_type.value}")
        
        # Get prediction for near future
        prediction = self.predict_resource_usage(resource_type, 300)  # 5 minutes
        
        if prediction.predicted_usage > 90:
            logger.warning(f"Predicted {resource_type.value} usage: {prediction.predicted_usage:.1f}% in 5 minutes")
            # Take preemptive action
    
    def _trigger_predictive_scaling(self, resource_type: ResourceType):
        """Trigger predictive scaling based on forecasts."""
        logger.info(f"Triggering predictive scaling for {resource_type.value}")
        
        # Get multiple time horizon predictions
        predictions = []
        for horizon in [300, 600, 1800]:  # 5, 10, 30 minutes
            pred = self.predict_resource_usage(resource_type, horizon)
            predictions.append(pred)
        
        # Check if scaling is needed
        needs_scaling = any(p.predicted_usage > 80 for p in predictions)
        
        if needs_scaling:
            logger.info(f"Predictive scaling recommended for {resource_type.value}")
            # In production, this would trigger auto-scaling
    
    def _optimize_allocation(self):
        """Optimize resource allocation based on current strategy."""
        if self.allocation_strategy == AllocationStrategy.PREDICTIVE:
            self._predictive_allocation()
        elif self.allocation_strategy == AllocationStrategy.AGGRESSIVE:
            self._aggressive_allocation()
        elif self.allocation_strategy == AllocationStrategy.CONSERVATIVE:
            self._conservative_allocation()
        else:  # BALANCED
            self._balanced_allocation()
    
    def _predictive_allocation(self):
        """Predictive resource allocation based on forecasts."""
        # Get predictions for all resource types
        predictions = {}
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.DISK]:
            predictions[resource_type] = self.predict_resource_usage(resource_type)
        
        # Adjust allocations based on predictions
        for resource_type, prediction in predictions.items():
            if prediction.predicted_usage > 75 and prediction.confidence > 0.7:
                # Preemptively allocate more resources
                self._increase_resource_allocation(resource_type)
    
    def _increase_resource_allocation(self, resource_type: ResourceType):
        """Increase allocation for specific resource type."""
        logger.debug(f"Increasing allocation for {resource_type.value}")
        # Implementation would depend on the specific resource type
        # and available allocation mechanisms
    
    def _aggressive_allocation(self):
        """Aggressive allocation strategy - maximize performance."""
        pass  # Implementation for aggressive allocation
    
    def _conservative_allocation(self):
        """Conservative allocation strategy - minimize resource usage."""
        pass  # Implementation for conservative allocation
    
    def _balanced_allocation(self):
        """Balanced allocation strategy - balance performance and efficiency."""
        pass  # Implementation for balanced allocation
    
    def get_resource_forecast(self, hours: int = 24) -> Dict[str, List[ResourcePrediction]]:
        """Get resource forecast for specified time period."""
        forecasts = {}
        
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.DISK]:
            resource_forecasts = []
            
            # Generate predictions at hourly intervals
            for hour in range(hours):
                time_horizon = hour * 3600  # Convert to seconds
                prediction = self.predict_resource_usage(resource_type, time_horizon)
                resource_forecasts.append(prediction)
            
            forecasts[resource_type.value] = resource_forecasts
        
        return forecasts
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary."""
        if not self.resource_history:
            return {"status": "no_data"}
        
        current_metrics = self.resource_history[-1]
        
        # Get predictions
        predictions = {}
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.DISK]:
            predictions[resource_type.value] = self.predict_resource_usage(resource_type)
        
        # Calculate resource efficiency
        efficiency = self._calculate_resource_efficiency()
        
        return {
            "timestamp": current_metrics.timestamp,
            "current_usage": {
                "cpu": current_metrics.cpu_usage,
                "memory": current_metrics.memory_usage,
                "disk": current_metrics.disk_usage,
                "load_average": current_metrics.load_average,
                "process_count": current_metrics.process_count
            },
            "predictions": {
                resource: {
                    "predicted_usage": pred.predicted_usage,
                    "confidence": pred.confidence,
                    "factors": pred.factors
                }
                for resource, pred in predictions.items()
            },
            "efficiency": efficiency,
            "allocation_strategy": self.allocation_strategy.value,
            "monitoring_duration": time.time() - self.resource_history[0].timestamp if self.resource_history else 0
        }
    
    def _calculate_resource_efficiency(self) -> Dict[str, float]:
        """Calculate resource efficiency metrics."""
        if len(self.resource_history) < 2:
            return {}
        
        recent_metrics = list(self.resource_history)[-10:]
        
        # Calculate average utilization
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        avg_disk = np.mean([m.disk_usage for m in recent_metrics])
        
        # Calculate efficiency (utilization without waste)
        # Optimal range is 60-80% for most resources
        def calculate_efficiency(usage):
            if 60 <= usage <= 80:
                return 1.0
            elif usage < 60:
                return usage / 60  # Underutilization
            else:
                return max(0, 1 - (usage - 80) / 20)  # Overutilization
        
        return {
            "cpu_efficiency": calculate_efficiency(avg_cpu),
            "memory_efficiency": calculate_efficiency(avg_memory),
            "disk_efficiency": calculate_efficiency(avg_disk),
            "overall_efficiency": (
                calculate_efficiency(avg_cpu) +
                calculate_efficiency(avg_memory) +
                calculate_efficiency(avg_disk)
            ) / 3
        }
    
    def save_state(self, path: str):
        """Save resource manager state."""
        state = {
            'resource_history': [m.to_dict() for m in self.resource_history],
            'allocation_strategy': self.allocation_strategy.value,
            'resource_limits': self.resource_limits,
            'seasonal_patterns': dict(self.seasonal_patterns),
            'trend_analyzers': dict(self.trend_analyzers)
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Resource manager state saved to {path}")


if __name__ == "__main__":
    # Generation 3 resource management demonstration
    logger.info("=== GENERATION 3: INTELLIGENT RESOURCE MANAGEMENT DEMO ===")
    
    # Initialize resource manager
    resource_manager = PredictiveResourceManager(
        prediction_horizon=300,
        allocation_strategy=AllocationStrategy.PREDICTIVE
    )
    
    try:
        # Start monitoring
        resource_manager.start_monitoring(interval_seconds=10)
        
        # Run for demonstration
        time.sleep(60)  # Monitor for 1 minute
        
        # Get resource predictions
        logger.info("Resource Predictions:")
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.DISK]:
            prediction = resource_manager.predict_resource_usage(resource_type)
            logger.info(f"  {resource_type.value}: {prediction.predicted_usage:.1f}% (confidence: {prediction.confidence:.2f})")
        
        # Get comprehensive summary
        summary = resource_manager.get_resource_summary()
        logger.info("Resource Summary:")
        logger.info(f"  Current CPU: {summary['current_usage']['cpu']:.1f}%")
        logger.info(f"  Current Memory: {summary['current_usage']['memory']:.1f}%")
        logger.info(f"  Overall Efficiency: {summary['efficiency']['overall_efficiency']:.2f}")
        
        # Get forecast
        forecast = resource_manager.get_resource_forecast(hours=6)
        logger.info("6-hour Forecast:")
        for resource, predictions in forecast.items():
            avg_usage = np.mean([p.predicted_usage for p in predictions])
            logger.info(f"  {resource}: {avg_usage:.1f}% average")
        
        # Save state
        resource_manager.save_state('resource_manager_state.json')
        
    finally:
        resource_manager.stop_monitoring()
    
    logger.info("=== GENERATION 3 RESOURCE MANAGEMENT COMPLETE ===")