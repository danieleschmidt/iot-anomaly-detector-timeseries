"""Comprehensive monitoring and observability system for IoT anomaly detection."""

import time
import json
import logging
import threading
import queue
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import statistics

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    timestamp: float
    level: AlertLevel
    component: str
    metric: str
    message: str
    current_value: float
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class HealthCheck:
    """Component health check result."""
    component: str
    timestamp: float
    healthy: bool
    response_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Metric:
    """Individual metric with history."""
    
    def __init__(self, name: str, metric_type: MetricType, max_points: int = 1000):
        self.name = name
        self.metric_type = metric_type
        self.max_points = max_points
        self.data_points: deque = deque(maxlen=max_points)
        self.labels: Dict[str, str] = {}
        self._lock = threading.RLock()
    
    def add_point(self, value: float, labels: Optional[Dict[str, str]] = None, timestamp: Optional[float] = None) -> None:
        """Add data point to metric."""
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            point = MetricPoint(timestamp=timestamp, value=value, labels=labels or {})
            self.data_points.append(point)
    
    def get_latest_value(self) -> Optional[float]:
        """Get most recent value."""
        with self._lock:
            if self.data_points:
                return self.data_points[-1].value
            return None
    
    def get_values(self, time_window_seconds: Optional[int] = None) -> List[float]:
        """Get values within time window."""
        with self._lock:
            if time_window_seconds is None:
                return [p.value for p in self.data_points]
            
            cutoff_time = time.time() - time_window_seconds
            return [p.value for p in self.data_points if p.timestamp >= cutoff_time]
    
    def get_statistics(self, time_window_seconds: Optional[int] = None) -> Dict[str, float]:
        """Get statistical summary."""
        values = self.get_values(time_window_seconds)
        
        if not values:
            return {}
        
        stats = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "latest": values[-1]
        }
        
        if len(values) > 1:
            stats["std"] = statistics.stdev(values)
            stats["median"] = statistics.median(values)
            
            # Calculate percentiles if numpy available
            if NUMPY_AVAILABLE:
                values_array = np.array(values)
                stats["p50"] = np.percentile(values_array, 50)
                stats["p95"] = np.percentile(values_array, 95)
                stats["p99"] = np.percentile(values_array, 99)
        
        return stats


class AlertRule:
    """Alert rule definition."""
    
    def __init__(
        self,
        rule_id: str,
        metric_name: str,
        condition: str,
        threshold: float,
        level: AlertLevel = AlertLevel.WARNING,
        time_window_seconds: int = 300,
        min_samples: int = 5,
        cooldown_seconds: int = 300
    ):
        self.rule_id = rule_id
        self.metric_name = metric_name
        self.condition = condition  # 'gt', 'lt', 'eq', 'ne'
        self.threshold = threshold
        self.level = level
        self.time_window_seconds = time_window_seconds
        self.min_samples = min_samples
        self.cooldown_seconds = cooldown_seconds
        
        self.last_alert_time: Optional[float] = None
        self.active_alert: Optional[Alert] = None
    
    def evaluate(self, metric: Metric, current_time: float) -> Optional[Alert]:
        """Evaluate alert rule against metric."""
        # Check cooldown period
        if (self.last_alert_time and 
            current_time - self.last_alert_time < self.cooldown_seconds):
            return None
        
        # Get values for evaluation
        values = metric.get_values(self.time_window_seconds)
        if len(values) < self.min_samples:
            return None
        
        # Calculate evaluation metric (using latest value by default)
        eval_value = values[-1]
        
        # Check condition
        trigger_alert = False
        if self.condition == "gt":
            trigger_alert = eval_value > self.threshold
        elif self.condition == "lt":
            trigger_alert = eval_value < self.threshold
        elif self.condition == "eq":
            trigger_alert = abs(eval_value - self.threshold) < 1e-9
        elif self.condition == "ne":
            trigger_alert = abs(eval_value - self.threshold) >= 1e-9
        
        if trigger_alert:
            self.last_alert_time = current_time
            
            alert = Alert(
                alert_id=f"{self.rule_id}_{int(current_time * 1000)}",
                timestamp=current_time,
                level=self.level,
                component="monitoring",
                metric=self.metric_name,
                message=f"Metric {self.metric_name} {self.condition} {self.threshold}",
                current_value=eval_value,
                threshold=self.threshold,
                metadata={
                    "rule_id": self.rule_id,
                    "time_window": self.time_window_seconds,
                    "sample_count": len(values)
                }
            )
            
            self.active_alert = alert
            return alert
        
        # Check if active alert should be resolved
        elif self.active_alert and not self.active_alert.resolved:
            self.active_alert.resolved = True
            self.active_alert.resolution_time = current_time
        
        return None


class PerformanceProfiler:
    """Performance profiling and timing."""
    
    def __init__(self, monitoring_system):
        self.monitoring = monitoring_system
        self._active_timers: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def start_timer(self, operation_name: str) -> str:
        """Start timing an operation."""
        timer_id = f"{operation_name}_{threading.current_thread().ident}_{time.time()}"
        
        with self._lock:
            self._active_timers[timer_id] = time.time()
        
        return timer_id
    
    def end_timer(self, timer_id: str, labels: Optional[Dict[str, str]] = None) -> float:
        """End timing and record metric."""
        with self._lock:
            if timer_id not in self._active_timers:
                return 0.0
            
            start_time = self._active_timers[timer_id]
            duration = time.time() - start_time
            del self._active_timers[timer_id]
        
        # Extract operation name
        operation_name = timer_id.split('_')[0]
        metric_name = f"operation_duration_{operation_name}"
        
        # Record timing metric
        self.monitoring.record_metric(metric_name, duration * 1000, MetricType.TIMER, labels)
        
        return duration
    
    def time_function(self, func_name: str, labels: Optional[Dict[str, str]] = None):
        """Decorator for timing functions."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                timer_id = self.start_timer(func_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_timer(timer_id, labels)
            return wrapper
        return decorator


class SystemResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self, monitoring_system):
        self.monitoring = monitoring_system
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        self.monitor_interval = 10  # seconds
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
    
    def _monitoring_loop(self) -> None:
        """Resource monitoring loop."""
        while self.is_monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
                time.sleep(5)  # Brief pause on error
    
    def _collect_system_metrics(self) -> None:
        """Collect system resource metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.monitoring.record_metric("system_cpu_percent", cpu_percent, MetricType.GAUGE)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.monitoring.record_metric("system_memory_percent", memory.percent, MetricType.GAUGE)
            self.monitoring.record_metric("system_memory_available_gb", memory.available / (1024**3), MetricType.GAUGE)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.monitoring.record_metric("system_disk_percent", disk_percent, MetricType.GAUGE)
            
            # Network metrics
            network = psutil.net_io_counters()
            self.monitoring.record_metric("system_network_bytes_sent", network.bytes_sent, MetricType.COUNTER)
            self.monitoring.record_metric("system_network_bytes_recv", network.bytes_recv, MetricType.COUNTER)
            
            # Process metrics
            process = psutil.Process()
            self.monitoring.record_metric("process_cpu_percent", process.cpu_percent(), MetricType.GAUGE)
            self.monitoring.record_metric("process_memory_mb", process.memory_info().rss / (1024**2), MetricType.GAUGE)
            self.monitoring.record_metric("process_threads", process.num_threads(), MetricType.GAUGE)
            
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")


class ComprehensiveMonitoring:
    """Main monitoring and observability system."""
    
    def __init__(
        self,
        metrics_retention_seconds: int = 3600,
        alert_retention_count: int = 1000,
        health_check_interval: int = 60
    ):
        self.metrics_retention = metrics_retention_seconds
        self.alert_retention_count = alert_retention_count
        self.health_check_interval = health_check_interval
        
        # Core components
        self.metrics: Dict[str, Metric] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alerts: deque = deque(maxlen=alert_retention_count)
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Handlers and callbacks
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.health_check_functions: Dict[str, Callable[[], HealthCheck]] = {}
        
        # Background processing
        self.processing_queue = queue.Queue()
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False
        
        # Specialized monitors
        self.profiler = PerformanceProfiler(self)
        self.resource_monitor = SystemResourceMonitor(self)
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
        self._start_processing()
    
    def _start_processing(self) -> None:
        """Start background processing."""
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
    
    def _processing_loop(self) -> None:
        """Background processing loop."""
        last_health_check = 0
        last_alert_evaluation = 0
        
        while self.is_processing:
            try:
                current_time = time.time()
                
                # Process queued items
                self._process_queue_items()
                
                # Periodic health checks
                if current_time - last_health_check > self.health_check_interval:
                    self._run_health_checks()
                    last_health_check = current_time
                
                # Alert rule evaluation
                if current_time - last_alert_evaluation > 10:  # Every 10 seconds
                    self._evaluate_alert_rules()
                    last_alert_evaluation = current_time
                
                time.sleep(1)  # 1 second processing interval
                
            except Exception as e:
                self.logger.error(f"Monitoring processing error: {e}")
                time.sleep(5)
    
    def _process_queue_items(self) -> None:
        """Process queued monitoring items."""
        processed_count = 0
        max_batch_size = 100
        
        while not self.processing_queue.empty() and processed_count < max_batch_size:
            try:
                item = self.processing_queue.get_nowait()
                self._process_monitoring_item(item)
                processed_count += 1
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing monitoring item: {e}")
    
    def _process_monitoring_item(self, item: Dict[str, Any]) -> None:
        """Process individual monitoring item."""
        item_type = item.get("type")
        
        if item_type == "metric":
            self._process_metric_item(item)
        elif item_type == "alert":
            self._process_alert_item(item)
    
    def _process_metric_item(self, item: Dict[str, Any]) -> None:
        """Process metric recording."""
        name = item["name"]
        value = item["value"]
        metric_type = item["metric_type"]
        labels = item.get("labels")
        timestamp = item.get("timestamp")
        
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(name, metric_type)
            
            self.metrics[name].add_point(value, labels, timestamp)
    
    def _process_alert_item(self, item: Dict[str, Any]) -> None:
        """Process alert."""
        alert = item["alert"]
        
        with self._lock:
            self.alerts.append(alert)
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler error: {e}")
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """Record a metric value."""
        try:
            item = {
                "type": "metric",
                "name": name,
                "value": value,
                "metric_type": metric_type,
                "labels": labels,
                "timestamp": timestamp
            }
            self.processing_queue.put_nowait(item)
        except queue.Full:
            self.logger.warning(f"Monitoring queue full, dropping metric {name}")
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter metric."""
        current_value = 0
        if name in self.metrics:
            latest = self.metrics[name].get_latest_value()
            if latest is not None:
                current_value = latest
        
        self.record_metric(name, current_value + 1, MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge metric value."""
        self.record_metric(name, value, MetricType.GAUGE, labels)
    
    def record_timing(self, name: str, duration_ms: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record timing metric."""
        self.record_metric(name, duration_ms, MetricType.TIMER, labels)
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        with self._lock:
            self.alert_rules[rule.rule_id] = rule
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove alert rule."""
        with self._lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                return True
            return False
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add alert handler."""
        self.alert_handlers.append(handler)
    
    def _evaluate_alert_rules(self) -> None:
        """Evaluate all alert rules."""
        current_time = time.time()
        
        with self._lock:
            for rule in self.alert_rules.values():
                if rule.metric_name in self.metrics:
                    metric = self.metrics[rule.metric_name]
                    alert = rule.evaluate(metric, current_time)
                    
                    if alert:
                        item = {"type": "alert", "alert": alert}
                        try:
                            self.processing_queue.put_nowait(item)
                        except queue.Full:
                            self.logger.warning("Monitoring queue full, dropping alert")
    
    def add_health_check(self, name: str, check_function: Callable[[], HealthCheck]) -> None:
        """Add health check function."""
        self.health_check_functions[name] = check_function
    
    def _run_health_checks(self) -> None:
        """Run all health checks."""
        for name, check_func in self.health_check_functions.items():
            try:
                health_check = check_func()
                with self._lock:
                    self.health_checks[name] = health_check
                
                # Generate alert if unhealthy
                if not health_check.healthy:
                    alert = Alert(
                        alert_id=f"health_{name}_{int(time.time() * 1000)}",
                        timestamp=health_check.timestamp,
                        level=AlertLevel.ERROR,
                        component=name,
                        metric="health_status",
                        message=f"Health check failed: {health_check.error_message}",
                        current_value=0.0,
                        threshold=1.0,
                        metadata={"health_check": True}
                    )
                    
                    item = {"type": "alert", "alert": alert}
                    try:
                        self.processing_queue.put_nowait(item)
                    except queue.Full:
                        pass
                        
            except Exception as e:
                self.logger.error(f"Health check {name} failed: {e}")
    
    def get_metric_statistics(self, name: str, time_window_seconds: Optional[int] = None) -> Optional[Dict[str, float]]:
        """Get statistics for a metric."""
        with self._lock:
            if name in self.metrics:
                return self.metrics[name].get_statistics(time_window_seconds)
            return None
    
    def get_recent_alerts(self, count: int = 50, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get recent alerts."""
        with self._lock:
            recent_alerts = list(self.alerts)[-count:] if count > 0 else list(self.alerts)
        
        if level:
            recent_alerts = [a for a in recent_alerts if a.level == level]
        
        return sorted(recent_alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview."""
        current_time = time.time()
        
        with self._lock:
            metrics_summary = {
                name: metric.get_latest_value()
                for name, metric in self.metrics.items()
                if metric.get_latest_value() is not None
            }
            
            health_summary = {
                name: check.healthy
                for name, check in self.health_checks.items()
            }
            
            recent_alerts = self.get_recent_alerts(10)
            alert_counts = defaultdict(int)
            for alert in recent_alerts:
                alert_counts[alert.level.value] += 1
        
        return {
            "timestamp": current_time,
            "metrics_count": len(self.metrics),
            "active_alert_rules": len(self.alert_rules),
            "recent_alerts_count": len(recent_alerts),
            "alert_counts_by_level": dict(alert_counts),
            "health_checks": health_summary,
            "key_metrics": metrics_summary,
            "queue_size": self.processing_queue.qsize()
        }
    
    def export_metrics(self, output_path: str, time_window_seconds: Optional[int] = None) -> None:
        """Export metrics to file."""
        export_data = {
            "timestamp": time.time(),
            "time_window_seconds": time_window_seconds,
            "metrics": {}
        }
        
        with self._lock:
            for name, metric in self.metrics.items():
                export_data["metrics"][name] = {
                    "type": metric.metric_type.value,
                    "statistics": metric.get_statistics(time_window_seconds),
                    "data_points": [
                        {
                            "timestamp": point.timestamp,
                            "value": point.value,
                            "labels": point.labels
                        }
                        for point in (metric.get_values(time_window_seconds) 
                                     if time_window_seconds 
                                     else metric.data_points)
                    ]
                }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {output_path}")
    
    def shutdown(self) -> None:
        """Shutdown monitoring system."""
        self.is_processing = False
        self.resource_monitor.stop_monitoring()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)


# Utility functions and decorators
def create_monitoring_system(**kwargs) -> ComprehensiveMonitoring:
    """Create monitoring system with configuration."""
    return ComprehensiveMonitoring(**kwargs)


def default_alert_handler(alert: Alert) -> None:
    """Default alert handler that logs alerts."""
    level_map = {
        AlertLevel.INFO: logging.INFO,
        AlertLevel.WARNING: logging.WARNING,
        AlertLevel.ERROR: logging.ERROR,
        AlertLevel.CRITICAL: logging.CRITICAL
    }
    
    logger = logging.getLogger("monitoring.alerts")
    logger.log(
        level_map.get(alert.level, logging.WARNING),
        f"ALERT [{alert.level.value}] {alert.component}.{alert.metric}: {alert.message} "
        f"(current: {alert.current_value}, threshold: {alert.threshold})"
    )


# Example usage
if __name__ == "__main__":
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="Comprehensive Monitoring System")
    parser.add_argument("--duration", type=int, default=60, help="Test duration (seconds)")
    parser.add_argument("--export-metrics", help="Export metrics to file")
    
    args = parser.parse_args()
    
    # Initialize monitoring system
    monitoring = create_monitoring_system()
    monitoring.add_alert_handler(default_alert_handler)
    
    # Add some alert rules
    monitoring.add_alert_rule(AlertRule(
        rule_id="high_cpu",
        metric_name="system_cpu_percent",
        condition="gt",
        threshold=80.0,
        level=AlertLevel.WARNING
    ))
    
    monitoring.add_alert_rule(AlertRule(
        rule_id="high_memory",
        metric_name="system_memory_percent", 
        condition="gt",
        threshold=90.0,
        level=AlertLevel.ERROR
    ))
    
    # Add health checks
    def dummy_health_check() -> HealthCheck:
        start_time = time.time()
        healthy = random.random() > 0.1  # 90% healthy
        response_time = random.uniform(10, 100)  # 10-100ms
        
        return HealthCheck(
            component="dummy_service",
            timestamp=time.time(),
            healthy=healthy,
            response_time_ms=response_time,
            error_message=None if healthy else "Service unavailable"
        )
    
    monitoring.add_health_check("dummy_service", dummy_health_check)
    
    print(f"Starting monitoring test for {args.duration} seconds...")
    
    # Generate test metrics
    start_time = time.time()
    while time.time() - start_time < args.duration:
        # Simulate various metrics
        monitoring.set_gauge("test_temperature", random.gauss(25.0, 5.0))
        monitoring.set_gauge("test_humidity", random.uniform(30, 70))
        monitoring.increment_counter("test_requests")
        
        # Simulate timing
        with monitoring.profiler.time_function("test_operation")():
            time.sleep(random.uniform(0.01, 0.1))
        
        time.sleep(1)
    
    # Get final overview
    overview = monitoring.get_system_overview()
    print(f"\nSystem Overview:")
    print(json.dumps(overview, indent=2))
    
    # Export metrics if requested
    if args.export_metrics:
        monitoring.export_metrics(args.export_metrics)
        print(f"Metrics exported to {args.export_metrics}")
    
    # Get recent alerts
    recent_alerts = monitoring.get_recent_alerts(5)
    if recent_alerts:
        print(f"\nRecent Alerts:")
        for alert in recent_alerts:
            print(f"  {alert.level.value}: {alert.message}")
    
    monitoring.shutdown()
    print("Monitoring system shutdown complete")