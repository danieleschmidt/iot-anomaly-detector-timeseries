"""Robust monitoring and health check system for Generation 2."""

import numpy as np
import pandas as pd
import logging
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics to monitor."""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    SYSTEM = "system"
    DATA_QUALITY = "data_quality"
    SECURITY = "security"


@dataclass
class Alert:
    """Alert data structure."""
    timestamp: datetime
    level: AlertLevel
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['level'] = self.level.value
        result['metric_type'] = self.metric_type.value
        return result


@dataclass
class HealthMetrics:
    """System health metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    model_latency: float
    prediction_accuracy: float
    data_quality_score: float
    error_rate: float
    throughput: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class RobustCircuitBreaker:
    """Enhanced circuit breaker with multiple failure modes."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_calls = 0
        
        self._lock = threading.Lock()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker."""
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.half_open_calls = 0
                    logger.info("Circuit breaker: Attempting reset (HALF_OPEN)")
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            elif self.state == "HALF_OPEN":
                if self.half_open_calls >= self.half_open_max_calls:
                    raise Exception("Circuit breaker: Too many calls in HALF_OPEN state")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self.state == "HALF_OPEN":
                self.half_open_calls += 1
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker: Reset to CLOSED state")
            else:
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker: Opened due to {self.failure_count} failures")


class AdvancedHealthMonitor:
    """Comprehensive health monitoring with predictive alerts."""
    
    def __init__(
        self,
        alert_callback: Optional[Callable] = None,
        metrics_retention_hours: int = 24
    ):
        self.alert_callback = alert_callback or self._default_alert_callback
        self.metrics_retention_hours = metrics_retention_hours
        
        self.metrics_history: List[HealthMetrics] = []
        self.alerts_history: List[Alert] = []
        self.thresholds = self._initialize_thresholds()
        
        self._monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # Circuit breakers for critical components
        self.model_circuit_breaker = RobustCircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30
        )
        self.data_circuit_breaker = RobustCircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
    
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize monitoring thresholds."""
        return {
            'cpu_usage': {'warning': 70.0, 'critical': 85.0, 'emergency': 95.0},
            'memory_usage': {'warning': 75.0, 'critical': 90.0, 'emergency': 98.0},
            'disk_usage': {'warning': 80.0, 'critical': 90.0, 'emergency': 95.0},
            'model_latency': {'warning': 100.0, 'critical': 500.0, 'emergency': 1000.0},
            'prediction_accuracy': {'warning': 0.85, 'critical': 0.75, 'emergency': 0.65},
            'data_quality_score': {'warning': 0.9, 'critical': 0.8, 'emergency': 0.7},
            'error_rate': {'warning': 0.01, 'critical': 0.05, 'emergency': 0.1},
            'throughput': {'warning': 10.0, 'critical': 5.0, 'emergency': 1.0}
        }
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous monitoring."""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Health monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join()
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                metrics = self._collect_metrics()
                self._store_metrics(metrics)
                self._check_thresholds(metrics)
                self._cleanup_old_data()
                
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics."""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        # Application metrics (simulated for demo)
        model_latency = self._measure_model_latency()
        prediction_accuracy = self._calculate_prediction_accuracy()
        data_quality_score = self._assess_data_quality()
        error_rate = self._calculate_error_rate()
        throughput = self._measure_throughput()
        
        return HealthMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            model_latency=model_latency,
            prediction_accuracy=prediction_accuracy,
            data_quality_score=data_quality_score,
            error_rate=error_rate,
            throughput=throughput
        )
    
    def _measure_model_latency(self) -> float:
        """Measure model inference latency."""
        try:
            start_time = time.time()
            # Simulate model inference
            time.sleep(0.01)  # Simulated processing time
            return (time.time() - start_time) * 1000  # Convert to milliseconds
        except Exception:
            return 1000.0  # High latency on error
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate recent prediction accuracy."""
        # Simulated accuracy calculation
        base_accuracy = 0.92
        noise = np.random.normal(0, 0.02)
        return max(0.0, min(1.0, base_accuracy + noise))
    
    def _assess_data_quality(self) -> float:
        """Assess current data quality."""
        # Simulated data quality assessment
        base_quality = 0.95
        noise = np.random.normal(0, 0.01)
        return max(0.0, min(1.0, base_quality + noise))
    
    def _calculate_error_rate(self) -> float:
        """Calculate recent error rate."""
        # Simulated error rate
        base_error_rate = 0.005
        noise = np.random.exponential(0.001)
        return min(1.0, base_error_rate + noise)
    
    def _measure_throughput(self) -> float:
        """Measure system throughput."""
        # Simulated throughput measurement
        base_throughput = 50.0
        noise = np.random.normal(0, 5.0)
        return max(0.0, base_throughput + noise)
    
    def _store_metrics(self, metrics: HealthMetrics):
        """Store metrics in history."""
        with self._lock:
            self.metrics_history.append(metrics)
    
    def _check_thresholds(self, metrics: HealthMetrics):
        """Check metrics against thresholds and generate alerts."""
        metric_values = {
            'cpu_usage': metrics.cpu_usage,
            'memory_usage': metrics.memory_usage,
            'disk_usage': metrics.disk_usage,
            'model_latency': metrics.model_latency,
            'prediction_accuracy': metrics.prediction_accuracy,
            'data_quality_score': metrics.data_quality_score,
            'error_rate': metrics.error_rate,
            'throughput': metrics.throughput
        }
        
        for metric_name, value in metric_values.items():
            thresholds = self.thresholds[metric_name]
            alert_level = self._determine_alert_level(value, thresholds, metric_name)
            
            if alert_level:
                self._create_alert(metric_name, value, thresholds, alert_level)
    
    def _determine_alert_level(
        self,
        value: float,
        thresholds: Dict[str, float],
        metric_name: str
    ) -> Optional[AlertLevel]:
        """Determine alert level based on value and thresholds."""
        # For accuracy-based metrics (higher is better)
        if metric_name in ['prediction_accuracy', 'data_quality_score']:
            if value < thresholds['emergency']:
                return AlertLevel.EMERGENCY
            elif value < thresholds['critical']:
                return AlertLevel.CRITICAL
            elif value < thresholds['warning']:
                return AlertLevel.WARNING
        
        # For throughput (higher is better, but measured differently)
        elif metric_name == 'throughput':
            if value < thresholds['emergency']:
                return AlertLevel.EMERGENCY
            elif value < thresholds['critical']:
                return AlertLevel.CRITICAL
            elif value < thresholds['warning']:
                return AlertLevel.WARNING
        
        # For other metrics (lower is better)
        else:
            if value > thresholds['emergency']:
                return AlertLevel.EMERGENCY
            elif value > thresholds['critical']:
                return AlertLevel.CRITICAL
            elif value > thresholds['warning']:
                return AlertLevel.WARNING
        
        return None
    
    def _create_alert(
        self,
        metric_name: str,
        value: float,
        thresholds: Dict[str, float],
        alert_level: AlertLevel
    ):
        """Create and process alert."""
        # Determine metric type
        metric_type_mapping = {
            'cpu_usage': MetricType.SYSTEM,
            'memory_usage': MetricType.SYSTEM,
            'disk_usage': MetricType.SYSTEM,
            'model_latency': MetricType.PERFORMANCE,
            'prediction_accuracy': MetricType.ACCURACY,
            'data_quality_score': MetricType.DATA_QUALITY,
            'error_rate': MetricType.PERFORMANCE,
            'throughput': MetricType.PERFORMANCE
        }
        
        alert = Alert(
            timestamp=datetime.now(),
            level=alert_level,
            metric_type=metric_type_mapping[metric_name],
            message=f"{metric_name.replace('_', ' ').title()}: {value:.2f}",
            value=value,
            threshold=thresholds[alert_level.value],
            metadata={'metric_name': metric_name, 'thresholds': thresholds}
        )
        
        with self._lock:
            self.alerts_history.append(alert)
        
        self.alert_callback(alert)
    
    def _cleanup_old_data(self):
        """Remove old metrics and alerts."""
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        
        with self._lock:
            self.metrics_history = [
                m for m in self.metrics_history
                if m.timestamp > cutoff_time
            ]
            self.alerts_history = [
                a for a in self.alerts_history
                if a.timestamp > cutoff_time
            ]
    
    def _default_alert_callback(self, alert: Alert):
        """Default alert handling."""
        logger.warning(f"ALERT [{alert.level.value.upper()}]: {alert.message}")
    
    def get_recent_metrics(self, hours: int = 1) -> List[HealthMetrics]:
        """Get metrics from recent hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        with self._lock:
            return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def get_recent_alerts(self, hours: int = 1) -> List[Alert]:
        """Get alerts from recent hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        with self._lock:
            return [a for a in self.alerts_history if a.timestamp > cutoff_time]
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        recent_metrics = self.get_recent_metrics(hours=1)
        recent_alerts = self.get_recent_alerts(hours=1)
        
        if not recent_metrics:
            return {"status": "no_data", "message": "No recent metrics available"}
        
        latest_metrics = recent_metrics[-1]
        
        # Calculate trends
        trends = self._calculate_trends(recent_metrics)
        
        # Categorize alerts
        alert_summary = self._summarize_alerts(recent_alerts)
        
        # Overall health score
        health_score = self._calculate_health_score(latest_metrics, recent_alerts)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "health_score": health_score,
            "status": self._determine_overall_status(health_score),
            "latest_metrics": latest_metrics.to_dict(),
            "trends": trends,
            "alert_summary": alert_summary,
            "circuit_breaker_status": {
                "model": self.model_circuit_breaker.state,
                "data": self.data_circuit_breaker.state
            },
            "metrics_count": len(recent_metrics),
            "alerts_count": len(recent_alerts)
        }
    
    def _calculate_trends(self, metrics: List[HealthMetrics]) -> Dict[str, str]:
        """Calculate trends for key metrics."""
        if len(metrics) < 2:
            return {}
        
        trends = {}
        metric_names = ['cpu_usage', 'memory_usage', 'model_latency', 'prediction_accuracy']
        
        for metric_name in metric_names:
            values = [getattr(m, metric_name) for m in metrics]
            if len(values) >= 2:
                recent_avg = np.mean(values[-5:])  # Last 5 readings
                older_avg = np.mean(values[-10:-5] if len(values) >= 10 else values[:-5])
                
                if recent_avg > older_avg * 1.1:
                    trends[metric_name] = "increasing"
                elif recent_avg < older_avg * 0.9:
                    trends[metric_name] = "decreasing"
                else:
                    trends[metric_name] = "stable"
        
        return trends
    
    def _summarize_alerts(self, alerts: List[Alert]) -> Dict[str, int]:
        """Summarize alerts by level."""
        summary = {level.value: 0 for level in AlertLevel}
        for alert in alerts:
            summary[alert.level.value] += 1
        return summary
    
    def _calculate_health_score(
        self,
        metrics: HealthMetrics,
        recent_alerts: List[Alert]
    ) -> float:
        """Calculate overall health score (0-100)."""
        base_score = 100.0
        
        # Deduct points for high resource usage
        if metrics.cpu_usage > 80:
            base_score -= (metrics.cpu_usage - 80) * 0.5
        if metrics.memory_usage > 80:
            base_score -= (metrics.memory_usage - 80) * 0.5
        
        # Deduct points for poor performance
        if metrics.model_latency > 100:
            base_score -= min(20, (metrics.model_latency - 100) * 0.1)
        
        # Deduct points for low accuracy
        if metrics.prediction_accuracy < 0.9:
            base_score -= (0.9 - metrics.prediction_accuracy) * 100
        
        # Deduct points for recent alerts
        for alert in recent_alerts:
            if alert.level == AlertLevel.EMERGENCY:
                base_score -= 10
            elif alert.level == AlertLevel.CRITICAL:
                base_score -= 5
            elif alert.level == AlertLevel.WARNING:
                base_score -= 2
        
        return max(0.0, min(100.0, base_score))
    
    def _determine_overall_status(self, health_score: float) -> str:
        """Determine overall system status."""
        if health_score >= 90:
            return "excellent"
        elif health_score >= 75:
            return "good"
        elif health_score >= 60:
            return "warning"
        elif health_score >= 40:
            return "critical"
        else:
            return "emergency"
    
    def save_state(self, path: str):
        """Save monitoring state to file."""
        state = {
            'metrics_history': [m.to_dict() for m in self.metrics_history],
            'alerts_history': [a.to_dict() for a in self.alerts_history],
            'thresholds': self.thresholds
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Monitoring state saved to {path}")


class ReliabilityTester:
    """Test system reliability under various conditions."""
    
    def __init__(self, monitor: AdvancedHealthMonitor):
        self.monitor = monitor
        self.test_results = []
    
    def run_load_test(self, duration_seconds: int = 60, requests_per_second: int = 10):
        """Run load test simulation."""
        logger.info(f"Starting load test: {requests_per_second} RPS for {duration_seconds}s")
        
        start_time = time.time()
        request_count = 0
        error_count = 0
        
        while time.time() - start_time < duration_seconds:
            try:
                # Simulate request processing
                time.sleep(1.0 / requests_per_second)
                request_count += 1
                
                # Simulate occasional errors
                if np.random.random() < 0.05:  # 5% error rate
                    error_count += 1
                    raise Exception("Simulated error")
                
            except Exception:
                error_count += 1
        
        success_rate = (request_count - error_count) / request_count
        logger.info(f"Load test complete: {success_rate:.2%} success rate")
        
        return {
            'duration': duration_seconds,
            'requests_per_second': requests_per_second,
            'total_requests': request_count,
            'errors': error_count,
            'success_rate': success_rate
        }
    
    def run_chaos_test(self):
        """Run chaos engineering test."""
        logger.info("Starting chaos test...")
        
        # Simulate various failure modes
        test_scenarios = [
            "high_cpu_load",
            "memory_pressure",
            "network_latency",
            "disk_io_errors"
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            logger.info(f"Testing scenario: {scenario}")
            start_time = time.time()
            
            # Simulate failure scenario
            if scenario == "high_cpu_load":
                self._simulate_cpu_load()
            elif scenario == "memory_pressure":
                self._simulate_memory_pressure()
            elif scenario == "network_latency":
                self._simulate_network_latency()
            elif scenario == "disk_io_errors":
                self._simulate_disk_errors()
            
            duration = time.time() - start_time
            results[scenario] = {
                'duration': duration,
                'status': 'completed'
            }
        
        logger.info("Chaos test complete")
        return results
    
    def _simulate_cpu_load(self):
        """Simulate high CPU load."""
        # Simulate CPU intensive task
        end_time = time.time() + 5  # 5 seconds
        while time.time() < end_time:
            sum(i * i for i in range(1000))
    
    def _simulate_memory_pressure(self):
        """Simulate memory pressure."""
        # Allocate memory temporarily
        memory_hog = [np.random.random(1000000) for _ in range(10)]
        time.sleep(2)
        del memory_hog
    
    def _simulate_network_latency(self):
        """Simulate network latency."""
        time.sleep(1)  # Simulate network delay
    
    def _simulate_disk_errors(self):
        """Simulate disk I/O errors."""
        # Simulate disk operations
        try:
            with open('/tmp/test_file', 'w') as f:
                f.write('test data')
            Path('/tmp/test_file').unlink()
        except Exception:
            pass


if __name__ == "__main__":
    # Generation 2 robustness demonstration
    logger.info("=== GENERATION 2: ROBUST MONITORING DEMO ===")
    
    # Initialize monitoring system
    monitor = AdvancedHealthMonitor()
    
    # Start monitoring
    monitor.start_monitoring(interval_seconds=5)
    
    try:
        # Run for demonstration
        time.sleep(30)
        
        # Generate health report
        health_report = monitor.generate_health_report()
        logger.info("Health Report:")
        logger.info(f"  Status: {health_report['status']}")
        logger.info(f"  Health Score: {health_report['health_score']:.1f}")
        logger.info(f"  Metrics Count: {health_report['metrics_count']}")
        logger.info(f"  Alerts Count: {health_report['alerts_count']}")
        
        # Test reliability
        reliability_tester = ReliabilityTester(monitor)
        load_test_results = reliability_tester.run_load_test(duration_seconds=10)
        logger.info(f"Load test results: {load_test_results}")
        
        # Save monitoring state
        monitor.save_state('monitoring_state.json')
        
    finally:
        monitor.stop_monitoring()
    
    logger.info("=== GENERATION 2 ROBUSTNESS IMPLEMENTATION COMPLETE ===")