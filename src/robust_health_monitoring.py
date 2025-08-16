"""Robust Health Monitoring System for IoT Anomaly Detection Pipeline.

This module provides comprehensive health monitoring, alerting, and automatic
recovery mechanisms to ensure system reliability and uptime.
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import psutil

from .circuit_breaker import CircuitBreaker
from .logging_config import get_logger


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric definition."""
    name: str
    current_value: float
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    description: str = ""
    last_updated: float = field(default_factory=time.time)


@dataclass
class Alert:
    """System alert definition."""
    id: str
    severity: AlertSeverity
    component: str
    message: str
    details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class ComponentHealth:
    """Health status for a system component."""
    name: str
    status: HealthStatus
    metrics: Dict[str, HealthMetric]
    last_check: float = field(default_factory=time.time)
    error_count: int = 0
    uptime_start: float = field(default_factory=time.time)


class RobustHealthMonitoring:
    """Comprehensive health monitoring and alerting system."""

    def __init__(
        self,
        check_interval: float = 5.0,
        enable_auto_recovery: bool = True,
        enable_alerting: bool = True,
        alert_callbacks: Optional[List[Callable]] = None
    ):
        """Initialize the health monitoring system.
        
        Args:
            check_interval: Interval between health checks (seconds)
            enable_auto_recovery: Enable automatic recovery mechanisms
            enable_alerting: Enable alert generation
            alert_callbacks: List of callback functions for alerts
        """
        self.logger = get_logger(__name__)
        self.check_interval = check_interval
        self.enable_auto_recovery = enable_auto_recovery
        self.enable_alerting = enable_alerting
        self.alert_callbacks = alert_callbacks or []

        # System state
        self.components: Dict[str, ComponentHealth] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_history = deque(maxlen=1000)
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))

        # Threading
        self.monitoring_lock = threading.RLock()
        self.is_running = False
        self.monitoring_task = None

        # Circuit breakers for auto-recovery
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Recovery strategies
        self.recovery_strategies: Dict[str, Callable] = {}

        # System metrics
        self.system_start_time = time.time()

        # Initialize core components
        self._initialize_core_components()

        self.logger.info("Health monitoring system initialized")

    def _initialize_core_components(self) -> None:
        """Initialize monitoring for core system components."""

        # System Resources
        self.register_component(
            "system_resources",
            {
                "cpu_usage": HealthMetric("cpu_usage", 0.0, 80.0, 95.0, "%", "CPU utilization"),
                "memory_usage": HealthMetric("memory_usage", 0.0, 80.0, 95.0, "%", "Memory utilization"),
                "disk_usage": HealthMetric("disk_usage", 0.0, 85.0, 95.0, "%", "Disk utilization"),
                "load_average": HealthMetric("load_average", 0.0, 2.0, 5.0, "", "System load average")
            }
        )

        # Application Performance
        self.register_component(
            "application_performance",
            {
                "response_time": HealthMetric("response_time", 0.0, 100.0, 500.0, "ms", "Average response time"),
                "error_rate": HealthMetric("error_rate", 0.0, 5.0, 10.0, "%", "Error rate percentage"),
                "throughput": HealthMetric("throughput", 0.0, 0.0, 0.0, "ops/sec", "Operations per second"),
                "queue_depth": HealthMetric("queue_depth", 0.0, 100.0, 500.0, "items", "Processing queue depth")
            }
        )

        # Model Performance
        self.register_component(
            "model_performance",
            {
                "inference_latency": HealthMetric("inference_latency", 0.0, 50.0, 200.0, "ms", "Model inference latency"),
                "prediction_confidence": HealthMetric("prediction_confidence", 1.0, 0.7, 0.5, "", "Average prediction confidence"),
                "model_accuracy": HealthMetric("model_accuracy", 1.0, 0.8, 0.7, "", "Model accuracy estimate"),
                "drift_score": HealthMetric("drift_score", 0.0, 0.3, 0.5, "", "Data drift detection score")
            }
        )

        # Data Pipeline
        self.register_component(
            "data_pipeline",
            {
                "data_ingestion_rate": HealthMetric("data_ingestion_rate", 0.0, 0.0, 0.0, "samples/sec", "Data ingestion rate"),
                "validation_failures": HealthMetric("validation_failures", 0.0, 5.0, 10.0, "%", "Data validation failure rate"),
                "processing_lag": HealthMetric("processing_lag", 0.0, 10.0, 30.0, "sec", "Data processing lag")
            }
        )

    def register_component(self, name: str, metrics: Dict[str, HealthMetric]) -> None:
        """Register a new component for monitoring.
        
        Args:
            name: Component name
            metrics: Dictionary of metrics to monitor
        """
        with self.monitoring_lock:
            self.components[name] = ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY,
                metrics=metrics
            )

            # Initialize circuit breaker for component
            self.circuit_breakers[name] = CircuitBreaker(
                failure_threshold=3,
                timeout_duration=60.0
            )

        self.logger.info(f"Registered component: {name} with {len(metrics)} metrics")

    def update_metric(self, component: str, metric_name: str, value: float) -> None:
        """Update a specific metric value.
        
        Args:
            component: Component name
            metric_name: Metric name
            value: New metric value
        """
        with self.monitoring_lock:
            if component in self.components and metric_name in self.components[component].metrics:
                metric = self.components[component].metrics[metric_name]
                metric.current_value = value
                metric.last_updated = time.time()

                # Store in history
                self.metrics_history[f"{component}.{metric_name}"].append({
                    'timestamp': time.time(),
                    'value': value
                })

                # Check thresholds and generate alerts if needed
                self._check_metric_thresholds(component, metric_name, metric)

    def _check_metric_thresholds(self, component: str, metric_name: str, metric: HealthMetric) -> None:
        """Check metric against thresholds and generate alerts."""
        if not self.enable_alerting:
            return

        alert_id = f"{component}.{metric_name}"
        current_value = metric.current_value

        # Determine alert level
        alert_level = None
        if current_value >= metric.threshold_critical:
            alert_level = AlertSeverity.CRITICAL
        elif current_value >= metric.threshold_warning:
            alert_level = AlertSeverity.WARNING

        # Generate or clear alert
        if alert_level:
            if alert_id not in self.alerts or self.alerts[alert_id].severity != alert_level:
                alert = Alert(
                    id=alert_id,
                    severity=alert_level,
                    component=component,
                    message=f"{metric.description} is {alert_level.value}: {current_value}{metric.unit}",
                    details={
                        'metric_name': metric_name,
                        'current_value': current_value,
                        'threshold_warning': metric.threshold_warning,
                        'threshold_critical': metric.threshold_critical,
                        'unit': metric.unit
                    }
                )

                self._generate_alert(alert)
        # Clear existing alert if metric is now healthy
        elif alert_id in self.alerts and not self.alerts[alert_id].resolved:
            self._resolve_alert(alert_id)

    def _generate_alert(self, alert: Alert) -> None:
        """Generate a new alert."""
        with self.monitoring_lock:
            self.alerts[alert.id] = alert
            self.alert_history.append(alert)

        # Update component status
        self._update_component_status(alert.component)

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

        # Log alert
        self.logger.warning(f"ALERT [{alert.severity.value.upper()}] {alert.component}: {alert.message}")

        # Trigger auto-recovery if enabled
        if self.enable_auto_recovery and alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            self._trigger_auto_recovery(alert.component, alert)

    def _resolve_alert(self, alert_id: str) -> None:
        """Resolve an existing alert."""
        if alert_id in self.alerts:
            with self.monitoring_lock:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].timestamp = time.time()

            component = alert_id.split('.')[0]
            self._update_component_status(component)

            self.logger.info(f"Alert resolved: {alert_id}")

    def _update_component_status(self, component_name: str) -> None:
        """Update overall component health status."""
        if component_name not in self.components:
            return

        component = self.components[component_name]

        # Check for active alerts
        critical_alerts = [
            alert for alert in self.alerts.values()
            if alert.component == component_name and not alert.resolved and alert.severity == AlertSeverity.CRITICAL
        ]

        warning_alerts = [
            alert for alert in self.alerts.values()
            if alert.component == component_name and not alert.resolved and alert.severity == AlertSeverity.WARNING
        ]

        # Determine status
        if critical_alerts:
            component.status = HealthStatus.CRITICAL
        elif warning_alerts:
            component.status = HealthStatus.WARNING
        elif component.error_count > 0:
            component.status = HealthStatus.DEGRADED
        else:
            component.status = HealthStatus.HEALTHY

        component.last_check = time.time()

    def _trigger_auto_recovery(self, component: str, alert: Alert) -> None:
        """Trigger automatic recovery for a component."""
        if component in self.recovery_strategies:
            try:
                self.logger.info(f"Triggering auto-recovery for {component}")
                recovery_func = self.recovery_strategies[component]
                recovery_func(alert)

                # Record recovery attempt
                self.circuit_breakers[component].record_success()

            except Exception as e:
                self.logger.error(f"Auto-recovery failed for {component}: {e}")
                self.circuit_breakers[component].record_failure()

    def register_recovery_strategy(self, component: str, recovery_func: Callable[[Alert], None]) -> None:
        """Register an auto-recovery strategy for a component.
        
        Args:
            component: Component name
            recovery_func: Function to call for recovery (takes Alert as parameter)
        """
        self.recovery_strategies[component] = recovery_func
        self.logger.info(f"Registered recovery strategy for {component}")

    async def start_monitoring(self) -> None:
        """Start the health monitoring loop."""
        self.is_running = True
        self.logger.info("Starting health monitoring")

        try:
            while self.is_running:
                await self._perform_health_check()
                await asyncio.sleep(self.check_interval)
        except Exception as e:
            self.logger.error(f"Health monitoring error: {e}")
        finally:
            self.is_running = False

    def stop_monitoring(self) -> None:
        """Stop the health monitoring loop."""
        self.is_running = False
        self.logger.info("Stopped health monitoring")

    async def _perform_health_check(self) -> None:
        """Perform comprehensive health check."""
        try:
            # Update system resource metrics
            await self._update_system_metrics()

            # Check component circuit breakers
            self._check_circuit_breakers()

            # Perform custom health checks
            await self._perform_custom_checks()

        except Exception as e:
            self.logger.error(f"Health check error: {e}")

    async def _update_system_metrics(self) -> None:
        """Update system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.update_metric("system_resources", "cpu_usage", cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            self.update_metric("system_resources", "memory_usage", memory.percent)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.update_metric("system_resources", "disk_usage", disk_percent)

            # Load average (Unix/Linux systems)
            try:
                load_avg = psutil.getloadavg()[0]  # 1-minute load average
                self.update_metric("system_resources", "load_average", load_avg)
            except AttributeError:
                # Windows doesn't have load average
                pass

        except Exception as e:
            self.logger.error(f"System metrics update failed: {e}")

    def _check_circuit_breakers(self) -> None:
        """Check circuit breaker states and update component status."""
        for component_name, circuit_breaker in self.circuit_breakers.items():
            if component_name in self.components:
                component = self.components[component_name]

                if not circuit_breaker.can_execute():
                    # Circuit breaker is open - mark component as degraded
                    if component.status == HealthStatus.HEALTHY:
                        component.status = HealthStatus.DEGRADED
                        component.error_count += 1

                        # Generate alert
                        alert = Alert(
                            id=f"{component_name}.circuit_breaker",
                            severity=AlertSeverity.WARNING,
                            component=component_name,
                            message=f"Circuit breaker open for {component_name}",
                            details={'circuit_breaker_state': circuit_breaker.state.value}
                        )
                        self._generate_alert(alert)

    async def _perform_custom_checks(self) -> None:
        """Perform custom health checks."""
        # This can be extended with specific application health checks
        pass

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        with self.monitoring_lock:
            summary = {
                'overall_status': self._calculate_overall_status(),
                'components': {},
                'active_alerts': len([a for a in self.alerts.values() if not a.resolved]),
                'uptime_seconds': time.time() - self.system_start_time,
                'last_check': time.time()
            }

            # Component details
            for name, component in self.components.items():
                summary['components'][name] = {
                    'status': component.status.value,
                    'metrics': {
                        metric_name: {
                            'value': metric.current_value,
                            'unit': metric.unit,
                            'threshold_warning': metric.threshold_warning,
                            'threshold_critical': metric.threshold_critical,
                            'last_updated': metric.last_updated
                        }
                        for metric_name, metric in component.metrics.items()
                    },
                    'error_count': component.error_count,
                    'uptime': time.time() - component.uptime_start
                }

            return summary

    def _calculate_overall_status(self) -> str:
        """Calculate overall system health status."""
        if not self.components:
            return HealthStatus.UNKNOWN.value

        statuses = [comp.status for comp in self.components.values()]

        if any(s == HealthStatus.CRITICAL for s in statuses):
            return HealthStatus.CRITICAL.value
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED.value
        elif any(s == HealthStatus.WARNING for s in statuses):
            return HealthStatus.WARNING.value
        else:
            return HealthStatus.HEALTHY.value

    def get_alerts(self, include_resolved: bool = False, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get alerts with optional filtering.
        
        Args:
            include_resolved: Include resolved alerts
            severity: Filter by severity level
            
        Returns:
            List of alerts matching criteria
        """
        alerts = list(self.alerts.values())

        if not include_resolved:
            alerts = [a for a in alerts if not a.resolved]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if alert was acknowledged, False otherwise
        """
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            self.logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False

    def get_metrics_history(self, component: str, metric: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical metrics data.
        
        Args:
            component: Component name
            metric: Metric name
            hours: Number of hours of history to return
            
        Returns:
            List of historical data points
        """
        key = f"{component}.{metric}"
        if key not in self.metrics_history:
            return []

        cutoff_time = time.time() - (hours * 3600)
        return [
            point for point in self.metrics_history[key]
            if point['timestamp'] >= cutoff_time
        ]

    def export_health_report(self, filepath: str) -> None:
        """Export comprehensive health report to file.
        
        Args:
            filepath: Output file path
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'health_summary': self.get_health_summary(),
            'alerts': [
                {
                    'id': alert.id,
                    'severity': alert.severity.value,
                    'component': alert.component,
                    'message': alert.message,
                    'timestamp': alert.timestamp,
                    'acknowledged': alert.acknowledged,
                    'resolved': alert.resolved
                }
                for alert in self.alert_history
            ],
            'metrics_summary': {}
        }

        # Add metrics summary
        for component_name, component in self.components.items():
            report['metrics_summary'][component_name] = {}
            for metric_name, metric in component.metrics.items():
                history = self.get_metrics_history(component_name, metric_name, 1)
                if history:
                    values = [h['value'] for h in history]
                    report['metrics_summary'][component_name][metric_name] = {
                        'current': metric.current_value,
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'count': len(values)
                    }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Health report exported to {filepath}")


# Example alert callback functions
def log_alert_callback(alert: Alert) -> None:
    """Example callback that logs alerts."""
    logger = get_logger("alert_callback")
    logger.warning(f"Alert: {alert.severity.value} - {alert.component} - {alert.message}")


def email_alert_callback(alert: Alert) -> None:
    """Example callback for email alerts (placeholder implementation)."""
    # In a real implementation, this would send an email
    logger = get_logger("email_alerts")
    if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
        logger.info(f"Would send email alert: {alert.message}")


# Example recovery strategies
def restart_component_strategy(alert: Alert) -> None:
    """Example recovery strategy that restarts a component."""
    logger = get_logger("recovery")
    logger.info(f"Recovery: Restarting component {alert.component}")
    # In a real implementation, this would restart the component


def scale_resources_strategy(alert: Alert) -> None:
    """Example recovery strategy that scales resources."""
    logger = get_logger("recovery")
    if "cpu_usage" in alert.id or "memory_usage" in alert.id:
        logger.info(f"Recovery: Scaling resources for {alert.component}")
        # In a real implementation, this would trigger resource scaling


# CLI Interface
def main() -> None:
    """CLI entry point for health monitoring."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Robust Health Monitoring System"
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=5.0,
        help="Health check interval in seconds"
    )
    parser.add_argument(
        "--export-report",
        help="Export health report to file"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Monitoring duration in seconds"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = get_logger(__name__)

    # Create health monitor
    health_monitor = RobustHealthMonitoring(
        check_interval=args.check_interval,
        enable_auto_recovery=True,
        enable_alerting=True,
        alert_callbacks=[log_alert_callback, email_alert_callback]
    )

    # Register recovery strategies
    health_monitor.register_recovery_strategy("system_resources", restart_component_strategy)
    health_monitor.register_recovery_strategy("application_performance", scale_resources_strategy)

    async def run_monitoring():
        logger.info(f"Starting health monitoring for {args.duration} seconds")

        # Start monitoring
        monitoring_task = asyncio.create_task(health_monitor.start_monitoring())

        # Wait for specified duration
        await asyncio.sleep(args.duration)

        # Stop monitoring
        health_monitor.stop_monitoring()
        monitoring_task.cancel()

        # Print health summary
        summary = health_monitor.get_health_summary()
        print("\n" + "="*50)
        print("HEALTH MONITORING SUMMARY")
        print("="*50)
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Active Alerts: {summary['active_alerts']}")
        print(f"Uptime: {summary['uptime_seconds']:.1f} seconds")

        for comp_name, comp_data in summary['components'].items():
            print(f"\n{comp_name}: {comp_data['status']}")
            for metric_name, metric_data in comp_data['metrics'].items():
                print(f"  {metric_name}: {metric_data['value']:.2f}{metric_data['unit']}")

        # Export report if requested
        if args.export_report:
            health_monitor.export_health_report(args.export_report)
            print(f"\nHealth report exported to: {args.export_report}")

    asyncio.run(run_monitoring())


if __name__ == "__main__":
    main()
