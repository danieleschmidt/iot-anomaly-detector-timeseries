"""
Monitoring Service

Business logic for system monitoring, metrics collection, and health checks.
"""

import logging
import psutil
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict
import json
import numpy as np
import pandas as pd
from threading import Thread, Lock
import gc

logger = logging.getLogger(__name__)


class MetricType:
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MonitoringService:
    """
    Service for system monitoring and observability.
    
    Provides health checks, performance monitoring, resource tracking,
    and metrics collection for the anomaly detection system.
    """
    
    def __init__(
        self,
        metrics_dir: str = "metrics",
        retention_hours: int = 24,
        collection_interval: int = 60,
        enable_auto_collection: bool = True
    ):
        """
        Initialize the monitoring service.
        
        Args:
            metrics_dir: Directory for metrics storage
            retention_hours: Hours to retain metrics
            collection_interval: Seconds between metric collections
            enable_auto_collection: Whether to auto-collect metrics
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.retention_hours = retention_hours
        self.collection_interval = collection_interval
        self.enable_auto_collection = enable_auto_collection
        
        # Metrics storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._custom_metrics: Dict[str, Any] = {}
        self._health_status: Dict[str, Any] = {}
        
        # Performance tracking
        self._performance_history = deque(maxlen=1000)
        self._operation_timings: Dict[str, List[float]] = defaultdict(list)
        
        # Resource monitoring
        self._resource_history = deque(maxlen=1000)
        self._alert_thresholds = self._get_default_thresholds()
        
        # Thread safety
        self._lock = Lock()
        self._collection_thread = None
        
        # Start auto-collection if enabled
        if enable_auto_collection:
            self._start_auto_collection()
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collect current system metrics.
        
        Returns:
            System metrics snapshot
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': self._collect_cpu_metrics(),
            'memory': self._collect_memory_metrics(),
            'disk': self._collect_disk_metrics(),
            'network': self._collect_network_metrics(),
            'process': self._collect_process_metrics()
        }
        
        # Store in history
        with self._lock:
            self._resource_history.append(metrics)
        
        # Check for alerts
        alerts = self._check_resource_alerts(metrics)
        if alerts:
            metrics['alerts'] = alerts
        
        return metrics
    
    def collect_application_metrics(self) -> Dict[str, Any]:
        """
        Collect application-specific metrics.
        
        Returns:
            Application metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_metrics': self._collect_model_metrics(),
            'data_metrics': self._collect_data_metrics(),
            'api_metrics': self._collect_api_metrics(),
            'cache_metrics': self._collect_cache_metrics()
        }
        
        # Add custom metrics
        metrics['custom'] = self._custom_metrics.copy()
        
        return metrics
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: str = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a custom metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Optional metric labels
        """
        metric_data = {
            'timestamp': datetime.now().isoformat(),
            'value': value,
            'type': metric_type,
            'labels': labels or {}
        }
        
        with self._lock:
            self._metrics[name].append(metric_data)
    
    def record_operation_timing(
        self,
        operation: str,
        duration: float,
        success: bool = True
    ) -> None:
        """
        Record operation timing.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            success: Whether operation succeeded
        """
        timing_data = {
            'operation': operation,
            'duration': duration,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        with self._lock:
            self._operation_timings[operation].append(duration)
            self._performance_history.append(timing_data)
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status.
        
        Returns:
            Health status report
        """
        # Collect current metrics
        system_metrics = self.collect_system_metrics()
        app_metrics = self.collect_application_metrics()
        
        # Determine health status
        health_checks = {
            'system': self._check_system_health(system_metrics),
            'application': self._check_application_health(app_metrics),
            'dependencies': self._check_dependencies_health()
        }
        
        # Overall health
        overall_health = 'healthy'
        if any(check['status'] == 'unhealthy' for check in health_checks.values()):
            overall_health = 'unhealthy'
        elif any(check['status'] == 'degraded' for check in health_checks.values()):
            overall_health = 'degraded'
        
        self._health_status = {
            'status': overall_health,
            'timestamp': datetime.now().isoformat(),
            'checks': health_checks,
            'metrics': {
                'system': system_metrics,
                'application': app_metrics
            }
        }
        
        return self._health_status
    
    def get_performance_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get performance report for time range.
        
        Args:
            start_time: Start of report period
            end_time: End of report period
            
        Returns:
            Performance report
        """
        # Filter performance history
        history = list(self._performance_history)
        
        if start_time:
            history = [
                h for h in history
                if datetime.fromisoformat(h['timestamp']) >= start_time
            ]
        
        if end_time:
            history = [
                h for h in history
                if datetime.fromisoformat(h['timestamp']) <= end_time
            ]
        
        if not history:
            return {'message': 'No performance data available for specified period'}
        
        # Calculate statistics by operation
        operation_stats = {}
        for operation, timings in self._operation_timings.items():
            if timings:
                operation_stats[operation] = {
                    'count': len(timings),
                    'mean': np.mean(timings),
                    'median': np.median(timings),
                    'min': np.min(timings),
                    'max': np.max(timings),
                    'p95': np.percentile(timings, 95),
                    'p99': np.percentile(timings, 99)
                }
        
        # Overall statistics
        all_durations = [h['duration'] for h in history]
        success_rate = sum(1 for h in history if h['success']) / len(history) * 100
        
        return {
            'period': {
                'start': start_time.isoformat() if start_time else 'all',
                'end': end_time.isoformat() if end_time else 'now'
            },
            'total_operations': len(history),
            'success_rate': success_rate,
            'overall_stats': {
                'mean_duration': np.mean(all_durations),
                'median_duration': np.median(all_durations),
                'p95_duration': np.percentile(all_durations, 95),
                'p99_duration': np.percentile(all_durations, 99)
            },
            'operation_stats': operation_stats,
            'slowest_operations': self._get_slowest_operations(history, 10)
        }
    
    def get_resource_usage_trends(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get resource usage trends.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Resource usage trends
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter resource history
        history = [
            h for h in self._resource_history
            if datetime.fromisoformat(h['timestamp']) >= cutoff_time
        ]
        
        if not history:
            return {'message': 'No resource data available'}
        
        # Calculate trends
        trends = {
            'cpu': self._calculate_trend([h['cpu']['percent'] for h in history]),
            'memory': self._calculate_trend([h['memory']['percent'] for h in history]),
            'disk': self._calculate_trend([h['disk']['percent'] for h in history])
        }
        
        # Find peak usage times
        peak_cpu = max(history, key=lambda x: x['cpu']['percent'])
        peak_memory = max(history, key=lambda x: x['memory']['percent'])
        
        return {
            'period_hours': hours,
            'data_points': len(history),
            'trends': trends,
            'peak_usage': {
                'cpu': {
                    'value': peak_cpu['cpu']['percent'],
                    'timestamp': peak_cpu['timestamp']
                },
                'memory': {
                    'value': peak_memory['memory']['percent'],
                    'timestamp': peak_memory['timestamp']
                }
            },
            'current_usage': history[-1] if history else None
        }
    
    def set_alert_threshold(
        self,
        metric: str,
        threshold: float,
        comparison: str = '>'
    ) -> Dict[str, Any]:
        """
        Set alert threshold for a metric.
        
        Args:
            metric: Metric name
            threshold: Threshold value
            comparison: Comparison operator
            
        Returns:
            Threshold configuration
        """
        self._alert_thresholds[metric] = {
            'threshold': threshold,
            'comparison': comparison,
            'enabled': True
        }
        
        logger.info(f"Alert threshold set: {metric} {comparison} {threshold}")
        
        return {
            'metric': metric,
            'threshold': threshold,
            'comparison': comparison,
            'status': 'configured'
        }
    
    def export_metrics(
        self,
        format: str = 'prometheus',
        output_path: Optional[str] = None
    ) -> str:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format (prometheus, json, csv)
            output_path: Optional output file path
            
        Returns:
            Exported metrics string or file path
        """
        logger.info(f"Exporting metrics in {format} format")
        
        if format == 'prometheus':
            output = self._export_prometheus_format()
        elif format == 'json':
            output = self._export_json_format()
        elif format == 'csv':
            output = self._export_csv_format()
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(output)
            return str(path)
        
        return output
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate data for monitoring dashboard.
        
        Returns:
            Dashboard-ready data
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'health': self.get_health_status(),
            'system_metrics': self.collect_system_metrics(),
            'application_metrics': self.collect_application_metrics(),
            'performance': self.get_performance_report(),
            'trends': self.get_resource_usage_trends(hours=6),
            'alerts': self._get_active_alerts()
        }
    
    def cleanup_old_metrics(self) -> Dict[str, Any]:
        """
        Clean up metrics older than retention period.
        
        Returns:
            Cleanup summary
        """
        logger.info("Cleaning up old metrics")
        
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        cleaned_metrics = 0
        cleaned_performance = 0
        
        with self._lock:
            # Clean metrics
            for metric_name, values in self._metrics.items():
                original_len = len(values)
                filtered = [
                    v for v in values
                    if datetime.fromisoformat(v['timestamp']) >= cutoff_time
                ]
                self._metrics[metric_name] = deque(filtered, maxlen=10000)
                cleaned_metrics += original_len - len(filtered)
            
            # Clean performance history
            original_len = len(self._performance_history)
            self._performance_history = deque(
                [h for h in self._performance_history
                 if datetime.fromisoformat(h['timestamp']) >= cutoff_time],
                maxlen=1000
            )
            cleaned_performance = original_len - len(self._performance_history)
        
        # Force garbage collection
        gc.collect()
        
        return {
            'cleaned_metrics': cleaned_metrics,
            'cleaned_performance': cleaned_performance,
            'retention_hours': self.retention_hours,
            'timestamp': datetime.now().isoformat()
        }
    
    def _collect_cpu_metrics(self) -> Dict[str, Any]:
        """Collect CPU metrics."""
        return {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }
    
    def _collect_memory_metrics(self) -> Dict[str, Any]:
        """Collect memory metrics."""
        mem = psutil.virtual_memory()
        return {
            'percent': mem.percent,
            'total': mem.total,
            'available': mem.available,
            'used': mem.used,
            'free': mem.free
        }
    
    def _collect_disk_metrics(self) -> Dict[str, Any]:
        """Collect disk metrics."""
        disk = psutil.disk_usage('/')
        return {
            'percent': disk.percent,
            'total': disk.total,
            'used': disk.used,
            'free': disk.free
        }
    
    def _collect_network_metrics(self) -> Dict[str, Any]:
        """Collect network metrics."""
        net = psutil.net_io_counters()
        return {
            'bytes_sent': net.bytes_sent,
            'bytes_recv': net.bytes_recv,
            'packets_sent': net.packets_sent,
            'packets_recv': net.packets_recv,
            'errin': net.errin,
            'errout': net.errout
        }
    
    def _collect_process_metrics(self) -> Dict[str, Any]:
        """Collect process-specific metrics."""
        process = psutil.Process()
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_percent': process.memory_percent(),
            'num_threads': process.num_threads(),
            'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0,
            'create_time': process.create_time()
        }
    
    def _collect_model_metrics(self) -> Dict[str, Any]:
        """Collect model-related metrics."""
        # In production, would collect actual model metrics
        return {
            'models_loaded': len(self._custom_metrics.get('models', [])),
            'inference_count': self._custom_metrics.get('inference_count', 0),
            'training_count': self._custom_metrics.get('training_count', 0),
            'avg_inference_time': self._custom_metrics.get('avg_inference_time', 0)
        }
    
    def _collect_data_metrics(self) -> Dict[str, Any]:
        """Collect data processing metrics."""
        return {
            'records_processed': self._custom_metrics.get('records_processed', 0),
            'data_quality_score': self._custom_metrics.get('data_quality_score', 100),
            'drift_score': self._custom_metrics.get('drift_score', 0)
        }
    
    def _collect_api_metrics(self) -> Dict[str, Any]:
        """Collect API metrics."""
        return {
            'requests_total': self._custom_metrics.get('api_requests', 0),
            'errors_total': self._custom_metrics.get('api_errors', 0),
            'avg_response_time': self._custom_metrics.get('avg_response_time', 0)
        }
    
    def _collect_cache_metrics(self) -> Dict[str, Any]:
        """Collect cache metrics."""
        return {
            'hit_rate': self._custom_metrics.get('cache_hit_rate', 0),
            'size_mb': self._custom_metrics.get('cache_size_mb', 0),
            'evictions': self._custom_metrics.get('cache_evictions', 0)
        }
    
    def _check_system_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check system health based on metrics."""
        issues = []
        
        # Check CPU
        if metrics['cpu']['percent'] > 90:
            issues.append('High CPU usage')
        
        # Check memory
        if metrics['memory']['percent'] > 90:
            issues.append('High memory usage')
        
        # Check disk
        if metrics['disk']['percent'] > 90:
            issues.append('Low disk space')
        
        status = 'healthy'
        if issues:
            status = 'degraded' if len(issues) == 1 else 'unhealthy'
        
        return {
            'status': status,
            'issues': issues,
            'metrics': metrics
        }
    
    def _check_application_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check application health."""
        issues = []
        
        # Check model metrics
        if metrics['model_metrics']['models_loaded'] == 0:
            issues.append('No models loaded')
        
        # Check API metrics
        if metrics['api_metrics']['errors_total'] > 100:
            issues.append('High API error rate')
        
        status = 'healthy' if not issues else 'degraded'
        
        return {
            'status': status,
            'issues': issues
        }
    
    def _check_dependencies_health(self) -> Dict[str, Any]:
        """Check health of external dependencies."""
        # In production, would check actual dependencies
        return {
            'status': 'healthy',
            'dependencies': {
                'database': 'healthy',
                'cache': 'healthy',
                'storage': 'healthy'
            }
        }
    
    def _check_resource_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for resource alerts."""
        alerts = []
        
        for metric_path, threshold_config in self._alert_thresholds.items():
            if not threshold_config.get('enabled', True):
                continue
            
            # Extract metric value
            value = self._get_metric_value(metrics, metric_path)
            if value is None:
                continue
            
            # Check threshold
            threshold = threshold_config['threshold']
            comparison = threshold_config['comparison']
            
            triggered = False
            if comparison == '>' and value > threshold:
                triggered = True
            elif comparison == '<' and value < threshold:
                triggered = True
            elif comparison == '>=' and value >= threshold:
                triggered = True
            elif comparison == '<=' and value <= threshold:
                triggered = True
            
            if triggered:
                alerts.append({
                    'metric': metric_path,
                    'value': value,
                    'threshold': threshold,
                    'comparison': comparison,
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def _get_metric_value(self, metrics: Dict[str, Any], path: str) -> Optional[float]:
        """Extract metric value from nested dict."""
        parts = path.split('.')
        value = metrics
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return float(value) if isinstance(value, (int, float)) else None
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend from values."""
        if len(values) < 2:
            return {'direction': 'stable', 'change': 0}
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend direction
        if slope > 0.1:
            direction = 'increasing'
        elif slope < -0.1:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        return {
            'direction': direction,
            'change': float(slope),
            'current': float(values[-1]),
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
    
    def _get_slowest_operations(
        self,
        history: List[Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get slowest operations from history."""
        sorted_ops = sorted(history, key=lambda x: x['duration'], reverse=True)
        return sorted_ops[:limit]
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        # Check current metrics against thresholds
        current_metrics = self.collect_system_metrics()
        return self._check_resource_alerts(current_metrics)
    
    def _get_default_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Get default alert thresholds."""
        return {
            'cpu.percent': {'threshold': 90, 'comparison': '>', 'enabled': True},
            'memory.percent': {'threshold': 90, 'comparison': '>', 'enabled': True},
            'disk.percent': {'threshold': 90, 'comparison': '>', 'enabled': True}
        }
    
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for metric_name, values in self._metrics.items():
            if values:
                latest = values[-1]
                # Format: metric_name{label="value"} value timestamp
                labels = ','.join([f'{k}="{v}"' for k, v in latest.get('labels', {}).items()])
                line = f"{metric_name}"
                if labels:
                    line += f"{{{labels}}}"
                line += f" {latest['value']}"
                lines.append(line)
        
        return '\n'.join(lines)
    
    def _export_json_format(self) -> str:
        """Export metrics in JSON format."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'performance': list(self._performance_history),
            'resources': list(self._resource_history)
        }
        
        for metric_name, values in self._metrics.items():
            export_data['metrics'][metric_name] = [v for v in values]
        
        return json.dumps(export_data, indent=2)
    
    def _export_csv_format(self) -> str:
        """Export metrics in CSV format."""
        # Convert to DataFrame for easy CSV export
        rows = []
        
        for metric_name, values in self._metrics.items():
            for value in values:
                row = {
                    'metric': metric_name,
                    'value': value['value'],
                    'timestamp': value['timestamp'],
                    'type': value.get('type', 'gauge')
                }
                # Add labels as columns
                for k, v in value.get('labels', {}).items():
                    row[f'label_{k}'] = v
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            return df.to_csv(index=False)
        
        return "metric,value,timestamp,type\n"
    
    def _start_auto_collection(self) -> None:
        """Start automatic metric collection thread."""
        def collect_loop():
            while self.enable_auto_collection:
                try:
                    self.collect_system_metrics()
                    self.collect_application_metrics()
                except Exception as e:
                    logger.error(f"Error in metric collection: {e}")
                
                time.sleep(self.collection_interval)
        
        self._collection_thread = Thread(target=collect_loop, daemon=True)
        self._collection_thread.start()
        logger.info("Started automatic metric collection")