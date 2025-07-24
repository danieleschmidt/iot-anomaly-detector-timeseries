"""Centralized logging configuration for IoT Anomaly Detector."""

import logging
import logging.handlers
import os
import re
import sys
import psutil
import threading
import time
from collections import deque, defaultdict
from pathlib import Path
from typing import Optional, Dict, Any

from .config import get_config


class SensitiveDataFilter(logging.Filter):
    """Filter to mask sensitive data in log messages."""
    
    # Enhanced patterns for sensitive data
    SENSITIVE_PATTERNS = [
        (re.compile(r'password[=:\s]+[^\s]+', re.IGNORECASE), 'password=***'),
        (re.compile(r'api[_\s]*key[=:\s]+[^\s]+', re.IGNORECASE), 'api_key=***'),
        (re.compile(r'token[=:\s]+[^\s]+', re.IGNORECASE), 'token=***'),
        (re.compile(r'secret[=:\s]+[^\s]+', re.IGNORECASE), 'secret=***'),
        (re.compile(r'://[^:]+:[^@]+@', re.IGNORECASE), '://user:***@'),  # DB connection strings
        # Additional security patterns
        (re.compile(r'/home/[^/\s]+', re.IGNORECASE), '/home/[USER]'),  # Home directories
        (re.compile(r'\\\\Users\\\\[^\\\\s]+', re.IGNORECASE), r'\\Users\\[USER]'),  # Windows user dirs
        (re.compile(r'/[^\s]*(?:/[^\s]*){3,}', re.IGNORECASE), '[PATH_REDACTED]'),  # Long paths
        (re.compile(r'[A-Z]:\\[^\s]*(?:\\[^\s]*){2,}', re.IGNORECASE), '[PATH_REDACTED]'),  # Windows paths
        (re.compile(r'auth[=:\s]+[^\s]+', re.IGNORECASE), 'auth=***'),
        (re.compile(r'bearer\s+[^\s]+', re.IGNORECASE), 'bearer ***'),
        (re.compile(r'ssh-[a-z0-9]+ [^\s]+', re.IGNORECASE), 'ssh-key ***'),  # SSH keys
    ]
    
    def filter(self, record) -> bool:
        """Filter sensitive data from log records."""
        if hasattr(record, 'msg') and record.msg:
            message = str(record.msg)
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                message = pattern.sub(replacement, message)
            record.msg = message
        return True


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record) -> str:
        """Format log record with structured data."""
        # Start with default format
        formatted = super().format(record)
        
        # Add structured data if available
        extra_data = []
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 'funcName',
                          'created', 'msecs', 'relativeCreated', 'thread',
                          'threadName', 'processName', 'process', 'getMessage',
                          'exc_info', 'exc_text', 'stack_info', 'asctime', 'taskName']:
                extra_data.append(f"{key}={value}")
        
        if extra_data:
            formatted += f" | {' '.join(extra_data)}"
        
        return formatted


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_sensitive_filter: bool = True
) -> None:
    """Set up centralized logging configuration.
    
    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        Path to log file. If None, logs only to console.
    max_file_size : int
        Maximum size of log file before rotation (bytes)
    backup_count : int
        Number of backup files to keep
    enable_sensitive_filter : bool
        Whether to enable sensitive data filtering
    """
    
    # Clear existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # Create formatters
    console_formatter = StructuredFormatter()
    file_formatter = StructuredFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    
    if enable_sensitive_filter:
        console_handler.addFilter(SensitiveDataFilter())
    
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        
        if enable_sensitive_filter:
            file_handler.addFilter(SensitiveDataFilter())
        
        root_logger.addHandler(file_handler)
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={level}, file={log_file}")


def setup_logging_from_config() -> None:
    """Set up logging using configuration from config system and environment."""
    
    # Get configuration values
    get_config()
    
    # Check for environment overrides
    log_level = os.getenv("IOT_LOG_LEVEL", "INFO")
    log_file = os.getenv("IOT_LOG_FILE", None)
    max_file_size = int(os.getenv("IOT_LOG_MAX_SIZE", "10485760"))  # 10MB default
    backup_count = int(os.getenv("IOT_LOG_BACKUP_COUNT", "5"))
    
    # Set up logging
    setup_logging(
        level=log_level,
        log_file=log_file,
        max_file_size=max_file_size,
        backup_count=backup_count
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__ or module name)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    # Ensure the logger inherits from root logger but has appropriate level
    if not logger.handlers:
        logger.setLevel(logging.NOTSET)  # This makes it inherit from parent
    return logger


def log_function_call(func):
    """Decorator to log function calls with parameters and timing."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function entry
        logger.debug(f"Entering {func.__name__}", extra={
            "function": func.__name__,
            "args_count": len(args),
            "kwargs_count": len(kwargs)
        })
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log successful completion
            logger.debug(f"Completed {func.__name__}", extra={
                "function": func.__name__,
                "duration": round(duration, 3),
                "status": "success"
            })
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            logger.error(f"Error in {func.__name__}: {e}", extra={
                "function": func.__name__,
                "duration": round(duration, 3),
                "status": "error",
                "error_type": type(e).__name__
            })
            raise
    
    return wrapper


def log_performance(operation: str, threshold: float = 1.0):
    """Decorator to log performance metrics for operations.
    
    Parameters
    ----------
    operation : str
        Name of the operation being measured
    threshold : float
        Time threshold in seconds above which to log a warning
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            log_level = logging.WARNING if duration > threshold else logging.INFO
            
            logger.log(log_level, f"Performance: {operation}", extra={
                "operation": operation,
                "duration": round(duration, 3),
                "threshold": threshold,
                "slow": duration > threshold
            })
            
            return result
        
        return wrapper
    return decorator


class PerformanceMetrics:
    """Comprehensive performance metrics collection and monitoring."""
    
    def __init__(self, buffer_size: int = 1000):
        """Initialize performance metrics collector.
        
        Parameters
        ----------
        buffer_size : int
            Maximum number of metrics to keep in memory
        """
        self.buffer_size = buffer_size
        self.metrics = defaultdict(lambda: deque(maxlen=buffer_size))
        self.counters = defaultdict(int)
        self.timers = {}
        self.logger = get_logger(__name__)
        self._lock = threading.Lock()
        
        # System monitoring
        self.process = psutil.Process()
        self.start_time = time.time()
        
        # GPU monitoring (if available)
        self.gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import GPUtil
            return len(GPUtil.getGPUs()) > 0
        except ImportError:
            return False
    
    def record_timing(self, operation: str, duration: float, **metadata) -> None:
        """Record timing metrics for an operation.
        
        Parameters
        ----------
        operation : str
            Name of the operation
        duration : float
            Duration in seconds
        **metadata
            Additional metadata to store with the metric
        """
        with self._lock:
            metric_entry = {
                'timestamp': time.time(),
                'operation': operation,
                'duration': duration,
                'metadata': metadata
            }
            self.metrics['timing'].append(metric_entry)
            self.counters[f'{operation}_count'] += 1
            
            # Log slow operations
            threshold = metadata.get('threshold', 1.0)
            if duration > threshold:
                self.logger.warning(
                    f"Slow operation detected: {operation}",
                    extra={
                        'operation': operation,
                        'duration': duration,
                        'threshold': threshold,
                        **metadata
                    }
                )
    
    def record_memory_usage(self, operation: str, **metadata) -> None:
        """Record current memory usage.
        
        Parameters
        ----------
        operation : str
            Context/operation name
        **metadata
            Additional metadata
        """
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            with self._lock:
                metric_entry = {
                    'timestamp': time.time(),
                    'operation': operation,
                    'rss_mb': memory_info.rss / (1024 * 1024),
                    'vms_mb': memory_info.vms / (1024 * 1024),
                    'memory_percent': memory_percent,
                    'metadata': metadata
                }
                self.metrics['memory'].append(metric_entry)
                
        except Exception as e:
            self.logger.error(f"Error recording memory usage: {e}")
    
    def record_gpu_usage(self, operation: str, **metadata) -> None:
        """Record GPU utilization if available.
        
        Parameters
        ----------
        operation : str
            Context/operation name
        **metadata
            Additional metadata
        """
        if not self.gpu_available:
            return
            
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            with self._lock:
                for i, gpu in enumerate(gpus):
                    metric_entry = {
                        'timestamp': time.time(),
                        'operation': operation,
                        'gpu_id': i,
                        'gpu_name': gpu.name,
                        'utilization_percent': gpu.load * 100,
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'temperature_c': gpu.temperature,
                        'metadata': metadata
                    }
                    self.metrics['gpu'].append(metric_entry)
                    
        except Exception as e:
            self.logger.error(f"Error recording GPU usage: {e}")
    
    def record_custom_metric(self, metric_type: str, value: Any, operation: str = None, **metadata) -> None:
        """Record custom metric.
        
        Parameters
        ----------
        metric_type : str
            Type of metric (e.g., 'cache_hit_rate', 'queue_size')
        value : Any
            Metric value
        operation : str, optional
            Associated operation
        **metadata
            Additional metadata
        """
        with self._lock:
            metric_entry = {
                'timestamp': time.time(),
                'metric_type': metric_type,
                'value': value,
                'operation': operation,
                'metadata': metadata
            }
            self.metrics[metric_type].append(metric_entry)
    
    def increment_counter(self, counter_name: str, increment: int = 1) -> None:
        """Increment a counter metric.
        
        Parameters
        ----------
        counter_name : str
            Name of the counter
        increment : int
            Amount to increment by
        """
        with self._lock:
            self.counters[counter_name] += increment
    
    def get_summary_stats(self, operation: str = None, last_n: int = None) -> Dict[str, Any]:
        """Get summary statistics for metrics.
        
        Parameters
        ----------
        operation : str, optional
            Filter by operation name
        last_n : int, optional
            Only consider last N entries
            
        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        with self._lock:
            stats = {
                'counters': dict(self.counters),
                'uptime_seconds': time.time() - self.start_time
            }
            
            # Timing statistics
            timing_metrics = list(self.metrics['timing'])
            if operation:
                timing_metrics = [m for m in timing_metrics if m['operation'] == operation]
            if last_n:
                timing_metrics = timing_metrics[-last_n:]
            
            if timing_metrics:
                durations = [m['duration'] for m in timing_metrics]
                stats['timing'] = {
                    'count': len(durations),
                    'min': min(durations),
                    'max': max(durations),
                    'avg': sum(durations) / len(durations),
                    'total': sum(durations)
                }
            
            # Memory statistics (most recent)
            memory_metrics = list(self.metrics['memory'])
            if memory_metrics:
                latest_memory = memory_metrics[-1]
                stats['memory'] = {
                    'current_rss_mb': latest_memory['rss_mb'],
                    'current_vms_mb': latest_memory['vms_mb'],
                    'current_percent': latest_memory['memory_percent']
                }
            
            # GPU statistics (most recent)
            gpu_metrics = list(self.metrics['gpu'])
            if gpu_metrics:
                latest_gpu = gpu_metrics[-1]
                stats['gpu'] = {
                    'utilization_percent': latest_gpu['utilization_percent'],
                    'memory_used_mb': latest_gpu['memory_used_mb'],
                    'memory_percent': latest_gpu['memory_percent'],
                    'temperature_c': latest_gpu['temperature_c']
                }
            
            return stats
    
    def export_metrics(self, output_path: str, format: str = 'json') -> None:
        """Export metrics to file.
        
        Parameters
        ----------
        output_path : str
            Output file path
        format : str
            Export format ('json' or 'csv')
        """
        import json
        import pandas as pd
        
        with self._lock:
            if format.lower() == 'json':
                export_data = {
                    'counters': dict(self.counters),
                    'metrics': {k: list(v) for k, v in self.metrics.items()},
                    'export_timestamp': time.time()
                }
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                # Flatten all metrics for CSV export
                all_records = []
                
                for metric_type, entries in self.metrics.items():
                    for entry in entries:
                        record = {
                            'metric_type': metric_type,
                            'timestamp': entry['timestamp'],
                            **entry
                        }
                        # Flatten metadata
                        if 'metadata' in record:
                            for k, v in record['metadata'].items():
                                record[f'meta_{k}'] = v
                            del record['metadata']
                        all_records.append(record)
                
                df = pd.DataFrame(all_records)
                df.to_csv(output_path, index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported metrics to {output_path}")


# Global performance metrics instance
_performance_metrics = PerformanceMetrics()


def get_performance_metrics() -> PerformanceMetrics:
    """Get the global performance metrics instance."""
    return _performance_metrics


def performance_monitor(operation: str = None, threshold: float = 1.0, 
                       track_memory: bool = False, track_gpu: bool = False):
    """Enhanced decorator for comprehensive performance monitoring.
    
    Parameters
    ----------
    operation : str, optional
        Operation name (defaults to function name)
    threshold : float
        Time threshold for slow operation warnings
    track_memory : bool
        Whether to track memory usage
    track_gpu : bool
        Whether to track GPU usage
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            metrics = get_performance_metrics()
            logger = get_logger(func.__module__)
            
            # Record initial state
            if track_memory:
                metrics.record_memory_usage(f"{op_name}_start")
            if track_gpu:
                metrics.record_gpu_usage(f"{op_name}_start")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics
                metrics.record_timing(
                    op_name, duration, 
                    threshold=threshold,
                    status='success',
                    args_count=len(args),
                    kwargs_count=len(kwargs)
                )
                
                if track_memory:
                    metrics.record_memory_usage(f"{op_name}_end")
                if track_gpu:
                    metrics.record_gpu_usage(f"{op_name}_end")
                
                # Log performance
                log_level = logging.WARNING if duration > threshold else logging.DEBUG
                logger.log(log_level, f"Performance: {op_name} completed", extra={
                    "operation": op_name,
                    "duration": round(duration, 3),
                    "threshold": threshold,
                    "slow": duration > threshold
                })
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error metrics
                metrics.record_timing(
                    op_name, duration,
                    threshold=threshold,
                    status='error',
                    error_type=type(e).__name__
                )
                metrics.increment_counter(f"{op_name}_errors")
                
                logger.error(f"Performance: {op_name} failed", extra={
                    "operation": op_name,
                    "duration": round(duration, 3),
                    "error_type": type(e).__name__
                })
                
                raise
        
        return wrapper
    return decorator


class PerformanceMonitor:
    """Context manager for monitoring performance of code blocks."""
    
    def __init__(self, operation: str, track_memory: bool = False, track_gpu: bool = False):
        """Initialize performance monitor context.
        
        Parameters
        ----------
        operation : str
            Name of the operation being monitored
        track_memory : bool
            Whether to track memory usage
        track_gpu : bool
            Whether to track GPU usage
        """
        self.operation = operation
        self.track_memory = track_memory
        self.track_gpu = track_gpu
        self.metrics = get_performance_metrics()
        self.logger = get_logger(__name__)
        self.start_time = None
    
    def __enter__(self):
        """Enter the monitoring context."""
        self.start_time = time.time()
        
        if self.track_memory:
            self.metrics.record_memory_usage(f"{self.operation}_start")
        if self.track_gpu:
            self.metrics.record_gpu_usage(f"{self.operation}_start")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the monitoring context."""
        duration = time.time() - self.start_time
        
        # Record metrics
        status = 'error' if exc_type else 'success'
        self.metrics.record_timing(
            self.operation, duration,
            status=status,
            error_type=exc_type.__name__ if exc_type else None
        )
        
        if self.track_memory:
            self.metrics.record_memory_usage(f"{self.operation}_end")
        if self.track_gpu:
            self.metrics.record_gpu_usage(f"{self.operation}_end")
        
        # Log result
        if exc_type:
            self.logger.error(f"Performance: {self.operation} failed", extra={
                "operation": self.operation,
                "duration": round(duration, 3),
                "error_type": exc_type.__name__
            })
        else:
            self.logger.debug(f"Performance: {self.operation} completed", extra={
                "operation": self.operation,
                "duration": round(duration, 3)
            })


# Initialize logging when module is imported
setup_logging_from_config()