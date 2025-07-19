"""Centralized logging configuration for IoT Anomaly Detector."""

import logging
import logging.handlers
import os
import re
import sys
from pathlib import Path
from typing import Optional

from .config import get_config


class SensitiveDataFilter(logging.Filter):
    """Filter to mask sensitive data in log messages."""
    
    # Patterns for sensitive data
    SENSITIVE_PATTERNS = [
        (re.compile(r'password[=:\s]+[^\s]+', re.IGNORECASE), 'password=***'),
        (re.compile(r'api[_\s]*key[=:\s]+[^\s]+', re.IGNORECASE), 'api_key=***'),
        (re.compile(r'token[=:\s]+[^\s]+', re.IGNORECASE), 'token=***'),
        (re.compile(r'secret[=:\s]+[^\s]+', re.IGNORECASE), 'secret=***'),
        (re.compile(r'://[^:]+:[^@]+@', re.IGNORECASE), '://user:***@'),  # DB connection strings
    ]
    
    def filter(self, record):
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
    
    def format(self, record):
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
    config = get_config()
    
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


# Initialize logging when module is imported
setup_logging_from_config()