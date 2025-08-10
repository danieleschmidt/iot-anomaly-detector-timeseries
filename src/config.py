"""Configuration management for IoT Anomaly Detector."""

import os
import yaml
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import json

from .security_utils import validate_file_path, sanitize_error_message, validate_file_size


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    database: str = "anomaly_detection"
    username: str = "postgres"
    password: str = ""
    connection_pool_size: int = 5
    connection_timeout: int = 30


@dataclass
class CacheConfig:
    """Caching configuration settings"""
    enabled: bool = True
    backend: str = "redis"  # redis, memory, file
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    ttl_seconds: int = 3600
    max_memory_mb: int = 256


@dataclass
class MonitoringConfig:
    """Monitoring and observability settings"""
    enabled: bool = True
    metrics_port: int = 8080
    health_check_interval: int = 30
    log_level: str = "INFO"
    export_metrics: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_usage": 85.0,
        "memory_usage": 85.0,
        "disk_usage": 90.0,
        "error_rate": 0.05
    })


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    enable_authentication: bool = False
    jwt_secret: Optional[str] = None
    jwt_expiry_hours: int = 24
    rate_limit_per_minute: int = 100
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_ssl: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None


@dataclass
class ModelConfig:
    """Model-specific configuration"""
    auto_retrain: bool = False
    retrain_threshold: float = 0.15  # Drift threshold
    model_registry_path: str = "saved_models"
    backup_models: int = 5
    validation_split: float = 0.2
    early_stopping_patience: int = 10


class Config:
    """Configuration manager with support for files, environment variables, and defaults."""
    
    # Default configuration values
    DEFAULTS = {
        'window_size': 30,
        'lstm_units': 32,
        'latent_dim': 16,
        'batch_size': 32,
        'epochs': 100,
        'threshold_factor': 3.0,
        'anomaly_start': 200,
        'anomaly_length': 20,
        'anomaly_magnitude': 3.0,
    }
    
    # Environment variable prefix
    ENV_PREFIX = "IOT_"
    
    def __init__(self, config_file: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration from file, environment variables, and defaults.
        
        Parameters
        ----------
        config_file : str, optional
            Path to YAML configuration file
        config_dict : dict, optional
            Configuration dictionary (used for testing)
        """
        self._config = {}
        
        # Start with defaults
        self._config.update(self.DEFAULTS)
        
        # Initialize structured configs
        self.database = DatabaseConfig()
        self.cache = CacheConfig()
        self.monitoring = MonitoringConfig()
        self.security = SecurityConfig()
        self.model = ModelConfig()
        
        # Override with file configuration if provided
        if config_file:
            self._load_from_file(config_file)
        
        # Override with test config if provided (for testing)
        if config_dict:
            self._config.update(config_dict)
        
        # Override with environment variables
        self._load_from_env()
        
        # Validate configuration
        self._validate()
        
        # Set as instance attributes
        self._set_attributes()
        
        logging.info(f"Configuration loaded: {self._config}")
    
    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from YAML file."""
        try:
            # Validate file path for security
            validated_path = validate_file_path(config_file)
            
            # Validate file size (config files should be small)
            validate_file_size(validated_path, max_size_mb=1.0)
            
            with open(validated_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                # Update core config
                self._config.update(file_config.get('core', {}))
                
                # Update structured configs
                if 'database' in file_config:
                    for key, value in file_config['database'].items():
                        if hasattr(self.database, key):
                            setattr(self.database, key, value)
                            
                if 'cache' in file_config:
                    for key, value in file_config['cache'].items():
                        if hasattr(self.cache, key):
                            setattr(self.cache, key, value)
                            
                if 'monitoring' in file_config:
                    for key, value in file_config['monitoring'].items():
                        if hasattr(self.monitoring, key):
                            setattr(self.monitoring, key, value)
                            
                if 'security' in file_config:
                    for key, value in file_config['security'].items():
                        if hasattr(self.security, key):
                            setattr(self.security, key, value)
                            
                if 'model' in file_config:
                    for key, value in file_config['model'].items():
                        if hasattr(self.model, key):
                            setattr(self.model, key, value)
                            
                logging.info(f"Loaded configuration from {sanitize_error_message(config_file)}")
        except (FileNotFoundError, ValueError) as e:
            # Re-raise validation errors as-is
            raise e
        except yaml.YAMLError as e:
            sanitized_error = sanitize_error_message(str(e))
            raise ValueError(f"Invalid YAML configuration file: {sanitized_error}") from e
        except Exception as e:
            sanitized_error = sanitize_error_message(str(e))
            raise ValueError(f"Unable to read configuration file: {sanitized_error}") from e
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_config = {}
        
        for key in self.DEFAULTS.keys():
            env_key = f"{self.ENV_PREFIX}{key.upper()}"
            env_value = os.getenv(env_key)
            
            if env_value is not None:
                # Convert to appropriate type based on default
                default_value = self.DEFAULTS[key]
                if isinstance(default_value, int):
                    try:
                        env_config[key] = int(env_value)
                    except ValueError:
                        logging.warning(f"Invalid integer value for {env_key}: {env_value}")
                        continue
                elif isinstance(default_value, float):
                    try:
                        env_config[key] = float(env_value)
                    except ValueError:
                        logging.warning(f"Invalid float value for {env_key}: {env_value}")
                        continue
                else:
                    env_config[key] = env_value
        
        if env_config:
            self._config.update(env_config)
            logging.info(f"Loaded environment variables: {list(env_config.keys())}")
    
    def _validate(self) -> None:
        """Validate configuration values."""
        # Validate positive values
        positive_fields = ['window_size', 'lstm_units', 'latent_dim', 'batch_size', 
                          'epochs', 'threshold_factor', 'anomaly_length']
        
        for field in positive_fields:
            value = self._config.get(field)
            if value is not None and value <= 0:
                raise ValueError(f"{field} must be positive, got {value}")
        
        # Validate specific constraints
        if self._config.get('anomaly_start', 0) < 0:
            raise ValueError("anomaly_start must be non-negative")
        
        if self._config.get('anomaly_magnitude', 0) < 0:
            raise ValueError("anomaly_magnitude must be non-negative")
    
    def _set_attributes(self) -> None:
        """Set configuration values as instance attributes."""
        for key, value in self._config.items():
            setattr(self, key.upper(), value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()
    
    def save(self, config_file: str) -> None:
        """Save current configuration to YAML file."""
        try:
            with open(config_file, 'w') as f:
                yaml.safe_dump(self._config, f, default_flow_style=False)
            logging.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logging.error(f"Failed to save configuration to {config_file}: {e}")
            raise ValueError(f"Unable to save configuration to {config_file}: {e}")
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self._config})"


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config(config_file: Optional[str] = None) -> Config:
    """Reload the global configuration."""
    global config
    config = Config(config_file)
    return config