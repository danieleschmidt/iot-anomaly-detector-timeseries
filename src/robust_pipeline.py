"""
Robust IoT Anomaly Detection Pipeline
Generation 2: Enhanced error handling, validation, logging, and resilience
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union, List
import joblib
import time
import traceback
from contextlib import contextmanager
import os
import psutil
import gc

from .data_preprocessor import DataPreprocessor
from .autoencoder_model import build_autoencoder, get_training_callbacks
from .anomaly_detector import AnomalyDetector
from .data_validator import DataValidator, ValidationLevel
from .circuit_breaker import CircuitBreaker
from .retry_manager import RetryManager
from .security_utils import validate_file_path, sanitize_error_message
from .logging_config import setup_logging

# Enhanced logging configuration
logger = logging.getLogger(__name__)


class RobustAnomalyPipeline:
    """Production-ready anomaly detection pipeline with comprehensive error handling."""
    
    def __init__(
        self,
        window_size: int = 30,
        latent_dim: int = 16,
        lstm_units: int = 32,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        max_memory_usage_gb: float = 4.0,
        enable_validation: bool = True,
        validation_level: ValidationLevel = ValidationLevel.MODERATE,
        retry_config: Optional[Dict[str, Any]] = None,
        circuit_breaker_config: Optional[Dict[str, Any]] = None
    ):
        # Core parameters
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.max_memory_usage_gb = max_memory_usage_gb
        
        # Validation settings
        self.enable_validation = enable_validation
        self.validation_level = validation_level
        
        # Components
        self.preprocessor = None
        self.model = None
        self.detector = None
        self.validator = DataValidator(validation_level) if enable_validation else None
        
        # Resilience components
        self.retry_manager = RetryManager(**(retry_config or {}))
        circuit_breaker_defaults = {'name': 'robust_pipeline'}
        circuit_breaker_defaults.update(circuit_breaker_config or {})
        self.circuit_breaker = CircuitBreaker(**circuit_breaker_defaults)
        
        # Performance tracking
        self.metrics = {
            'training_time': 0,
            'prediction_time': 0,
            'memory_usage': {},
            'error_count': 0,
            'retry_count': 0
        }
        
        # Setup logging
        setup_logging(level=logging.INFO)
        logger.info("RobustAnomalyPipeline initialized with enhanced error handling")
    
    @contextmanager
    def memory_monitor(self, operation: str):
        """Context manager to monitor memory usage during operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        
        logger.info(f"Starting {operation} - Initial memory: {initial_memory:.2f}GB")
        
        try:
            yield
        finally:
            final_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
            peak_memory = max(initial_memory, final_memory)
            
            self.metrics['memory_usage'][operation] = {
                'initial_gb': initial_memory,
                'final_gb': final_memory,
                'peak_gb': peak_memory
            }
            
            logger.info(f"Completed {operation} - Final memory: {final_memory:.2f}GB, Peak: {peak_memory:.2f}GB")
            
            # Memory cleanup
            if peak_memory > self.max_memory_usage_gb:
                logger.warning(f"High memory usage detected ({peak_memory:.2f}GB), forcing garbage collection")
                gc.collect()
    
    def _validate_inputs(self, **kwargs) -> None:
        """Comprehensive input validation."""
        if 'data_path' in kwargs:
            path = kwargs['data_path']
            if not isinstance(path, (str, Path)):
                raise TypeError(f"data_path must be string or Path, got {type(path)}")
            if not Path(path).exists():
                raise FileNotFoundError(f"Data file not found: {path}")
            
            # Security validation
            try:
                validate_file_path(str(path))
            except Exception as e:
                raise ValueError(f"Invalid file path: {sanitize_error_message(str(e))}")
        
        if 'window_size' in kwargs:
            if not isinstance(kwargs['window_size'], int) or kwargs['window_size'] <= 0:
                raise ValueError("window_size must be a positive integer")
        
        if 'threshold' in kwargs and kwargs['threshold'] is not None:
            if not isinstance(kwargs['threshold'], (int, float)) or kwargs['threshold'] < 0:
                raise ValueError("threshold must be a non-negative number")
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate time series data with comprehensive error handling."""
        
        @self.circuit_breaker.call
        @self.retry_manager.retry
        def _load_data_with_resilience(path: str) -> pd.DataFrame:
            with self.memory_monitor("data_loading"):
                logger.info(f"Loading data from {sanitize_error_message(path)}")
                
                # Input validation
                self._validate_inputs(data_path=path)
                
                # Load data with error handling
                try:
                    df = pd.read_csv(path)
                    
                    if df.empty:
                        raise ValueError("Loaded data is empty")
                    
                    logger.info(f"Successfully loaded {len(df)} samples with {len(df.columns)} features")
                    
                    # Data validation if enabled
                    if self.validator:
                        validation_result = self.validator.validate_dataframe(df)
                        
                        if not validation_result.is_valid:
                            if self.validation_level == ValidationLevel.STRICT:
                                raise ValueError(f"Data validation failed: {validation_result.summary()}")
                            else:
                                logger.warning(f"Data validation issues detected: {validation_result.summary()}")
                    
                    return df
                    
                except pd.errors.EmptyDataError:
                    raise ValueError(f"Data file is empty or invalid: {path}")
                except pd.errors.ParserError as e:
                    raise ValueError(f"Failed to parse CSV file: {sanitize_error_message(str(e))}")
                except MemoryError:
                    raise RuntimeError(f"Insufficient memory to load data file: {path}")
                except Exception as e:
                    self.metrics['error_count'] += 1
                    raise RuntimeError(f"Unexpected error loading data: {sanitize_error_message(str(e))}")
        
        try:
            return _load_data_with_resilience(data_path)
        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"Failed to load data after all retries: {e}")
            raise
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Robust data preparation with comprehensive error handling."""
        
        @self.retry_manager.retry
        def _prepare_data_with_resilience(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            with self.memory_monitor("data_preparation"):
                logger.info("Starting robust data preparation")
                
                # Input validation
                if data.empty:
                    raise ValueError("Input DataFrame is empty")
                
                if len(data) < self.window_size:
                    raise ValueError(f"Data length ({len(data)}) is smaller than window size ({self.window_size})")
                
                # Check for sufficient memory
                estimated_memory = len(data) * len(data.columns) * 8 / 1024 / 1024 / 1024  # GB
                if estimated_memory > self.max_memory_usage_gb:
                    logger.warning(f"Estimated memory usage ({estimated_memory:.2f}GB) exceeds limit")
                
                try:
                    # Initialize preprocessor with validation
                    if self.preprocessor is None:
                        self.preprocessor = DataPreprocessor(
                            enable_validation=self.enable_validation,
                            validation_level=self.validation_level
                        )
                    
                    # Scale the data
                    scaled_data = self.preprocessor.fit_transform(data)
                    
                    # Validate scaled data
                    if np.any(np.isnan(scaled_data)) or np.any(np.isinf(scaled_data)):
                        raise ValueError("Scaling produced NaN or infinite values")
                    
                    # Create windowed sequences with memory monitoring
                    sequences = []
                    for i in range(len(scaled_data) - self.window_size + 1):
                        sequences.append(scaled_data[i:i + self.window_size])
                        
                        # Check memory usage periodically
                        if i % 1000 == 0 and i > 0:
                            current_memory = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                            if current_memory > self.max_memory_usage_gb:
                                logger.warning("Memory usage limit reached during sequence creation")
                                gc.collect()
                    
                    if not sequences:
                        raise ValueError("No sequences could be created from the data")
                    
                    sequences = np.array(sequences)
                    logger.info(f"Created {len(sequences)} sequences of shape {sequences.shape}")
                    
                    # Validate sequences
                    if sequences.shape[0] == 0:
                        raise ValueError("No valid sequences generated")
                    
                    return sequences, sequences
                    
                except MemoryError:
                    gc.collect()
                    raise RuntimeError("Insufficient memory for data preparation")
                except Exception as e:
                    self.metrics['error_count'] += 1
                    raise RuntimeError(f"Data preparation failed: {sanitize_error_message(str(e))}")
        
        try:
            return _prepare_data_with_resilience(df)
        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"Data preparation failed after all retries: {e}")
            raise
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build model with comprehensive validation and error handling."""
        
        @self.retry_manager.retry
        def _build_model_with_resilience(shape: Tuple[int, int]) -> None:
            with self.memory_monitor("model_building"):
                logger.info(f"Building robust model with input shape {shape}")
                
                # Input validation
                if len(shape) != 2:
                    raise ValueError(f"Expected 2D input shape, got {len(shape)}D")
                
                if shape[0] <= 0 or shape[1] <= 0:
                    raise ValueError(f"Invalid input shape dimensions: {shape}")
                
                try:
                    self.model = build_autoencoder(
                        input_shape=shape,
                        latent_dim=self.latent_dim,
                        lstm_units=self.lstm_units,
                        dropout_rate=0.2,  # Increased for robustness
                        learning_rate=0.001,
                        use_regularization=True
                    )
                    
                    # Validate model
                    if self.model is None:
                        raise RuntimeError("Model building returned None")
                    
                    # Check model parameters
                    param_count = self.model.count_params()
                    if param_count == 0:
                        raise RuntimeError("Model has no trainable parameters")
                    
                    logger.info(f"Model built successfully with {param_count} parameters")
                    
                except Exception as e:
                    self.metrics['error_count'] += 1
                    raise RuntimeError(f"Model building failed: {sanitize_error_message(str(e))}")
        
        try:
            _build_model_with_resilience(input_shape)
        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"Model building failed after all retries: {e}")
            raise
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Robust model training with comprehensive monitoring."""
        
        @self.circuit_breaker.call
        def _train_with_resilience(features: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
            with self.memory_monitor("model_training"):
                logger.info("Starting robust model training")
                start_time = time.time()
                
                # Input validation
                if features.shape != targets.shape:
                    raise ValueError(f"Feature shape {features.shape} != target shape {targets.shape}")
                
                if np.any(np.isnan(features)) or np.any(np.isnan(targets)):
                    raise ValueError("Training data contains NaN values")
                
                try:
                    # Build model if not exists
                    if self.model is None:
                        self.build_model(input_shape=features.shape[1:])
                    
                    # Enhanced callbacks with error handling
                    callbacks = get_training_callbacks(patience=15, min_delta=0.0001)
                    
                    # Training with error handling
                    history = self.model.fit(
                        features, targets,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        validation_split=self.validation_split,
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    # Validate training results
                    if not history.history:
                        raise RuntimeError("Training history is empty")
                    
                    final_loss = history.history['loss'][-1]
                    final_val_loss = history.history['val_loss'][-1]
                    
                    # Check for training issues
                    if np.isnan(final_loss) or np.isinf(final_loss):
                        raise RuntimeError("Training resulted in NaN/infinite loss")
                    
                    if final_val_loss > final_loss * 3:  # Significant overfitting
                        logger.warning(f"Potential overfitting detected: val_loss={final_val_loss:.4f}, train_loss={final_loss:.4f}")
                    
                    training_time = time.time() - start_time
                    self.metrics['training_time'] = training_time
                    
                    logger.info(f"Training completed successfully in {training_time:.2f}s")
                    logger.info(f"Final loss: {final_loss:.4f}, Val loss: {final_val_loss:.4f}")
                    
                    return history.history
                    
                except Exception as e:
                    self.metrics['error_count'] += 1
                    raise RuntimeError(f"Training failed: {sanitize_error_message(str(e))}")
        
        try:
            return _train_with_resilience(X, y)
        except Exception as e:
            self.metrics['error_count'] += 1
            logger.error(f"Training failed after circuit breaker: {e}")
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline health status."""
        return {
            'components': {
                'preprocessor': self.preprocessor is not None,
                'model': self.model is not None,
                'detector': self.detector is not None,
                'validator': self.validator is not None
            },
            'circuit_breaker': {
                'state': self.circuit_breaker.state.name,
                'failure_count': self.circuit_breaker.failure_count,
                'success_count': self.circuit_breaker.success_count
            },
            'metrics': self.metrics.copy(),
            'memory_usage_gb': psutil.Process().memory_info().rss / 1024 / 1024 / 1024
        }
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker for recovery operations."""
        self.circuit_breaker.reset()
        logger.info("Circuit breaker reset")


def main():
    """Example usage of the robust pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust IoT Anomaly Detection Pipeline")
    parser.add_argument("--data-path", required=True, help="Path to CSV data file")
    parser.add_argument("--model-path", default="saved_models/robust_autoencoder.h5", help="Model output path")
    parser.add_argument("--scaler-path", default="saved_models/robust_scaler.pkl", help="Scaler output path")
    parser.add_argument("--window-size", type=int, default=30, help="Sequence window size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--validation-level", choices=['strict', 'moderate', 'permissive'], 
                       default='moderate', help="Validation strictness level")
    
    args = parser.parse_args()
    
    # Map validation level
    validation_levels = {
        'strict': ValidationLevel.STRICT,
        'moderate': ValidationLevel.MODERATE,
        'permissive': ValidationLevel.PERMISSIVE
    }
    
    # Initialize robust pipeline
    pipeline = RobustAnomalyPipeline(
        window_size=args.window_size,
        epochs=args.epochs,
        validation_level=validation_levels[args.validation_level]
    )
    
    try:
        # Load and prepare data
        df = pipeline.load_data(args.data_path)
        X, y = pipeline.prepare_data(df)
        
        # Train model
        history = pipeline.train(X, y)
        
        # Print health status
        health = pipeline.get_health_status()
        print(f"\nPipeline Health Status:")
        print(f"Components ready: {all(health['components'].values())}")
        print(f"Circuit breaker state: {health['circuit_breaker']['state']}")
        print(f"Error count: {health['metrics']['error_count']}")
        print(f"Memory usage: {health['memory_usage_gb']:.2f}GB")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())