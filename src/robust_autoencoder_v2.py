"""Robust autoencoder for Generation 2: MAKE IT ROBUST - Enhanced reliability and error handling."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import joblib
import time
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import multiprocessing as mp

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    callbacks = None

from .smart_autoencoder_lite import SmartAutoencoderLite
from .logging_config import get_logger, setup_logging
from .config import get_config
from .circuit_breaker import CircuitBreaker
from .retry_manager import RetryManager
from .robust_error_handler import RobustErrorHandler


class RobustAutoencoderV2(SmartAutoencoderLite):
    """Robust LSTM Autoencoder with enhanced error handling and reliability - Generation 2."""
    
    def __init__(
        self,
        window_size: int = 30,
        encoding_dim: int = 16,
        learning_rate: float = 0.001,
        scaler_type: str = "standard",
        random_seed: int = 42,
        enable_circuit_breaker: bool = True,
        enable_retry: bool = True,
        max_retries: int = 3,
        enable_checkpointing: bool = True,
        enable_early_stopping: bool = True,
        enable_lr_scheduling: bool = True,
        enable_validation: bool = True
    ):
        """Initialize the robust autoencoder with enhanced features.
        
        Args:
            window_size: Size of the sliding window for sequences
            encoding_dim: Dimension of the encoding layer
            learning_rate: Learning rate for training
            scaler_type: Type of scaler ('standard' or 'minmax')
            random_seed: Random seed for reproducibility
            enable_circuit_breaker: Enable circuit breaker pattern
            enable_retry: Enable retry mechanism
            max_retries: Maximum number of retries
            enable_checkpointing: Enable model checkpointing
            enable_early_stopping: Enable early stopping
            enable_lr_scheduling: Enable learning rate scheduling
            enable_validation: Enable input validation
        """
        super().__init__(window_size, encoding_dim, learning_rate, scaler_type, random_seed)
        
        # Enhanced reliability features
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.enable_checkpointing = enable_checkpointing
        self.enable_early_stopping = enable_early_stopping
        self.enable_lr_scheduling = enable_lr_scheduling
        self.enable_validation = enable_validation
        
        # Initialize reliability components
        if enable_circuit_breaker:
            from .circuit_breaker import CircuitBreakerConfig
            config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60.0
            )
            self.circuit_breaker = CircuitBreaker(
                name="robust_autoencoder",
                config=config,
                allowed_exceptions=[Exception]
            )
        
        if enable_retry:
            self.retry_manager = RetryManager(
                max_retries=max_retries,
                base_delay=1.0,
                max_delay=30.0,
                backoff_multiplier=2.0,
                jitter=True
            )
        
        self.error_handler = RobustErrorHandler()
        
        # Training state
        self.training_history = {}
        self.validation_metrics = {}
        self.checkpoint_path = None
        self.best_weights = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("Initialized Robust Autoencoder V2 with enhanced reliability features")
    
    def _validate_input_data(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """Comprehensive input data validation."""
        if not self.enable_validation:
            return
        
        # Type validation
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise TypeError(f"Data must be DataFrame or ndarray, got {type(data)}")
        
        # Convert to DataFrame if necessary
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError(f"Data array must be 2D, got {data.ndim}D")
            data = pd.DataFrame(data)
        
        # Size validation
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        
        if len(data) < self.window_size:
            raise ValueError(f"Data length ({len(data)}) must be >= window_size ({self.window_size})")
        
        # Check for all NaN columns
        if data.isnull().all().any():
            null_columns = data.columns[data.isnull().all()].tolist()
            raise ValueError(f"Columns with all NaN values: {null_columns}")
        
        # Check for constant columns (no variance)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            constant_cols = []
            for col in numeric_cols:
                if data[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                self.logger.warning(f"Constant columns detected (may affect training): {constant_cols}")
        
        # Check data ranges for extreme values
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                q99 = col_data.quantile(0.99)
                q01 = col_data.quantile(0.01)
                extreme_range = q99 - q01
                
                if extreme_range == 0:
                    self.logger.warning(f"Column {col} has no variance")
                elif abs(q99) > 1e6 or abs(q01) > 1e6:
                    self.logger.warning(f"Column {col} has extreme values: range [{q01:.2e}, {q99:.2e}]")
        
        self.logger.info("Input data validation passed")
    
    def _create_robust_callbacks(
        self,
        checkpoint_path: Optional[str] = None,
        monitor: str = 'val_loss',
        patience: int = 15,
        min_delta: float = 1e-6
    ) -> List[callbacks.Callback]:
        """Create robust training callbacks."""
        callback_list = []
        
        if self.enable_early_stopping:
            early_stopping = callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            )
            callback_list.append(early_stopping)
        
        if self.enable_lr_scheduling:
            # Reduce learning rate on plateau
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.7,
                patience=int(patience * 0.6),
                min_lr=1e-7,
                verbose=1,
                mode='min'
            )
            callback_list.append(reduce_lr)
        
        if self.enable_checkpointing and checkpoint_path:
            # Model checkpointing
            checkpoint = callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='min'
            )
            callback_list.append(checkpoint)
            self.checkpoint_path = checkpoint_path
        
        # Custom callback for logging and monitoring
        class RobustTrainingCallback(callbacks.Callback):
            def __init__(self, parent_logger):
                super().__init__()
                self.logger = parent_logger
                self.epoch_start_time = None
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
                self.logger.debug(f"Starting epoch {epoch + 1}")
            
            def on_epoch_end(self, epoch, logs=None):
                epoch_time = time.time() - self.epoch_start_time
                logs = logs or {}
                
                log_msg = f"Epoch {epoch + 1} completed in {epoch_time:.2f}s"
                if 'loss' in logs:
                    log_msg += f" - loss: {logs['loss']:.6f}"
                if 'val_loss' in logs:
                    log_msg += f" - val_loss: {logs['val_loss']:.6f}"
                
                self.logger.info(log_msg)
        
        callback_list.append(RobustTrainingCallback(self.logger))
        
        return callback_list
    
    def _safe_fit(
        self,
        sequences: np.ndarray,
        epochs: int,
        batch_size: int,
        validation_split: float,
        callbacks_list: List[callbacks.Callback],
        verbose: int
    ) -> Dict[str, Any]:
        """Safely train the model with error handling."""
        
        def _train():
            return self.model.fit(
                sequences, sequences,  # Autoencoder: input = output
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks_list,
                verbose=verbose,
                shuffle=True
            )
        
        if self.enable_circuit_breaker:
            history = self.circuit_breaker.call(_train)
        else:
            history = _train()
        
        return history.history
    
    def fit(
        self,
        data: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train the robust autoencoder with enhanced error handling.
        
        Args:
            data: Training data (normal patterns only)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            verbose: Verbosity level for training
            checkpoint_dir: Directory for model checkpoints
            
        Returns:
            Training history and metrics dictionary
        """
        with self._lock:
            start_time = time.time()
            self.logger.info(f"Starting robust training on {len(data)} samples")
            
            try:
                # Validate input data
                self._validate_input_data(data)
                
                # Store feature names
                self.feature_names = data.columns.tolist()
                
                # Scale the data with error handling
                try:
                    scaled_data = self.scaler.fit_transform(data.values)
                except Exception as e:
                    self.logger.error(f"Scaling failed: {e}")
                    raise ValueError(f"Data scaling failed: {e}") from e
                
                # Create sequences
                sequences = self._prepare_sequences(scaled_data)
                self.logger.info(f"Created {len(sequences)} sequences of length {self.window_size}")
                
                if len(sequences) < 10:
                    raise ValueError(f"Insufficient sequences for training: {len(sequences)} (minimum 10 required)")
                
                # Create model if not exists
                if self.model is None:
                    input_shape = (self.window_size, len(self.feature_names))
                    self._create_model(input_shape)
                
                # Setup checkpoint directory
                if checkpoint_dir and self.enable_checkpointing:
                    checkpoint_path = Path(checkpoint_dir)
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    checkpoint_file = checkpoint_path / "robust_autoencoder_checkpoint.h5"
                else:
                    checkpoint_file = None
                
                # Create callbacks
                callbacks_list = self._create_robust_callbacks(
                    checkpoint_path=str(checkpoint_file) if checkpoint_file else None,
                    patience=max(10, epochs // 10)  # Dynamic patience
                )
                
                # Train with retry mechanism
                if self.enable_retry:
                    history = self.retry_manager.execute_with_retry(
                        lambda: self._safe_fit(
                            sequences, epochs, batch_size, validation_split, callbacks_list, verbose
                        )
                    )
                else:
                    history = self._safe_fit(
                        sequences, epochs, batch_size, validation_split, callbacks_list, verbose
                    )
                
                # Store training results
                self.training_history = history
                self.is_trained = True
                training_time = time.time() - start_time
                
                # Calculate validation metrics
                final_epoch = len(history['loss']) - 1
                self.validation_metrics = {
                    'final_training_loss': float(history['loss'][final_epoch]),
                    'final_validation_loss': float(history.get('val_loss', [0])[final_epoch]) if 'val_loss' in history else None,
                    'best_training_loss': float(min(history['loss'])),
                    'best_validation_loss': float(min(history['val_loss'])) if 'val_loss' in history else None,
                    'epochs_completed': len(history['loss']),
                    'training_time_seconds': training_time,
                    'convergence_rate': self._calculate_convergence_rate(history['loss']),
                    'overfitting_score': self._calculate_overfitting_score(history)
                }
                
                # Load best checkpoint if available
                if checkpoint_file and checkpoint_file.exists():
                    try:
                        self.model = keras.models.load_model(str(checkpoint_file))
                        self.logger.info("Loaded best model from checkpoint")
                    except Exception as e:
                        self.logger.warning(f"Could not load checkpoint: {e}")
                
                self.logger.info(f"Robust training completed successfully in {training_time:.2f} seconds")
                self.logger.info(f"Final training loss: {self.validation_metrics['final_training_loss']:.6f}")
                if self.validation_metrics['final_validation_loss']:
                    self.logger.info(f"Final validation loss: {self.validation_metrics['final_validation_loss']:.6f}")
                
                return {
                    'training_time_seconds': training_time,
                    'training_samples': len(data),
                    'feature_count': len(self.feature_names),
                    'sequences_created': len(sequences),
                    'epochs_completed': self.validation_metrics['epochs_completed'],
                    'history': history,
                    'validation_metrics': self.validation_metrics,
                    'model_architecture': self.get_model_summary()
                }
                
            except Exception as e:
                self.error_handler.handle_error(e, context="training")
                self.logger.error(f"Training failed: {e}")
                raise
    
    def _calculate_convergence_rate(self, loss_history: List[float]) -> float:
        """Calculate the convergence rate from loss history."""
        if len(loss_history) < 2:
            return 0.0
        
        # Calculate average rate of loss decrease
        improvements = []
        for i in range(1, len(loss_history)):
            if loss_history[i-1] > loss_history[i]:
                improvement = (loss_history[i-1] - loss_history[i]) / loss_history[i-1]
                improvements.append(improvement)
        
        return float(np.mean(improvements)) if improvements else 0.0
    
    def _calculate_overfitting_score(self, history: Dict[str, List[float]]) -> float:
        """Calculate overfitting score."""
        if 'val_loss' not in history or len(history['val_loss']) < 2:
            return 0.0
        
        train_loss = history['loss']
        val_loss = history['val_loss']
        
        # Find the epoch with minimum validation loss
        min_val_epoch = np.argmin(val_loss)
        
        if min_val_epoch >= len(train_loss) - 1:
            return 0.0
        
        # Calculate gap between training and validation loss at the end
        final_gap = val_loss[-1] - train_loss[-1]
        min_gap = val_loss[min_val_epoch] - train_loss[min_val_epoch]
        
        # Overfitting score: how much the gap increased from the best point
        overfitting_score = max(0.0, (final_gap - min_gap) / max(abs(min_gap), 1e-6))
        
        return float(overfitting_score)
    
    def predict_robust(
        self,
        data: pd.DataFrame,
        batch_size: Optional[int] = None,
        timeout_seconds: float = 300.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with robust error handling and timeout.
        
        Args:
            data: Input data for prediction
            batch_size: Optional batch size for prediction
            timeout_seconds: Timeout for prediction operation
            
        Returns:
            Tuple of (reconstructed_sequences, reconstruction_errors)
        """
        with self._lock:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")
            
            start_time = time.time()
            self.logger.info(f"Starting robust prediction for {len(data)} samples")
            
            try:
                # Validate input data
                self._validate_input_data(data)
                
                # Validate feature consistency
                if self.feature_names and data.columns.tolist() != self.feature_names:
                    self.logger.warning("Feature names don't match training data")
                
                # Scale the data
                try:
                    scaled_data = self.scaler.transform(data.values)
                except Exception as e:
                    self.logger.error(f"Data scaling failed during prediction: {e}")
                    raise ValueError(f"Data scaling failed: {e}") from e
                
                # Create sequences
                sequences = self._prepare_sequences(scaled_data)
                
                if len(sequences) == 0:
                    raise ValueError(f"Not enough data to create sequences. Need at least {self.window_size} samples")
                
                # Determine batch size
                if batch_size is None:
                    # Adaptive batch size based on available memory and data size
                    batch_size = min(32, max(1, len(sequences) // 10))
                
                # Predict with timeout
                def _predict_with_timeout():
                    return self.model.predict(sequences, batch_size=batch_size, verbose=0)
                
                # Execute with timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_predict_with_timeout)
                    try:
                        reconstructions = future.result(timeout=timeout_seconds)
                    except FuturesTimeoutError:
                        future.cancel()
                        raise TimeoutError(f"Prediction timed out after {timeout_seconds} seconds")
                
                # Calculate reconstruction errors (MSE for each sequence)
                reconstruction_errors = np.mean((sequences - reconstructions) ** 2, axis=(1, 2))
                
                prediction_time = time.time() - start_time
                self.logger.info(f"Robust prediction completed in {prediction_time:.2f} seconds")
                self.logger.info(f"Generated predictions for {len(sequences)} sequences")
                
                return reconstructions, reconstruction_errors
                
            except Exception as e:
                self.error_handler.handle_error(e, context="prediction")
                self.logger.error(f"Prediction failed: {e}")
                raise
    
    def detect_anomalies_robust(
        self,
        data: pd.DataFrame,
        threshold: Optional[float] = None,
        contamination_rate: float = 0.1,
        use_statistical_threshold: bool = True,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Detect anomalies with enhanced statistical methods.
        
        Args:
            data: Input data for anomaly detection
            threshold: Fixed threshold for anomaly detection
            contamination_rate: Expected contamination rate for threshold calculation
            use_statistical_threshold: Use statistical methods for threshold
            confidence_level: Confidence level for statistical threshold
            
        Returns:
            Tuple of (anomaly_flags, threshold_used, statistics)
        """
        try:
            # Get predictions
            _, reconstruction_errors = self.predict_robust(data)
            
            # Calculate threshold
            if threshold is None:
                if use_statistical_threshold:
                    # Use statistical approach with confidence intervals
                    mean_error = np.mean(reconstruction_errors)
                    std_error = np.std(reconstruction_errors)
                    
                    # Calculate threshold based on confidence level
                    from scipy import stats
                    z_score = stats.norm.ppf(confidence_level)
                    threshold = mean_error + z_score * std_error
                    
                    self.logger.info(f"Statistical threshold calculated: {threshold:.6f} (confidence={confidence_level})")
                else:
                    # Use percentile-based approach
                    threshold = np.percentile(reconstruction_errors, (1 - contamination_rate) * 100)
                    self.logger.info(f"Percentile threshold calculated: {threshold:.6f}")
            
            # Detect anomalies
            anomalies = reconstruction_errors > threshold
            
            # Calculate detailed statistics
            anomaly_count = np.sum(anomalies)
            statistics = {
                'total_sequences': len(anomalies),
                'anomaly_count': int(anomaly_count),
                'anomaly_rate': float(anomaly_count / len(anomalies)),
                'mean_reconstruction_error': float(np.mean(reconstruction_errors)),
                'std_reconstruction_error': float(np.std(reconstruction_errors)),
                'median_reconstruction_error': float(np.median(reconstruction_errors)),
                'max_reconstruction_error': float(np.max(reconstruction_errors)),
                'min_reconstruction_error': float(np.min(reconstruction_errors)),
                'threshold_used': float(threshold),
                'q75_reconstruction_error': float(np.percentile(reconstruction_errors, 75)),
                'q95_reconstruction_error': float(np.percentile(reconstruction_errors, 95)),
                'q99_reconstruction_error': float(np.percentile(reconstruction_errors, 99))
            }
            
            self.logger.info(f"Robust anomaly detection completed")
            self.logger.info(f"Detected {anomaly_count} anomalies ({statistics['anomaly_rate']:.2%} rate)")
            
            return anomalies, threshold, statistics
            
        except Exception as e:
            self.error_handler.handle_error(e, context="anomaly_detection")
            self.logger.error(f"Anomaly detection failed: {e}")
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the model."""
        status = {
            'is_trained': self.is_trained,
            'model_available': self.model is not None,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'window_size': self.window_size,
            'encoding_dim': self.encoding_dim,
            'scaler_fitted': hasattr(self.scaler, 'scale_') if self.scaler else False,
            'circuit_breaker_enabled': self.enable_circuit_breaker,
            'retry_enabled': self.enable_retry,
            'validation_enabled': self.enable_validation
        }
        
        # Add circuit breaker status
        if self.enable_circuit_breaker and hasattr(self, 'circuit_breaker'):
            status['circuit_breaker_status'] = {
                'state': self.circuit_breaker.state.name,
                'failure_count': self.circuit_breaker.failure_count,
                'success_count': self.circuit_breaker.success_count
            }
        
        # Add training metrics if available
        if self.validation_metrics:
            status['training_metrics'] = self.validation_metrics.copy()
        
        return status
    
    def save_robust(self, model_path: str, scaler_path: str, metadata_path: Optional[str] = None) -> None:
        """Save the model with enhanced metadata and validation."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            # Create directories if they don't exist
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.model.save(model_path)
            
            # Save scaler and enhanced metadata
            scaler_data = {
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'window_size': self.window_size,
                'encoding_dim': self.encoding_dim,
                'scaler_type': self.scaler_type,
                'learning_rate': self.learning_rate,
                'random_seed': self.random_seed,
                'validation_metrics': self.validation_metrics,
                'training_history_summary': {
                    'epochs_completed': len(self.training_history.get('loss', [])),
                    'final_loss': self.training_history.get('loss', [])[-1] if self.training_history.get('loss') else None,
                    'best_loss': min(self.training_history.get('loss', [])) if self.training_history.get('loss') else None
                },
                'model_config': {
                    'enable_circuit_breaker': self.enable_circuit_breaker,
                    'enable_retry': self.enable_retry,
                    'max_retries': self.max_retries,
                    'enable_validation': self.enable_validation
                },
                'save_timestamp': time.time(),
                'tensorflow_version': tf.__version__ if TENSORFLOW_AVAILABLE else None
            }
            
            joblib.dump(scaler_data, scaler_path)
            
            # Save additional metadata if requested
            if metadata_path:
                Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
                detailed_metadata = {
                    **scaler_data,
                    'full_training_history': self.training_history,
                    'model_summary': self.get_model_summary(),
                    'health_status': self.get_health_status()
                }
                joblib.dump(detailed_metadata, metadata_path)
                self.logger.info(f"Saved detailed metadata to {metadata_path}")
            
            self.logger.info(f"Robust model saved to {model_path} and scaler to {scaler_path}")
            
        except Exception as e:
            self.error_handler.handle_error(e, context="model_saving")
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    @classmethod
    def load_robust(
        cls, 
        model_path: str, 
        scaler_path: str,
        validate_integrity: bool = True
    ) -> 'RobustAutoencoderV2':
        """Load a robust model with integrity validation."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required but not available")
        
        try:
            # Load scaler and metadata
            scaler_data = joblib.load(scaler_path)
            
            # Create instance with loaded parameters
            model_config = scaler_data.get('model_config', {})
            instance = cls(
                window_size=scaler_data['window_size'],
                encoding_dim=scaler_data['encoding_dim'],
                scaler_type=scaler_data.get('scaler_type', 'standard'),
                learning_rate=scaler_data.get('learning_rate', 0.001),
                random_seed=scaler_data.get('random_seed', 42),
                **model_config
            )
            
            # Load model
            instance.model = keras.models.load_model(model_path)
            instance.scaler = scaler_data['scaler']
            instance.feature_names = scaler_data['feature_names']
            instance.validation_metrics = scaler_data.get('validation_metrics', {})
            instance.is_trained = True
            
            # Validate model integrity if requested
            if validate_integrity:
                instance._validate_model_integrity()
            
            instance.logger.info(f"Loaded robust model from {model_path} and scaler from {scaler_path}")
            
            return instance
            
        except Exception as e:
            logger = get_logger(cls.__name__)
            logger.error(f"Failed to load robust model: {e}")
            raise
    
    def _validate_model_integrity(self) -> None:
        """Validate the integrity of the loaded model."""
        try:
            # Check model structure
            if self.model is None:
                raise ValueError("Model is None")
            
            # Check input shape
            expected_input_shape = (None, self.window_size, len(self.feature_names))
            actual_input_shape = self.model.input_shape
            
            if actual_input_shape[1:] != expected_input_shape[1:]:
                raise ValueError(f"Input shape mismatch: expected {expected_input_shape}, got {actual_input_shape}")
            
            # Test with dummy data
            dummy_input = np.random.randn(1, self.window_size, len(self.feature_names))
            dummy_output = self.model.predict(dummy_input, verbose=0)
            
            if dummy_output.shape != dummy_input.shape:
                raise ValueError(f"Output shape mismatch: expected {dummy_input.shape}, got {dummy_output.shape}")
            
            self.logger.info("Model integrity validation passed")
            
        except Exception as e:
            self.logger.error(f"Model integrity validation failed: {e}")
            raise ValueError(f"Model integrity check failed: {e}") from e