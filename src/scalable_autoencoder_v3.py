"""Scalable autoencoder for Generation 3: MAKE IT SCALE - Performance optimization and distributed processing."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import joblib
import time
import warnings
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import gc
import psutil
from functools import lru_cache, wraps
import asyncio

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, mixed_precision
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    callbacks = None

from .robust_autoencoder_v2 import RobustAutoencoderV2
from .logging_config import get_logger, setup_logging
from .config import get_config
from .caching_strategy import CacheManager, cache_prediction
try:
    from .adaptive_cache import AdaptiveCache
    ADAPTIVE_CACHE_AVAILABLE = True
except ImportError:
    ADAPTIVE_CACHE_AVAILABLE = False
    AdaptiveCache = None
try:
    from .resource_pool_manager import ResourcePoolManager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False
    ResourcePoolManager = None
except RuntimeError:
    RESOURCE_MANAGER_AVAILABLE = False
    ResourcePoolManager = None


def optimize_memory(func):
    """Decorator for memory optimization."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()  # Clean up before execution
        result = func(*args, **kwargs)
        gc.collect()  # Clean up after execution
        return result
    return wrapper


def profile_performance(func):
    """Decorator for performance profiling."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'performance_stats'):
            self.performance_stats = {}
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(self, *args, **kwargs)
            success = True
        except Exception as e:
            result = e
            success = False
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            func_name = func.__name__
            if func_name not in self.performance_stats:
                self.performance_stats[func_name] = []
            
            self.performance_stats[func_name].append({
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'success': success,
                'timestamp': end_time
            })
        
        return result
    return wrapper


class ScalableAutoencoderV3(RobustAutoencoderV2):
    """Scalable LSTM Autoencoder with performance optimization and distributed processing - Generation 3."""
    
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
        enable_validation: bool = True,
        # New scalability features
        enable_mixed_precision: bool = True,
        enable_distributed_training: bool = False,
        enable_caching: bool = True,
        cache_size: int = 1000,
        enable_parallel_processing: bool = True,
        max_workers: Optional[int] = None,
        enable_memory_optimization: bool = True,
        enable_performance_profiling: bool = True,
        batch_size_optimizer: bool = True,
        enable_adaptive_batching: bool = True,
        target_memory_usage_mb: int = 2048
    ):
        """Initialize the scalable autoencoder with advanced optimization features.
        
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
            enable_mixed_precision: Enable mixed precision training
            enable_distributed_training: Enable distributed training
            enable_caching: Enable prediction caching
            cache_size: Size of prediction cache
            enable_parallel_processing: Enable parallel processing
            max_workers: Maximum number of worker threads/processes
            enable_memory_optimization: Enable memory optimization
            enable_performance_profiling: Enable performance profiling
            batch_size_optimizer: Enable automatic batch size optimization
            enable_adaptive_batching: Enable adaptive batch sizing
            target_memory_usage_mb: Target memory usage in MB
        """
        super().__init__(
            window_size, encoding_dim, learning_rate, scaler_type, random_seed,
            enable_circuit_breaker, enable_retry, max_retries, enable_checkpointing,
            enable_early_stopping, enable_lr_scheduling, enable_validation
        )
        
        # Scalability features
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_distributed_training = enable_distributed_training
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.enable_parallel_processing = enable_parallel_processing
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.enable_memory_optimization = enable_memory_optimization
        self.enable_performance_profiling = enable_performance_profiling
        self.batch_size_optimizer = batch_size_optimizer
        self.enable_adaptive_batching = enable_adaptive_batching
        self.target_memory_usage_mb = target_memory_usage_mb
        
        # Initialize performance tracking
        if enable_performance_profiling:
            self.performance_stats = {}
            self.memory_tracker = []
            self.throughput_tracker = []
        
        # Initialize caching
        if enable_caching:
            if ADAPTIVE_CACHE_AVAILABLE:
                self.cache = AdaptiveCache(
                    max_size=cache_size,
                    ttl_seconds=3600,  # 1 hour TTL
                    cleanup_interval=300  # 5 minutes
                )
            else:
                self.cache = {}  # Simple dict cache as fallback
            self.cache_manager = CacheManager(maxsize=cache_size)
        
        # Resource management
        if RESOURCE_MANAGER_AVAILABLE:
            self.resource_manager = ResourcePoolManager(
                max_memory_mb=target_memory_usage_mb,
                max_cpu_percent=80.0,
                cleanup_interval=60
            )
        else:
            self.resource_manager = None
        
        # Threading and processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = None  # Initialized on demand
        
        # Performance optimization state
        self.optimal_batch_size = None
        self.batch_size_history = []
        self.training_stats = {}
        
        # Configure mixed precision
        if enable_mixed_precision and TENSORFLOW_AVAILABLE:
            self._configure_mixed_precision()
        
        self.logger.info(f"Initialized Scalable Autoencoder V3 with advanced optimization features")
        self.logger.info(f"Max workers: {self.max_workers}, Mixed precision: {enable_mixed_precision}")
    
    def _configure_mixed_precision(self):
        """Configure mixed precision training for performance optimization."""
        try:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            self.logger.info("Mixed precision training enabled")
        except Exception as e:
            self.logger.warning(f"Could not enable mixed precision: {e}")
            self.enable_mixed_precision = False
    
    @optimize_memory
    @profile_performance
    def _create_optimized_model(self, input_shape: Tuple[int, int]) -> None:
        """Create an optimized LSTM autoencoder model with performance enhancements."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required but not available")
        
        # Input layer with optimization
        inputs = keras.Input(shape=input_shape, name='encoder_input')
        
        # Encoder with optimizations
        encoder = layers.LSTM(
            self.encoding_dim,
            return_state=True,
            return_sequences=False,
            dropout=0.1,  # Regularization
            recurrent_dropout=0.1,
            name='encoder_lstm',
            implementation=2,  # GPU-optimized implementation
            use_bias=True
        )(inputs)
        
        encoder_outputs, state_h, state_c = encoder[0], encoder[1], encoder[2]
        encoder_states = [state_h, state_c]
        
        # Bottleneck layer for compression
        bottleneck = layers.Dense(
            self.encoding_dim // 2,
            activation='relu',
            name='bottleneck'
        )(encoder_outputs)
        
        # Expansion layer
        expansion = layers.Dense(
            self.encoding_dim,
            activation='relu',
            name='expansion'
        )(bottleneck)
        
        # Decoder with optimizations
        decoder_inputs = layers.RepeatVector(
            input_shape[0],
            name='decoder_repeat'
        )(expansion)
        
        decoder_lstm = layers.LSTM(
            self.encoding_dim,
            return_sequences=True,
            dropout=0.1,
            recurrent_dropout=0.1,
            name='decoder_lstm',
            implementation=2,  # GPU-optimized implementation
            use_bias=True
        )(decoder_inputs, initial_state=encoder_states)
        
        # Output layer with mixed precision considerations
        outputs = layers.TimeDistributed(
            layers.Dense(input_shape[1], activation='linear', dtype='float32'),
            name='decoder_output'
        )(decoder_lstm)
        
        # Create model
        self.model = keras.Model(inputs, outputs, name='scalable_autoencoder_v3')
        
        # Compile with optimizations
        optimizer = self._create_optimized_optimizer()
        loss = self._get_optimized_loss()
        metrics = self._get_optimized_metrics()
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            run_eagerly=False  # Enable graph mode for better performance
        )
        
        self.logger.info(f"Created optimized LSTM autoencoder model with encoding_dim={self.encoding_dim}")
        self.logger.info(f"Model parameters: {self.model.count_params()}")
    
    def _create_optimized_optimizer(self):
        """Create optimized optimizer for training."""
        if self.enable_mixed_precision:
            # Use loss scaling for mixed precision
            optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                clipnorm=1.0  # Gradient clipping
            )
            # Wrap with loss scaling for mixed precision
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        else:
            optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                clipnorm=1.0
            )
        return optimizer
    
    def _get_optimized_loss(self):
        """Get optimized loss function."""
        if self.enable_mixed_precision:
            # Use float32 loss for numerical stability
            return keras.losses.MeanSquaredError(dtype='float32')
        else:
            return 'mean_squared_error'
    
    def _get_optimized_metrics(self):
        """Get optimized metrics."""
        return ['mean_absolute_error', 'mean_squared_error']
    
    def _optimize_batch_size(self, data_size: int, feature_count: int) -> int:
        """Automatically optimize batch size based on available memory and data."""
        if not self.batch_size_optimizer:
            return 32  # Default
        
        if self.optimal_batch_size is not None:
            return self.optimal_batch_size
        
        # Calculate memory-aware batch size
        available_memory_mb = self.resource_manager.get_available_memory_mb() if self.resource_manager else 2048
        target_memory_mb = min(available_memory_mb * 0.6, self.target_memory_usage_mb * 0.7)
        
        # Estimate memory per sample (rough calculation)
        bytes_per_sample = self.window_size * feature_count * 4  # float32
        samples_per_mb = (1024 * 1024) / bytes_per_sample
        max_batch_size = int(target_memory_mb * samples_per_mb)
        
        # Consider data size and processing capabilities
        ideal_batch_size = min(
            max_batch_size,
            max(16, data_size // 50),  # At least 50 batches
            512  # Upper limit
        )
        
        # Round to power of 2 for GPU optimization
        optimal_batch_size = 2 ** int(np.log2(ideal_batch_size))
        optimal_batch_size = max(16, optimal_batch_size)  # Minimum 16
        
        self.optimal_batch_size = optimal_batch_size
        self.logger.info(f"Optimized batch size: {optimal_batch_size} (memory budget: {target_memory_mb:.1f}MB)")
        
        return optimal_batch_size
    
    @optimize_memory
    @profile_performance
    def fit_scalable(
        self,
        data: pd.DataFrame,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        validation_split: float = 0.2,
        verbose: int = 1,
        checkpoint_dir: Optional[str] = None,
        use_multiprocessing: bool = False,
        workers: int = 1
    ) -> Dict[str, Any]:
        """Train the scalable autoencoder with performance optimizations.
        
        Args:
            data: Training data (normal patterns only)
            epochs: Number of training epochs
            batch_size: Batch size (None for auto-optimization)
            validation_split: Fraction of data to use for validation
            verbose: Verbosity level for training
            checkpoint_dir: Directory for model checkpoints
            use_multiprocessing: Use multiprocessing for data loading
            workers: Number of workers for data loading
            
        Returns:
            Training history and performance metrics
        """
        with self._lock:
            start_time = time.time()
            self.logger.info(f"Starting scalable training on {len(data)} samples")
            
            try:
                # Memory and resource monitoring
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
                if self.resource_manager:
                    self.resource_manager.start_monitoring()
                
                # Validate input data
                self._validate_input_data(data)
                
                # Store feature names
                self.feature_names = data.columns.tolist()
                
                # Optimize batch size if not provided
                if batch_size is None:
                    batch_size = self._optimize_batch_size(len(data), len(self.feature_names))
                
                # Scale the data with parallel processing
                scaled_data = self._parallel_scale_data(data)
                
                # Create sequences with memory optimization
                sequences = self._create_sequences_optimized(scaled_data)
                self.logger.info(f"Created {len(sequences)} sequences of length {self.window_size}")
                
                # Create optimized model if not exists
                if self.model is None:
                    input_shape = (self.window_size, len(self.feature_names))
                    self._create_optimized_model(input_shape)
                
                # Setup optimized callbacks
                callbacks_list = self._create_optimized_callbacks(
                    checkpoint_dir, len(sequences), batch_size
                )
                
                # Configure data pipeline for performance
                train_dataset, val_dataset = self._create_optimized_datasets(
                    sequences, batch_size, validation_split, use_multiprocessing, workers
                )
                
                # Train with optimizations
                self.logger.info(f"Starting training with batch_size={batch_size}")
                
                if val_dataset is not None:
                    history = self.model.fit(
                        train_dataset,
                        epochs=epochs,
                        validation_data=val_dataset,
                        callbacks=callbacks_list,
                        verbose=verbose
                    )
                else:
                    history = self.model.fit(
                        train_dataset,
                        epochs=epochs,
                        validation_split=validation_split,
                        callbacks=callbacks_list,
                        verbose=verbose
                    )
                
                # Store training results and performance metrics
                self.training_history = history.history
                self.is_trained = True
                training_time = time.time() - start_time
                
                # Calculate performance metrics
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = final_memory - initial_memory
                
                # Update batch size history
                self.batch_size_history.append({
                    'batch_size': batch_size,
                    'training_time': training_time,
                    'memory_used': memory_used,
                    'final_loss': history.history['loss'][-1] if history.history.get('loss') else None,
                    'epochs_completed': len(history.history.get('loss', []))
                })
                
                # Store training statistics
                self.training_stats = {
                    'training_time_seconds': training_time,
                    'memory_usage_mb': memory_used,
                    'sequences_processed': len(sequences),
                    'batch_size_used': batch_size,
                    'throughput_sequences_per_second': len(sequences) / training_time,
                    'memory_efficiency_sequences_per_mb': len(sequences) / max(memory_used, 1),
                    'optimal_batch_size': self.optimal_batch_size,
                    'resource_utilization': self.resource_manager.get_utilization_stats() if self.resource_manager else {}
                }
                
                self.logger.info(f"Scalable training completed in {training_time:.2f} seconds")
                self.logger.info(f"Memory used: {memory_used:.1f}MB, Throughput: {self.training_stats['throughput_sequences_per_second']:.1f} seq/s")
                
                return {
                    'training_time_seconds': training_time,
                    'training_samples': len(data),
                    'sequences_created': len(sequences),
                    'batch_size_used': batch_size,
                    'epochs_completed': len(history.history.get('loss', [])),
                    'history': history.history,
                    'performance_metrics': self.training_stats,
                    'memory_usage_mb': memory_used,
                    'model_architecture': self.get_model_summary()
                }
                
            except Exception as e:
                self.error_handler.handle_error(e, context="scalable_training")
                self.logger.error(f"Scalable training failed: {e}")
                raise
            finally:
                if self.resource_manager:
                    self.resource_manager.stop_monitoring()
    
    def _parallel_scale_data(self, data: pd.DataFrame) -> np.ndarray:
        """Scale data using parallel processing for large datasets."""
        if len(data) < 10000 or not self.enable_parallel_processing:
            # Use standard scaling for small datasets
            return self.scaler.fit_transform(data.values)
        
        # Parallel scaling for large datasets
        chunk_size = max(1000, len(data) // self.max_workers)
        chunks = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Fit scaler on sample
        sample_size = min(10000, len(data))
        sample_indices = np.random.choice(len(data), sample_size, replace=False)
        self.scaler.fit(data.iloc[sample_indices].values)
        
        # Transform chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.scaler.transform, chunk.values)
                for chunk in chunks
            ]
            
            scaled_chunks = []
            for future in as_completed(futures):
                scaled_chunks.append(future.result())
        
        # Combine results
        return np.vstack(scaled_chunks)
    
    def _create_sequences_optimized(self, data: np.ndarray) -> np.ndarray:
        """Create sequences with memory optimization."""
        if not self.enable_memory_optimization or len(data) < 50000:
            return self._prepare_sequences(data)
        
        # Use memory-mapped arrays for large datasets
        num_sequences = len(data) - self.window_size + 1
        sequences = np.memmap(
            'temp_sequences.dat',
            dtype=np.float32,
            mode='w+',
            shape=(num_sequences, self.window_size, data.shape[1])
        )
        
        # Fill sequences in chunks to manage memory
        chunk_size = min(10000, num_sequences // 4)
        for start_idx in range(0, num_sequences, chunk_size):
            end_idx = min(start_idx + chunk_size, num_sequences)
            
            for i in range(start_idx, end_idx):
                sequences[i] = data[i:i + self.window_size]
        
        return sequences
    
    def _create_optimized_datasets(
        self,
        sequences: np.ndarray,
        batch_size: int,
        validation_split: float,
        use_multiprocessing: bool,
        workers: int
    ) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
        """Create optimized TensorFlow datasets."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")
        
        # Split data
        split_idx = int(len(sequences) * (1 - validation_split))
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:] if validation_split > 0 else None
        
        # Create training dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_sequences))
        train_dataset = train_dataset.shuffle(buffer_size=min(10000, len(train_sequences)))
        train_dataset = train_dataset.batch(batch_size)
        
        # Optimize dataset pipeline
        train_dataset = train_dataset.cache()  # Cache for multiple epochs
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Create validation dataset
        val_dataset = None
        if val_sequences is not None and len(val_sequences) > 0:
            val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_sequences))
            val_dataset = val_dataset.batch(batch_size)
            val_dataset = val_dataset.cache()
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset
    
    def _create_optimized_callbacks(
        self,
        checkpoint_dir: Optional[str],
        num_sequences: int,
        batch_size: int
    ) -> List[callbacks.Callback]:
        """Create optimized callbacks for scalable training."""
        callback_list = super()._create_robust_callbacks(checkpoint_dir)
        
        # Add performance monitoring callback
        class PerformanceMonitorCallback(callbacks.Callback):
            def __init__(self, parent_autoencoder):
                super().__init__()
                self.parent = parent_autoencoder
                self.epoch_times = []
                self.batch_times = []
                self.memory_usage = []
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
                
            def on_epoch_end(self, epoch, logs=None):
                epoch_time = time.time() - self.epoch_start_time
                self.epoch_times.append(epoch_time)
                
                # Memory monitoring
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                self.memory_usage.append(memory_mb)
                
                # Update parent's performance tracking
                if hasattr(self.parent, 'memory_tracker'):
                    self.parent.memory_tracker.append(memory_mb)
                
                logs = logs or {}
                throughput = num_sequences / epoch_time if epoch_time > 0 else 0
                self.parent.logger.debug(
                    f"Epoch {epoch + 1} performance: {epoch_time:.2f}s, "
                    f"{throughput:.1f} seq/s, {memory_mb:.1f}MB"
                )
        
        callback_list.append(PerformanceMonitorCallback(self))
        
        # Add adaptive batch size callback if enabled
        if self.enable_adaptive_batching:
            class AdaptiveBatchSizeCallback(callbacks.Callback):
                def __init__(self, parent_autoencoder):
                    super().__init__()
                    self.parent = parent_autoencoder
                    self.performance_window = []
                
                def on_epoch_end(self, epoch, logs=None):
                    if epoch < 5:  # Skip first few epochs
                        return
                    
                    logs = logs or {}
                    loss = logs.get('loss', float('inf'))
                    val_loss = logs.get('val_loss', float('inf'))
                    
                    # Track performance
                    self.performance_window.append({
                        'epoch': epoch,
                        'loss': loss,
                        'val_loss': val_loss,
                        'batch_size': batch_size
                    })
                    
                    # Keep only recent history
                    if len(self.performance_window) > 10:
                        self.performance_window.pop(0)
                    
                    # Analyze trends and suggest batch size adjustments
                    if len(self.performance_window) >= 5:
                        recent_losses = [p['loss'] for p in self.performance_window[-3:]]
                        if all(recent_losses[i] > recent_losses[i-1] for i in range(1, len(recent_losses))):
                            # Loss is increasing, might need smaller batch size
                            self.parent.logger.debug("Loss trend suggests smaller batch size might help")
            
            callback_list.append(AdaptiveBatchSizeCallback(self))
        
        return callback_list
    
    @lru_cache(maxsize=128)
    @profile_performance
    def predict_cached(
        self,
        data_hash: str,
        data_shape: Tuple[int, int],
        batch_size: Optional[int] = None
    ) -> Tuple[str, str]:  # Returns cache keys for reconstructions and errors
        """Cached prediction method for frequently accessed data."""
        # This is a placeholder - actual implementation would use the cache
        # The real prediction happens in predict_scalable
        return f"recon_{data_hash}", f"errors_{data_hash}"
    
    @optimize_memory
    @profile_performance
    def predict_scalable(
        self,
        data: pd.DataFrame,
        batch_size: Optional[int] = None,
        use_parallel: bool = True,
        timeout_seconds: float = 300.0,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with scalable optimizations.
        
        Args:
            data: Input data for prediction
            batch_size: Optional batch size for prediction (None for auto)
            use_parallel: Enable parallel processing
            timeout_seconds: Timeout for prediction operation
            use_cache: Use prediction caching
            
        Returns:
            Tuple of (reconstructed_sequences, reconstruction_errors)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        start_time = time.time()
        self.logger.info(f"Starting scalable prediction for {len(data)} samples")
        
        try:
            # Check cache if enabled
            if use_cache and self.enable_caching:
                data_hash = self._hash_dataframe(data)
                if ADAPTIVE_CACHE_AVAILABLE and hasattr(self.cache, 'get'):
                    cached_result = self.cache.get(data_hash)
                else:
                    cached_result = self.cache.get(data_hash) if isinstance(self.cache, dict) else None
                
                if cached_result is not None:
                    self.logger.info("Using cached prediction result")
                    return cached_result
            
            # Validate and prepare data
            self._validate_input_data(data)
            
            # Scale data with parallel processing if enabled
            if use_parallel and len(data) > 5000:
                scaled_data = self._parallel_transform_data(data)
            else:
                scaled_data = self.scaler.transform(data.values)
            
            # Create sequences with optimization
            sequences = self._create_sequences_optimized(scaled_data)
            
            if len(sequences) == 0:
                raise ValueError(f"Not enough data to create sequences. Need at least {self.window_size} samples")
            
            # Optimize batch size for prediction
            if batch_size is None:
                available_memory = self.resource_manager.get_available_memory_mb() if self.resource_manager else 1024
                batch_size = self._calculate_prediction_batch_size(len(sequences), available_memory)
            
            # Perform prediction with batching
            reconstructions = self._predict_in_batches(sequences, batch_size, timeout_seconds)
            
            # Calculate reconstruction errors
            reconstruction_errors = np.mean((sequences - reconstructions) ** 2, axis=(1, 2))
            
            # Cache result if enabled
            if use_cache and self.enable_caching:
                if ADAPTIVE_CACHE_AVAILABLE and hasattr(self.cache, 'put'):
                    self.cache.put(data_hash, (reconstructions, reconstruction_errors))
                elif isinstance(self.cache, dict):
                    self.cache[data_hash] = (reconstructions, reconstruction_errors)
            
            prediction_time = time.time() - start_time
            throughput = len(sequences) / prediction_time
            
            # Update throughput tracking
            if hasattr(self, 'throughput_tracker'):
                self.throughput_tracker.append(throughput)
            
            self.logger.info(f"Scalable prediction completed in {prediction_time:.2f} seconds")
            self.logger.info(f"Throughput: {throughput:.1f} sequences/second")
            
            return reconstructions, reconstruction_errors
            
        except Exception as e:
            self.error_handler.handle_error(e, context="scalable_prediction")
            self.logger.error(f"Scalable prediction failed: {e}")
            raise
    
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Create hash of dataframe for caching."""
        import hashlib
        return hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()
    
    def _parallel_transform_data(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data using parallel processing."""
        chunk_size = max(1000, len(data) // self.max_workers)
        chunks = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.scaler.transform, chunk.values)
                for chunk in chunks
            ]
            
            scaled_chunks = []
            for future in as_completed(futures):
                scaled_chunks.append(future.result())
        
        return np.vstack(scaled_chunks)
    
    def _calculate_prediction_batch_size(self, num_sequences: int, available_memory_mb: float) -> int:
        """Calculate optimal batch size for prediction."""
        # Estimate memory per sequence
        memory_per_sequence_mb = (self.window_size * len(self.feature_names) * 4) / (1024 * 1024)
        
        # Use 50% of available memory for safety
        max_sequences_in_memory = int((available_memory_mb * 0.5) / memory_per_sequence_mb)
        
        # Calculate batch size
        batch_size = min(
            max_sequences_in_memory,
            max(16, num_sequences // 100),  # At least 100 batches
            256  # Upper limit
        )
        
        return max(1, batch_size)
    
    def _predict_in_batches(self, sequences: np.ndarray, batch_size: int, timeout_seconds: float) -> np.ndarray:
        """Predict in batches with timeout and progress monitoring."""
        num_sequences = len(sequences)
        num_batches = (num_sequences + batch_size - 1) // batch_size
        
        reconstructions = np.zeros_like(sequences)
        
        start_time = time.time()
        
        for i in range(0, num_sequences, batch_size):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Prediction timed out after {timeout_seconds} seconds")
            
            end_idx = min(i + batch_size, num_sequences)
            batch = sequences[i:end_idx]
            
            # Predict batch
            batch_reconstructions = self.model.predict(batch, verbose=0)
            reconstructions[i:end_idx] = batch_reconstructions
            
            # Progress logging
            if i // batch_size % 10 == 0:  # Log every 10 batches
                progress = (i + batch_size) / num_sequences * 100
                elapsed = time.time() - start_time
                self.logger.debug(f"Prediction progress: {progress:.1f}% ({elapsed:.1f}s elapsed)")
        
        return reconstructions
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'training_performance': self.training_stats,
            'batch_size_optimization': {
                'optimal_batch_size': self.optimal_batch_size,
                'batch_size_history': self.batch_size_history[-10:],  # Last 10 entries
            },
            'memory_tracking': {
                'peak_memory_mb': max(self.memory_tracker) if hasattr(self, 'memory_tracker') and self.memory_tracker else 0,
                'average_memory_mb': np.mean(self.memory_tracker) if hasattr(self, 'memory_tracker') and self.memory_tracker else 0,
                'memory_history': self.memory_tracker[-20:] if hasattr(self, 'memory_tracker') else []
            },
            'throughput_metrics': {
                'peak_throughput': max(self.throughput_tracker) if hasattr(self, 'throughput_tracker') and self.throughput_tracker else 0,
                'average_throughput': np.mean(self.throughput_tracker) if hasattr(self, 'throughput_tracker') and self.throughput_tracker else 0,
                'throughput_history': self.throughput_tracker[-20:] if hasattr(self, 'throughput_tracker') else []
            },
            'resource_utilization': self.resource_manager.get_utilization_stats() if hasattr(self, 'resource_manager') else {},
            'cache_statistics': self.cache.get_stats() if self.enable_caching and hasattr(self, 'cache') and hasattr(self.cache, 'get_stats') else {},
            'optimization_features': {
                'mixed_precision_enabled': self.enable_mixed_precision,
                'parallel_processing_enabled': self.enable_parallel_processing,
                'caching_enabled': self.enable_caching,
                'memory_optimization_enabled': self.enable_memory_optimization,
                'adaptive_batching_enabled': self.enable_adaptive_batching,
                'max_workers': self.max_workers
            }
        }
        
        # Add function-level performance stats if available
        if hasattr(self, 'performance_stats') and self.performance_stats:
            report['function_performance'] = {}
            for func_name, stats in self.performance_stats.items():
                if stats:
                    report['function_performance'][func_name] = {
                        'call_count': len(stats),
                        'average_duration': np.mean([s['duration'] for s in stats]),
                        'total_duration': sum([s['duration'] for s in stats]),
                        'average_memory_delta': np.mean([s['memory_delta'] for s in stats]),
                        'success_rate': np.mean([s['success'] for s in stats])
                    }
        
        return report
    
    def cleanup_resources(self):
        """Cleanup resources to free memory."""
        try:
            if hasattr(self, 'thread_pool') and self.thread_pool:
                self.thread_pool.shutdown(wait=False)
            
            if hasattr(self, 'process_pool') and self.process_pool:
                self.process_pool.shutdown(wait=False)
            
            if hasattr(self, 'resource_manager'):
                self.resource_manager.cleanup()
            
            if hasattr(self, 'cache'):
                if hasattr(self.cache, 'clear'):
                    self.cache.clear()
                elif isinstance(self.cache, dict):
                    self.cache.clear()
            
            # Clean up temporary files
            temp_files = ['temp_sequences.dat']
            for temp_file in temp_files:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("Resources cleaned up successfully")
            
        except Exception as e:
            self.logger.warning(f"Error during resource cleanup: {e}")
    
    def __del__(self):
        """Destructor to cleanup resources."""
        try:
            self.cleanup_resources()
        except:
            pass  # Ignore errors during destruction