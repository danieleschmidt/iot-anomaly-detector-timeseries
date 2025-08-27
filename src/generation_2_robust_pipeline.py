"""Generation 2 Robust Pipeline: MAKE IT ROBUST - Enhanced reliability and comprehensive monitoring."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import joblib
import time
import warnings
import threading
import json
import sqlite3
from datetime import datetime, timedelta
import hashlib

from .robust_autoencoder_v2 import RobustAutoencoderV2
from .data_preprocessor import DataPreprocessor
from .data_validator import DataValidator, ValidationLevel
from .logging_config import get_logger, setup_logging
from .config import get_config
from .circuit_breaker import CircuitBreaker
from .retry_manager import RetryManager
from .robust_error_handler import RobustErrorHandler
from .health_monitoring import HealthMonitor
from .comprehensive_monitoring import ComprehensiveMonitoring as ComprehensiveMonitor


class Generation2RobustPipeline:
    """Robust pipeline with enhanced reliability, monitoring, and error handling - Generation 2."""
    
    def __init__(
        self,
        window_size: int = 30,
        encoding_dim: int = 16,
        contamination_rate: float = 0.1,
        enable_logging: bool = True,
        enable_monitoring: bool = True,
        enable_data_validation: bool = True,
        validation_level: str = "moderate",
        enable_health_checks: bool = True,
        enable_backup: bool = True,
        backup_interval_minutes: int = 30,
        max_memory_usage_mb: int = 1024,
        enable_performance_tracking: bool = True
    ):
        """Initialize Generation 2 robust pipeline.
        
        Args:
            window_size: Size of sliding window for sequences
            encoding_dim: Encoding dimension for autoencoder
            contamination_rate: Expected contamination rate
            enable_logging: Whether to enable comprehensive logging
            enable_monitoring: Whether to enable system monitoring
            enable_data_validation: Whether to enable data validation
            validation_level: Data validation strictness level
            enable_health_checks: Whether to enable health monitoring
            enable_backup: Whether to enable automated backups
            backup_interval_minutes: Backup interval in minutes
            max_memory_usage_mb: Maximum memory usage limit
            enable_performance_tracking: Whether to track performance metrics
        """
        self.window_size = window_size
        self.encoding_dim = encoding_dim
        self.contamination_rate = contamination_rate
        self.max_memory_usage_mb = max_memory_usage_mb
        
        # Setup logging
        if enable_logging:
            setup_logging(level="INFO")
        
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.autoencoder = None
        self.preprocessor = DataPreprocessor(enable_validation=enable_data_validation)
        
        # Data validation
        if enable_data_validation:
            validation_levels = {
                "strict": ValidationLevel.STRICT,
                "moderate": ValidationLevel.MODERATE,
                "permissive": ValidationLevel.PERMISSIVE
            }
            self.data_validator = DataValidator(
                validation_level=validation_levels.get(validation_level.lower(), ValidationLevel.MODERATE)
            )
        else:
            self.data_validator = None
        
        # Error handling and reliability
        self.error_handler = RobustErrorHandler()
        from .circuit_breaker import CircuitBreakerConfig
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=300.0
        )
        self.circuit_breaker = CircuitBreaker(
            name="generation2_pipeline",
            config=config,
            allowed_exceptions=[Exception]
        )
        self.retry_manager = RetryManager(
            max_retries=3,
            base_delay=2.0,
            max_delay=60.0,
            backoff_multiplier=2.0,
            jitter=True
        )
        
        # Monitoring and health checks
        if enable_monitoring:
            self.monitor = ComprehensiveMonitor()
            self.monitor.start()
        else:
            self.monitor = None
        
        if enable_health_checks:
            self.health_monitor = HealthMonitor()
            self.health_monitor.start()
        else:
            self.health_monitor = None
        
        # State tracking
        self.is_trained = False
        self.training_metadata = {}
        self.performance_history = []
        self.error_log = []
        
        # Backup and persistence
        self.enable_backup = enable_backup
        self.backup_interval_minutes = backup_interval_minutes
        self.last_backup_time = None
        
        # Performance tracking
        self.enable_performance_tracking = enable_performance_tracking
        if enable_performance_tracking:
            self.performance_metrics = {
                'training_times': [],
                'prediction_times': [],
                'memory_usage': [],
                'error_rates': [],
                'throughput_samples_per_second': []
            }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Database for persistent logging
        self.db_path = None
        self._init_database()
        
        self.logger.info("Initialized Generation 2 Robust Pipeline with comprehensive monitoring")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for persistent logging."""
        try:
            self.db_path = Path("logs/pipeline_database.db")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create tables for logging
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    duration_seconds REAL,
                    samples_count INTEGER,
                    epochs_completed INTEGER,
                    final_loss REAL,
                    validation_loss REAL,
                    status TEXT,
                    error_message TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    duration_seconds REAL,
                    samples_count INTEGER,
                    anomalies_detected INTEGER,
                    threshold_used REAL,
                    mean_error REAL,
                    status TEXT,
                    error_message TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    operation_type TEXT,
                    duration_seconds REAL,
                    memory_usage_mb REAL,
                    cpu_usage_percent REAL,
                    throughput REAL,
                    error_rate REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Initialized database at {self.db_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize database: {e}")
            self.db_path = None
    
    def _log_to_database(self, table: str, data: Dict[str, Any]) -> None:
        """Log data to SQLite database."""
        if not self.db_path:
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Prepare data with timestamp
            data_with_timestamp = {'timestamp': datetime.now(), **data}
            
            # Insert based on table
            if table == 'training_logs':
                cursor.execute('''
                    INSERT INTO training_logs 
                    (timestamp, duration_seconds, samples_count, epochs_completed, 
                     final_loss, validation_loss, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data_with_timestamp['timestamp'],
                    data_with_timestamp.get('duration_seconds'),
                    data_with_timestamp.get('samples_count'),
                    data_with_timestamp.get('epochs_completed'),
                    data_with_timestamp.get('final_loss'),
                    data_with_timestamp.get('validation_loss'),
                    data_with_timestamp.get('status'),
                    data_with_timestamp.get('error_message')
                ))
            
            elif table == 'prediction_logs':
                cursor.execute('''
                    INSERT INTO prediction_logs 
                    (timestamp, duration_seconds, samples_count, anomalies_detected,
                     threshold_used, mean_error, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data_with_timestamp['timestamp'],
                    data_with_timestamp.get('duration_seconds'),
                    data_with_timestamp.get('samples_count'),
                    data_with_timestamp.get('anomalies_detected'),
                    data_with_timestamp.get('threshold_used'),
                    data_with_timestamp.get('mean_error'),
                    data_with_timestamp.get('status'),
                    data_with_timestamp.get('error_message')
                ))
            
            elif table == 'performance_metrics':
                cursor.execute('''
                    INSERT INTO performance_metrics 
                    (timestamp, operation_type, duration_seconds, memory_usage_mb,
                     cpu_usage_percent, throughput, error_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data_with_timestamp['timestamp'],
                    data_with_timestamp.get('operation_type'),
                    data_with_timestamp.get('duration_seconds'),
                    data_with_timestamp.get('memory_usage_mb'),
                    data_with_timestamp.get('cpu_usage_percent'),
                    data_with_timestamp.get('throughput'),
                    data_with_timestamp.get('error_rate')
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Could not log to database: {e}")
    
    def _validate_and_prepare_data(
        self,
        data_path: str,
        target_column: Optional[str] = None,
        auto_fix: bool = True
    ) -> pd.DataFrame:
        """Comprehensive data validation and preparation."""
        self.logger.info(f"Starting data validation and preparation for {data_path}")
        
        try:
            # Load data
            data = pd.read_csv(data_path)
            self.logger.info(f"Loaded {len(data)} samples with {len(data.columns)} features")
            
            # Remove target column if specified
            if target_column and target_column in data.columns:
                data = data.drop(columns=[target_column])
                self.logger.info(f"Removed target column: {target_column}")
            
            # Data validation with comprehensive checks
            if self.data_validator:
                validation_report = self.data_validator.validate(data, auto_fix=auto_fix)
                
                if not validation_report['is_valid']:
                    error_msg = f"Data validation failed: {validation_report.get('error_summary', 'Unknown errors')}"
                    self.logger.error(error_msg)
                    
                    if validation_report.get('critical_errors'):
                        raise ValueError(error_msg)
                    else:
                        self.logger.warning("Data has non-critical issues, proceeding with caution")
                
                # Use auto-fixed data if available
                if auto_fix and 'cleaned_data' in validation_report:
                    data = validation_report['cleaned_data']
                    self.logger.info("Using auto-fixed data")
            
            # Additional preprocessing
            data = self.preprocessor.clean_and_prepare(data)
            
            # Final validation checks
            if len(data) == 0:
                raise ValueError("Data is empty after preprocessing")
            
            if len(data) < self.window_size:
                raise ValueError(f"Insufficient data: {len(data)} samples, need at least {self.window_size}")
            
            # Check data quality
            null_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            if null_percentage > 50:
                self.logger.warning(f"High percentage of missing values: {null_percentage:.1f}%")
            
            self.logger.info(f"Data preparation completed: {len(data)} samples, {len(data.columns)} features")
            return data
            
        except Exception as e:
            self.error_handler.handle_error(e, context="data_preparation")
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    def train_robust(
        self,
        train_data_path: str,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        save_model: bool = True,
        model_dir: str = "saved_models_v2",
        auto_backup: bool = True
    ) -> Dict[str, Any]:
        """Train the pipeline with comprehensive error handling and monitoring.
        
        Args:
            train_data_path: Path to training data (normal patterns)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation split ratio
            save_model: Whether to save the trained model
            model_dir: Directory to save models
            auto_backup: Whether to create automatic backups
            
        Returns:
            Training results with comprehensive metrics
        """
        with self._lock:
            start_time = time.time()
            training_id = hashlib.md5(f"{train_data_path}_{start_time}".encode()).hexdigest()[:8]
            
            self.logger.info(f"Starting robust training pipeline (ID: {training_id})")
            
            training_success = False
            error_message = None
            
            try:
                # Monitor system resources
                if self.monitor:
                    initial_metrics = self.monitor.get_current_metrics()
                    self.logger.info(f"Initial system metrics: {initial_metrics}")
                
                # Validate and prepare data
                train_data = self._validate_and_prepare_data(train_data_path)
                
                # Check memory requirements
                estimated_memory_mb = self._estimate_memory_usage(len(train_data), len(train_data.columns))
                if estimated_memory_mb > self.max_memory_usage_mb:
                    self.logger.warning(f"Estimated memory usage ({estimated_memory_mb}MB) exceeds limit ({self.max_memory_usage_mb}MB)")
                
                # Initialize robust autoencoder
                self.autoencoder = RobustAutoencoderV2(
                    window_size=self.window_size,
                    encoding_dim=self.encoding_dim,
                    enable_circuit_breaker=True,
                    enable_retry=True,
                    max_retries=3,
                    enable_checkpointing=True,
                    enable_early_stopping=True,
                    enable_lr_scheduling=True,
                    enable_validation=True
                )
                
                # Setup checkpoint directory
                checkpoint_dir = Path(model_dir) / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Training with circuit breaker and retry
                def _train_with_monitoring():
                    return self.autoencoder.fit(
                        train_data,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        verbose=1,
                        checkpoint_dir=str(checkpoint_dir)
                    )
                
                # Execute training with retry mechanism
                training_results = self.retry_manager.execute_with_retry(
                    lambda: self.circuit_breaker.call(_train_with_monitoring)
                )
                
                self.is_trained = True
                training_success = True
                training_time = time.time() - start_time
                
                # Save model if requested
                if save_model:
                    model_path = Path(model_dir)
                    model_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save with enhanced metadata
                    self.autoencoder.save_robust(
                        str(model_path / "robust_autoencoder_v2.h5"),
                        str(model_path / "robust_scaler_v2.pkl"),
                        str(model_path / "metadata_v2.pkl")
                    )
                    
                    # Save pipeline configuration
                    pipeline_config = {
                        'window_size': self.window_size,
                        'encoding_dim': self.encoding_dim,
                        'contamination_rate': self.contamination_rate,
                        'training_data_path': train_data_path,
                        'training_samples': len(train_data),
                        'feature_count': len(train_data.columns),
                        'training_time_seconds': training_time,
                        'training_id': training_id,
                        'training_timestamp': datetime.now().isoformat(),
                        'system_info': self.monitor.get_current_metrics() if self.monitor else None,
                        'validation_enabled': self.data_validator is not None,
                        'backup_enabled': self.enable_backup
                    }
                    
                    joblib.dump(pipeline_config, model_path / "pipeline_config_v2.pkl")
                    
                    # Create backup if enabled
                    if auto_backup and self.enable_backup:
                        self._create_backup(model_path)
                    
                    self.logger.info(f"Saved robust model and configuration to {model_path}")
                
                # Store training metadata
                self.training_metadata = {
                    'training_id': training_id,
                    'training_time_seconds': training_time,
                    'training_samples': len(train_data),
                    'feature_count': len(train_data.columns),
                    'final_results': training_results,
                    'system_metrics': self.monitor.get_current_metrics() if self.monitor else None,
                    'health_status': self.health_monitor.get_health_report() if self.health_monitor else None
                }
                
                # Update performance tracking
                if self.enable_performance_tracking:
                    self.performance_metrics['training_times'].append(training_time)
                    if self.monitor:
                        memory_mb = self.monitor.get_memory_usage()
                        self.performance_metrics['memory_usage'].append(memory_mb)
                
                # Log to database
                self._log_to_database('training_logs', {
                    'duration_seconds': training_time,
                    'samples_count': len(train_data),
                    'epochs_completed': training_results.get('epochs_completed', 0),
                    'final_loss': training_results.get('validation_metrics', {}).get('final_training_loss'),
                    'validation_loss': training_results.get('validation_metrics', {}).get('final_validation_loss'),
                    'status': 'SUCCESS',
                    'error_message': None
                })
                
                # Prepare comprehensive results
                results = {
                    'training_id': training_id,
                    'training_time_seconds': training_time,
                    'training_samples': len(train_data),
                    'feature_count': len(train_data.columns),
                    'epochs_completed': training_results.get('epochs_completed', 0),
                    'training_results': training_results,
                    'system_metrics': self.monitor.get_current_metrics() if self.monitor else None,
                    'health_status': self.health_monitor.get_health_report() if self.health_monitor else None,
                    'memory_estimate_mb': estimated_memory_mb,
                    'circuit_breaker_stats': {
                        'state': self.circuit_breaker.state.name,
                        'failure_count': self.circuit_breaker.failure_count,
                        'success_count': self.circuit_breaker.success_count
                    },
                    'data_validation_enabled': self.data_validator is not None,
                    'monitoring_enabled': self.monitor is not None
                }
                
                self.logger.info(f"Robust training completed successfully in {training_time:.2f} seconds (ID: {training_id})")
                
                return results
                
            except Exception as e:
                training_time = time.time() - start_time
                error_message = str(e)
                
                # Log error to database
                self._log_to_database('training_logs', {
                    'duration_seconds': training_time,
                    'samples_count': 0,
                    'epochs_completed': 0,
                    'final_loss': None,
                    'validation_loss': None,
                    'status': 'FAILED',
                    'error_message': error_message
                })
                
                self.error_handler.handle_error(e, context=f"robust_training_{training_id}")
                self.logger.error(f"Robust training failed (ID: {training_id}): {e}")
                raise
    
    def _estimate_memory_usage(self, samples: int, features: int) -> float:
        """Estimate memory usage for training."""
        # Rough estimation based on data size and model complexity
        data_size_mb = (samples * features * 8) / (1024 * 1024)  # Assuming float64
        sequences_size_mb = ((samples - self.window_size + 1) * self.window_size * features * 4) / (1024 * 1024)  # float32
        model_size_mb = (self.encoding_dim * features * 4) / (1024 * 1024)  # Rough model size
        overhead_mb = 100  # TensorFlow overhead
        
        total_mb = data_size_mb + sequences_size_mb + model_size_mb + overhead_mb
        return total_mb
    
    def _create_backup(self, model_path: Path) -> None:
        """Create backup of trained models."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = model_path.parent / "backups" / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model files to backup
            import shutil
            for file_path in model_path.glob("*"):
                if file_path.is_file():
                    shutil.copy2(file_path, backup_dir / file_path.name)
            
            self.last_backup_time = datetime.now()
            self.logger.info(f"Created backup at {backup_dir}")
            
        except Exception as e:
            self.logger.warning(f"Backup creation failed: {e}")
    
    def detect_anomalies_robust(
        self,
        data_path: str,
        threshold: Optional[float] = None,
        use_statistical_threshold: bool = True,
        confidence_level: float = 0.95,
        output_path: Optional[str] = None,
        include_detailed_stats: bool = True
    ) -> Dict[str, Any]:
        """Detect anomalies with robust error handling and comprehensive statistics.
        
        Args:
            data_path: Path to data for anomaly detection
            threshold: Optional fixed threshold
            use_statistical_threshold: Use statistical threshold calculation
            confidence_level: Confidence level for statistical threshold
            output_path: Optional path to save results
            include_detailed_stats: Include detailed statistical analysis
            
        Returns:
            Comprehensive detection results
        """
        with self._lock:
            if not self.is_trained:
                raise ValueError("Pipeline must be trained before anomaly detection")
            
            start_time = time.time()
            detection_id = hashlib.md5(f"{data_path}_{start_time}".encode()).hexdigest()[:8]
            
            self.logger.info(f"Starting robust anomaly detection (ID: {detection_id})")
            
            detection_success = False
            error_message = None
            
            try:
                # Monitor system resources
                if self.monitor:
                    initial_metrics = self.monitor.get_current_metrics()
                
                # Prepare data
                data = self._validate_and_prepare_data(data_path, validate_data=True)
                
                # Detect anomalies with robust methods
                anomalies, threshold_used, basic_stats = self.autoencoder.detect_anomalies_robust(
                    data,
                    threshold=threshold,
                    contamination_rate=self.contamination_rate,
                    use_statistical_threshold=use_statistical_threshold,
                    confidence_level=confidence_level
                )
                
                detection_time = time.time() - start_time
                detection_success = True
                
                # Calculate additional statistics if requested
                detailed_stats = basic_stats.copy()
                if include_detailed_stats:
                    _, reconstruction_errors = self.autoencoder.predict_robust(data)
                    
                    # Statistical analysis
                    detailed_stats.update({
                        'reconstruction_error_distribution': {
                            'q10': float(np.percentile(reconstruction_errors, 10)),
                            'q25': float(np.percentile(reconstruction_errors, 25)),
                            'q75': float(np.percentile(reconstruction_errors, 75)),
                            'q90': float(np.percentile(reconstruction_errors, 90)),
                            'iqr': float(np.percentile(reconstruction_errors, 75) - np.percentile(reconstruction_errors, 25)),
                            'skewness': float(pd.Series(reconstruction_errors).skew()),
                            'kurtosis': float(pd.Series(reconstruction_errors).kurtosis())
                        },
                        'anomaly_analysis': {
                            'anomaly_indices': np.where(anomalies)[0].tolist(),
                            'anomaly_scores': reconstruction_errors[anomalies].tolist() if np.any(anomalies) else [],
                            'max_anomaly_score': float(np.max(reconstruction_errors[anomalies])) if np.any(anomalies) else 0.0,
                            'anomaly_score_threshold_ratio': float(np.max(reconstruction_errors[anomalies]) / threshold_used) if np.any(anomalies) else 0.0
                        },
                        'temporal_analysis': self._analyze_temporal_patterns(anomalies, reconstruction_errors),
                        'confidence_intervals': self._calculate_confidence_intervals(reconstruction_errors, confidence_level)
                    })
                
                # Save results if requested
                if output_path:
                    output_df = pd.DataFrame({
                        'sequence_index': range(len(anomalies)),
                        'is_anomaly': anomalies.astype(int),
                        'reconstruction_error': reconstruction_errors if 'reconstruction_errors' in locals() else [0] * len(anomalies),
                        'threshold': threshold_used,
                        'confidence_level': confidence_level,
                        'detection_id': detection_id,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    output_df.to_csv(output_path, index=False)
                    self.logger.info(f"Saved detection results to {output_path}")
                
                # Update performance tracking
                if self.enable_performance_tracking:
                    self.performance_metrics['prediction_times'].append(detection_time)
                    throughput = len(data) / detection_time
                    self.performance_metrics['throughput_samples_per_second'].append(throughput)
                
                # Log to database
                self._log_to_database('prediction_logs', {
                    'duration_seconds': detection_time,
                    'samples_count': len(data),
                    'anomalies_detected': int(detailed_stats['anomaly_count']),
                    'threshold_used': threshold_used,
                    'mean_error': detailed_stats['mean_reconstruction_error'],
                    'status': 'SUCCESS',
                    'error_message': None
                })
                
                # Prepare comprehensive results
                results = {
                    'detection_id': detection_id,
                    'detection_time_seconds': detection_time,
                    'data_samples': len(data),
                    'statistics': detailed_stats,
                    'anomaly_flags': anomalies.tolist(),
                    'threshold_used': threshold_used,
                    'confidence_level': confidence_level,
                    'system_metrics': self.monitor.get_current_metrics() if self.monitor else None,
                    'health_status': self.health_monitor.get_health_report() if self.health_monitor else None,
                    'throughput_samples_per_second': len(data) / detection_time,
                    'circuit_breaker_stats': {
                        'state': self.circuit_breaker.state.name,
                        'failure_count': self.circuit_breaker.failure_count,
                        'success_count': self.circuit_breaker.success_count
                    }
                }
                
                self.logger.info(f"Robust anomaly detection completed in {detection_time:.2f} seconds (ID: {detection_id})")
                self.logger.info(f"Detected {detailed_stats['anomaly_count']} anomalies ({detailed_stats['anomaly_rate']:.2%} rate)")
                
                return results
                
            except Exception as e:
                detection_time = time.time() - start_time
                error_message = str(e)
                
                # Log error to database
                self._log_to_database('prediction_logs', {
                    'duration_seconds': detection_time,
                    'samples_count': 0,
                    'anomalies_detected': 0,
                    'threshold_used': None,
                    'mean_error': None,
                    'status': 'FAILED',
                    'error_message': error_message
                })
                
                self.error_handler.handle_error(e, context=f"robust_detection_{detection_id}")
                self.logger.error(f"Robust anomaly detection failed (ID: {detection_id}): {e}")
                raise
    
    def _analyze_temporal_patterns(self, anomalies: np.ndarray, errors: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal patterns in anomalies."""
        if not np.any(anomalies):
            return {'anomaly_clusters': [], 'max_cluster_size': 0, 'cluster_count': 0}
        
        anomaly_indices = np.where(anomalies)[0]
        
        # Find clusters (consecutive anomalies)
        clusters = []
        current_cluster = [anomaly_indices[0]]
        
        for i in range(1, len(anomaly_indices)):
            if anomaly_indices[i] - anomaly_indices[i-1] == 1:
                current_cluster.append(anomaly_indices[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [anomaly_indices[i]]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return {
            'anomaly_clusters': [{'start': cluster[0], 'end': cluster[-1], 'size': len(cluster)} for cluster in clusters],
            'max_cluster_size': max(len(cluster) for cluster in clusters) if clusters else 0,
            'cluster_count': len(clusters),
            'isolated_anomalies': sum(1 for cluster in clusters if len(cluster) == 1),
            'clustered_anomalies': sum(1 for cluster in clusters if len(cluster) > 1)
        }
    
    def _calculate_confidence_intervals(self, errors: np.ndarray, confidence_level: float) -> Dict[str, float]:
        """Calculate confidence intervals for reconstruction errors."""
        try:
            from scipy import stats
            
            mean_error = np.mean(errors)
            std_error = np.std(errors, ddof=1)
            n = len(errors)
            
            # Calculate confidence interval for the mean
            t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 1)
            margin_error = t_critical * (std_error / np.sqrt(n))
            
            return {
                'mean_lower_bound': float(mean_error - margin_error),
                'mean_upper_bound': float(mean_error + margin_error),
                'margin_of_error': float(margin_error),
                'confidence_level': confidence_level,
                'sample_size': n
            }
        except ImportError:
            self.logger.warning("scipy not available for confidence interval calculation")
            return {}
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status and health metrics."""
        status = {
            'pipeline_info': {
                'is_trained': self.is_trained,
                'window_size': self.window_size,
                'encoding_dim': self.encoding_dim,
                'contamination_rate': self.contamination_rate,
                'training_metadata': self.training_metadata
            },
            'system_monitoring': {
                'monitoring_enabled': self.monitor is not None,
                'health_monitoring_enabled': self.health_monitor is not None,
                'current_metrics': self.monitor.get_current_metrics() if self.monitor else None,
                'health_report': self.health_monitor.get_health_report() if self.health_monitor else None
            },
            'reliability_features': {
                'circuit_breaker_state': self.circuit_breaker.state.name,
                'circuit_breaker_stats': {
                    'failure_count': self.circuit_breaker.failure_count,
                    'success_count': self.circuit_breaker.success_count
                },
                'retry_manager_enabled': True,
                'data_validation_enabled': self.data_validator is not None,
                'backup_enabled': self.enable_backup,
                'last_backup_time': self.last_backup_time.isoformat() if self.last_backup_time else None
            },
            'performance_metrics': self.performance_metrics if self.enable_performance_tracking else None,
            'database_logging': {
                'enabled': self.db_path is not None,
                'database_path': str(self.db_path) if self.db_path else None
            }
        }
        
        # Add autoencoder health status if available
        if self.autoencoder:
            status['autoencoder_health'] = self.autoencoder.get_health_status()
        
        return status
    
    @classmethod
    def load_robust_pipeline(
        cls,
        model_dir: str = "saved_models_v2",
        validate_integrity: bool = True
    ) -> 'Generation2RobustPipeline':
        """Load a previously trained robust pipeline with integrity validation."""
        model_path = Path(model_dir)
        
        # Load pipeline configuration
        pipeline_config = joblib.load(model_path / "pipeline_config_v2.pkl")
        
        # Create pipeline instance
        pipeline = cls(
            window_size=pipeline_config['window_size'],
            encoding_dim=pipeline_config['encoding_dim'],
            contamination_rate=pipeline_config['contamination_rate'],
            enable_monitoring=True,
            enable_data_validation=pipeline_config.get('validation_enabled', True),
            enable_backup=pipeline_config.get('backup_enabled', True)
        )
        
        # Load autoencoder with integrity validation
        pipeline.autoencoder = RobustAutoencoderV2.load_robust(
            str(model_path / "robust_autoencoder_v2.h5"),
            str(model_path / "robust_scaler_v2.pkl"),
            validate_integrity=validate_integrity
        )
        
        pipeline.is_trained = True
        pipeline.training_metadata = {
            'loaded_from': str(model_dir),
            'original_training_time': pipeline_config.get('training_time_seconds'),
            'original_training_samples': pipeline_config.get('training_samples'),
            'load_timestamp': datetime.now().isoformat()
        }
        
        pipeline.logger.info(f"Loaded robust pipeline from {model_dir}")
        
        return pipeline
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'monitor') and self.monitor:
                self.monitor.stop()
            if hasattr(self, 'health_monitor') and self.health_monitor:
                self.health_monitor.stop()
        except Exception:
            pass  # Ignore cleanup errors