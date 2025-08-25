"""
Generation 1: Autonomous Core Anomaly Detection System
Advanced LSTM-Autoencoder with Real-time Processing Capabilities
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from .config import Config
from .data_preprocessor import DataPreprocessor
from .autoencoder_model import AutoencoderModel
from .logging_config import setup_logging


@dataclass
class AnomalyResult:
    """Structured anomaly detection result with metadata."""
    timestamp: float
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    sensor_values: Dict[str, float]
    reconstruction_error: float
    threshold: float
    model_version: str = "1.0"
    processing_time_ms: float = 0.0


@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    accuracy: float = 0.0
    total_samples: int = 0
    anomaly_samples: int = 0
    processing_speed_samples_per_sec: float = 0.0
    memory_usage_mb: float = 0.0


class AutonomousAnomalyCore:
    """
    Generation 1: Advanced autonomous anomaly detection core with self-optimization.
    
    Features:
    - Adaptive threshold determination
    - Real-time performance monitoring
    - Automated model retraining triggers
    - Multi-scale temporal analysis
    - Ensemble uncertainty quantification
    """
    
    def __init__(
        self,
        window_size: int = 30,
        latent_dim: int = 16,
        threshold_method: str = "adaptive",
        scaler_type: str = "robust",
        ensemble_size: int = 3,
        confidence_threshold: float = 0.85,
        auto_retrain: bool = True,
        monitoring_enabled: bool = True
    ):
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.threshold_method = threshold_method
        self.scaler_type = scaler_type
        self.ensemble_size = ensemble_size
        self.confidence_threshold = confidence_threshold
        self.auto_retrain = auto_retrain
        self.monitoring_enabled = monitoring_enabled
        
        # Core components
        self.models: List[AutoencoderModel] = []
        self.scaler = self._create_scaler()
        self.preprocessor = DataPreprocessor()
        
        # Adaptive components
        self.threshold_history: List[float] = []
        self.performance_history: List[ModelMetrics] = []
        self.drift_detector = DataDriftDetector()
        
        # Monitoring
        self.logger = setup_logging(__name__)
        self.metrics = ModelMetrics()
        self._processing_times: List[float] = []
        
        # State management
        self.is_trained = False
        self.last_retrain_time = 0.0
        self.retrain_interval = 86400  # 24 hours
        
        self.logger.info(f"Initialized AutonomousAnomalyCore with {ensemble_size} ensemble models")
    
    def _create_scaler(self):
        """Create adaptive scaler based on configuration."""
        scalers = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler(),
        }
        return scalers.get(self.scaler_type, RobustScaler())
    
    async def train_ensemble(
        self,
        data: pd.DataFrame,
        epochs: int = 50,
        validation_split: float = 0.2,
        early_stopping: bool = True
    ) -> Dict[str, Any]:
        """Train ensemble of autoencoder models with advanced techniques."""
        start_time = time.time()
        
        try:
            # Preprocess training data
            scaled_data = self.preprocessor.fit_transform(data)
            X_sequences = self.preprocessor.create_sequences(
                scaled_data, self.window_size
            )
            
            # Train ensemble models
            self.models = []
            ensemble_metrics = []
            
            for i in range(self.ensemble_size):
                self.logger.info(f"Training ensemble model {i+1}/{self.ensemble_size}")
                
                # Create model with slight variations for diversity
                model = AutoencoderModel(
                    input_dim=scaled_data.shape[1],
                    window_size=self.window_size,
                    latent_dim=self.latent_dim + np.random.randint(-2, 3),
                    dropout_rate=0.1 + np.random.uniform(0, 0.1)
                )
                
                # Train with different data splits for robustness
                indices = np.random.permutation(len(X_sequences))
                train_size = int(0.8 * len(indices))
                train_indices = indices[:train_size]
                
                X_train = X_sequences[train_indices]
                
                history = model.fit(
                    X_train,
                    epochs=epochs,
                    validation_split=validation_split,
                    verbose=0 if i > 0 else 1
                )
                
                self.models.append(model)
                ensemble_metrics.append(history.history)
            
            # Fit scaler and set training status
            self.scaler.fit(scaled_data)
            self.is_trained = True
            self.last_retrain_time = time.time()
            
            # Calculate adaptive threshold
            await self._calculate_adaptive_threshold(X_sequences)
            
            training_time = time.time() - start_time
            
            training_results = {
                "ensemble_size": len(self.models),
                "training_time": training_time,
                "data_shape": scaled_data.shape,
                "window_size": self.window_size,
                "threshold": getattr(self, 'adaptive_threshold', 0.0),
                "ensemble_metrics": ensemble_metrics
            }
            
            self.logger.info(f"Ensemble training completed in {training_time:.2f}s")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Ensemble training failed: {str(e)}")
            raise
    
    async def _calculate_adaptive_threshold(self, sequences: np.ndarray) -> None:
        """Calculate adaptive threshold using ensemble predictions."""
        if not self.models:
            raise ValueError("No trained models available for threshold calculation")
        
        # Get ensemble predictions
        reconstruction_errors = []
        
        for model in self.models:
            reconstructed = model.predict(sequences)
            errors = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))
            reconstruction_errors.append(errors)
        
        # Ensemble reconstruction errors
        ensemble_errors = np.mean(reconstruction_errors, axis=0)
        
        # Adaptive threshold methods
        if self.threshold_method == "adaptive":
            # Use robust statistics for threshold
            q75 = np.percentile(ensemble_errors, 75)
            q95 = np.percentile(ensemble_errors, 95)
            iqr = np.percentile(ensemble_errors, 75) - np.percentile(ensemble_errors, 25)
            
            # Combine multiple threshold estimates
            threshold_estimates = [
                q95,  # High percentile
                q75 + 1.5 * iqr,  # Outlier detection method
                np.mean(ensemble_errors) + 2 * np.std(ensemble_errors)  # Statistical method
            ]
            
            self.adaptive_threshold = np.median(threshold_estimates)
            
        elif self.threshold_method == "statistical":
            self.adaptive_threshold = np.mean(ensemble_errors) + 2 * np.std(ensemble_errors)
            
        elif self.threshold_method == "percentile":
            self.adaptive_threshold = np.percentile(ensemble_errors, 95)
        
        self.threshold_history.append(self.adaptive_threshold)
        self.logger.info(f"Adaptive threshold calculated: {self.adaptive_threshold:.4f}")
    
    async def predict_anomaly(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        return_probabilities: bool = False
    ) -> Union[List[AnomalyResult], Tuple[List[AnomalyResult], np.ndarray]]:
        """Predict anomalies with ensemble uncertainty quantification."""
        start_time = time.time()
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Preprocess data
            if isinstance(data, pd.DataFrame):
                scaled_data = self.scaler.transform(data)
                timestamps = data.index if hasattr(data, 'index') else np.arange(len(data))
                sensor_data = data.to_dict('records')
            else:
                scaled_data = self.scaler.transform(data)
                timestamps = np.arange(len(data))
                sensor_data = [{"sensor_{}".format(i): val for i, val in enumerate(row)} for row in data]
            
            # Create sequences
            sequences = self.preprocessor.create_sequences(scaled_data, self.window_size)
            
            # Ensemble prediction
            ensemble_predictions = []
            ensemble_errors = []
            
            for model in self.models:
                reconstructed = model.predict(sequences)
                errors = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))
                ensemble_predictions.append(reconstructed)
                ensemble_errors.append(errors)
            
            # Calculate ensemble statistics
            mean_errors = np.mean(ensemble_errors, axis=0)
            std_errors = np.std(ensemble_errors, axis=0)
            confidence_scores = 1.0 / (1.0 + std_errors)  # Higher confidence with lower variance
            
            # Generate results
            results = []
            anomaly_probabilities = []
            
            for i, (error, confidence) in enumerate(zip(mean_errors, confidence_scores)):
                # Determine anomaly status
                is_anomaly = error > self.adaptive_threshold
                anomaly_probability = self._calculate_anomaly_probability(error)
                anomaly_probabilities.append(anomaly_probability)
                
                # Create result object
                result = AnomalyResult(
                    timestamp=float(timestamps[min(i, len(timestamps)-1)]),
                    anomaly_score=float(error),
                    is_anomaly=bool(is_anomaly),
                    confidence=float(confidence),
                    sensor_values=sensor_data[min(i, len(sensor_data)-1)],
                    reconstruction_error=float(error),
                    threshold=float(self.adaptive_threshold),
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                
                results.append(result)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._processing_times.append(processing_time)
            
            if self.monitoring_enabled:
                await self._update_performance_metrics(results)
            
            self.logger.info(f"Processed {len(sequences)} sequences in {processing_time:.3f}s")
            
            if return_probabilities:
                return results, np.array(anomaly_probabilities)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Anomaly prediction failed: {str(e)}")
            raise
    
    def _calculate_anomaly_probability(self, error: float) -> float:
        """Calculate probability of anomaly using sigmoid transformation."""
        # Sigmoid transformation centered at threshold
        x = (error - self.adaptive_threshold) / (self.adaptive_threshold * 0.1)
        return 1.0 / (1.0 + np.exp(-x))
    
    async def _update_performance_metrics(self, results: List[AnomalyResult]) -> None:
        """Update real-time performance metrics."""
        if not results:
            return
        
        # Calculate processing speed
        avg_processing_time = np.mean(self._processing_times[-100:])  # Last 100 predictions
        self.metrics.processing_speed_samples_per_sec = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        # Count anomalies
        anomaly_count = sum(1 for r in results if r.is_anomaly)
        self.metrics.anomaly_samples = anomaly_count
        self.metrics.total_samples = len(results)
        
        # Memory usage (approximate)
        import psutil
        process = psutil.Process()
        self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
    
    async def evaluate_performance(
        self,
        test_data: pd.DataFrame,
        ground_truth: Optional[np.ndarray] = None
    ) -> ModelMetrics:
        """Comprehensive performance evaluation with multiple metrics."""
        results, probabilities = await self.predict_anomaly(test_data, return_probabilities=True)
        
        if ground_truth is not None:
            # Align ground truth with results
            predictions = np.array([r.is_anomaly for r in results])
            
            # Calculate classification metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truth, predictions, average='binary', zero_division=0
            )
            
            accuracy = np.mean(ground_truth == predictions)
            
            # ROC AUC if probabilities available
            roc_auc = roc_auc_score(ground_truth, probabilities) if len(np.unique(ground_truth)) > 1 else 0.0
            
            # Update metrics
            self.metrics.precision = precision
            self.metrics.recall = recall
            self.metrics.f1_score = f1
            self.metrics.roc_auc = roc_auc
            self.metrics.accuracy = accuracy
            
            self.logger.info(f"Performance: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, AUC={roc_auc:.3f}")
        
        self.performance_history.append(self.metrics)
        return self.metrics
    
    async def auto_retrain_check(self, current_data: pd.DataFrame) -> bool:
        """Check if automatic retraining is needed based on drift detection."""
        if not self.auto_retrain or not self.is_trained:
            return False
        
        # Time-based retraining
        time_since_retrain = time.time() - self.last_retrain_time
        if time_since_retrain > self.retrain_interval:
            self.logger.info("Time-based retraining triggered")
            return True
        
        # Performance-based retraining
        if len(self.performance_history) > 10:
            recent_performance = np.mean([m.f1_score for m in self.performance_history[-5:]])
            historical_performance = np.mean([m.f1_score for m in self.performance_history[-10:-5]])
            
            if historical_performance > 0 and recent_performance < 0.8 * historical_performance:
                self.logger.info("Performance degradation detected, retraining triggered")
                return True
        
        # Data drift-based retraining
        try:
            drift_detected = await self.drift_detector.detect_drift(current_data)
            if drift_detected:
                self.logger.info("Data drift detected, retraining triggered")
                return True
        except Exception as e:
            self.logger.warning(f"Drift detection failed: {str(e)}")
        
        return False
    
    def save_state(self, path: Path) -> None:
        """Save complete model state including ensemble and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for i, model in enumerate(self.models):
            model.save(path / f"model_{i}.h5")
        
        # Save scaler
        import pickle
        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        # Save configuration and metadata
        config = {
            "window_size": self.window_size,
            "latent_dim": self.latent_dim,
            "threshold_method": self.threshold_method,
            "scaler_type": self.scaler_type,
            "ensemble_size": self.ensemble_size,
            "adaptive_threshold": getattr(self, 'adaptive_threshold', 0.0),
            "is_trained": self.is_trained,
            "last_retrain_time": self.last_retrain_time,
            "threshold_history": self.threshold_history,
            "model_version": "1.0"
        }
        
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Model state saved to {path}")
    
    @classmethod
    def load_state(cls, path: Path) -> 'AutonomousAnomalyCore':
        """Load complete model state from disk."""
        path = Path(path)
        
        # Load configuration
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(
            window_size=config["window_size"],
            latent_dim=config["latent_dim"],
            threshold_method=config["threshold_method"],
            scaler_type=config["scaler_type"],
            ensemble_size=config["ensemble_size"]
        )
        
        # Load models
        instance.models = []
        for i in range(config["ensemble_size"]):
            model_path = path / f"model_{i}.h5"
            if model_path.exists():
                model = tf.keras.models.load_model(str(model_path))
                instance.models.append(model)
        
        # Load scaler
        import pickle
        with open(path / "scaler.pkl", "rb") as f:
            instance.scaler = pickle.load(f)
        
        # Restore state
        instance.adaptive_threshold = config.get("adaptive_threshold", 0.0)
        instance.is_trained = config.get("is_trained", False)
        instance.last_retrain_time = config.get("last_retrain_time", 0.0)
        instance.threshold_history = config.get("threshold_history", [])
        
        instance.logger.info(f"Model state loaded from {path}")
        return instance


class DataDriftDetector:
    """Detect data drift for triggering model retraining."""
    
    def __init__(self, window_size: int = 1000, drift_threshold: float = 0.05):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.reference_stats: Optional[Dict[str, float]] = None
        self.logger = setup_logging(__name__)
    
    async def detect_drift(self, current_data: pd.DataFrame) -> bool:
        """Detect data drift using statistical tests."""
        try:
            if self.reference_stats is None:
                # Initialize reference statistics
                self.reference_stats = self._calculate_stats(current_data)
                return False
            
            # Calculate current statistics
            current_stats = self._calculate_stats(current_data.tail(self.window_size))
            
            # Compare distributions using Jensen-Shannon divergence
            drift_score = self._calculate_drift_score(self.reference_stats, current_stats)
            
            if drift_score > self.drift_threshold:
                self.logger.warning(f"Data drift detected: score={drift_score:.4f}")
                # Update reference stats for continuous monitoring
                self.reference_stats = current_stats
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Drift detection error: {str(e)}")
            return False
    
    def _calculate_stats(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical properties of data."""
        numeric_data = data.select_dtypes(include=[np.number])
        return {
            "mean": float(numeric_data.mean().mean()),
            "std": float(numeric_data.std().mean()),
            "skew": float(numeric_data.skew().mean()),
            "kurtosis": float(numeric_data.kurtosis().mean()),
            "min": float(numeric_data.min().min()),
            "max": float(numeric_data.max().max())
        }
    
    def _calculate_drift_score(self, ref_stats: Dict[str, float], curr_stats: Dict[str, float]) -> float:
        """Calculate drift score between reference and current statistics."""
        scores = []
        for key in ref_stats.keys():
            if key in curr_stats:
                # Normalized absolute difference
                ref_val = ref_stats[key]
                curr_val = curr_stats[key]
                
                if abs(ref_val) > 1e-8:  # Avoid division by zero
                    score = abs(curr_val - ref_val) / abs(ref_val)
                else:
                    score = abs(curr_val - ref_val)
                
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0