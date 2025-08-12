"""Advanced Multi-Modal Anomaly Detection with Adaptive Learning.

This module implements a sophisticated anomaly detection system that combines
multiple detection approaches with adaptive learning capabilities for enhanced
accuracy and robustness in IoT environments.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

try:
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras import layers, models
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    warnings.warn("Advanced dependencies not available. Some features will be disabled.")

from .logging_config import get_logger
from .data_preprocessor import DataPreprocessor


@dataclass
class DetectionResult:
    """Container for anomaly detection results."""
    
    anomaly_scores: np.ndarray
    anomaly_predictions: np.ndarray
    confidence_scores: np.ndarray
    detection_method: str
    metadata: Dict[str, Any]


@dataclass
class EnsembleWeights:
    """Weights for ensemble methods."""
    
    lstm_weight: float = 0.4
    isolation_forest_weight: float = 0.3
    one_class_svm_weight: float = 0.2
    statistical_weight: float = 0.1


class BaseDetector(ABC):
    """Abstract base class for anomaly detectors."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")
        self.is_trained = False
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Train the detector on normal data."""
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> DetectionResult:
        """Detect anomalies in the given data."""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the trained detector."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load a trained detector."""
        pass


class LSTMAutoencoderDetector(BaseDetector):
    """LSTM Autoencoder-based anomaly detector."""
    
    def __init__(self, window_size: int = 30, latent_dim: int = 16, lstm_units: int = 64):
        super().__init__("LSTM_Autoencoder")
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.model: Optional[Model] = None
        self.threshold: float = 0.0
        self.scaler = StandardScaler() if DEPENDENCIES_AVAILABLE else None
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build advanced LSTM autoencoder architecture."""
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        # Encoder
        inputs = layers.Input(shape=input_shape)
        x = layers.LSTM(self.lstm_units, return_sequences=True, dropout=0.2)(inputs)
        x = layers.LSTM(self.lstm_units // 2, return_sequences=True, dropout=0.2)(x)
        encoded = layers.LSTM(self.latent_dim)(x)
        
        # Decoder
        x = layers.RepeatVector(input_shape[0])(encoded)
        x = layers.LSTM(self.latent_dim, return_sequences=True, dropout=0.2)(x)
        x = layers.LSTM(self.lstm_units // 2, return_sequences=True, dropout=0.2)(x)
        x = layers.LSTM(self.lstm_units, return_sequences=True, dropout=0.2)(x)
        decoded = layers.TimeDistributed(layers.Dense(input_shape[1]))(x)
        
        model = models.Model(inputs, decoded)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, data: np.ndarray) -> None:
        """Train the LSTM autoencoder on normal data."""
        if not DEPENDENCIES_AVAILABLE:
            self.logger.warning("Dependencies not available, using dummy implementation")
            self.is_trained = True
            return
        
        self.logger.info(f"Training LSTM autoencoder on {len(data)} samples")
        
        # Normalize data
        data_scaled = self.scaler.fit_transform(data.reshape(-1, data.shape[-1]))
        data_scaled = data_scaled.reshape(data.shape)
        
        # Build model
        self.model = self._build_model((data.shape[1], data.shape[2]))
        
        # Train model
        history = self.model.fit(
            data_scaled, data_scaled,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Calculate threshold based on training data reconstruction error
        train_predictions = self.model.predict(data_scaled, verbose=0)
        train_errors = np.mean(np.square(data_scaled - train_predictions), axis=(1, 2))
        self.threshold = np.percentile(train_errors, 95)
        
        self.is_trained = True
        self.logger.info(f"Training completed. Threshold: {self.threshold:.4f}")
    
    def predict(self, data: np.ndarray) -> DetectionResult:
        """Detect anomalies using reconstruction error."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before prediction")
        
        if not DEPENDENCIES_AVAILABLE:
            # Dummy implementation
            return DetectionResult(
                anomaly_scores=np.random.random(len(data)),
                anomaly_predictions=(np.random.random(len(data)) > 0.5).astype(int),
                confidence_scores=np.random.random(len(data)),
                detection_method="LSTM_Autoencoder_Dummy",
                metadata={"threshold": 0.5}
            )
        
        # Normalize data
        data_scaled = self.scaler.transform(data.reshape(-1, data.shape[-1]))
        data_scaled = data_scaled.reshape(data.shape)
        
        # Compute reconstruction error
        reconstructed = self.model.predict(data_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(data_scaled - reconstructed), axis=(1, 2))
        
        # Generate predictions
        anomaly_predictions = (reconstruction_errors > self.threshold).astype(int)
        
        # Calculate confidence scores
        confidence_scores = np.abs(reconstruction_errors - self.threshold) / self.threshold
        
        return DetectionResult(
            anomaly_scores=reconstruction_errors,
            anomaly_predictions=anomaly_predictions,
            confidence_scores=confidence_scores,
            detection_method="LSTM_Autoencoder",
            metadata={"threshold": self.threshold, "mean_error": reconstruction_errors.mean()}
        )
    
    def save(self, path: Path) -> None:
        """Save the trained LSTM detector."""
        path.mkdir(parents=True, exist_ok=True)
        
        if DEPENDENCIES_AVAILABLE and self.model:
            self.model.save(path / "lstm_model.h5")
        
        with open(path / "lstm_metadata.pkl", "wb") as f:
            pickle.dump({
                "threshold": self.threshold,
                "window_size": self.window_size,
                "latent_dim": self.latent_dim,
                "lstm_units": self.lstm_units,
                "scaler": self.scaler
            }, f)
    
    def load(self, path: Path) -> None:
        """Load a trained LSTM detector."""
        if DEPENDENCIES_AVAILABLE:
            model_path = path / "lstm_model.h5"
            if model_path.exists():
                self.model = load_model(str(model_path))
        
        with open(path / "lstm_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            self.threshold = metadata["threshold"]
            self.window_size = metadata["window_size"]
            self.latent_dim = metadata["latent_dim"]
            self.lstm_units = metadata["lstm_units"]
            self.scaler = metadata["scaler"]
        
        self.is_trained = True


class IsolationForestDetector(BaseDetector):
    """Isolation Forest-based anomaly detector."""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        super().__init__("IsolationForest")
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        ) if DEPENDENCIES_AVAILABLE else None
    
    def fit(self, data: np.ndarray) -> None:
        """Train the Isolation Forest detector."""
        if not DEPENDENCIES_AVAILABLE:
            self.logger.warning("Dependencies not available, using dummy implementation")
            self.is_trained = True
            return
        
        # Flatten sequences to features
        data_flat = data.reshape(len(data), -1)
        self.model.fit(data_flat)
        self.is_trained = True
        self.logger.info("Isolation Forest training completed")
    
    def predict(self, data: np.ndarray) -> DetectionResult:
        """Detect anomalies using Isolation Forest."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before prediction")
        
        if not DEPENDENCIES_AVAILABLE:
            # Dummy implementation
            return DetectionResult(
                anomaly_scores=np.random.random(len(data)),
                anomaly_predictions=(np.random.random(len(data)) > 0.5).astype(int),
                confidence_scores=np.random.random(len(data)),
                detection_method="IsolationForest_Dummy",
                metadata={"contamination": self.contamination}
            )
        
        # Flatten sequences to features
        data_flat = data.reshape(len(data), -1)
        
        # Get anomaly scores and predictions
        anomaly_scores = -self.model.decision_function(data_flat)  # Negative for higher is more anomalous
        anomaly_predictions = (self.model.predict(data_flat) == -1).astype(int)
        
        # Normalize confidence scores
        confidence_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        
        return DetectionResult(
            anomaly_scores=anomaly_scores,
            anomaly_predictions=anomaly_predictions,
            confidence_scores=confidence_scores,
            detection_method="IsolationForest",
            metadata={"contamination": self.contamination}
        )
    
    def save(self, path: Path) -> None:
        """Save the trained Isolation Forest detector."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "isolation_forest.pkl", "wb") as f:
            pickle.dump(self.model, f)
    
    def load(self, path: Path) -> None:
        """Load a trained Isolation Forest detector."""
        with open(path / "isolation_forest.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True


class StatisticalDetector(BaseDetector):
    """Statistical anomaly detector using z-score and IQR methods."""
    
    def __init__(self, z_threshold: float = 3.0, iqr_factor: float = 1.5):
        super().__init__("Statistical")
        self.z_threshold = z_threshold
        self.iqr_factor = iqr_factor
        self.feature_stats = {}
    
    def fit(self, data: np.ndarray) -> None:
        """Calculate statistical parameters from normal data."""
        self.logger.info("Computing statistical parameters")
        
        # Calculate statistics for each feature
        for feature_idx in range(data.shape[2]):
            feature_data = data[:, :, feature_idx].flatten()
            
            self.feature_stats[feature_idx] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'q1': np.percentile(feature_data, 25),
                'q3': np.percentile(feature_data, 75),
                'iqr': np.percentile(feature_data, 75) - np.percentile(feature_data, 25)
            }
        
        self.is_trained = True
        self.logger.info("Statistical parameter calculation completed")
    
    def predict(self, data: np.ndarray) -> DetectionResult:
        """Detect anomalies using statistical methods."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before prediction")
        
        anomaly_scores = np.zeros(len(data))
        
        for sequence_idx, sequence in enumerate(data):
            sequence_score = 0
            
            for feature_idx in range(sequence.shape[1]):
                feature_values = sequence[:, feature_idx]
                stats = self.feature_stats[feature_idx]
                
                # Z-score based detection
                z_scores = np.abs((feature_values - stats['mean']) / (stats['std'] + 1e-8))
                z_anomalies = np.sum(z_scores > self.z_threshold)
                
                # IQR based detection
                lower_bound = stats['q1'] - self.iqr_factor * stats['iqr']
                upper_bound = stats['q3'] + self.iqr_factor * stats['iqr']
                iqr_anomalies = np.sum((feature_values < lower_bound) | (feature_values > upper_bound))
                
                # Combine scores
                feature_score = (z_anomalies + iqr_anomalies) / len(feature_values)
                sequence_score += feature_score
            
            anomaly_scores[sequence_idx] = sequence_score / data.shape[2]
        
        # Generate predictions based on score threshold
        score_threshold = np.percentile(anomaly_scores, 95)
        anomaly_predictions = (anomaly_scores > score_threshold).astype(int)
        
        # Normalize confidence scores
        if anomaly_scores.max() > anomaly_scores.min():
            confidence_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        else:
            confidence_scores = np.zeros_like(anomaly_scores)
        
        return DetectionResult(
            anomaly_scores=anomaly_scores,
            anomaly_predictions=anomaly_predictions,
            confidence_scores=confidence_scores,
            detection_method="Statistical",
            metadata={
                "z_threshold": self.z_threshold,
                "iqr_factor": self.iqr_factor,
                "score_threshold": score_threshold
            }
        )
    
    def save(self, path: Path) -> None:
        """Save the statistical detector parameters."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "statistical_params.pkl", "wb") as f:
            pickle.dump({
                "feature_stats": self.feature_stats,
                "z_threshold": self.z_threshold,
                "iqr_factor": self.iqr_factor
            }, f)
    
    def load(self, path: Path) -> None:
        """Load statistical detector parameters."""
        with open(path / "statistical_params.pkl", "rb") as f:
            data = pickle.load(f)
            self.feature_stats = data["feature_stats"]
            self.z_threshold = data["z_threshold"]
            self.iqr_factor = data["iqr_factor"]
        self.is_trained = True


class AdaptiveMultiModalDetector:
    """Advanced multi-modal anomaly detector with adaptive learning."""
    
    def __init__(
        self,
        window_size: int = 30,
        ensemble_weights: Optional[EnsembleWeights] = None,
        adaptation_rate: float = 0.01
    ):
        self.window_size = window_size
        self.ensemble_weights = ensemble_weights or EnsembleWeights()
        self.adaptation_rate = adaptation_rate
        self.logger = get_logger(__name__)
        
        # Initialize detectors
        self.detectors = {
            "lstm": LSTMAutoencoderDetector(window_size=window_size),
            "isolation_forest": IsolationForestDetector(),
            "statistical": StatisticalDetector()
        }
        
        # Performance tracking
        self.performance_history = []
        self.is_trained = False
    
    def fit(self, data: np.ndarray, parallel: bool = True) -> None:
        """Train all detectors on the provided data."""
        self.logger.info(f"Training multi-modal detector on {len(data)} samples")
        
        if parallel and len(self.detectors) > 1:
            # Parallel training
            with ThreadPoolExecutor(max_workers=len(self.detectors)) as executor:
                futures = {}
                for name, detector in self.detectors.items():
                    future = executor.submit(detector.fit, data)
                    futures[future] = name
                
                for future in as_completed(futures):
                    detector_name = futures[future]
                    try:
                        future.result()
                        self.logger.info(f"Completed training for {detector_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to train {detector_name}: {e}")
        else:
            # Sequential training
            for name, detector in self.detectors.items():
                try:
                    detector.fit(data)
                    self.logger.info(f"Completed training for {name}")
                except Exception as e:
                    self.logger.error(f"Failed to train {name}: {e}")
        
        self.is_trained = True
        self.logger.info("Multi-modal detector training completed")
    
    def predict(self, data: np.ndarray, method: str = "ensemble") -> DetectionResult:
        """Detect anomalies using the specified method."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before prediction")
        
        if method == "ensemble":
            return self._ensemble_predict(data)
        elif method in self.detectors:
            return self.detectors[method].predict(data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _ensemble_predict(self, data: np.ndarray) -> DetectionResult:
        """Combine predictions from multiple detectors."""
        results = {}
        
        # Get predictions from all detectors
        for name, detector in self.detectors.items():
            if detector.is_trained:
                try:
                    results[name] = detector.predict(data)
                except Exception as e:
                    self.logger.warning(f"Failed to get prediction from {name}: {e}")
        
        if not results:
            raise ValueError("No trained detectors available for ensemble prediction")
        
        # Combine scores using weighted average
        combined_scores = np.zeros(len(data))
        combined_predictions = np.zeros(len(data))
        combined_confidence = np.zeros(len(data))
        total_weight = 0
        
        for name, result in results.items():
            weight = getattr(self.ensemble_weights, f"{name}_weight", 0.25)
            combined_scores += weight * result.anomaly_scores
            combined_predictions += weight * result.anomaly_predictions
            combined_confidence += weight * result.confidence_scores
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_scores /= total_weight
            combined_predictions /= total_weight
            combined_confidence /= total_weight
        
        # Convert to binary predictions
        threshold = np.median(combined_predictions)
        final_predictions = (combined_predictions > threshold).astype(int)
        
        return DetectionResult(
            anomaly_scores=combined_scores,
            anomaly_predictions=final_predictions,
            confidence_scores=combined_confidence,
            detection_method="Ensemble",
            metadata={
                "ensemble_weights": self.ensemble_weights.__dict__,
                "detectors_used": list(results.keys()),
                "threshold": threshold
            }
        )
    
    def adapt_weights(self, true_labels: np.ndarray, predictions: Dict[str, DetectionResult]) -> None:
        """Adapt ensemble weights based on performance feedback."""
        if len(true_labels) == 0:
            return
        
        # Calculate performance metrics for each detector
        detector_performance = {}
        
        for name, result in predictions.items():
            # Calculate F1 score as performance metric
            tp = np.sum((true_labels == 1) & (result.anomaly_predictions == 1))
            fp = np.sum((true_labels == 0) & (result.anomaly_predictions == 1))
            fn = np.sum((true_labels == 1) & (result.anomaly_predictions == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            detector_performance[name] = f1_score
        
        # Update weights based on performance
        total_performance = sum(detector_performance.values())
        if total_performance > 0:
            for name, performance in detector_performance.items():
                current_weight = getattr(self.ensemble_weights, f"{name}_weight", 0.25)
                new_weight = current_weight + self.adaptation_rate * (performance / total_performance - current_weight)
                setattr(self.ensemble_weights, f"{name}_weight", max(0.01, min(1.0, new_weight)))
        
        # Normalize weights
        total_weight = sum([
            self.ensemble_weights.lstm_weight,
            self.ensemble_weights.isolation_forest_weight,
            self.ensemble_weights.statistical_weight
        ])
        
        if total_weight > 0:
            self.ensemble_weights.lstm_weight /= total_weight
            self.ensemble_weights.isolation_forest_weight /= total_weight
            self.ensemble_weights.statistical_weight /= total_weight
        
        self.logger.info(f"Updated ensemble weights: {self.ensemble_weights.__dict__}")
    
    def save(self, path: Path) -> None:
        """Save all trained detectors."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save individual detectors
        for name, detector in self.detectors.items():
            if detector.is_trained:
                detector_path = path / name
                detector.save(detector_path)
        
        # Save ensemble configuration
        with open(path / "ensemble_config.pkl", "wb") as f:
            pickle.dump({
                "ensemble_weights": self.ensemble_weights,
                "adaptation_rate": self.adaptation_rate,
                "window_size": self.window_size,
                "performance_history": self.performance_history
            }, f)
    
    def load(self, path: Path) -> None:
        """Load all trained detectors."""
        # Load individual detectors
        for name, detector in self.detectors.items():
            detector_path = path / name
            if detector_path.exists():
                try:
                    detector.load(detector_path)
                    self.logger.info(f"Loaded {name} detector")
                except Exception as e:
                    self.logger.warning(f"Failed to load {name} detector: {e}")
        
        # Load ensemble configuration
        config_path = path / "ensemble_config.pkl"
        if config_path.exists():
            with open(config_path, "rb") as f:
                config = pickle.load(f)
                self.ensemble_weights = config["ensemble_weights"]
                self.adaptation_rate = config["adaptation_rate"]
                self.window_size = config["window_size"]
                self.performance_history = config.get("performance_history", [])
        
        self.is_trained = True
    
    def get_detector_status(self) -> Dict[str, bool]:
        """Get training status of all detectors."""
        return {name: detector.is_trained for name, detector in self.detectors.items()}
    
    def benchmark(self, test_data: np.ndarray, true_labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Benchmark all detectors against test data."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before benchmarking")
        
        results = {}
        
        for name, detector in self.detectors.items():
            if detector.is_trained:
                try:
                    prediction_result = detector.predict(test_data)
                    predictions = prediction_result.anomaly_predictions
                    
                    # Calculate metrics
                    tp = np.sum((true_labels == 1) & (predictions == 1))
                    fp = np.sum((true_labels == 0) & (predictions == 1))
                    fn = np.sum((true_labels == 1) & (predictions == 0))
                    tn = np.sum((true_labels == 0) & (predictions == 0))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0
                    
                    results[name] = {
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score,
                        "accuracy": accuracy,
                        "true_positives": int(tp),
                        "false_positives": int(fp),
                        "true_negatives": int(tn),
                        "false_negatives": int(fn)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Failed to benchmark {name}: {e}")
                    results[name] = {"error": str(e)}
        
        return results