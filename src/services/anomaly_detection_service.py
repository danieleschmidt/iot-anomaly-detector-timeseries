"""
Anomaly Detection Service

Core business logic for detecting anomalies in IoT sensor data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from ..anomaly_detector import AnomalyDetector
from ..data_preprocessor import DataPreprocessor
from ..model_manager import ModelManager
from ..caching_strategy import CachingStrategy

logger = logging.getLogger(__name__)


class AnomalyDetectionService:
    """
    Service for orchestrating anomaly detection operations.
    
    This service coordinates between various components to provide
    high-level anomaly detection functionality.
    """
    
    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        cache_strategy: Optional[CachingStrategy] = None,
        enable_caching: bool = True
    ):
        """
        Initialize the anomaly detection service.
        
        Args:
            model_manager: Model management instance
            cache_strategy: Caching strategy for performance
            enable_caching: Whether to enable result caching
        """
        self.model_manager = model_manager or ModelManager()
        self.cache_strategy = cache_strategy or CachingStrategy()
        self.enable_caching = enable_caching
        self.preprocessor = DataPreprocessor()
        self._detectors: Dict[str, AnomalyDetector] = {}
        
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        model_version: Optional[str] = None,
        threshold: Optional[float] = None,
        quantile: Optional[float] = None,
        window_size: int = 30,
        step_size: int = 1
    ) -> Dict[str, Any]:
        """
        Detect anomalies in the provided sensor data.
        
        Args:
            data: Sensor data DataFrame
            model_version: Specific model version to use
            threshold: Manual threshold for anomaly detection
            quantile: Quantile-based threshold (0-1)
            window_size: Size of sliding window
            step_size: Step size for sliding window
            
        Returns:
            Dictionary containing detection results and metadata
        """
        start_time = datetime.now()
        
        # Get model and detector
        detector = self._get_or_create_detector(model_version)
        
        # Check cache if enabled
        cache_key = self._generate_cache_key(data, model_version, threshold, quantile)
        if self.enable_caching:
            cached_result = self.cache_strategy.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for key: {cache_key}")
                return cached_result
        
        # Preprocess data
        logger.info("Preprocessing sensor data")
        processed_data = self.preprocessor.fit_transform(data)
        
        # Create windows
        windows = self._create_windows(processed_data, window_size, step_size)
        
        # Detect anomalies
        logger.info(f"Running anomaly detection on {len(windows)} windows")
        if quantile is not None:
            detector.set_threshold_from_quantile(windows, quantile)
            threshold = detector.threshold
        elif threshold is not None:
            detector.threshold = threshold
        
        predictions = detector.predict_batch(windows)
        reconstruction_errors = detector.get_reconstruction_errors(windows)
        
        # Analyze results
        anomaly_indices = np.where(predictions == 1)[0]
        anomaly_timestamps = self._get_anomaly_timestamps(
            data, anomaly_indices, window_size, step_size
        )
        
        # Calculate statistics
        stats = self._calculate_statistics(
            predictions, reconstruction_errors, threshold
        )
        
        # Prepare results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version or 'latest',
            'threshold': threshold,
            'quantile': quantile,
            'window_size': window_size,
            'total_windows': len(windows),
            'anomalies_detected': len(anomaly_indices),
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_timestamps': anomaly_timestamps,
            'reconstruction_errors': reconstruction_errors.tolist(),
            'statistics': stats,
            'processing_time': (datetime.now() - start_time).total_seconds()
        }
        
        # Cache results if enabled
        if self.enable_caching:
            self.cache_strategy.set(cache_key, results, ttl=3600)
        
        # Log summary
        logger.info(
            f"Detection complete: {len(anomaly_indices)} anomalies found "
            f"in {len(windows)} windows (threshold: {threshold:.4f})"
        )
        
        return results
    
    def detect_streaming(
        self,
        data_point: Dict[str, float],
        model_version: Optional[str] = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect anomalies in streaming data.
        
        Args:
            data_point: Single data point from sensor stream
            model_version: Model version to use
            threshold: Anomaly threshold
            
        Returns:
            Detection result for the data point
        """
        # Convert to DataFrame for consistency
        df = pd.DataFrame([data_point])
        
        # Use regular detection with window size 1
        result = self.detect_anomalies(
            df, model_version, threshold, window_size=1, step_size=1
        )
        
        # Simplify result for streaming
        is_anomaly = result['anomalies_detected'] > 0
        return {
            'timestamp': datetime.now().isoformat(),
            'is_anomaly': is_anomaly,
            'reconstruction_error': result['reconstruction_errors'][0] if result['reconstruction_errors'] else 0,
            'threshold': threshold,
            'data_point': data_point
        }
    
    def analyze_pattern(
        self,
        data: pd.DataFrame,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Analyze patterns in historical data.
        
        Args:
            data: Historical sensor data
            start_time: Start of analysis period
            end_time: End of analysis period
            
        Returns:
            Pattern analysis results
        """
        # Filter by time range if provided
        if start_time or end_time:
            data = self._filter_by_time(data, start_time, end_time)
        
        # Detect anomalies
        detection_results = self.detect_anomalies(data)
        
        # Analyze patterns
        patterns = {
            'total_data_points': len(data),
            'anomaly_rate': detection_results['anomalies_detected'] / detection_results['total_windows'],
            'peak_anomaly_periods': self._identify_peak_periods(
                detection_results['anomaly_timestamps']
            ),
            'anomaly_clusters': self._cluster_anomalies(
                detection_results['anomaly_indices']
            ),
            'severity_distribution': self._calculate_severity_distribution(
                detection_results['reconstruction_errors'],
                detection_results['threshold']
            )
        }
        
        return patterns
    
    def evaluate_model_performance(
        self,
        test_data: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        model_version: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test sensor data
            labels: Ground truth labels (optional)
            model_version: Model version to evaluate
            
        Returns:
            Performance metrics
        """
        # Run detection
        results = self.detect_anomalies(test_data, model_version)
        
        metrics = {
            'mean_reconstruction_error': np.mean(results['reconstruction_errors']),
            'std_reconstruction_error': np.std(results['reconstruction_errors']),
            'anomaly_percentage': results['anomalies_detected'] / results['total_windows'] * 100
        }
        
        # Calculate classification metrics if labels provided
        if labels is not None:
            predictions = np.zeros(results['total_windows'])
            predictions[results['anomaly_indices']] = 1
            
            # Ensure labels match prediction length
            if len(labels) >= len(predictions):
                labels = labels[:len(predictions)]
                
                tp = np.sum((predictions == 1) & (labels == 1))
                fp = np.sum((predictions == 1) & (labels == 0))
                fn = np.sum((predictions == 0) & (labels == 1))
                tn = np.sum((predictions == 0) & (labels == 0))
                
                metrics.update({
                    'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                    'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
                })
        
        return metrics
    
    def get_anomaly_explanation(
        self,
        data_point: pd.DataFrame,
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get explanation for why a data point is anomalous.
        
        Args:
            data_point: Single data point to explain
            model_version: Model version to use
            
        Returns:
            Explanation of anomaly detection
        """
        detector = self._get_or_create_detector(model_version)
        
        # Get reconstruction error
        processed = self.preprocessor.transform(data_point)
        reconstruction_error = detector.get_reconstruction_errors(
            processed.values.reshape(1, -1)
        )[0]
        
        # Calculate feature contributions
        original = processed.values[0]
        reconstructed = detector.model.predict(original.reshape(1, 1, -1))[0, 0]
        feature_errors = np.abs(original - reconstructed)
        
        # Identify top contributing features
        feature_names = data_point.columns.tolist()
        contributions = sorted(
            zip(feature_names, feature_errors),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'reconstruction_error': float(reconstruction_error),
            'threshold': detector.threshold,
            'is_anomaly': reconstruction_error > detector.threshold,
            'top_contributing_features': contributions[:5],
            'feature_contributions': dict(contributions)
        }
    
    def _get_or_create_detector(self, model_version: Optional[str] = None) -> AnomalyDetector:
        """Get or create an anomaly detector for the specified model version."""
        version = model_version or 'latest'
        
        if version not in self._detectors:
            model_path, scaler_path = self.model_manager.get_model_paths(version)
            self._detectors[version] = AnomalyDetector(
                str(model_path),
                str(scaler_path) if scaler_path else None
            )
        
        return self._detectors[version]
    
    def _create_windows(
        self,
        data: np.ndarray,
        window_size: int,
        step_size: int
    ) -> np.ndarray:
        """Create sliding windows from data."""
        windows = []
        for i in range(0, len(data) - window_size + 1, step_size):
            windows.append(data[i:i + window_size])
        return np.array(windows)
    
    def _generate_cache_key(
        self,
        data: pd.DataFrame,
        model_version: Optional[str],
        threshold: Optional[float],
        quantile: Optional[float]
    ) -> str:
        """Generate cache key for detection results."""
        data_hash = hash(pd.util.hash_pandas_object(data).sum())
        return f"anomaly_{data_hash}_{model_version}_{threshold}_{quantile}"
    
    def _get_anomaly_timestamps(
        self,
        data: pd.DataFrame,
        anomaly_indices: np.ndarray,
        window_size: int,
        step_size: int
    ) -> List[str]:
        """Get timestamps for detected anomalies."""
        timestamps = []
        
        # Check if data has a timestamp column
        if 'timestamp' in data.columns:
            for idx in anomaly_indices:
                window_start = idx * step_size
                window_end = window_start + window_size
                if window_start < len(data):
                    timestamps.append(str(data.iloc[window_start]['timestamp']))
        else:
            # Use indices as timestamps
            timestamps = [f"window_{idx}" for idx in anomaly_indices]
        
        return timestamps
    
    def _calculate_statistics(
        self,
        predictions: np.ndarray,
        reconstruction_errors: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:
        """Calculate detection statistics."""
        return {
            'mean_error': float(np.mean(reconstruction_errors)),
            'std_error': float(np.std(reconstruction_errors)),
            'min_error': float(np.min(reconstruction_errors)),
            'max_error': float(np.max(reconstruction_errors)),
            'median_error': float(np.median(reconstruction_errors)),
            'anomaly_rate': float(np.mean(predictions)),
            'threshold': float(threshold),
            'percentile_95': float(np.percentile(reconstruction_errors, 95)),
            'percentile_99': float(np.percentile(reconstruction_errors, 99))
        }
    
    def _filter_by_time(
        self,
        data: pd.DataFrame,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> pd.DataFrame:
        """Filter data by time range."""
        if 'timestamp' not in data.columns:
            return data
        
        filtered = data.copy()
        if start_time:
            filtered = filtered[filtered['timestamp'] >= start_time]
        if end_time:
            filtered = filtered[filtered['timestamp'] <= end_time]
        
        return filtered
    
    def _identify_peak_periods(
        self,
        anomaly_timestamps: List[str],
        window_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Identify periods with high anomaly concentration."""
        if not anomaly_timestamps:
            return []
        
        # Group anomalies by time windows
        # This is a simplified implementation
        peak_periods = []
        if len(anomaly_timestamps) > 10:
            peak_periods.append({
                'period': f"Peak detected",
                'anomaly_count': len(anomaly_timestamps),
                'start': anomaly_timestamps[0],
                'end': anomaly_timestamps[-1]
            })
        
        return peak_periods
    
    def _cluster_anomalies(
        self,
        anomaly_indices: np.ndarray,
        min_cluster_size: int = 3
    ) -> List[Dict[str, Any]]:
        """Identify clusters of consecutive anomalies."""
        if len(anomaly_indices) == 0:
            return []
        
        clusters = []
        current_cluster = [anomaly_indices[0]]
        
        for i in range(1, len(anomaly_indices)):
            if anomaly_indices[i] - anomaly_indices[i-1] <= 2:
                current_cluster.append(anomaly_indices[i])
            else:
                if len(current_cluster) >= min_cluster_size:
                    clusters.append({
                        'start_index': int(current_cluster[0]),
                        'end_index': int(current_cluster[-1]),
                        'size': len(current_cluster)
                    })
                current_cluster = [anomaly_indices[i]]
        
        # Check last cluster
        if len(current_cluster) >= min_cluster_size:
            clusters.append({
                'start_index': int(current_cluster[0]),
                'end_index': int(current_cluster[-1]),
                'size': len(current_cluster)
            })
        
        return clusters
    
    def _calculate_severity_distribution(
        self,
        reconstruction_errors: List[float],
        threshold: float
    ) -> Dict[str, int]:
        """Calculate distribution of anomaly severities."""
        severities = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for error in reconstruction_errors:
            if error <= threshold:
                continue  # Not an anomaly
            elif error <= threshold * 1.5:
                severities['low'] += 1
            elif error <= threshold * 2:
                severities['medium'] += 1
            elif error <= threshold * 3:
                severities['high'] += 1
            else:
                severities['critical'] += 1
        
        return severities