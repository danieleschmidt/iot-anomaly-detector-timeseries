"""
Data Drift Detection System for IoT Anomaly Detection

This module provides comprehensive data drift detection capabilities using
statistical tests and monitoring to maintain model accuracy over time.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
from scipy import stats
from scipy.stats import wasserstein_distance

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DriftDetectionConfig:
    """Configuration for data drift detection parameters."""
    
    # Statistical test thresholds
    ks_threshold: float = 0.05  # Kolmogorov-Smirnov p-value threshold
    psi_threshold: float = 0.25  # Population Stability Index threshold
    wasserstein_threshold: float = 0.3  # Wasserstein distance threshold
    
    # Data requirements
    min_samples: int = 100  # Minimum samples required for detection
    detection_window_size: int = 1000  # Size of detection window
    
    # Alert configuration
    enable_alerts: bool = True
    alert_cooldown_hours: int = 24  # Minimum hours between alerts for same feature
    
    # Feature binning for PSI
    n_bins: int = 10
    bin_method: str = 'quantile'  # 'quantile' or 'equal_width'
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.ks_threshold < 1:
            raise ValueError("ks_threshold must be between 0 and 1")
        if self.psi_threshold < 0:
            raise ValueError("psi_threshold must be non-negative")
        if self.wasserstein_threshold < 0:
            raise ValueError("wasserstein_threshold must be non-negative")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if self.detection_window_size < self.min_samples:
            raise ValueError("detection_window_size must be >= min_samples")
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2")


@dataclass
class DriftResult:
    """Results from drift detection analysis."""
    
    timestamp: datetime
    ks_statistic: float
    ks_p_value: float
    psi_score: float
    wasserstein_distance: float
    drift_detected: bool
    feature_drifts: Dict[str, bool] = field(default_factory=dict)
    drift_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'ks_statistic': float(self.ks_statistic),
            'ks_p_value': float(self.ks_p_value),
            'psi_score': float(self.psi_score),
            'wasserstein_distance': float(self.wasserstein_distance),
            'drift_detected': bool(self.drift_detected),
            'feature_drifts': {k: bool(v) for k, v in self.feature_drifts.items()},
            'drift_scores': {
                feature: {metric: float(score) for metric, score in scores.items()}
                for feature, scores in self.drift_scores.items()
            },
            'summary': self.summary
        }


@dataclass
class DriftAlert:
    """Alert generated when drift is detected."""
    
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    feature_alerts: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)
    
    @classmethod
    def from_drift_result(cls, result: DriftResult, config: DriftDetectionConfig) -> 'DriftAlert':
        """Generate alert from drift detection result."""
        # Determine severity based on drift metrics
        severity = 'LOW'
        if result.psi_score > config.psi_threshold * 2:
            severity = 'CRITICAL'
        elif result.psi_score > config.psi_threshold * 1.5:
            severity = 'HIGH'
        elif result.drift_detected:
            severity = 'MEDIUM'
        
        # Generate message
        drifted_features = [f for f, drifted in result.feature_drifts.items() if drifted]
        message = f"Data drift detected in {len(drifted_features)} features: {', '.join(drifted_features[:3])}"
        if len(drifted_features) > 3:
            message += f" and {len(drifted_features) - 3} more"
        
        # Feature-specific alerts
        feature_alerts = {}
        for feature, scores in result.drift_scores.items():
            alerts = []
            if scores.get('psi', 0) > config.psi_threshold:
                alerts.append(f"High PSI score: {scores['psi']:.3f}")
            if scores.get('ks_pvalue', 1) < config.ks_threshold:
                alerts.append(f"Significant KS test: p={scores['ks_pvalue']:.4f}")
            if scores.get('wasserstein', 0) > config.wasserstein_threshold:
                alerts.append(f"High Wasserstein distance: {scores['wasserstein']:.3f}")
            
            if alerts:
                feature_alerts[feature] = "; ".join(alerts)
        
        # Generate recommendations
        recommendations = []
        if severity in ['HIGH', 'CRITICAL']:
            recommendations.extend([
                "Consider retraining the model with recent data",
                "Investigate potential changes in data collection process",
                "Review feature engineering pipeline for issues"
            ])
        elif severity == 'MEDIUM':
            recommendations.extend([
                "Monitor drift trends closely",
                "Consider updating model validation thresholds",
                "Investigate specific drifted features"
            ])
        else:
            recommendations.append("Continue monitoring data quality")
        
        return cls(
            severity=severity,
            message=message,
            feature_alerts=feature_alerts,
            timestamp=result.timestamp,
            recommendations=recommendations
        )


class DataDriftDetector:
    """
    Comprehensive data drift detection system using multiple statistical methods.
    
    Supports Kolmogorov-Smirnov test, Population Stability Index (PSI),
    and Wasserstein distance for detecting distribution changes.
    """
    
    def __init__(self, config: Optional[DriftDetectionConfig] = None):
        """
        Initialize drift detector.
        
        Parameters
        ----------
        config : DriftDetectionConfig, optional
            Configuration for drift detection. Uses defaults if None.
        """
        self.config = config or DriftDetectionConfig()
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_stats: Dict[str, Any] = {}
        self.drift_history: List[DriftResult] = []
        self.alert_history: List[DriftAlert] = []
        
        logger.info(f"DataDriftDetector initialized with config: "
                   f"ks_threshold={self.config.ks_threshold}, "
                   f"psi_threshold={self.config.psi_threshold}")
    
    def set_reference_data(self, data: pd.DataFrame) -> None:
        """
        Set reference/baseline data for drift detection.
        
        Parameters
        ----------
        data : pd.DataFrame
            Reference dataset to compare against
        """
        if data.empty:
            raise ValueError("Reference data cannot be empty")
        
        if len(data) < self.config.min_samples:
            raise ValueError(f"Reference data must have at least {self.config.min_samples} samples")
        
        # Keep only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("Reference data must contain at least one numeric column")
        
        if len(numeric_data.columns) < len(data.columns):
            dropped_cols = set(data.columns) - set(numeric_data.columns)
            logger.warning(f"Dropped non-numeric columns: {dropped_cols}")
        
        self.reference_data = numeric_data.copy()
        self._compute_reference_stats()
        
        logger.info(f"Reference data set: {self.reference_data.shape[0]} samples, "
                   f"{self.reference_data.shape[1]} features")
    
    def _compute_reference_stats(self) -> None:
        """Compute reference statistics for efficient drift detection."""
        if self.reference_data is None:
            return
        
        self.reference_stats = {}
        
        for column in self.reference_data.columns:
            col_data = self.reference_data[column].dropna()
            
            # Compute bins for PSI calculation
            if self.config.bin_method == 'quantile':
                _, bin_edges = pd.qcut(col_data, q=self.config.n_bins, 
                                     retbins=True, duplicates='drop')
            else:  # equal_width
                _, bin_edges = pd.cut(col_data, bins=self.config.n_bins, 
                                    retbins=True, duplicates='drop')
            
            # Compute reference bin frequencies
            ref_counts, _ = np.histogram(col_data, bins=bin_edges)
            ref_freqs = ref_counts / len(col_data)
            
            self.reference_stats[column] = {
                'bin_edges': bin_edges,
                'ref_frequencies': ref_freqs,
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max()
            }
    
    def _kolmogorov_smirnov_test(self, new_data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        Perform Kolmogorov-Smirnov test for each feature.
        
        Parameters
        ----------
        new_data : pd.DataFrame
            New data to test for drift
            
        Returns
        -------
        Tuple[List[float], List[float]]
            KS statistics and p-values for each feature
        """
        ks_statistics = []
        p_values = []
        
        for column in self.reference_data.columns:
            if column not in new_data.columns:
                logger.warning(f"Column {column} not found in new data, skipping")
                ks_statistics.append(0.0)
                p_values.append(1.0)
                continue
            
            ref_col = self.reference_data[column].dropna()
            new_col = new_data[column].dropna()
            
            if len(new_col) == 0:
                logger.warning(f"No valid data for column {column}")
                ks_statistics.append(0.0)
                p_values.append(1.0)
                continue
            
            try:
                ks_stat, p_val = stats.ks_2samp(ref_col, new_col)
                ks_statistics.append(ks_stat)
                p_values.append(p_val)
            except Exception as e:
                logger.error(f"KS test failed for column {column}: {e}")
                ks_statistics.append(0.0)
                p_values.append(1.0)
        
        return ks_statistics, p_values
    
    def _population_stability_index(self, new_data: pd.DataFrame) -> List[float]:
        """
        Calculate Population Stability Index for each feature.
        
        Parameters
        ----------
        new_data : pd.DataFrame
            New data to calculate PSI against reference
            
        Returns
        -------
        List[float]
            PSI scores for each feature
        """
        psi_scores = []
        
        for column in self.reference_data.columns:
            if column not in new_data.columns:
                logger.warning(f"Column {column} not found in new data for PSI")
                psi_scores.append(0.0)
                continue
            
            new_col = new_data[column].dropna()
            if len(new_col) == 0:
                psi_scores.append(0.0)
                continue
            
            try:
                # Get reference statistics
                ref_stats = self.reference_stats[column]
                bin_edges = ref_stats['bin_edges']
                ref_freqs = ref_stats['ref_frequencies']
                
                # Calculate new data frequencies using same bins
                new_counts, _ = np.histogram(new_col, bins=bin_edges)
                new_freqs = new_counts / len(new_col)
                
                # Calculate PSI
                psi = 0.0
                for i in range(len(ref_freqs)):
                    ref_freq = max(ref_freqs[i], 1e-10)  # Avoid log(0)
                    new_freq = max(new_freqs[i], 1e-10)
                    
                    psi += (new_freq - ref_freq) * np.log(new_freq / ref_freq)
                
                psi_scores.append(psi)
                
            except Exception as e:
                logger.error(f"PSI calculation failed for column {column}: {e}")
                psi_scores.append(0.0)
        
        return psi_scores
    
    def _wasserstein_distance(self, new_data: pd.DataFrame) -> List[float]:
        """
        Calculate Wasserstein distance for each feature.
        
        Parameters
        ----------
        new_data : pd.DataFrame
            New data to calculate distance against reference
            
        Returns
        -------
        List[float]
            Wasserstein distances for each feature
        """
        distances = []
        
        for column in self.reference_data.columns:
            if column not in new_data.columns:
                distances.append(0.0)
                continue
            
            ref_col = self.reference_data[column].dropna()
            new_col = new_data[column].dropna()
            
            if len(new_col) == 0:
                distances.append(0.0)
                continue
            
            try:
                # Normalize by reference standard deviation for interpretability
                ref_std = self.reference_stats[column]['std']
                if ref_std > 0:
                    distance = wasserstein_distance(ref_col, new_col) / ref_std
                else:
                    distance = wasserstein_distance(ref_col, new_col)
                
                distances.append(distance)
                
            except Exception as e:
                logger.error(f"Wasserstein distance calculation failed for column {column}: {e}")
                distances.append(0.0)
        
        return distances
    
    def detect_drift(self, new_data: pd.DataFrame) -> DriftResult:
        """
        Detect drift in new data compared to reference data.
        
        Parameters
        ----------
        new_data : pd.DataFrame
            New data to test for drift
            
        Returns
        -------
        DriftResult
            Comprehensive drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data must be set before drift detection")
        
        if len(new_data) < self.config.min_samples:
            raise ValueError(f"New data must have at least {self.config.min_samples} samples")
        
        # Keep only numeric columns that exist in reference data
        numeric_new_data = new_data.select_dtypes(include=[np.number])
        common_columns = list(set(self.reference_data.columns) & set(numeric_new_data.columns))
        
        if not common_columns:
            raise ValueError("No common numeric columns between reference and new data")
        
        new_data_filtered = numeric_new_data[common_columns]
        
        logger.info(f"Detecting drift in {len(new_data_filtered)} samples across {len(common_columns)} features")
        
        # Perform drift detection tests
        ks_statistics, ks_p_values = self._kolmogorov_smirnov_test(new_data_filtered)
        psi_scores = self._population_stability_index(new_data_filtered)
        wasserstein_distances = self._wasserstein_distance(new_data_filtered)
        
        # Determine drift for each feature
        feature_drifts = {}
        drift_scores = {}
        
        for i, column in enumerate(common_columns):
            # Drift detected if any test indicates drift
            ks_drift = ks_p_values[i] < self.config.ks_threshold
            psi_drift = psi_scores[i] > self.config.psi_threshold
            wasserstein_drift = wasserstein_distances[i] > self.config.wasserstein_threshold
            
            feature_drifts[column] = ks_drift or psi_drift or wasserstein_drift
            
            drift_scores[column] = {
                'ks_statistic': ks_statistics[i],
                'ks_pvalue': ks_p_values[i],
                'psi': psi_scores[i],
                'wasserstein': wasserstein_distances[i]
            }
        
        # Overall drift detection
        overall_drift = any(feature_drifts.values())
        
        # Aggregate scores
        avg_ks_stat = np.mean(ks_statistics)
        min_ks_pvalue = np.min(ks_p_values)
        max_psi = np.max(psi_scores)
        max_wasserstein = np.max(wasserstein_distances)
        
        # Create result
        result = DriftResult(
            timestamp=datetime.now(),
            ks_statistic=avg_ks_stat,
            ks_p_value=min_ks_pvalue,
            psi_score=max_psi,
            wasserstein_distance=max_wasserstein,
            drift_detected=overall_drift,
            feature_drifts=feature_drifts,
            drift_scores=drift_scores,
            summary={
                'total_features': len(common_columns),
                'drifted_features': sum(feature_drifts.values()),
                'drift_rate': sum(feature_drifts.values()) / len(common_columns),
                'max_psi_feature': common_columns[np.argmax(psi_scores)] if psi_scores else None,
                'min_ks_pvalue_feature': common_columns[np.argmin(ks_p_values)] if ks_p_values else None
            }
        )
        
        # Store in history
        self.drift_history.append(result)
        
        # Generate alert if needed
        if overall_drift and self.config.enable_alerts:
            alert = DriftAlert.from_drift_result(result, self.config)
            self.alert_history.append(alert)
            logger.warning(f"Drift alert generated: {alert.message}")
        
        logger.info(f"Drift detection completed: {sum(feature_drifts.values())}/{len(common_columns)} features drifted")
        
        return result
    
    def get_drift_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get summary of drift detection over specified period.
        
        Parameters
        ----------
        days : int, default 30
            Number of days to include in summary
            
        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        if not self.drift_history:
            return {
                'total_detections': 0,
                'drift_rate': 0.0,
                'feature_drift_rates': {},
                'recent_drift_trend': 'stable'
            }
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_results = [r for r in self.drift_history if r.timestamp >= cutoff_date]
        
        if not recent_results:
            recent_results = self.drift_history[-10:]  # Last 10 if no recent ones
        
        # Calculate summary statistics
        total_detections = len(recent_results)
        drift_detections = sum(1 for r in recent_results if r.drift_detected)
        drift_rate = drift_detections / total_detections if total_detections > 0 else 0.0
        
        # Feature-level drift rates
        feature_drift_rates = {}
        if recent_results:
            all_features = set()
            for result in recent_results:
                all_features.update(result.feature_drifts.keys())
            
            for feature in all_features:
                feature_drifts = [r.feature_drifts.get(feature, False) for r in recent_results]
                feature_drift_rates[feature] = sum(feature_drifts) / len(feature_drifts)
        
        # Trend analysis
        if len(recent_results) >= 3:
            recent_drift_rates = [r.summary.get('drift_rate', 0) for r in recent_results[-3:]]
            if recent_drift_rates[-1] > recent_drift_rates[0]:
                trend = 'increasing'
            elif recent_drift_rates[-1] < recent_drift_rates[0] * 0.8:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'total_detections': total_detections,
            'drift_rate': drift_rate,
            'feature_drift_rates': feature_drift_rates,
            'recent_drift_trend': trend,
            'period_days': days,
            'most_drifted_feature': max(feature_drift_rates.items(), 
                                      key=lambda x: x[1], default=(None, 0))[0],
            'avg_psi_score': np.mean([r.psi_score for r in recent_results]),
            'avg_wasserstein_distance': np.mean([r.wasserstein_distance for r in recent_results])
        }
    
    def export_drift_history(self, filepath: str) -> None:
        """
        Export drift detection history to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to export file
        """
        export_data = {
            'config': {
                'ks_threshold': self.config.ks_threshold,
                'psi_threshold': self.config.psi_threshold,
                'wasserstein_threshold': self.config.wasserstein_threshold,
                'min_samples': self.config.min_samples
            },
            'reference_data_info': {
                'shape': list(self.reference_data.shape) if self.reference_data is not None else None,
                'columns': list(self.reference_data.columns) if self.reference_data is not None else []
            },
            'drift_history': [result.to_dict() for result in self.drift_history],
            'summary': self.get_drift_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Drift history exported to {filepath}")
    
    def reset(self) -> None:
        """Reset detector state, clearing all history and reference data."""
        self.reference_data = None
        self.reference_stats = {}
        self.drift_history = []
        self.alert_history = []
        logger.info("Drift detector reset")


def create_drift_detector_from_training_data(
    training_data_path: str,
    config: Optional[DriftDetectionConfig] = None
) -> DataDriftDetector:
    """
    Create and configure drift detector from training data file.
    
    Parameters
    ----------
    training_data_path : str
        Path to training data CSV file
    config : DriftDetectionConfig, optional
        Custom configuration
        
    Returns
    -------
    DataDriftDetector
        Configured drift detector
    """
    if not Path(training_data_path).exists():
        raise FileNotFoundError(f"Training data file not found: {training_data_path}")
    
    try:
        training_data = pd.read_csv(training_data_path)
        logger.info(f"Loaded training data: {training_data.shape}")
        
        detector = DataDriftDetector(config=config)
        detector.set_reference_data(training_data)
        
        return detector
        
    except Exception as e:
        logger.error(f"Failed to create drift detector from {training_data_path}: {e}")
        raise