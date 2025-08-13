"""Smart data preprocessing with automatic feature engineering for Generation 1."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Tuple, Optional, List
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartPreprocessor:
    """Enhanced preprocessing with automatic feature engineering and adaptive scaling."""
    
    def __init__(self, scaler_type: str = 'auto', window_size: int = 30, step_size: int = 1):
        self.scaler_type = scaler_type
        self.window_size = window_size
        self.step_size = step_size
        self.scaler = None
        self.imputer = None
        self.feature_stats = {}
        self.is_fitted = False
        
    def fit(self, data: np.ndarray) -> 'SmartPreprocessor':
        """Fit preprocessor to data with automatic parameter selection."""
        logger.info("Fitting smart preprocessor...")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        # Analyze data characteristics
        self._analyze_data_characteristics(data)
        
        # Select optimal scaler
        if self.scaler_type == 'auto':
            self.scaler_type = self._select_optimal_scaler(data)
        
        # Initialize scaler
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        self.scaler = scalers[self.scaler_type]
        
        # Initialize imputer
        self.imputer = SimpleImputer(strategy='median')
        
        # Fit components
        reshaped_data = data.reshape(-1, data.shape[-1])
        clean_data = self.imputer.fit_transform(reshaped_data)
        self.scaler.fit(clean_data)
        
        self.is_fitted = True
        logger.info(f"Preprocessor fitted with {self.scaler_type} scaler")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data with fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        # Handle missing values
        original_shape = data.shape
        reshaped_data = data.reshape(-1, data.shape[-1])
        clean_data = self.imputer.transform(reshaped_data)
        
        # Scale data
        scaled_data = self.scaler.transform(clean_data)
        scaled_data = scaled_data.reshape(original_shape)
        
        return scaled_data
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data in one step."""
        return self.fit(data).transform(data)
    
    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sliding window sequences with enhanced features."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        sequences = []
        for i in range(0, len(data) - self.window_size + 1, self.step_size):
            sequence = data[i:i + self.window_size]
            
            # Add engineered features
            enhanced_sequence = self._add_features(sequence)
            sequences.append(enhanced_sequence)
        
        return np.array(sequences)
    
    def _analyze_data_characteristics(self, data: np.ndarray):
        """Analyze data to determine optimal preprocessing strategy."""
        reshaped_data = data.reshape(-1, data.shape[-1])
        
        self.feature_stats = {
            'mean': np.mean(reshaped_data, axis=0),
            'std': np.std(reshaped_data, axis=0),
            'median': np.median(reshaped_data, axis=0),
            'q25': np.percentile(reshaped_data, 25, axis=0),
            'q75': np.percentile(reshaped_data, 75, axis=0),
            'skewness': self._calculate_skewness(reshaped_data),
            'outlier_ratio': self._calculate_outlier_ratio(reshaped_data)
        }
        
        logger.info(f"Data analysis complete: {reshaped_data.shape[1]} features, {reshaped_data.shape[0]} samples")
    
    def _select_optimal_scaler(self, data: np.ndarray) -> str:
        """Select optimal scaler based on data characteristics."""
        reshaped_data = data.reshape(-1, data.shape[-1])
        
        # Calculate metrics for scaler selection
        avg_skewness = np.mean(np.abs(self.feature_stats['skewness']))
        avg_outlier_ratio = np.mean(self.feature_stats['outlier_ratio'])
        
        if avg_outlier_ratio > 0.1:  # High outlier ratio
            scaler_type = 'robust'
        elif avg_skewness > 1.0:  # High skewness
            scaler_type = 'robust'
        else:
            scaler_type = 'standard'
        
        logger.info(f"Selected {scaler_type} scaler (skewness: {avg_skewness:.2f}, outliers: {avg_outlier_ratio:.2f})")
        return scaler_type
    
    def _add_features(self, sequence: np.ndarray) -> np.ndarray:
        """Add engineered features to sequence."""
        original_features = sequence.shape[1]
        enhanced_features = []
        
        for i in range(original_features):
            feature_data = sequence[:, i]
            
            # Original feature
            enhanced_features.append(feature_data)
            
            # Moving statistics (every 5 features to avoid explosion)
            if i % 5 == 0:
                # Moving average
                ma_5 = self._moving_average(feature_data, 5)
                enhanced_features.append(ma_5)
                
                # Moving standard deviation
                std_5 = self._moving_std(feature_data, 5)
                enhanced_features.append(std_5)
        
        return np.column_stack(enhanced_features)
    
    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average."""
        if len(data) < window:
            return np.full_like(data, np.mean(data))
        
        ma = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            ma[i] = np.mean(data[start_idx:i+1])
        
        return ma
    
    def _moving_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving standard deviation."""
        if len(data) < window:
            return np.full_like(data, np.std(data))
        
        std = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            std[i] = np.std(data[start_idx:i+1])
        
        return std
    
    def _calculate_skewness(self, data: np.ndarray) -> np.ndarray:
        """Calculate skewness for each feature."""
        skewness = np.zeros(data.shape[1])
        for i in range(data.shape[1]):
            mean_val = np.mean(data[:, i])
            std_val = np.std(data[:, i])
            if std_val > 0:
                skewness[i] = np.mean(((data[:, i] - mean_val) / std_val) ** 3)
        
        return skewness
    
    def _calculate_outlier_ratio(self, data: np.ndarray) -> np.ndarray:
        """Calculate outlier ratio using IQR method."""
        outlier_ratios = np.zeros(data.shape[1])
        
        for i in range(data.shape[1]):
            q25, q75 = np.percentile(data[:, i], [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            outliers = (data[:, i] < lower_bound) | (data[:, i] > upper_bound)
            outlier_ratios[i] = np.sum(outliers) / len(data[:, i])
        
        return outlier_ratios
    
    def save(self, path: str):
        """Save preprocessor to file."""
        save_data = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'scaler_type': self.scaler_type,
            'window_size': self.window_size,
            'step_size': self.step_size,
            'feature_stats': self.feature_stats,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SmartPreprocessor':
        """Load preprocessor from file."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        preprocessor = cls()
        preprocessor.scaler = save_data['scaler']
        preprocessor.imputer = save_data['imputer']
        preprocessor.scaler_type = save_data['scaler_type']
        preprocessor.window_size = save_data['window_size']
        preprocessor.step_size = save_data['step_size']
        preprocessor.feature_stats = save_data['feature_stats']
        preprocessor.is_fitted = save_data['is_fitted']
        
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor


class DataQualityChecker:
    """Real-time data quality monitoring."""
    
    def __init__(self, reference_stats: Dict[str, Any]):
        self.reference_stats = reference_stats
        self.alerts = []
        
    def check_data_quality(self, data: np.ndarray) -> Dict[str, Any]:
        """Check data quality against reference statistics."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        current_stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'missing_ratio': np.sum(np.isnan(data)) / data.size
        }
        
        quality_report = {
            'timestamp': pd.Timestamp.now(),
            'sample_count': len(data),
            'missing_data': current_stats['missing_ratio'],
            'drift_detected': False,
            'alerts': []
        }
        
        # Check for data drift
        if 'mean' in self.reference_stats:
            mean_drift = np.mean(np.abs(current_stats['mean'] - self.reference_stats['mean']))
            if mean_drift > 2 * np.mean(self.reference_stats['std']):
                quality_report['drift_detected'] = True
                quality_report['alerts'].append(f"Mean drift detected: {mean_drift:.4f}")
        
        # Check missing data
        if current_stats['missing_ratio'] > 0.05:  # 5% threshold
            quality_report['alerts'].append(f"High missing data: {current_stats['missing_ratio']:.2%}")
        
        return quality_report


def create_enhanced_training_pipeline(
    data_path: str,
    window_size: int = 30,
    validation_split: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, SmartPreprocessor]:
    """Create complete preprocessing pipeline for training."""
    logger.info("=== ENHANCED PREPROCESSING PIPELINE ===")
    
    # Load data
    if Path(data_path).suffix == '.csv':
        data = pd.read_csv(data_path).values
    else:
        data = np.load(data_path)
    
    logger.info(f"Loaded data with shape: {data.shape}")
    
    # Initialize preprocessor
    preprocessor = SmartPreprocessor(window_size=window_size)
    
    # Fit and transform data
    scaled_data = preprocessor.fit_transform(data)
    
    # Create sequences
    sequences = preprocessor.create_sequences(scaled_data)
    logger.info(f"Created {len(sequences)} sequences with shape: {sequences[0].shape}")
    
    # Split data
    split_idx = int(len(sequences) * (1 - validation_split))
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    logger.info(f"Training sequences: {len(train_sequences)}, Validation: {len(val_sequences)}")
    
    return train_sequences, val_sequences, preprocessor


if __name__ == "__main__":
    # Generation 1 preprocessing demonstration
    logger.info("=== GENERATION 1: SMART PREPROCESSING DEMO ===")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Generate time series with different characteristics
    data = np.zeros((n_samples, n_features))
    
    for i in range(n_features):
        # Different patterns per feature
        t = np.linspace(0, 10, n_samples)
        if i == 0:  # Sinusoidal
            data[:, i] = np.sin(t) + np.random.normal(0, 0.1, n_samples)
        elif i == 1:  # Exponential decay
            data[:, i] = np.exp(-t/5) + np.random.normal(0, 0.05, n_samples)
        elif i == 2:  # Linear trend with outliers
            data[:, i] = 0.1 * t + np.random.normal(0, 0.2, n_samples)
            # Add outliers
            outlier_indices = np.random.choice(n_samples, 50, replace=False)
            data[outlier_indices, i] += np.random.normal(0, 2, 50)
        elif i == 3:  # High frequency noise
            data[:, i] = np.sin(5*t) + np.random.normal(0, 0.5, n_samples)
        else:  # Missing values pattern
            data[:, i] = np.cos(t) + np.random.normal(0, 0.1, n_samples)
            # Add missing values
            missing_indices = np.random.choice(n_samples, 100, replace=False)
            data[missing_indices, i] = np.nan
    
    # Test smart preprocessor
    preprocessor = SmartPreprocessor(window_size=20)
    
    logger.info("Analyzing and preprocessing data...")
    scaled_data = preprocessor.fit_transform(data)
    
    logger.info("Creating sequences with enhanced features...")
    sequences = preprocessor.create_sequences(scaled_data)
    
    logger.info(f"Original data shape: {data.shape}")
    logger.info(f"Sequence shape: {sequences.shape}")
    logger.info(f"Enhanced features per sequence: {sequences.shape[2]}")
    
    # Test data quality checker
    quality_checker = DataQualityChecker(preprocessor.feature_stats)
    quality_report = quality_checker.check_data_quality(data[500:600])
    
    logger.info("Data quality report:")
    for key, value in quality_report.items():
        if key != 'timestamp':
            logger.info(f"  {key}: {value}")
    
    # Save preprocessor
    preprocessor.save('smart_preprocessor.pkl')
    
    logger.info("=== GENERATION 1 PREPROCESSING COMPLETE ===")