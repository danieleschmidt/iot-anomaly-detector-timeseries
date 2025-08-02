"""Test fixtures for sample data generation."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any


@pytest.fixture
def sample_sensor_data() -> pd.DataFrame:
    """Generate sample multivariate sensor data for testing."""
    np.random.seed(42)
    
    # Generate timestamps
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=i) for i in range(1000)]
    
    # Generate normal sensor readings
    temperature = 20 + 5 * np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.5, 1000)
    pressure = 1013 + 10 * np.cos(np.linspace(0, 8*np.pi, 1000)) + np.random.normal(0, 1, 1000)
    humidity = 50 + 20 * np.sin(np.linspace(0, 6*np.pi, 1000)) + np.random.normal(0, 2, 1000)
    vibration = 0.1 + 0.05 * np.random.normal(0, 1, 1000)
    
    # Add some anomalies
    anomaly_indices = [200, 201, 202, 500, 501, 800, 801, 802, 803]
    temperature[anomaly_indices] = temperature[anomaly_indices] + 15  # Temperature spikes
    pressure[anomaly_indices] = pressure[anomaly_indices] - 50  # Pressure drops
    vibration[anomaly_indices] = vibration[anomaly_indices] * 10  # Vibration spikes
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'pressure': pressure,
        'humidity': humidity,
        'vibration': vibration
    })
    
    return df


@pytest.fixture
def sample_anomaly_labels() -> pd.DataFrame:
    """Generate corresponding anomaly labels for sample data."""
    labels = np.zeros(1000)
    anomaly_indices = [200, 201, 202, 500, 501, 800, 801, 802, 803]
    labels[anomaly_indices] = 1
    
    return pd.DataFrame({
        'timestamp': [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(1000)],
        'is_anomaly': labels
    })


@pytest.fixture
def model_config() -> Dict[str, Any]:
    """Sample model configuration for testing."""
    return {
        'window_size': 50,
        'overlap': 0.5,
        'encoding_dim': 32,
        'epochs': 5,
        'batch_size': 32,
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'early_stopping_patience': 3,
        'feature_columns': ['temperature', 'pressure', 'humidity', 'vibration'],
        'anomaly_threshold': 0.95
    }


@pytest.fixture
def training_data_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate train/validation/test data splits."""
    np.random.seed(42)
    
    # Generate larger dataset for proper splitting
    n_samples = 2000
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n_samples)]
    
    # Normal patterns with seasonal variations
    t = np.linspace(0, 20*np.pi, n_samples)
    temperature = 20 + 5 * np.sin(t) + 2 * np.cos(2*t) + np.random.normal(0, 0.5, n_samples)
    pressure = 1013 + 10 * np.cos(t) + 3 * np.sin(3*t) + np.random.normal(0, 1, n_samples)
    humidity = 50 + 20 * np.sin(0.5*t) + 5 * np.cos(4*t) + np.random.normal(0, 2, n_samples)
    vibration = 0.1 + 0.05 * np.random.normal(0, 1, n_samples)
    
    # Add realistic anomalies
    anomaly_prob = 0.05  # 5% anomaly rate
    anomaly_mask = np.random.random(n_samples) < anomaly_prob
    
    temperature[anomaly_mask] += np.random.normal(10, 3, np.sum(anomaly_mask))
    pressure[anomaly_mask] += np.random.normal(-30, 10, np.sum(anomaly_mask))
    vibration[anomaly_mask] *= np.random.uniform(3, 8, np.sum(anomaly_mask))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'pressure': pressure,
        'humidity': humidity,
        'vibration': vibration,
        'is_anomaly': anomaly_mask.astype(int)
    })
    
    # Split data (70% train, 15% val, 15% test)
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df[:train_size].copy()
    val_df = df[train_size:train_size + val_size].copy()
    test_df = df[train_size + val_size:].copy()
    
    return train_df, val_df, test_df


@pytest.fixture
def streaming_data_generator():
    """Generator for streaming data simulation."""
    def generate_stream(n_batches: int = 10, batch_size: int = 100):
        np.random.seed(42)
        
        for batch_idx in range(n_batches):
            # Generate batch timestamps
            start_time = datetime.now() + timedelta(seconds=batch_idx * batch_size)
            timestamps = [start_time + timedelta(seconds=i) for i in range(batch_size)]
            
            # Generate sensor readings with some drift over time
            drift_factor = 1 + 0.1 * batch_idx  # Gradual drift
            
            temperature = 20 + 5 * np.sin(np.linspace(0, 2*np.pi, batch_size)) * drift_factor
            temperature += np.random.normal(0, 0.5, batch_size)
            
            pressure = 1013 + 10 * np.cos(np.linspace(0, 2*np.pi, batch_size))
            pressure += np.random.normal(0, 1, batch_size)
            
            humidity = 50 + 15 * np.sin(np.linspace(0, np.pi, batch_size))
            humidity += np.random.normal(0, 2, batch_size)
            
            vibration = 0.1 + 0.05 * np.random.normal(0, 1, batch_size)
            
            # Occasionally inject anomalies
            if batch_idx % 3 == 0:  # Every 3rd batch has anomalies
                anomaly_indices = np.random.choice(batch_size, size=2, replace=False)
                temperature[anomaly_indices] += 15
                vibration[anomaly_indices] *= 5
            
            batch_df = pd.DataFrame({
                'timestamp': timestamps,
                'temperature': temperature,
                'pressure': pressure,
                'humidity': humidity,
                'vibration': vibration
            })
            
            yield batch_df
    
    return generate_stream


@pytest.fixture
def model_artifacts_paths(tmp_path):
    """Temporary paths for model artifacts during testing."""
    artifacts = {
        'model_path': tmp_path / 'test_model.h5',
        'scaler_path': tmp_path / 'test_scaler.pkl',
        'metadata_path': tmp_path / 'test_metadata.json',
        'config_path': tmp_path / 'test_config.yaml',
        'logs_path': tmp_path / 'test_logs',
        'checkpoints_path': tmp_path / 'test_checkpoints'
    }
    
    # Create directories
    artifacts['logs_path'].mkdir(exist_ok=True)
    artifacts['checkpoints_path'].mkdir(exist_ok=True)
    
    return artifacts


@pytest.fixture
def performance_metrics():
    """Expected performance metrics for model validation."""
    return {
        'precision_threshold': 0.8,
        'recall_threshold': 0.7,
        'f1_threshold': 0.75,
        'auc_threshold': 0.85,
        'max_inference_time_ms': 100,
        'max_memory_mb': 500,
        'max_training_time_minutes': 10
    }


@pytest.fixture
def api_test_client():
    """Test client for API testing."""
    try:
        from fastapi.testclient import TestClient
        from src.model_serving_api import app
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI not available for API testing")


@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing."""
    class MockDB:
        def __init__(self):
            self.data = {}
            self.connected = True
        
        def insert(self, table: str, data: dict):
            if table not in self.data:
                self.data[table] = []
            self.data[table].append(data)
        
        def query(self, table: str, conditions: dict = None):
            if table not in self.data:
                return []
            
            results = self.data[table]
            if conditions:
                # Simple filtering
                for key, value in conditions.items():
                    results = [r for r in results if r.get(key) == value]
            
            return results
        
        def close(self):
            self.connected = False
    
    return MockDB()