"""
Pytest configuration and shared fixtures for the IoT Anomaly Detection system.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "data"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
FIXTURES_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration fixture."""
    return {
        "window_size": 10,
        "n_features": 3,
        "n_samples": 100,
        "latent_dim": 8,
        "batch_size": 16,
        "epochs": 2,
        "test_threshold": 0.5,
    }


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_sensor_data(test_config: Dict[str, Any]) -> pd.DataFrame:
    """Generate sample sensor data for testing."""
    np.random.seed(42)
    n_samples = test_config["n_samples"]
    n_features = test_config["n_features"]
    
    # Generate normal data with some patterns
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="1T")
    data = np.random.normal(0, 1, (n_samples, n_features))
    
    # Add some temporal patterns
    for i in range(n_features):
        data[:, i] += 0.5 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
    
    df = pd.DataFrame(data, columns=[f"sensor_{i}" for i in range(n_features)])
    df["timestamp"] = timestamps
    
    return df


@pytest.fixture
def sample_anomaly_data(sample_sensor_data: pd.DataFrame) -> pd.DataFrame:
    """Generate sample data with injected anomalies."""
    data = sample_sensor_data.copy()
    
    # Inject anomalies at specific indices
    anomaly_indices = [20, 21, 22, 50, 75, 76]
    for idx in anomaly_indices:
        if idx < len(data):
            # Make anomalies by multiplying by large factor
            data.iloc[idx, :-1] *= 3.0
    
    return data


@pytest.fixture
def sample_labels(sample_sensor_data: pd.DataFrame) -> pd.Series:
    """Generate sample anomaly labels."""
    labels = pd.Series(0, index=sample_sensor_data.index, name="anomaly")
    # Mark specific indices as anomalies
    anomaly_indices = [20, 21, 22, 50, 75, 76]
    for idx in anomaly_indices:
        if idx < len(labels):
            labels.iloc[idx] = 1
    return labels


@pytest.fixture
def sample_csv_file(sample_sensor_data: pd.DataFrame, temp_dir: Path) -> Path:
    """Create a temporary CSV file with sample data."""
    csv_path = temp_dir / "test_sensor_data.csv"
    sample_sensor_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_labels_file(sample_labels: pd.Series, temp_dir: Path) -> Path:
    """Create a temporary CSV file with sample labels."""
    labels_path = temp_dir / "test_labels.csv"
    sample_labels.to_csv(labels_path, index=False)
    return labels_path


@pytest.fixture
def mock_model():
    """Mock TensorFlow model for testing."""
    mock = Mock()
    mock.predict.return_value = np.random.random((10, 10, 3))
    mock.fit.return_value = Mock(history={"loss": [0.5, 0.3, 0.2]})
    mock.save.return_value = None
    return mock


@pytest.fixture
def mock_scaler():
    """Mock sklearn scaler for testing."""
    mock = Mock()
    mock.fit_transform.return_value = np.random.random((100, 3))
    mock.transform.return_value = np.random.random((100, 3))
    mock.inverse_transform.return_value = np.random.random((100, 3))
    return mock


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TENSORFLOW_LOG_LEVEL", "3")  # Suppress TF logs


@pytest.fixture
def mock_tensorflow():
    """Mock TensorFlow imports for unit tests."""
    with patch.dict("sys.modules", {
        "tensorflow": Mock(),
        "tensorflow.keras": Mock(),
        "tensorflow.keras.models": Mock(),
        "tensorflow.keras.layers": Mock(),
        "tensorflow.keras.optimizers": Mock(),
        "tensorflow.keras.callbacks": Mock(),
    }):
        yield


@pytest.fixture(scope="session")
def integration_test_data():
    """Larger dataset for integration tests."""
    np.random.seed(123)
    n_samples = 1000
    n_features = 5
    
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="1min")
    data = np.random.normal(0, 1, (n_samples, n_features))
    
    # Add realistic patterns
    for i in range(n_features):
        # Daily pattern
        daily_pattern = np.sin(2 * np.pi * np.arange(n_samples) / (24 * 60))
        # Weekly pattern  
        weekly_pattern = 0.3 * np.sin(2 * np.pi * np.arange(n_samples) / (7 * 24 * 60))
        # Trend
        trend = 0.001 * np.arange(n_samples)
        
        data[:, i] += daily_pattern + weekly_pattern + trend
    
    df = pd.DataFrame(data, columns=[f"sensor_{i}" for i in range(n_features)])
    df["timestamp"] = timestamps
    
    # Inject anomalies
    anomaly_ranges = [(100, 110), (300, 305), (500, 520), (800, 810)]
    labels = np.zeros(n_samples)
    
    for start, end in anomaly_ranges:
        # Spike anomalies
        df.iloc[start:end, :-1] *= np.random.uniform(2, 4, (end-start, n_features))
        labels[start:end] = 1
    
    return df, pd.Series(labels, name="anomaly")


@pytest.fixture
def performance_test_data():
    """Large dataset for performance testing."""
    np.random.seed(456)
    n_samples = 10000
    n_features = 10
    
    data = np.random.normal(0, 1, (n_samples, n_features))
    df = pd.DataFrame(data, columns=[f"sensor_{i}" for i in range(n_features)])
    
    return df


class TestMetrics:
    """Helper class for collecting test metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def record_time(self, operation: str, duration: float):
        """Record operation duration."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
    
    def record_accuracy(self, accuracy: float):
        """Record model accuracy."""
        self.metrics["accuracy"] = accuracy
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {}
        for operation, times in self.metrics.items():
            if isinstance(times, list):
                summary[f"{operation}_avg"] = np.mean(times)
                summary[f"{operation}_max"] = np.max(times)
                summary[f"{operation}_min"] = np.min(times)
            else:
                summary[operation] = times
        return summary


@pytest.fixture
def test_metrics():
    """Test metrics collection fixture."""
    return TestMetrics()


# Pytest markers
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"  
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark performance tests  
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Mark security tests
        if "security" in item.nodeid:
            item.add_marker(pytest.mark.security)
        
        # Mark slow tests
        if any(keyword in item.nodeid for keyword in ["slow", "large", "stress"]):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True, scope="session")
def cleanup_test_artifacts():
    """Clean up test artifacts after test session."""
    yield
    
    # Clean up any test files that might have been created
    test_patterns = [
        "test_*.h5",
        "test_*.pkl", 
        "test_*.csv",
        "test_*.json",
        "*.tmp",
    ]
    
    for pattern in test_patterns:
        for file_path in Path.cwd().glob(pattern):
            try:
                file_path.unlink()
            except OSError:
                pass  # File might be in use or already deleted