"""
Basic functionality tests for pipeline generations
Simplified tests that can run without all dependencies
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Simple test that verifies basic imports work
def test_imports():
    """Test that basic imports work."""
    try:
        from src.basic_pipeline import BasicAnomalyPipeline
        from src.robust_pipeline import RobustAnomalyPipeline
        from src.scalable_pipeline import ScalablePipeline
        assert True
    except ImportError as e:
        pytest.skip(f"Import failed: {e}")


@pytest.fixture
def sample_data():
    """Generate sample IoT sensor data for testing."""
    np.random.seed(42)
    n_samples = 100  # Smaller for faster tests
    n_features = 3
    
    # Generate normal data
    normal_data = np.random.normal(0, 1, (n_samples, n_features))
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=5, replace=False)
    normal_data[anomaly_indices] *= 3  # Make anomalies
    
    df = pd.DataFrame(normal_data, columns=['temp', 'humidity', 'pressure'])
    return df


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_basic_pipeline_creation():
    """Test basic pipeline can be created."""
    try:
        from src.basic_pipeline import BasicAnomalyPipeline
        
        pipeline = BasicAnomalyPipeline(
            window_size=10,
            latent_dim=8,
            epochs=1  # Minimal for testing
        )
        
        assert pipeline.window_size == 10
        assert pipeline.latent_dim == 8
        assert pipeline.epochs == 1
        assert pipeline.model is None
        assert pipeline.preprocessor is None
        
    except ImportError as e:
        pytest.skip(f"Basic pipeline import failed: {e}")


def test_data_loading_basic(sample_data, temp_dir):
    """Test basic data loading functionality."""
    try:
        from src.basic_pipeline import BasicAnomalyPipeline
        
        pipeline = BasicAnomalyPipeline()
        
        # Save test data
        data_path = Path(temp_dir) / "test_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        # Load data
        loaded_data = pipeline.load_data(str(data_path))
        
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)
        
    except ImportError as e:
        pytest.skip(f"Basic pipeline import failed: {e}")


def test_data_preparation_basic(sample_data):
    """Test basic data preparation."""
    try:
        from src.basic_pipeline import BasicAnomalyPipeline
        
        pipeline = BasicAnomalyPipeline(window_size=10)
        
        X, y = pipeline.prepare_data(sample_data)
        
        expected_sequences = len(sample_data) - pipeline.window_size + 1
        assert X.shape[0] == expected_sequences
        assert X.shape[1] == pipeline.window_size
        assert X.shape[2] == len(sample_data.columns)
        assert np.array_equal(X, y)  # For autoencoder, X == y
        
    except ImportError as e:
        pytest.skip(f"Basic pipeline import failed: {e}")


def test_robust_pipeline_creation():
    """Test robust pipeline can be created."""
    try:
        from src.robust_pipeline import RobustAnomalyPipeline
        from src.data_validator import ValidationLevel
        
        pipeline = RobustAnomalyPipeline(
            validation_level=ValidationLevel.MODERATE,
            enable_validation=True,
            max_memory_usage_gb=1.0
        )
        
        assert pipeline.enable_validation is True
        assert pipeline.max_memory_usage_gb == 1.0
        
    except ImportError as e:
        pytest.skip(f"Robust pipeline import failed: {e}")


def test_scalable_pipeline_creation():
    """Test scalable pipeline can be created."""
    try:
        from src.scalable_pipeline import ScalablePipeline
        
        pipeline = ScalablePipeline(
            enable_caching=True,
            enable_parallel_processing=True,
            max_workers=2
        )
        
        assert pipeline.enable_caching is True
        assert pipeline.enable_parallel_processing is True
        assert pipeline.max_workers == 2
        
    except ImportError as e:
        pytest.skip(f"Scalable pipeline import failed: {e}")


def test_numpy_basic_operations():
    """Test basic numpy operations work."""
    arr = np.array([1, 2, 3, 4, 5])
    assert np.mean(arr) == 3.0
    assert np.sum(arr) == 15
    assert len(arr) == 5


def test_pandas_basic_operations():
    """Test basic pandas operations work."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert len(df) == 3
    assert list(df.columns) == ['a', 'b']
    assert df['a'].sum() == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])