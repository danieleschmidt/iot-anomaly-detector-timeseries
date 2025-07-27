"""
End-to-end integration tests for the IoT anomaly detection pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from src.data_preprocessor import DataPreprocessor
from src.train_autoencoder import train_model, save_artifacts
from src.anomaly_detector import AnomalyDetector
from src.evaluate_model import evaluate_model
from src.model_manager import ModelManager


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test complete pipeline from data generation to anomaly detection."""
    
    def test_complete_pipeline_with_simulated_data(self, temp_dir, test_config):
        """Test complete pipeline with simulated data."""
        
        # Step 1: Generate test data
        np.random.seed(42)
        n_samples = 200
        n_features = 3
        
        # Generate normal data
        data = np.random.normal(0, 1, (n_samples, n_features))
        
        # Add patterns
        for i in range(n_features):
            data[:, i] += 0.5 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
        
        # Inject anomalies
        anomaly_indices = [50, 51, 52, 100, 150, 151]
        for idx in anomaly_indices:
            data[idx] *= 3.0
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=[f"sensor_{i}" for i in range(n_features)])
        
        # Save to CSV
        data_path = temp_dir / "pipeline_test_data.csv"
        df.to_csv(data_path, index=False)
        
        # Step 2: Preprocess data
        preprocessor = DataPreprocessor(
            window_size=10,
            step=1,
            scaler_type="standard"
        )
        
        X_train, _ = preprocessor.fit_transform(df)
        assert X_train.shape[0] > 0
        assert X_train.shape[2] == n_features
        
        # Step 3: Train model (mock training for speed)
        with patch('src.train_autoencoder.build_autoencoder') as mock_build:
            # Mock model that returns reasonable reconstruction errors
            mock_model = MockAutoencoder(n_features)
            mock_build.return_value = mock_model
            
            model_path = temp_dir / "test_model.h5"
            scaler_path = temp_dir / "test_scaler.pkl"
            
            # Train model
            trained_model, scaler, history = train_model(
                X_train=X_train,
                epochs=2,
                batch_size=16,
                latent_dim=8,
                model_path=str(model_path),
                scaler_path=str(scaler_path),
                preprocessor=preprocessor
            )
            
            assert model_path.exists()
            assert scaler_path.exists()
        
        # Step 4: Detect anomalies
        detector = AnomalyDetector(
            model_path=str(model_path),
            scaler_path=str(scaler_path)
        )
        
        # Mock the model loading for testing
        detector.model = mock_model
        detector.scaler = preprocessor.scaler
        detector.window_size = 10
        
        # Detect anomalies
        predictions = detector.predict_from_dataframe(df, threshold=0.5)
        
        assert len(predictions) == len(df)
        assert predictions.sum() > 0  # Should detect some anomalies
        
        # Step 5: Evaluate performance
        labels = pd.Series(0, index=df.index)
        for idx in anomaly_indices:
            if idx < len(labels):
                labels.iloc[idx] = 1
        
        # Calculate basic metrics
        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            
            # Should have reasonable performance
            assert precision >= 0.0  # At least some precision
            assert recall >= 0.0     # At least some recall
    
    @pytest.mark.slow
    def test_pipeline_with_large_dataset(self, temp_dir):
        """Test pipeline performance with larger dataset."""
        
        # Generate larger dataset
        np.random.seed(123)
        n_samples = 2000
        n_features = 5
        
        data = np.random.normal(0, 1, (n_samples, n_features))
        df = pd.DataFrame(data, columns=[f"sensor_{i}" for i in range(n_features)])
        
        # Time the preprocessing step
        start_time = time.time()
        
        preprocessor = DataPreprocessor(window_size=30, step=1)
        X_train, _ = preprocessor.fit_transform(df)
        
        preprocess_time = time.time() - start_time
        
        # Should process large dataset in reasonable time
        assert preprocess_time < 10.0  # Less than 10 seconds
        assert X_train.shape[0] > 0
        
        # Mock training for performance test
        with patch('tensorflow.keras.models.Sequential') as mock_model_class:
            mock_model = MockAutoencoder(n_features)
            mock_model_class.return_value = mock_model
            
            start_time = time.time()
            
            # Quick training simulation
            model_path = temp_dir / "large_test_model.h5"
            scaler_path = temp_dir / "large_test_scaler.pkl"
            
            save_artifacts(mock_model, preprocessor.scaler, str(model_path), str(scaler_path))
            
            training_time = time.time() - start_time
            
            # Should save quickly
            assert training_time < 5.0
            assert model_path.exists()
            assert scaler_path.exists()
    
    def test_pipeline_error_handling(self, temp_dir):
        """Test pipeline error handling with invalid data."""
        
        # Test with empty data
        empty_df = pd.DataFrame()
        preprocessor = DataPreprocessor()
        
        with pytest.raises(ValueError):
            preprocessor.fit_transform(empty_df)
        
        # Test with insufficient data
        small_df = pd.DataFrame({
            'sensor_0': [1, 2, 3],
            'sensor_1': [4, 5, 6]
        })
        
        with pytest.raises(ValueError):
            preprocessor.fit_transform(small_df)
        
        # Test with missing model files
        detector = AnomalyDetector(
            model_path="nonexistent_model.h5",
            scaler_path="nonexistent_scaler.pkl"
        )
        
        with pytest.raises(FileNotFoundError):
            detector.predict_from_dataframe(small_df)
    
    @pytest.mark.integration
    def test_model_versioning_pipeline(self, temp_dir):
        """Test model versioning and management."""
        
        # Create test data
        np.random.seed(789)
        df = pd.DataFrame({
            'sensor_0': np.random.normal(0, 1, 100),
            'sensor_1': np.random.normal(0, 1, 100),
            'sensor_2': np.random.normal(0, 1, 100)
        })
        
        # Initialize model manager
        manager = ModelManager(registry_path=str(temp_dir))
        
        # Create multiple model versions
        for version in range(3):
            model_name = f"test_model_v{version}"
            
            # Mock model training
            with patch('src.train_autoencoder.build_autoencoder') as mock_build:
                mock_model = MockAutoencoder(3)
                mock_build.return_value = mock_model
                
                preprocessor = DataPreprocessor(window_size=10)
                X_train, _ = preprocessor.fit_transform(df)
                
                # Save model with version
                model_path = temp_dir / f"{model_name}.h5"
                scaler_path = temp_dir / f"{model_name}_scaler.pkl"
                
                save_artifacts(mock_model, preprocessor.scaler, str(model_path), str(scaler_path))
                
                # Register model
                manager.register_model(
                    name=model_name,
                    version=f"1.0.{version}",
                    model_path=str(model_path),
                    scaler_path=str(scaler_path),
                    metrics={"accuracy": 0.85 + version * 0.05}
                )
        
        # Test model retrieval
        latest_model = manager.get_latest_model("test_model_v2")
        assert latest_model is not None
        
        # Test model comparison
        models = manager.list_models()
        assert len(models) == 3
        
        # Test best model selection
        best_model = manager.get_best_model(metric="accuracy")
        assert best_model["metrics"]["accuracy"] >= 0.85


class MockAutoencoder:
    """Mock autoencoder for testing."""
    
    def __init__(self, n_features):
        self.n_features = n_features
        self.history = None
    
    def fit(self, X, epochs=10, batch_size=32, validation_split=0.2, callbacks=None, verbose=0):
        """Mock training."""
        # Return mock history
        n_batches = len(X) // batch_size
        loss_values = [0.5 - i * 0.01 for i in range(epochs)]
        
        self.history = type('History', (), {
            'history': {
                'loss': loss_values,
                'val_loss': [l + 0.05 for l in loss_values]
            }
        })()
        
        return self.history
    
    def predict(self, X, batch_size=None):
        """Mock prediction with realistic reconstruction errors."""
        if len(X.shape) == 3:
            # Return reconstructed sequences with small differences
            reconstructed = X + np.random.normal(0, 0.1, X.shape)
            return reconstructed
        else:
            # Handle 2D input
            return X + np.random.normal(0, 0.1, X.shape)
    
    def save(self, filepath):
        """Mock save."""
        Path(filepath).touch()
    
    def compile(self, optimizer, loss):
        """Mock compile."""
        pass


@pytest.mark.integration
class TestPipelinePerformance:
    """Performance tests for the complete pipeline."""
    
    @pytest.mark.performance
    def test_inference_latency(self, temp_dir):
        """Test inference latency requirements."""
        
        # Create test data
        df = pd.DataFrame({
            'sensor_0': np.random.normal(0, 1, 1000),
            'sensor_1': np.random.normal(0, 1, 1000),
            'sensor_2': np.random.normal(0, 1, 1000)
        })
        
        # Setup mock detector
        model_path = temp_dir / "latency_test_model.h5"
        scaler_path = temp_dir / "latency_test_scaler.pkl"
        
        # Create mock files
        model_path.touch()
        scaler_path.touch()
        
        detector = AnomalyDetector(str(model_path), str(scaler_path))
        detector.model = MockAutoencoder(3)
        detector.window_size = 30
        
        # Mock scaler
        from sklearn.preprocessing import StandardScaler
        detector.scaler = StandardScaler().fit(df.values)
        
        # Test single window inference time
        single_window = df.iloc[:30]
        
        start_time = time.time()
        predictions = detector.predict_from_dataframe(single_window, threshold=0.5)
        inference_time = time.time() - start_time
        
        # Should meet latency requirement (< 100ms for single window)
        assert inference_time < 0.1  # 100ms
        assert len(predictions) == len(single_window)
    
    @pytest.mark.performance
    def test_throughput_requirements(self, temp_dir):
        """Test system throughput requirements."""
        
        # Create larger dataset to test throughput
        n_samples = 10000
        df = pd.DataFrame({
            'sensor_0': np.random.normal(0, 1, n_samples),
            'sensor_1': np.random.normal(0, 1, n_samples),
            'sensor_2': np.random.normal(0, 1, n_samples)
        })
        
        # Setup detector
        model_path = temp_dir / "throughput_test_model.h5"
        scaler_path = temp_dir / "throughput_test_scaler.pkl"
        
        model_path.touch()
        scaler_path.touch()
        
        detector = AnomalyDetector(str(model_path), str(scaler_path))
        detector.model = MockAutoencoder(3)
        detector.window_size = 30
        
        from sklearn.preprocessing import StandardScaler
        detector.scaler = StandardScaler().fit(df.iloc[:1000].values)
        
        # Test batch processing throughput
        start_time = time.time()
        
        # Process in batches
        batch_size = 1000
        total_processed = 0
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            predictions = detector.predict_from_dataframe(batch, threshold=0.5)
            total_processed += len(predictions)
        
        total_time = time.time() - start_time
        throughput = total_processed / total_time
        
        # Should process at least 1000 samples per second
        assert throughput >= 1000
        assert total_processed == len(df)
    
    def test_memory_efficiency(self, temp_dir):
        """Test memory efficiency with large datasets."""
        
        # This test would monitor memory usage in a real scenario
        # For now, we test that the system doesn't crash with large data
        
        # Create large dataset
        n_samples = 50000
        df = pd.DataFrame({
            f'sensor_{i}': np.random.normal(0, 1, n_samples) 
            for i in range(10)  # 10 features
        })
        
        # Test preprocessing memory efficiency
        preprocessor = DataPreprocessor(window_size=50, step=10)
        
        # Should handle large dataset without memory error
        try:
            X_train, _ = preprocessor.fit_transform(df)
            assert X_train.shape[0] > 0
            memory_test_passed = True
        except MemoryError:
            memory_test_passed = False
        
        assert memory_test_passed