"""Tests for Generation 1: MAKE IT WORK - Simple functionality."""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.smart_autoencoder_lite import SmartAutoencoderLite
from src.generation_1_pipeline import Generation1Pipeline


class TestSmartAutoencoderLite:
    """Test the lightweight autoencoder."""
    
    def test_autoencoder_creation(self):
        """Test basic autoencoder creation."""
        autoencoder = SmartAutoencoderLite(window_size=10, encoding_dim=8)
        
        assert autoencoder.window_size == 10
        assert autoencoder.encoding_dim == 8
        assert not autoencoder.is_trained
        assert autoencoder.model is None
        assert autoencoder.scaler is not None
    
    def test_sequence_preparation(self):
        """Test sequence preparation from time series."""
        autoencoder = SmartAutoencoderLite(window_size=5)
        
        # Create test data
        data = np.random.randn(20, 3)  # 20 timesteps, 3 features
        sequences = autoencoder._prepare_sequences(data)
        
        assert sequences.shape == (16, 5, 3)  # 20-5+1=16 sequences
    
    @pytest.mark.skipif(
        not hasattr(SmartAutoencoderLite, '_create_model') or 
        not SmartAutoencoderLite().__class__.__module__.endswith('smart_autoencoder_lite'),
        reason="TensorFlow not available"
    )
    def test_model_training_basic(self):
        """Test basic model training with synthetic data."""
        # Create synthetic normal data
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        
        autoencoder = SmartAutoencoderLite(window_size=10, encoding_dim=8)
        
        try:
            # Train with minimal epochs for testing
            history = autoencoder.fit(data, epochs=2, verbose=0)
            
            assert autoencoder.is_trained
            assert autoencoder.model is not None
            assert 'loss' in history
            assert len(history['loss']) <= 2  # Should have early stopped or completed
            
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_anomaly_detection_logic(self):
        """Test anomaly detection logic without training."""
        # Create mock reconstruction errors
        reconstruction_errors = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.12])
        
        # Test threshold calculation
        contamination_rate = 0.2  # Expect 20% anomalies
        threshold = np.percentile(reconstruction_errors, (1 - contamination_rate) * 100)
        anomalies = reconstruction_errors > threshold
        
        # Should detect roughly 20% as anomalies
        anomaly_rate = np.sum(anomalies) / len(anomalies)
        assert 0.1 <= anomaly_rate <= 0.4  # Allow some flexibility


class TestGeneration1Pipeline:
    """Test the Generation 1 pipeline."""
    
    def test_pipeline_creation(self):
        """Test basic pipeline creation."""
        pipeline = Generation1Pipeline(
            window_size=20,
            encoding_dim=12,
            contamination_rate=0.05,
            enable_logging=False
        )
        
        assert pipeline.window_size == 20
        assert pipeline.encoding_dim == 12
        assert pipeline.contamination_rate == 0.05
        assert not pipeline.is_trained
        assert pipeline.autoencoder is None
    
    def test_data_preparation(self):
        """Test data preparation functionality."""
        pipeline = Generation1Pipeline(window_size=5, enable_logging=False)  # Use smaller window
        
        # Create temporary CSV file with enough data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data = pd.DataFrame({
                'temp': [20.1, 21.2, 20.8, 19.9, 21.1, 20.5, 21.0, 20.2, 19.8, 21.3],
                'humidity': [65.2, 66.1, 64.8, 67.2, 65.9, 66.0, 65.5, 64.9, 67.0, 65.8],
                'pressure': [1013.2, 1012.8, 1013.5, 1013.1, 1012.9, 1013.0, 1012.7, 1013.3, 1013.4, 1012.6],
                'target': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]  # Target column to be removed
            })
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Test data loading and preparation
            prepared_data = pipeline.prepare_data(temp_path, target_column='target')
            
            assert len(prepared_data) == 10
            assert len(prepared_data.columns) == 3  # Target column removed
            assert 'target' not in prepared_data.columns
            assert all(col in prepared_data.columns for col in ['temp', 'humidity', 'pressure'])
            
        finally:
            Path(temp_path).unlink()  # Clean up
    
    def test_data_validation(self):
        """Test data validation logic."""
        pipeline = Generation1Pipeline(window_size=10, enable_logging=False)
        
        # Test empty data validation
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError, match="Data is empty"):
            pipeline._validate_data(empty_data)
        
        # Test insufficient data validation
        small_data = pd.DataFrame({'col1': [1, 2, 3]})  # Only 3 samples, need 10
        with pytest.raises(ValueError, match="Need at least 10 samples"):
            pipeline._validate_data(small_data)
    
    def test_pipeline_status(self):
        """Test pipeline status reporting."""
        pipeline = Generation1Pipeline(enable_logging=False)
        
        status = pipeline.get_status()
        
        assert isinstance(status, dict)
        assert 'is_trained' in status
        assert 'window_size' in status
        assert 'encoding_dim' in status
        assert 'contamination_rate' in status
        assert 'model_available' in status
        assert 'model_trained' in status
        
        assert not status['is_trained']
        assert not status['model_available']
        assert not status['model_trained']
    
    def test_missing_values_handling(self):
        """Test missing values handling."""
        pipeline = Generation1Pipeline(window_size=5, enable_logging=False)
        
        # Create data with missing values - enough samples
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data = pd.DataFrame({
                'temp': [20.1, np.nan, 20.8, 19.9, 21.1, 20.3, 21.2, np.nan, 20.0, 21.0],
                'humidity': [65.2, 66.1, np.nan, 67.2, 65.9, 66.5, 65.0, 66.8, 64.5, 67.1]
            })
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            prepared_data = pipeline.prepare_data(temp_path)
            
            # Should have no missing values after preparation
            assert not prepared_data.isnull().any().any()
            assert len(prepared_data) == 10
            
        finally:
            Path(temp_path).unlink()


class TestGeneration1Integration:
    """Integration tests for Generation 1 components."""
    
    def test_end_to_end_synthetic_data(self):
        """Test end-to-end pipeline with synthetic data."""
        # Create synthetic training data
        np.random.seed(42)
        train_data = pd.DataFrame({
            'sensor1': np.random.normal(20, 2, 200),  # Temperature-like
            'sensor2': np.random.normal(65, 5, 200),  # Humidity-like
            'sensor3': np.random.normal(1013, 10, 200)  # Pressure-like
        })
        
        # Create synthetic test data with some anomalies
        test_normal = pd.DataFrame({
            'sensor1': np.random.normal(20, 2, 80),
            'sensor2': np.random.normal(65, 5, 80),
            'sensor3': np.random.normal(1013, 10, 80)
        })
        
        test_anomalies = pd.DataFrame({
            'sensor1': np.random.normal(35, 5, 20),  # Temperature spikes
            'sensor2': np.random.normal(90, 10, 20),  # Humidity spikes
            'sensor3': np.random.normal(950, 20, 20)  # Pressure drops
        })
        
        test_data = pd.concat([test_normal, test_anomalies], ignore_index=True)
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = Path(temp_dir) / "train.csv"
            test_path = Path(temp_dir) / "test.csv"
            model_dir = Path(temp_dir) / "models"
            
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)
            
            # Initialize pipeline
            pipeline = Generation1Pipeline(
                window_size=10,
                encoding_dim=8,
                contamination_rate=0.2,
                enable_logging=False
            )
            
            try:
                # Train the pipeline
                train_results = pipeline.train(
                    str(train_path),
                    epochs=3,  # Minimal for testing
                    batch_size=16,
                    save_model=True,
                    model_dir=str(model_dir)
                )
                
                # Verify training results
                assert 'training_time_seconds' in train_results
                assert 'training_samples' in train_results
                assert train_results['training_samples'] == 200
                assert train_results['feature_count'] == 3
                assert pipeline.is_trained
                
                # Test anomaly detection
                detection_results = pipeline.detect_anomalies(str(test_path))
                
                # Verify detection results
                assert 'total_sequences' in detection_results
                assert 'anomaly_count' in detection_results
                assert 'anomaly_rate' in detection_results
                assert 'threshold_used' in detection_results
                
                # Should detect some anomalies
                assert detection_results['anomaly_count'] > 0
                assert 0 < detection_results['anomaly_rate'] < 1
                
                # Test model loading
                loaded_pipeline = Generation1Pipeline.load_trained_pipeline(str(model_dir))
                assert loaded_pipeline.is_trained
                assert loaded_pipeline.window_size == 10
                assert loaded_pipeline.encoding_dim == 8
                
                # Test loaded pipeline detection
                loaded_results = loaded_pipeline.detect_anomalies(str(test_path))
                assert loaded_results['total_sequences'] == detection_results['total_sequences']
                
            except ImportError:
                pytest.skip("TensorFlow not available for integration test")
    
    def test_pipeline_robustness(self):
        """Test pipeline robustness with edge cases."""
        pipeline = Generation1Pipeline(window_size=5, enable_logging=False)
        
        # Test with minimum viable data
        min_data = pd.DataFrame({
            'col1': list(range(10)),  # Exactly 10 samples for window_size=5
            'col2': [x * 2 for x in range(10)]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            min_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            prepared_data = pipeline.prepare_data(temp_path)
            assert len(prepared_data) == 10
            assert len(prepared_data.columns) == 2
            
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])