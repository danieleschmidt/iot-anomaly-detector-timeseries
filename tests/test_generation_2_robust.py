"""Tests for Generation 2: MAKE IT ROBUST - Enhanced reliability and monitoring."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch, Mock
from datetime import datetime

from src.robust_autoencoder_v2 import RobustAutoencoderV2
from src.generation_2_robust_pipeline import Generation2RobustPipeline


class TestRobustAutoencoderV2:
    """Test the robust autoencoder with enhanced features."""
    
    def test_robust_autoencoder_creation(self):
        """Test robust autoencoder creation with enhanced features."""
        autoencoder = RobustAutoencoderV2(
            window_size=10,
            encoding_dim=8,
            enable_circuit_breaker=True,
            enable_retry=True,
            max_retries=2,
            enable_validation=True
        )
        
        assert autoencoder.window_size == 10
        assert autoencoder.encoding_dim == 8
        assert autoencoder.enable_circuit_breaker
        assert autoencoder.enable_retry
        assert autoencoder.max_retries == 2
        assert autoencoder.enable_validation
        assert not autoencoder.is_trained
        assert autoencoder.model is None
        assert hasattr(autoencoder, 'circuit_breaker')
        assert hasattr(autoencoder, 'retry_manager')
        assert hasattr(autoencoder, 'error_handler')
    
    def test_input_data_validation(self):
        """Test comprehensive input data validation."""
        autoencoder = RobustAutoencoderV2(window_size=5, enable_validation=True)
        
        # Test empty data
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError, match="Data cannot be empty"):
            autoencoder._validate_input_data(empty_data)
        
        # Test insufficient data
        small_data = pd.DataFrame({'col1': [1, 2, 3]})  # Only 3 samples, need 5
        with pytest.raises(ValueError, match="Data length .* must be >= window_size"):
            autoencoder._validate_input_data(small_data)
        
        # Test all NaN column
        nan_data = pd.DataFrame({
            'good_col': [1, 2, 3, 4, 5, 6],
            'bad_col': [np.nan] * 6
        })
        with pytest.raises(ValueError, match="Columns with all NaN values"):
            autoencoder._validate_input_data(nan_data)
        
        # Test valid data (should not raise)
        valid_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6],
            'col2': [2.1, 2.2, 2.3, 2.4, 2.5, 2.6]
        })
        autoencoder._validate_input_data(valid_data)  # Should pass without error
    
    def test_robust_callbacks_creation(self):
        """Test creation of robust training callbacks."""
        autoencoder = RobustAutoencoderV2(
            enable_early_stopping=True,
            enable_lr_scheduling=True,
            enable_checkpointing=True
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.h5"
            
            try:
                callbacks_list = autoencoder._create_robust_callbacks(
                    checkpoint_path=str(checkpoint_path),
                    monitor='val_loss',
                    patience=10
                )
                
                # Should have early stopping, LR reduction, checkpoint, and custom callback
                assert len(callbacks_list) >= 3
                
                # Check callback types
                callback_types = [type(cb).__name__ for cb in callbacks_list]
                assert any('EarlyStopping' in name for name in callback_types)
                assert any('ReduceLROnPlateau' in name for name in callback_types)
                assert any('ModelCheckpoint' in name for name in callback_types)
                
            except ImportError:
                pytest.skip("TensorFlow not available")
    
    def test_health_status_reporting(self):
        """Test comprehensive health status reporting."""
        autoencoder = RobustAutoencoderV2(
            window_size=10,
            encoding_dim=8,
            enable_circuit_breaker=True,
            enable_retry=True,
            enable_validation=True
        )
        
        health_status = autoencoder.get_health_status()
        
        # Check required fields
        assert 'is_trained' in health_status
        assert 'model_available' in health_status
        assert 'feature_count' in health_status
        assert 'window_size' in health_status
        assert 'encoding_dim' in health_status
        assert 'circuit_breaker_enabled' in health_status
        assert 'retry_enabled' in health_status
        assert 'validation_enabled' in health_status
        
        # Check circuit breaker status
        assert 'circuit_breaker_status' in health_status
        cb_status = health_status['circuit_breaker_status']
        assert 'state' in cb_status
        assert 'failure_count' in cb_status
        assert 'success_count' in cb_status
        
        # Verify initial values
        assert not health_status['is_trained']
        assert not health_status['model_available']
        assert health_status['circuit_breaker_enabled']
        assert health_status['retry_enabled']
        assert health_status['validation_enabled']


class TestGeneration2RobustPipeline:
    """Test the Generation 2 robust pipeline."""
    
    def test_robust_pipeline_creation(self):
        """Test robust pipeline creation with all features."""
        pipeline = Generation2RobustPipeline(
            window_size=20,
            encoding_dim=12,
            contamination_rate=0.05,
            enable_logging=False,  # Disable for testing
            enable_monitoring=False,  # Disable for testing
            enable_data_validation=True,
            validation_level="moderate",
            enable_health_checks=False,  # Disable for testing
            enable_backup=True,
            enable_performance_tracking=True
        )
        
        assert pipeline.window_size == 20
        assert pipeline.encoding_dim == 12
        assert pipeline.contamination_rate == 0.05
        assert not pipeline.is_trained
        assert pipeline.autoencoder is None
        assert pipeline.data_validator is not None
        assert pipeline.enable_backup
        assert pipeline.enable_performance_tracking
        assert hasattr(pipeline, 'circuit_breaker')
        assert hasattr(pipeline, 'retry_manager')
        assert hasattr(pipeline, 'error_handler')
    
    def test_database_initialization(self):
        """Test SQLite database initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch the database path to use temp directory
            with patch.object(Path, 'mkdir'), patch('src.generation_2_robust_pipeline.Path') as mock_path:
                mock_db_path = Path(temp_dir) / "test_pipeline.db"
                mock_path.return_value = mock_db_path
                
                pipeline = Generation2RobustPipeline(
                    enable_logging=False,
                    enable_monitoring=False,
                    enable_health_checks=False
                )
                
                # Manually set the database path for testing
                pipeline.db_path = mock_db_path
                pipeline._init_database()
                
                # Check if database file exists and has correct tables
                if mock_db_path.exists():
                    conn = sqlite3.connect(str(mock_db_path))
                    cursor = conn.cursor()
                    
                    # Check if tables exist
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    expected_tables = ['training_logs', 'prediction_logs', 'performance_metrics']
                    for table in expected_tables:
                        assert table in tables
                    
                    conn.close()
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        pipeline = Generation2RobustPipeline(
            window_size=30,
            enable_logging=False,
            enable_monitoring=False,
            enable_health_checks=False
        )
        
        # Test with different data sizes
        memory_small = pipeline._estimate_memory_usage(100, 5)
        memory_large = pipeline._estimate_memory_usage(1000, 10)
        
        assert isinstance(memory_small, float)
        assert isinstance(memory_large, float)
        assert memory_large > memory_small  # Larger data should require more memory
        assert memory_small > 0  # Should be positive
    
    def test_comprehensive_data_validation(self):
        """Test comprehensive data validation and preparation."""
        pipeline = Generation2RobustPipeline(
            window_size=5,
            enable_logging=False,
            enable_monitoring=False,
            enable_health_checks=False,
            enable_data_validation=True
        )
        
        # Create test data with various issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data = pd.DataFrame({
                'temp': [20.1, 21.2, np.nan, 19.9, 21.1, 20.5, 21.0, 20.2, 19.8, 21.3],
                'humidity': [65.2, 66.1, 64.8, np.nan, 65.9, 66.0, 65.5, 64.9, 67.0, 65.8],
                'pressure': [1013.2, 1012.8, 1013.5, 1013.1, 1012.9, 1013.0, 1012.7, 1013.3, 1013.4, 1012.6],
                'target': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]  # Target column
            })
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Test data validation and preparation
            prepared_data = pipeline._validate_and_prepare_data(
                temp_path,
                target_column='target',
                auto_fix=True
            )
            
            assert len(prepared_data) == 10
            assert len(prepared_data.columns) == 3  # Target removed
            assert 'target' not in prepared_data.columns
            assert not prepared_data.isnull().any().any()  # No missing values after auto-fix
            
        finally:
            Path(temp_path).unlink()
    
    def test_temporal_pattern_analysis(self):
        """Test temporal pattern analysis for anomalies."""
        pipeline = Generation2RobustPipeline(
            enable_logging=False,
            enable_monitoring=False,
            enable_health_checks=False
        )
        
        # Test with clustered anomalies
        anomalies = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
        errors = np.random.randn(len(anomalies))
        
        temporal_analysis = pipeline._analyze_temporal_patterns(anomalies, errors)
        
        assert 'anomaly_clusters' in temporal_analysis
        assert 'max_cluster_size' in temporal_analysis
        assert 'cluster_count' in temporal_analysis
        assert 'isolated_anomalies' in temporal_analysis
        assert 'clustered_anomalies' in temporal_analysis
        
        # Check cluster detection
        assert temporal_analysis['cluster_count'] > 0
        assert temporal_analysis['max_cluster_size'] >= 1
        
        # Test with no anomalies
        no_anomalies = np.array([0, 0, 0, 0, 0])
        no_errors = np.random.randn(5)
        
        no_temporal = pipeline._analyze_temporal_patterns(no_anomalies, no_errors)
        assert no_temporal['cluster_count'] == 0
        assert no_temporal['max_cluster_size'] == 0
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        pipeline = Generation2RobustPipeline(
            enable_logging=False,
            enable_monitoring=False,
            enable_health_checks=False
        )
        
        # Generate test data
        np.random.seed(42)
        errors = np.random.normal(0.1, 0.05, 100)  # Mean=0.1, std=0.05
        
        confidence_intervals = pipeline._calculate_confidence_intervals(errors, 0.95)
        
        if confidence_intervals:  # Only test if scipy is available
            assert 'mean_lower_bound' in confidence_intervals
            assert 'mean_upper_bound' in confidence_intervals
            assert 'margin_of_error' in confidence_intervals
            assert 'confidence_level' in confidence_intervals
            assert 'sample_size' in confidence_intervals
            
            # Check that bounds make sense
            assert confidence_intervals['mean_lower_bound'] < confidence_intervals['mean_upper_bound']
            assert confidence_intervals['confidence_level'] == 0.95
            assert confidence_intervals['sample_size'] == 100
    
    def test_comprehensive_status_reporting(self):
        """Test comprehensive status reporting."""
        pipeline = Generation2RobustPipeline(
            window_size=15,
            encoding_dim=10,
            enable_logging=False,
            enable_monitoring=False,
            enable_health_checks=False,
            enable_backup=True,
            enable_performance_tracking=True
        )
        
        status = pipeline.get_comprehensive_status()
        
        # Check main sections
        assert 'pipeline_info' in status
        assert 'system_monitoring' in status
        assert 'reliability_features' in status
        assert 'performance_metrics' in status
        assert 'database_logging' in status
        
        # Check pipeline info
        pipeline_info = status['pipeline_info']
        assert pipeline_info['window_size'] == 15
        assert pipeline_info['encoding_dim'] == 10
        assert not pipeline_info['is_trained']
        
        # Check reliability features
        reliability = status['reliability_features']
        assert 'circuit_breaker_state' in reliability
        assert 'circuit_breaker_stats' in reliability
        assert reliability['retry_manager_enabled']
        assert reliability['backup_enabled']
        
        # Check performance metrics
        assert status['performance_metrics'] is not None
        assert 'training_times' in status['performance_metrics']
        assert 'prediction_times' in status['performance_metrics']
    
    def test_backup_functionality(self):
        """Test backup creation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Generation2RobustPipeline(
                enable_logging=False,
                enable_monitoring=False,
                enable_health_checks=False,
                enable_backup=True
            )
            
            model_path = Path(temp_dir) / "models"
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Create dummy model files
            (model_path / "model.h5").write_text("dummy model")
            (model_path / "scaler.pkl").write_text("dummy scaler")
            (model_path / "config.pkl").write_text("dummy config")
            
            # Create backup
            pipeline._create_backup(model_path)
            
            # Check if backup was created
            backup_dir = model_path.parent / "backups"
            assert backup_dir.exists()
            
            # Check if backup contains files
            backup_folders = list(backup_dir.glob("backup_*"))
            assert len(backup_folders) > 0
            
            # Check backup contents
            backup_folder = backup_folders[0]
            backup_files = list(backup_folder.glob("*"))
            assert len(backup_files) == 3  # Should have all 3 files
    
    def test_error_logging_to_database(self):
        """Test error logging to database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Generation2RobustPipeline(
                enable_logging=False,
                enable_monitoring=False,
                enable_health_checks=False
            )
            
            # Set up temporary database
            db_path = Path(temp_dir) / "test.db"
            pipeline.db_path = db_path
            pipeline._init_database()
            
            # Test logging training data
            training_data = {
                'duration_seconds': 120.5,
                'samples_count': 1000,
                'epochs_completed': 50,
                'final_loss': 0.001,
                'validation_loss': 0.002,
                'status': 'SUCCESS',
                'error_message': None
            }
            
            pipeline._log_to_database('training_logs', training_data)
            
            # Verify data was logged
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM training_logs")
            rows = cursor.fetchall()
            conn.close()
            
            assert len(rows) == 1
            row = rows[0]
            assert row[2] == 120.5  # duration_seconds
            assert row[3] == 1000   # samples_count
            assert row[4] == 50     # epochs_completed


class TestRobustIntegration:
    """Integration tests for Generation 2 robust components."""
    
    def test_robust_pipeline_error_handling(self):
        """Test robust pipeline error handling and recovery."""
        pipeline = Generation2RobustPipeline(
            window_size=5,
            enable_logging=False,
            enable_monitoring=False,
            enable_health_checks=False,
            enable_data_validation=True
        )
        
        # Test with invalid data path
        with pytest.raises(FileNotFoundError):
            pipeline._validate_and_prepare_data("nonexistent_file.csv")
        
        # Test circuit breaker functionality
        original_failure_count = pipeline.circuit_breaker.failure_count
        
        # Trigger multiple failures to test circuit breaker
        for _ in range(2):
            try:
                pipeline._validate_and_prepare_data("nonexistent_file.csv")
            except:
                pass  # Expected to fail
        
        # Circuit breaker should have recorded failures
        assert pipeline.circuit_breaker.failure_count > original_failure_count
    
    def test_robust_autoencoder_with_mock_training(self):
        """Test robust autoencoder with mocked training components."""
        # Create synthetic data for testing
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(0, 1, 50),
            'feature3': np.random.normal(0, 1, 50)
        })
        
        autoencoder = RobustAutoencoderV2(
            window_size=10,
            encoding_dim=8,
            enable_circuit_breaker=False,  # Disable for testing
            enable_retry=False,            # Disable for testing
            enable_validation=True
        )
        
        # Test data validation
        autoencoder._validate_input_data(data)  # Should pass
        
        # Test sequence preparation
        sequences = autoencoder._prepare_sequences(data.values)
        assert sequences.shape == (41, 10, 3)  # 50-10+1=41 sequences
        
        # Test health status
        health = autoencoder.get_health_status()
        assert not health['is_trained']
        assert not health['model_available']
        assert health['validation_enabled']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])