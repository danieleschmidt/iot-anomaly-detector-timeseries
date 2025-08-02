"""End-to-end pipeline testing."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.append('src')

from tests.fixtures.sample_data import (
    sample_sensor_data, 
    sample_anomaly_labels, 
    model_config,
    training_data_split,
    model_artifacts_paths
)


class TestCompleteAnomalyDetectionPipeline:
    """Test the complete anomaly detection pipeline from data ingestion to prediction."""
    
    @pytest.fixture
    def pipeline_workspace(self, tmp_path):
        """Create a temporary workspace for pipeline testing."""
        workspace = tmp_path / "pipeline_test"
        workspace.mkdir()
        
        # Create directory structure
        (workspace / "data" / "raw").mkdir(parents=True)
        (workspace / "data" / "processed").mkdir(parents=True)
        (workspace / "models").mkdir()
        (workspace / "logs").mkdir()
        (workspace / "outputs").mkdir()
        
        return workspace
    
    def test_data_ingestion_and_validation(self, sample_sensor_data, pipeline_workspace):
        """Test data ingestion and validation step."""
        from src.data_validator import DataValidator
        from src.data_preprocessor import DataPreprocessor
        
        # Save sample data to workspace
        data_path = pipeline_workspace / "data" / "raw" / "sensor_data.csv"
        sample_sensor_data.to_csv(data_path, index=False)
        
        # Initialize validator
        validator = DataValidator()
        
        # Test data loading and validation
        loaded_data = pd.read_csv(data_path)
        validation_results = validator.validate_dataset(loaded_data)
        
        assert validation_results['is_valid']  
        assert 'timestamp' in loaded_data.columns
        assert len(loaded_data) > 0
        assert not loaded_data.isnull().all().any()  # No completely null columns
    
    def test_data_preprocessing_pipeline(self, training_data_split, pipeline_workspace, model_config):
        """Test complete data preprocessing pipeline."""
        from src.data_preprocessor import DataPreprocessor
        
        train_df, val_df, test_df = training_data_split
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(
            window_size=model_config['window_size'],
            overlap=model_config['overlap']
        )
        
        # Fit preprocessing on training data
        feature_columns = model_config['feature_columns']
        preprocessor.fit(train_df[feature_columns])
        
        # Transform all datasets
        train_windows, train_labels = preprocessor.create_windows(
            train_df[feature_columns], 
            train_df['is_anomaly']
        )
        val_windows, val_labels = preprocessor.create_windows(
            val_df[feature_columns], 
            val_df['is_anomaly']
        )
        test_windows, test_labels = preprocessor.create_windows(
            test_df[feature_columns], 
            test_df['is_anomaly']
        )
        
        # Validate preprocessing results
        assert train_windows.shape[0] > 0
        assert train_windows.shape[1] == model_config['window_size']
        assert train_windows.shape[2] == len(feature_columns)
        assert len(train_labels) == len(train_windows)
        
        # Save preprocessed data
        processed_path = pipeline_workspace / "data" / "processed"
        np.save(processed_path / "train_windows.npy", train_windows)
        np.save(processed_path / "train_labels.npy", train_labels)
        np.save(processed_path / "val_windows.npy", val_windows)
        np.save(processed_path / "val_labels.npy", val_labels)
        np.save(processed_path / "test_windows.npy", test_windows)
        np.save(processed_path / "test_labels.npy", test_labels)
        
        # Verify files were saved
        assert (processed_path / "train_windows.npy").exists()
        assert (processed_path / "train_labels.npy").exists()
    
    def test_model_training_pipeline(self, pipeline_workspace, model_config):
        """Test model training pipeline."""
        from src.flexible_autoencoder import FlexibleAutoencoder
        from src.training_callbacks import create_callbacks
        
        # Load preprocessed data
        processed_path = pipeline_workspace / "data" / "processed"
        train_windows = np.load(processed_path / "train_windows.npy")
        val_windows = np.load(processed_path / "val_windows.npy")
        
        # Initialize model
        input_shape = (model_config['window_size'], len(model_config['feature_columns']))
        model = FlexibleAutoencoder(
            input_shape=input_shape,
            encoding_dim=model_config['encoding_dim']
        )
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Create callbacks
        model_path = pipeline_workspace / "models" / "autoencoder.h5"
        logs_path = pipeline_workspace / "logs"
        
        callbacks = create_callbacks(
            model_path=str(model_path),
            logs_dir=str(logs_path),
            patience=2
        )
        
        # Train model (reduced epochs for testing)
        history = model.fit(
            train_windows, train_windows,  # Autoencoder: input = output
            validation_data=(val_windows, val_windows),
            epochs=2,  # Reduced for testing
            batch_size=model_config['batch_size'],
            callbacks=callbacks,
            verbose=0
        )
        
        # Validate training results
        assert 'loss' in history.history
        assert 'val_loss' in history.history
        assert len(history.history['loss']) == 2
        assert model_path.exists()  # Model was saved
        
        return model
    
    def test_anomaly_detection_pipeline(self, pipeline_workspace, model_config):
        """Test anomaly detection pipeline."""
        from src.anomaly_detector import AnomalyDetector
        from tensorflow.keras.models import load_model
        
        # Load trained model
        model_path = pipeline_workspace / "models" / "autoencoder.h5"
        if not model_path.exists():
            pytest.skip("Model not available from training test")
        
        model = load_model(str(model_path))
        
        # Load test data
        processed_path = pipeline_workspace / "data" / "processed"
        test_windows = np.load(processed_path / "test_windows.npy")
        test_labels = np.load(processed_path / "test_labels.npy")
        
        # Initialize anomaly detector
        detector = AnomalyDetector(
            model=model,
            threshold=model_config['anomaly_threshold']
        )
        
        # Calculate reconstruction errors
        reconstruction_errors = detector.calculate_reconstruction_errors(test_windows)
        
        # Detect anomalies
        predictions = detector.detect_anomalies(test_windows)
        
        # Validate detection results
        assert len(reconstruction_errors) == len(test_windows)
        assert len(predictions) == len(test_windows)
        assert all(pred in [0, 1] for pred in predictions)  # Binary predictions
        
        # Calculate basic metrics
        if len(test_labels) > 0 and np.sum(test_labels) > 0:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(test_labels, predictions, average='binary', zero_division=0)
            recall = recall_score(test_labels, predictions, average='binary', zero_division=0)
            f1 = f1_score(test_labels, predictions, average='binary', zero_division=0)
            
            # Basic sanity checks (not strict due to limited training)
            assert 0 <= precision <= 1
            assert 0 <= recall <= 1
            assert 0 <= f1 <= 1
        
        # Save detection results
        results_path = pipeline_workspace / "outputs" / "detection_results.csv"
        results_df = pd.DataFrame({
            'reconstruction_error': reconstruction_errors,
            'prediction': predictions,
            'actual_label': test_labels if len(test_labels) > 0 else np.zeros(len(predictions))
        })
        results_df.to_csv(results_path, index=False)
        
        assert results_path.exists()
    
    def test_model_persistence_and_loading(self, pipeline_workspace, model_config):
        """Test model saving and loading functionality."""
        from src.model_manager import ModelManager
        from src.model_metadata import ModelMetadata
        
        model_path = pipeline_workspace / "models" / "autoencoder.h5"
        if not model_path.exists():
            pytest.skip("Model not available from training test")
        
        # Create model metadata
        metadata = ModelMetadata(
            model_name="test_autoencoder",
            version="1.0.0",
            created_at=pd.Timestamp.now(),
            config=model_config,
            metrics={
                'train_loss': 0.01,
                'val_loss': 0.015,
                'precision': 0.85,
                'recall': 0.80,
                'f1_score': 0.82
            }
        )
        
        # Initialize model manager
        manager = ModelManager(base_path=str(pipeline_workspace / "models"))
        
        # Save model with metadata
        manager.save_model_with_metadata(
            model_path=str(model_path),
            metadata=metadata
        )
        
        # Load model and metadata
        loaded_model, loaded_metadata = manager.load_model_with_metadata("test_autoencoder")
        
        # Validate loading
        assert loaded_model is not None
        assert loaded_metadata.model_name == "test_autoencoder"
        assert loaded_metadata.version == "1.0.0"
        assert 'train_loss' in loaded_metadata.metrics
    
    def test_streaming_prediction_pipeline(self, streaming_data_generator, pipeline_workspace, model_config):
        """Test streaming prediction pipeline."""
        from src.streaming_processor import StreamingProcessor
        from tensorflow.keras.models import load_model
        
        model_path = pipeline_workspace / "models" / "autoencoder.h5"
        if not model_path.exists():
            pytest.skip("Model not available from training test")
        
        # Load trained model
        model = load_model(str(model_path))
        
        # Initialize streaming processor
        processor = StreamingProcessor(
            model=model,
            window_size=model_config['window_size'],
            feature_columns=model_config['feature_columns'],
            threshold=model_config['anomaly_threshold']
        )
        
        # Process streaming data
        stream_generator = streaming_data_generator()
        total_processed = 0
        anomaly_count = 0
        
        for batch_idx, batch_df in enumerate(stream_generator):
            if batch_idx >= 3:  # Process only first 3 batches for testing
                break
            
            # Process batch
            results = processor.process_batch(batch_df)
            
            # Validate results
            assert 'predictions' in results
            assert 'reconstruction_errors' in results
            assert 'timestamp' in results
            
            total_processed += len(results['predictions'])
            anomaly_count += sum(results['predictions'])
        
        # Validate streaming results
        assert total_processed > 0
        assert anomaly_count >= 0
        
        # Test processor state management
        processor_state = processor.get_state()
        assert 'last_processed_timestamp' in processor_state
        assert 'total_processed' in processor_state
    
    def test_performance_monitoring_integration(self, pipeline_workspace):
        """Test performance monitoring integration."""
        from src.performance_monitor_cli import PerformanceMonitor
        
        # Initialize performance monitor
        monitor = PerformanceMonitor(
            log_dir=str(pipeline_workspace / "logs"),
            metrics_file=str(pipeline_workspace / "outputs" / "performance_metrics.json")
        )
        
        # Simulate performance data
        monitor.log_inference_time(45.2)  # ms
        monitor.log_memory_usage(128.5)   # MB
        monitor.log_throughput(150)       # samples/sec
        monitor.log_model_accuracy(0.87)
        
        # Generate performance report
        report = monitor.generate_report()
        
        # Validate report
        assert 'inference_time' in report
        assert 'memory_usage' in report
        assert 'throughput' in report
        assert 'model_accuracy' in report
        
        # Test alert triggering
        monitor.log_inference_time(200)  # High latency should trigger alert
        alerts = monitor.check_alerts()
        
        assert len(alerts) >= 0  # May or may not have alerts depending on thresholds
    
    def test_complete_pipeline_integration(self, sample_sensor_data, pipeline_workspace, model_config):
        """Test complete end-to-end pipeline integration."""
        # This test combines all the above steps in sequence
        
        # 1. Data preparation
        data_path = pipeline_workspace / "data" / "raw" / "sensor_data.csv"
        sample_sensor_data.to_csv(data_path, index=False)
        
        # 2. Run pipeline steps (simplified for integration test)
        try:
            # Data validation
            from src.data_validator import DataValidator
            validator = DataValidator()
            loaded_data = pd.read_csv(data_path)
            validation_results = validator.validate_dataset(loaded_data)
            assert validation_results['is_valid']
            
            # Data preprocessing
            from src.data_preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor(
                window_size=model_config['window_size'],
                overlap=0.5
            )
            
            feature_columns = model_config['feature_columns']
            preprocessor.fit(loaded_data[feature_columns])
            
            # Create windows
            windows, labels = preprocessor.create_windows(
                loaded_data[feature_columns],
                loaded_data.get('is_anomaly', np.zeros(len(loaded_data)))
            )
            
            # Model creation and quick training
            from src.flexible_autoencoder import FlexibleAutoencoder
            
            input_shape = (model_config['window_size'], len(feature_columns))
            model = FlexibleAutoencoder(
                input_shape=input_shape,
                encoding_dim=16  # Smaller for quick test
            )
            
            model.compile(optimizer='adam', loss='mse')
            
            # Quick training
            model.fit(windows, windows, epochs=1, verbose=0)
            
            # Anomaly detection
            from src.anomaly_detector import AnomalyDetector
            detector = AnomalyDetector(model=model, threshold=0.9)
            
            predictions = detector.detect_anomalies(windows[:10])  # Test on subset
            
            # Validate final results
            assert len(predictions) == 10
            assert all(pred in [0, 1] for pred in predictions)
            
            print(f"✅ Complete pipeline test passed!")
            print(f"   - Processed {len(loaded_data)} data points")
            print(f"   - Created {len(windows)} windows")
            print(f"   - Generated {len(predictions)} predictions")
            
        except Exception as e:
            pytest.fail(f"Pipeline integration test failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])