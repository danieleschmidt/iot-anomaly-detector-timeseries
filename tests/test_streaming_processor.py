"""Test streaming data processing functionality."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import queue
import threading
import time
from pathlib import Path
import tempfile
import json

# Import the module we'll create
from src.streaming_processor import StreamingProcessor, StreamingConfig


class TestStreamingProcessor:
    """Test streaming data processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StreamingConfig(
            window_size=10,
            step_size=1,
            batch_size=5,
            anomaly_threshold=0.5,
            buffer_size=100
        )
        
        # Mock model and preprocessor
        self.mock_model = Mock()
        self.mock_model.predict.return_value = np.random.random((5, 10, 3))
        
        self.mock_preprocessor = Mock()
        self.mock_preprocessor.create_sliding_windows.return_value = np.random.random((5, 10, 3))
        
    def test_streaming_processor_initialization(self):
        """Test streaming processor initialization."""
        with patch('src.streaming_processor.load_model'), \
             patch('src.streaming_processor.DataPreprocessor'):
            
            processor = StreamingProcessor(
                model_path="dummy.h5",
                config=self.config
            )
            
            assert processor.config == self.config
            assert processor.buffer.maxlen == self.config.buffer_size
            assert not processor.is_running
    
    def test_data_ingestion(self):
        """Test data ingestion into buffer."""
        with patch('src.streaming_processor.load_model'), \
             patch('src.streaming_processor.DataPreprocessor'):
            
            processor = StreamingProcessor("dummy.h5", self.config)
            
            # Test single data point ingestion
            data_point = {'timestamp': '2023-01-01T00:00:00', 'sensor1': 1.0, 'sensor2': 2.0}
            processor.ingest_data(data_point)
            
            assert len(processor.buffer) == 1
            assert processor.buffer[0] == data_point
    
    def test_batch_data_ingestion(self):
        """Test batch data ingestion."""
        with patch('src.streaming_processor.load_model'), \
             patch('src.streaming_processor.DataPreprocessor'):
            
            processor = StreamingProcessor("dummy.h5", self.config)
            
            # Test batch ingestion
            batch_data = [
                {'timestamp': f'2023-01-01T00:0{i}:00', 'sensor1': float(i), 'sensor2': float(i*2)}
                for i in range(5)
            ]
            processor.ingest_batch(batch_data)
            
            assert len(processor.buffer) == 5
    
    def test_buffer_overflow_handling(self):
        """Test buffer overflow handling with circular buffer."""
        small_config = StreamingConfig(buffer_size=3, window_size=5)
        
        with patch('src.streaming_processor.load_model'), \
             patch('src.streaming_processor.DataPreprocessor'):
            
            processor = StreamingProcessor("dummy.h5", small_config)
            
            # Add more data than buffer size
            for i in range(5):
                processor.ingest_data({'timestamp': f'2023-01-01T00:0{i}:00', 'value': i})
            
            # Buffer should only contain last 3 items
            assert len(processor.buffer) == 3
            assert processor.buffer[-1]['value'] == 4  # Most recent
    
    def test_window_creation_from_buffer(self):
        """Test window creation from streaming buffer."""
        with patch('src.streaming_processor.load_model') as mock_load, \
             patch('src.streaming_processor.DataPreprocessor') as mock_dp_class:
            
            mock_dp_instance = Mock()
            mock_dp_class.return_value = mock_dp_instance
            mock_dp_instance.create_sliding_windows.return_value = np.random.random((2, 5, 2))
            
            processor = StreamingProcessor("dummy.h5", StreamingConfig(window_size=5))
            
            # Add enough data to create windows
            for i in range(10):
                processor.ingest_data({'sensor1': float(i), 'sensor2': float(i*2)})
            
            windows = processor._create_windows_from_buffer()
            
            # Verify preprocessor was called
            mock_dp_instance.create_sliding_windows.assert_called_once()
            assert windows is not None
    
    def test_anomaly_detection(self):
        """Test real-time anomaly detection."""
        with patch('src.streaming_processor.load_model') as mock_load, \
             patch('src.streaming_processor.DataPreprocessor') as mock_dp_class:
            
            # Setup mocks
            mock_model = Mock()
            mock_model.predict.return_value = np.random.random((2, 5, 2))
            mock_load.return_value = mock_model
            
            mock_dp_instance = Mock()
            mock_dp_class.return_value = mock_dp_instance
            mock_dp_instance.create_sliding_windows.return_value = np.random.random((2, 5, 2))
            
            processor = StreamingProcessor("dummy.h5", self.config)
            
            # Add data and trigger detection
            for i in range(15):
                processor.ingest_data({'sensor1': float(i), 'sensor2': float(i*2)})
            
            results = processor.detect_anomalies()
            
            assert 'scores' in results
            assert 'anomalies' in results
            assert 'timestamp' in results
    
    def test_streaming_mode_start_stop(self):
        """Test starting and stopping streaming mode."""
        with patch('src.streaming_processor.load_model'), \
             patch('src.streaming_processor.DataPreprocessor'):
            
            processor = StreamingProcessor("dummy.h5", self.config)
            
            # Test start
            processor.start_streaming()
            assert processor.is_running
            assert processor.streaming_thread is not None
            
            # Test stop
            processor.stop_streaming()
            assert not processor.is_running
    
    def test_callback_system(self):
        """Test anomaly detection callback system."""
        callback_results = []
        
        def test_callback(anomaly_result):
            callback_results.append(anomaly_result)
        
        with patch('src.streaming_processor.load_model') as mock_load, \
             patch('src.streaming_processor.DataPreprocessor') as mock_dp_class:
            
            # Setup mocks for anomaly detection
            mock_model = Mock()
            mock_model.predict.return_value = np.array([[[0.1, 0.2]], [[0.9, 0.8]]])
            mock_load.return_value = mock_model
            
            mock_dp_instance = Mock()
            mock_dp_class.return_value = mock_dp_instance
            mock_dp_instance.create_sliding_windows.return_value = np.array([[[0.1, 0.2]], [[0.5, 0.6]]])
            
            config = StreamingConfig(anomaly_threshold=0.3)  # Low threshold to trigger anomaly
            processor = StreamingProcessor("dummy.h5", config)
            processor.add_anomaly_callback(test_callback)
            
            # Add data that should trigger anomaly detection
            for i in range(15):
                processor.ingest_data({'sensor1': float(i), 'sensor2': float(i*2)})
            
            # Manually trigger detection for testing
            processor.detect_anomalies()
            
            # Callback should have been called
            assert len(callback_results) > 0
    
    def test_config_validation(self):
        """Test streaming configuration validation."""
        # Test invalid window size
        with pytest.raises(ValueError, match="Window size must be positive"):
            StreamingConfig(window_size=0)
        
        # Test invalid batch size
        with pytest.raises(ValueError, match="Batch size must be positive"):
            StreamingConfig(batch_size=0)
        
        # Test invalid threshold
        with pytest.raises(ValueError, match="Anomaly threshold must be non-negative"):
            StreamingConfig(anomaly_threshold=-1)
    
    def test_performance_metrics(self):
        """Test performance monitoring in streaming mode."""
        with patch('src.streaming_processor.load_model'), \
             patch('src.streaming_processor.DataPreprocessor'):
            
            processor = StreamingProcessor("dummy.h5", self.config)
            
            # Process some data
            for i in range(10):
                processor.ingest_data({'sensor1': float(i), 'sensor2': float(i*2)})
            
            metrics = processor.get_performance_metrics()
            
            assert 'total_processed' in metrics
            assert 'processing_rate' in metrics
            assert 'buffer_utilization' in metrics
            assert 'anomalies_detected' in metrics
    
    def test_data_export(self):
        """Test exporting streaming results."""
        with patch('src.streaming_processor.load_model'), \
             patch('src.streaming_processor.DataPreprocessor'):
            
            processor = StreamingProcessor("dummy.h5", self.config)
            
            # Add some mock results
            processor.results_history = [
                {'timestamp': '2023-01-01T00:00:00', 'scores': [0.1, 0.2], 'anomalies': [False, False]},
                {'timestamp': '2023-01-01T00:01:00', 'scores': [0.8, 0.9], 'anomalies': [True, True]},
            ]
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                export_path = f.name
            
            try:
                processor.export_results(export_path, format='json')
                
                # Verify export
                with open(export_path, 'r') as f:
                    exported_data = json.load(f)
                
                assert len(exported_data) == 2
                assert 'timestamp' in exported_data[0]
                assert 'scores' in exported_data[0]
                
            finally:
                Path(export_path).unlink(missing_ok=True)


class TestStreamingConfig:
    """Test streaming configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StreamingConfig()
        
        assert config.window_size == 50
        assert config.step_size == 1
        assert config.batch_size == 32
        assert config.anomaly_threshold == 0.5
        assert config.buffer_size == 1000
        assert config.processing_interval == 1.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = StreamingConfig(
            window_size=100,
            batch_size=64,
            anomaly_threshold=0.7,
            processing_interval=2.0
        )
        
        assert config.window_size == 100
        assert config.batch_size == 64
        assert config.anomaly_threshold == 0.7
        assert config.processing_interval == 2.0
    
    def test_config_serialization(self):
        """Test configuration serialization/deserialization."""
        config = StreamingConfig(window_size=75, batch_size=16)
        
        config_dict = config.to_dict()
        restored_config = StreamingConfig.from_dict(config_dict)
        
        assert config.window_size == restored_config.window_size
        assert config.batch_size == restored_config.batch_size
        assert config.anomaly_threshold == restored_config.anomaly_threshold