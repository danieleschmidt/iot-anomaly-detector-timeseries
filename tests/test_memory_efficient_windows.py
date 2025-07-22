"""Test memory-efficient window creation functionality."""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_preprocessor import DataPreprocessor


class TestMemoryEfficientWindows:
    """Test memory-efficient window creation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        self.test_data = np.random.random((1000, 5))  # Large dataset for testing
    
    def test_window_generator_creation(self):
        """Test basic window generator functionality."""
        window_size = 10
        step = 1
        
        generator = self.preprocessor.create_windows_generator(
            self.test_data, window_size, step
        )
        
        # Test that it returns a generator
        assert hasattr(generator, '__iter__')
        assert hasattr(generator, '__next__')
        
        # Test first window
        first_window = next(generator)
        assert first_window.shape == (window_size, 5)
        np.testing.assert_array_equal(first_window, self.test_data[0:window_size])
        
    def test_window_generator_count(self):
        """Test that generator yields correct number of windows."""
        window_size = 50
        step = 10
        expected_count = (len(self.test_data) - window_size) // step + 1
        
        generator = self.preprocessor.create_windows_generator(
            self.test_data, window_size, step
        )
        
        windows = list(generator)
        assert len(windows) == expected_count
        
    def test_window_generator_step_size(self):
        """Test generator with different step sizes."""
        window_size = 10
        step = 5
        
        generator = self.preprocessor.create_windows_generator(
            self.test_data, window_size, step
        )
        
        windows = list(generator)
        
        # Check that windows are spaced correctly
        expected_starts = list(range(0, len(self.test_data) - window_size + 1, step))
        assert len(windows) == len(expected_starts)
        
        for i, window in enumerate(windows):
            expected_start = expected_starts[i]
            expected_window = self.test_data[expected_start:expected_start + window_size]
            np.testing.assert_array_equal(window, expected_window)
    
    def test_batched_window_processing(self):
        """Test batched processing of windows from generator."""
        window_size = 20
        batch_size = 32
        
        generator = self.preprocessor.create_windows_generator(
            self.test_data, window_size, step=1
        )
        
        # Process windows in batches
        batch_count = 0
        total_windows = 0
        
        for batch in self.preprocessor.process_windows_batched(generator, batch_size):
            assert batch.shape[0] <= batch_size  # Batch size constraint
            assert batch.shape[1] == window_size  # Window size maintained
            assert batch.shape[2] == 5  # Feature count maintained
            batch_count += 1
            total_windows += batch.shape[0]
        
        # Verify total windows processed
        expected_total = len(self.test_data) - window_size + 1
        assert total_windows == expected_total
        assert batch_count > 1  # Should have multiple batches
    
    def test_memory_usage_comparison(self):
        """Test memory efficiency compared to traditional approach."""
        # This is a conceptual test - in practice would need memory profiling
        window_size = 100
        large_data = np.random.random((10000, 10))  # Very large dataset
        
        # Test that generator doesn't load all windows at once
        generator = self.preprocessor.create_windows_generator(
            large_data, window_size, step=1
        )
        
        # Generator should be created instantly without memory spike
        assert generator is not None
        
        # Process first few windows only
        windows_processed = 0
        for window in generator:
            windows_processed += 1
            if windows_processed >= 5:
                break
        
        assert windows_processed == 5
    
    def test_window_generator_validation(self):
        """Test input validation for window generator."""
        data = np.random.random((100, 3))
        
        # Test invalid window size
        with pytest.raises(ValueError, match="Window size must be positive"):
            list(self.preprocessor.create_windows_generator(data, 0))
        
        # Test invalid step size
        with pytest.raises(ValueError, match="Step size must be positive"):
            list(self.preprocessor.create_windows_generator(data, 10, 0))
        
        # Test window size larger than data
        with pytest.raises(ValueError, match="Window size cannot be larger than data length"):
            list(self.preprocessor.create_windows_generator(data, 200))
    
    def test_sliding_windows_generator(self):
        """Test sliding windows generator with preprocessing."""
        window_size = 25
        data = np.random.random((500, 4))
        
        generator = self.preprocessor.create_sliding_windows_generator(
            data, window_size, step=1
        )
        
        # Test first few windows are properly scaled
        windows = []
        for i, window in enumerate(generator):
            windows.append(window)
            if i >= 2:
                break
        
        assert len(windows) == 3
        assert all(window.shape == (window_size, 4) for window in windows)
        
        # Verify scaling was applied (values should be in [0, 1] range for MinMaxScaler)
        for window in windows:
            assert window.min() >= 0.0
            assert window.max() <= 1.0
    
    def test_anomaly_detection_with_generator(self):
        """Test integration with anomaly detection using generators."""
        # Mock model for testing
        mock_model = Mock()
        mock_model.predict.return_value = np.random.random((10, 50, 4))
        
        window_size = 50
        batch_size = 10
        
        # Create generator
        generator = self.preprocessor.create_windows_generator(
            self.test_data[:, :4],  # Use 4 features to match mock
            window_size, step=1
        )
        
        # Simulate anomaly detection on batches
        total_scores = []
        for batch in self.preprocessor.process_windows_batched(generator, batch_size):
            # Mock reconstruction and scoring
            reconstructed = mock_model.predict(batch)
            scores = np.mean(np.square(batch - reconstructed), axis=(1, 2))
            total_scores.extend(scores)
        
        # Verify we got scores for the expected number of windows
        expected_windows = len(self.test_data) - window_size + 1
        assert len(total_scores) == expected_windows
    
    def test_large_dataset_processing(self):
        """Test processing of very large dataset using generators."""
        # Simulate processing a dataset too large for memory
        # In practice this would be much larger (GB scale)
        large_data = np.random.random((50000, 8))
        window_size = 100
        batch_size = 256
        
        generator = self.preprocessor.create_windows_generator(
            large_data, window_size, step=10
        )
        
        processed_batches = 0
        total_windows = 0
        
        for batch in self.preprocessor.process_windows_batched(generator, batch_size):
            processed_batches += 1
            total_windows += batch.shape[0]
            
            # Verify batch properties
            assert batch.shape[0] <= batch_size
            assert batch.shape[1] == window_size
            assert batch.shape[2] == 8
            
            # Simulate processing time
            if processed_batches >= 5:  # Process only first few batches for test
                break
        
        assert processed_batches == 5
        assert total_windows > 0
    
    def test_window_generator_with_preprocessing_pipeline(self):
        """Test window generator integration with full preprocessing pipeline."""
        # Create test data with some preprocessing requirements
        raw_data = np.random.random((2000, 6)) * 100  # Unscaled data
        
        # Test the full pipeline
        window_size = 75
        batch_size = 64
        
        # Create generator with preprocessing
        generator = self.preprocessor.create_sliding_windows_generator(
            raw_data, window_size, step=5
        )
        
        batch_count = 0
        for batch in self.preprocessor.process_windows_batched(generator, batch_size):
            # Verify preprocessing was applied
            assert batch.min() >= 0.0  # MinMaxScaler output
            assert batch.max() <= 1.0
            
            # Verify shapes
            assert batch.shape[1] == window_size
            assert batch.shape[2] == 6
            
            batch_count += 1
            if batch_count >= 3:  # Test first few batches
                break
        
        assert batch_count == 3


class TestWindowGeneratorUtilities:
    """Test utility functions for window generator processing."""
    
    def test_estimate_memory_usage(self):
        """Test memory usage estimation utility."""
        preprocessor = DataPreprocessor()
        
        # Test parameters
        data_shape = (100000, 10)
        window_size = 200
        batch_size = 128
        
        # Test memory estimation
        memory_info = preprocessor.estimate_window_memory_usage(
            data_shape, window_size, batch_size
        )
        
        assert 'traditional_memory_mb' in memory_info
        assert 'generator_memory_mb' in memory_info
        assert 'memory_savings_ratio' in memory_info
        
        # Generator should use significantly less memory
        assert memory_info['generator_memory_mb'] < memory_info['traditional_memory_mb']
        assert memory_info['memory_savings_ratio'] > 2.0
    
    def test_optimal_batch_size_calculation(self):
        """Test calculation of optimal batch size for memory efficiency."""
        preprocessor = DataPreprocessor()
        
        data_shape = (50000, 8)
        window_size = 100
        available_memory_mb = 1024  # 1 GB
        
        optimal_batch = preprocessor.calculate_optimal_batch_size(
            data_shape, window_size, available_memory_mb
        )
        
        assert isinstance(optimal_batch, int)
        assert optimal_batch > 0
        assert optimal_batch <= 1000  # Should be reasonable
    
    def test_progress_tracking_with_generator(self):
        """Test progress tracking during generator-based processing."""
        preprocessor = DataPreprocessor()
        data = np.random.random((5000, 5))
        
        window_size = 50
        batch_size = 100
        
        generator = preprocessor.create_windows_generator(data, window_size)
        
        progress_updates = []
        
        def progress_callback(current, total, elapsed_time):
            progress_updates.append({
                'current': current,
                'total': total,
                'elapsed': elapsed_time
            })
        
        # Process with progress tracking
        total_processed = preprocessor.process_windows_with_progress(
            generator, batch_size, progress_callback
        )
        
        expected_total = len(data) - window_size + 1
        assert total_processed == expected_total
        assert len(progress_updates) > 0
        
        # Verify progress updates are reasonable
        for update in progress_updates:
            assert update['current'] <= update['total']
            assert update['elapsed'] >= 0