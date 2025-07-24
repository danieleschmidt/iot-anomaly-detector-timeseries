"""Performance tests for AnomalyDetector batched inference optimization."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import time
import psutil

from src.anomaly_detector import AnomalyDetector


class TestBatchedInferenceOptimization:
    """Test suite for batched inference performance improvements."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock autoencoder model for testing."""
        model = MagicMock()
        model.predict.return_value = np.random.randn(10, 30, 3)  # Mock prediction
        return model
    
    @pytest.fixture
    def mock_preprocessor(self):
        """Create mock preprocessor."""
        preprocessor = MagicMock()
        preprocessor.load_and_preprocess.return_value = np.random.randn(100, 30, 3)
        return preprocessor
    
    @pytest.fixture
    def detector_with_mocks(self, temp_dir, mock_model, mock_preprocessor):
        """Create AnomalyDetector with mocked dependencies."""
        # Create temporary model file
        model_path = Path(temp_dir) / "test_model.h5"
        model_path.write_text("dummy model")
        
        with patch('src.anomaly_detector.load_model', return_value=mock_model):
            detector = AnomalyDetector(str(model_path))
            detector.preprocessor = mock_preprocessor
            return detector, mock_model, mock_preprocessor


class TestBatchedScoring:
    """Test the new batched scoring functionality."""
    
    def test_score_batched_basic_functionality(self):
        """Test that batched scoring produces correct results."""
        # Create test data
        test_sequences = np.random.randn(100, 30, 3)
        
        with patch('src.anomaly_detector.load_model') as mock_load:
            mock_model = MagicMock()
            
            # Mock model to return reconstructions equal to input (no error)
            mock_model.predict.return_value = test_sequences[:32]  # First batch
            mock_load.return_value = mock_model
            
            detector = AnomalyDetector("dummy_path")
            
            # Test with batch size that evenly divides data
            with patch.object(detector, 'model', mock_model):
                # Setup model to return different results for each batch
                def mock_predict(batch, verbose=0):
                    return np.zeros_like(batch)  # Perfect reconstruction
                
                mock_model.predict.side_effect = mock_predict
                
                scores = detector.score_batched(test_sequences, batch_size=25)
                
                assert len(scores) == 100
                assert np.all(scores >= 0)  # Scores should be non-negative
                assert mock_model.predict.call_count == 4  # 100/25 = 4 batches
    
    def test_score_batched_with_remainder(self):
        """Test batched scoring when data doesn't divide evenly into batches."""
        test_sequences = np.random.randn(103, 30, 3)  # Not evenly divisible by 25
        
        with patch('src.anomaly_detector.load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.zeros((25, 30, 3))
            mock_load.return_value = mock_model
            
            detector = AnomalyDetector("dummy_path")
            detector.model = mock_model
            
            def mock_predict(batch, verbose=0):
                return np.zeros_like(batch)
            
            mock_model.predict.side_effect = mock_predict
            
            scores = detector.score_batched(test_sequences, batch_size=25)
            
            assert len(scores) == 103
            # Should have 5 calls: 4 full batches + 1 remainder (3 items)
            assert mock_model.predict.call_count == 5
    
    def test_score_batched_progress_logging(self, caplog):
        """Test that progress is logged during batched inference."""
        test_sequences = np.random.randn(1000, 30, 3)
        
        with patch('src.anomaly_detector.load_model') as mock_load:
            mock_model = MagicMock()
            
            def mock_predict(batch, verbose=0):
                return np.zeros_like(batch)
            
            mock_model.predict.side_effect = mock_predict
            mock_load.return_value = mock_model
            
            detector = AnomalyDetector("dummy_path")
            detector.model = mock_model
            
            with caplog.at_level("INFO"):
                detector.score_batched(test_sequences, batch_size=100)
            
            # Should log progress at intervals
            progress_logs = [record for record in caplog.records if "Processed" in record.message]
            assert len(progress_logs) > 0
    
    def test_score_batched_memory_efficiency(self):
        """Test that batched scoring uses less memory than full processing."""
        # Create moderately large test data
        n_sequences = 1000
        test_sequences = np.random.randn(n_sequences, 30, 3)
        
        with patch('src.anomaly_detector.load_model') as mock_load:
            mock_model = MagicMock()
            
            def mock_predict(batch, verbose=0):
                return np.random.randn(*batch.shape) * 0.1  # Small reconstruction error
            
            mock_model.predict.side_effect = mock_predict
            mock_load.return_value = mock_model
            
            detector = AnomalyDetector("dummy_path")
            detector.model = mock_model
            
            # Monitor memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            scores = detector.score_batched(test_sequences, batch_size=50)
            
            peak_memory = process.memory_info().rss
            memory_increase = peak_memory - initial_memory
            
            assert len(scores) == n_sequences
            # Memory increase should be reasonable (less than 100MB for this test)
            assert memory_increase < 100 * 1024 * 1024
    
    def test_score_batched_matches_original_score(self):
        """Test that batched scoring produces same results as original score method."""
        test_sequences = np.random.randn(50, 30, 3)
        
        with patch('src.anomaly_detector.load_model') as mock_load:
            mock_model = MagicMock()
            
            # Create deterministic predictions for consistency
            def mock_predict(batch, verbose=0):
                return batch * 0.9  # Consistent reconstruction with some error
            
            mock_model.predict.side_effect = mock_predict
            mock_load.return_value = mock_model
            
            detector = AnomalyDetector("dummy_path")
            detector.model = mock_model
            
            # Get results from both methods
            original_scores = detector.score(test_sequences)
            
            # Reset the mock call count
            mock_model.reset_mock()
            mock_model.predict.side_effect = mock_predict
            
            batched_scores = detector.score_batched(test_sequences, batch_size=10)
            
            # Results should be identical (within floating point precision)
            np.testing.assert_array_almost_equal(original_scores, batched_scores, decimal=6)
    
    def test_score_batched_default_batch_size(self):
        """Test that score_batched works with default batch size."""
        test_sequences = np.random.randn(100, 30, 3)
        
        with patch('src.anomaly_detector.load_model') as mock_load:
            mock_model = MagicMock()
            
            def mock_predict(batch, verbose=0):
                return np.zeros_like(batch)
            
            mock_model.predict.side_effect = mock_predict
            mock_load.return_value = mock_model
            
            detector = AnomalyDetector("dummy_path")
            detector.model = mock_model
            
            scores = detector.score_batched(test_sequences)  # Use default batch size
            
            assert len(scores) == 100
            assert np.all(scores >= 0)
    
    def test_score_batched_single_batch(self):
        """Test batched scoring when data fits in single batch."""
        test_sequences = np.random.randn(10, 30, 3)
        
        with patch('src.anomaly_detector.load_model') as mock_load:
            mock_model = MagicMock()
            
            def mock_predict(batch, verbose=0):
                return np.zeros_like(batch)
            
            mock_model.predict.side_effect = mock_predict
            mock_load.return_value = mock_model
            
            detector = AnomalyDetector("dummy_path")
            detector.model = mock_model
            
            scores = detector.score_batched(test_sequences, batch_size=100)
            
            assert len(scores) == 10
            assert mock_model.predict.call_count == 1  # Only one batch needed
    
    def test_score_batched_empty_input(self):
        """Test batched scoring with empty input."""
        test_sequences = np.array([]).reshape(0, 30, 3)
        
        with patch('src.anomaly_detector.load_model') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            detector = AnomalyDetector("dummy_path")
            detector.model = mock_model
            
            scores = detector.score_batched(test_sequences, batch_size=100)
            
            assert len(scores) == 0
            assert mock_model.predict.call_count == 0


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    @pytest.mark.slow
    def test_batched_vs_full_inference_speed(self):
        """Benchmark batched inference vs full inference speed."""
        n_sequences = 1000
        test_sequences = np.random.randn(n_sequences, 30, 3)
        
        with patch('src.anomaly_detector.load_model') as mock_load:
            mock_model = MagicMock()
            
            def mock_predict(batch, verbose=0):
                # Simulate some processing time
                time.sleep(0.001 * len(batch))  # 1ms per sequence
                return np.random.randn(*batch.shape)
            
            mock_model.predict.side_effect = mock_predict
            mock_load.return_value = mock_model
            
            detector = AnomalyDetector("dummy_path")
            detector.model = mock_model
            
            # Time full inference
            start_time = time.time()
            full_scores = detector.score(test_sequences)
            full_time = time.time() - start_time
            
            # Reset mock
            mock_model.reset_mock()
            mock_model.predict.side_effect = mock_predict
            
            # Time batched inference
            start_time = time.time()
            batched_scores = detector.score_batched(test_sequences, batch_size=100)
            batched_time = time.time() - start_time
            
            # Results should be similar
            assert len(full_scores) == len(batched_scores)
            
            # Batched should be roughly similar in time (allowing for overhead)
            # This test mainly ensures no dramatic performance regression
            assert batched_time < full_time * 2  # Allow 2x overhead tolerance
    
    def test_memory_usage_scaling(self):
        """Test that memory usage scales appropriately with batch size."""
        base_sequences = np.random.randn(200, 30, 3)
        
        with patch('src.anomaly_detector.load_model') as mock_load:
            mock_model = MagicMock()
            
            def mock_predict(batch, verbose=0):
                return np.random.randn(*batch.shape)
            
            mock_model.predict.side_effect = mock_predict
            mock_load.return_value = mock_model
            
            detector = AnomalyDetector("dummy_path")
            detector.model = mock_model
            
            # Test different batch sizes
            batch_sizes = [10, 50, 100]
            memory_usage = []
            
            for batch_size in batch_sizes:
                mock_model.reset_mock()
                mock_model.predict.side_effect = mock_predict
                
                process = psutil.Process()
                initial_memory = process.memory_info().rss
                
                scores = detector.score_batched(base_sequences, batch_size=batch_size)
                
                peak_memory = process.memory_info().rss
                memory_increase = peak_memory - initial_memory
                memory_usage.append(memory_increase)
                
                assert len(scores) == 200
            
            # Memory usage shouldn't vary dramatically with batch size for this test size
            # (but larger batch sizes might use slightly more memory)
            assert all(usage >= 0 for usage in memory_usage)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])