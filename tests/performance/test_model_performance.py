"""
Performance tests for model training and inference.
"""

import pytest
import numpy as np
import pandas as pd
import time
from pathlib import Path
from unittest.mock import patch, Mock

from src.data_preprocessor import DataPreprocessor
from src.anomaly_detector import AnomalyDetector


@pytest.mark.performance
class TestModelPerformance:
    """Performance tests for model operations."""
    
    def test_preprocessing_performance(self, performance_test_data):
        """Test data preprocessing performance with large datasets."""
        
        # Test with different window sizes
        window_sizes = [10, 30, 50, 100]
        performance_results = {}
        
        for window_size in window_sizes:
            start_time = time.time()
            
            preprocessor = DataPreprocessor(
                window_size=window_size,
                step=1,
                scaler_type="standard"
            )
            
            X_train, _ = preprocessor.fit_transform(performance_test_data)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            performance_results[window_size] = {
                "time": processing_time,
                "samples_per_second": len(performance_test_data) / processing_time,
                "output_shape": X_train.shape
            }
            
            # Performance requirements
            assert processing_time < 30.0  # Should complete within 30 seconds
            assert performance_results[window_size]["samples_per_second"] > 100
        
        # Window size should not drastically affect performance
        time_ratios = []
        for i in range(1, len(window_sizes)):
            ratio = performance_results[window_sizes[i]]["time"] / performance_results[window_sizes[i-1]]["time"]
            time_ratios.append(ratio)
        
        # Time should not increase more than 3x with larger windows
        assert all(ratio < 3.0 for ratio in time_ratios)
    
    @pytest.mark.slow
    def test_batch_inference_performance(self, temp_dir):
        """Test batch inference performance."""
        
        # Create test data
        n_samples = 5000
        n_features = 8
        
        df = pd.DataFrame({
            f'sensor_{i}': np.random.normal(0, 1, n_samples) 
            for i in range(n_features)
        })
        
        # Setup mock detector
        model_path = temp_dir / "perf_model.h5"
        scaler_path = temp_dir / "perf_scaler.pkl"
        
        model_path.touch()
        scaler_path.touch()
        
        detector = AnomalyDetector(str(model_path), str(scaler_path))
        
        # Mock model with realistic performance characteristics
        mock_model = Mock()
        mock_model.predict.side_effect = lambda x, **kwargs: x + np.random.normal(0, 0.1, x.shape)
        detector.model = mock_model
        detector.window_size = 30
        
        # Mock scaler
        from sklearn.preprocessing import StandardScaler
        detector.scaler = StandardScaler().fit(df.iloc[:1000].values)
        
        # Test different batch sizes
        batch_sizes = [100, 500, 1000, 2000]
        batch_performance = {}
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            # Process in batches
            total_predictions = []
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                predictions = detector.predict_from_dataframe(batch, threshold=0.5)
                total_predictions.extend(predictions)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            batch_performance[batch_size] = {
                "time": total_time,
                "throughput": len(df) / total_time,
                "predictions": len(total_predictions)
            }
            
            # Performance requirements
            assert total_time < 60.0  # Should complete within 1 minute
            assert batch_performance[batch_size]["throughput"] > 500  # At least 500 samples/sec
        
        # Larger batches should be more efficient
        throughputs = [batch_performance[size]["throughput"] for size in batch_sizes]
        assert throughputs[-1] >= throughputs[0]  # Largest batch should be fastest
    
    def test_memory_usage_scaling(self, temp_dir):
        """Test memory usage with increasing data sizes."""
        
        data_sizes = [1000, 5000, 10000]
        memory_metrics = {}
        
        for size in data_sizes:
            # Create data of specific size
            df = pd.DataFrame({
                f'sensor_{i}': np.random.normal(0, 1, size) 
                for i in range(5)
            })
            
            # Measure preprocessing memory efficiency
            preprocessor = DataPreprocessor(window_size=30, step=1)
            
            start_time = time.time()
            X_train, _ = preprocessor.fit_transform(df)
            processing_time = time.time() - start_time
            
            memory_metrics[size] = {
                "processing_time": processing_time,
                "output_size": X_train.nbytes,
                "efficiency": X_train.nbytes / (processing_time * 1024 * 1024)  # MB/sec
            }
            
            # Memory should scale reasonably
            assert processing_time < size / 500  # At least 500 samples/sec
        
        # Check that efficiency doesn't degrade significantly with size
        efficiencies = [memory_metrics[size]["efficiency"] for size in data_sizes]
        min_efficiency = min(efficiencies)
        max_efficiency = max(efficiencies)
        
        # Efficiency should not vary by more than 50%
        assert min_efficiency / max_efficiency > 0.5
    
    @pytest.mark.slow
    def test_concurrent_inference(self, temp_dir):
        """Test performance under concurrent inference requests."""
        
        import threading
        import queue
        
        # Setup detector
        model_path = temp_dir / "concurrent_model.h5"
        scaler_path = temp_dir / "concurrent_scaler.pkl"
        
        model_path.touch()
        scaler_path.touch()
        
        detector = AnomalyDetector(str(model_path), str(scaler_path))
        
        # Mock fast model
        mock_model = Mock()
        mock_model.predict.side_effect = lambda x, **kwargs: x * 1.1
        detector.model = mock_model
        detector.window_size = 20
        
        from sklearn.preprocessing import StandardScaler
        sample_data = pd.DataFrame({
            f'sensor_{i}': np.random.normal(0, 1, 1000) 
            for i in range(3)
        })
        detector.scaler = StandardScaler().fit(sample_data.values)
        
        # Test concurrent requests
        num_threads = 5
        requests_per_thread = 10
        results_queue = queue.Queue()
        
        def worker():
            """Worker function for concurrent testing."""
            thread_results = []
            
            for _ in range(requests_per_thread):
                # Create small batch for each request
                test_batch = sample_data.iloc[:100].copy()
                
                start_time = time.time()
                predictions = detector.predict_from_dataframe(test_batch, threshold=0.5)
                end_time = time.time()
                
                thread_results.append({
                    "time": end_time - start_time,
                    "predictions": len(predictions)
                })
            
            results_queue.put(thread_results)
        
        # Start concurrent threads
        threads = []
        overall_start = time.time()
        
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join()
        
        overall_time = time.time() - overall_start
        
        # Collect results
        all_results = []
        while not results_queue.empty():
            thread_results = results_queue.get()
            all_results.extend(thread_results)
        
        # Performance validation
        total_requests = num_threads * requests_per_thread
        avg_request_time = sum(r["time"] for r in all_results) / len(all_results)
        requests_per_second = total_requests / overall_time
        
        assert len(all_results) == total_requests
        assert avg_request_time < 1.0  # Average request should be under 1 second
        assert requests_per_second > 10  # At least 10 requests/sec overall
        
        # No request should take too long
        max_request_time = max(r["time"] for r in all_results)
        assert max_request_time < 5.0  # No request over 5 seconds
    
    def test_model_loading_performance(self, temp_dir):
        """Test model loading and initialization performance."""
        
        # Create mock model files
        model_path = temp_dir / "load_test_model.h5"
        scaler_path = temp_dir / "load_test_scaler.pkl"
        
        model_path.touch()
        scaler_path.touch()
        
        # Test cold start performance
        load_times = []
        
        for _ in range(5):
            start_time = time.time()
            
            # Mock the actual file loading
            with patch('tensorflow.keras.models.load_model') as mock_load_model, \
                 patch('joblib.load') as mock_load_scaler:
                
                mock_load_model.return_value = Mock()
                mock_load_scaler.return_value = Mock()
                
                detector = AnomalyDetector(str(model_path), str(scaler_path))
                detector._load_model()
                detector._load_scaler()
            
            load_time = time.time() - start_time
            load_times.append(load_time)
        
        avg_load_time = sum(load_times) / len(load_times)
        max_load_time = max(load_times)
        
        # Model loading should be fast
        assert avg_load_time < 2.0  # Average under 2 seconds
        assert max_load_time < 5.0  # Maximum under 5 seconds
        
        # Loading times should be consistent
        load_time_std = np.std(load_times)
        assert load_time_std < 1.0  # Standard deviation under 1 second


@pytest.mark.performance
class TestScalabilityLimits:
    """Test system behavior at scale limits."""
    
    @pytest.mark.slow
    def test_maximum_window_size(self):
        """Test performance with very large window sizes."""
        
        # Test with increasingly large windows
        window_sizes = [100, 500, 1000]
        n_samples = 10000
        n_features = 10
        
        df = pd.DataFrame({
            f'sensor_{i}': np.random.normal(0, 1, n_samples) 
            for i in range(n_features)
        })
        
        for window_size in window_sizes:
            # Skip if window is larger than data
            if window_size >= n_samples:
                continue
            
            start_time = time.time()
            
            preprocessor = DataPreprocessor(
                window_size=window_size,
                step=window_size // 4,  # Use larger steps for efficiency
                scaler_type="standard"
            )
            
            X_train, _ = preprocessor.fit_transform(df)
            processing_time = time.time() - start_time
            
            # Should handle large windows within reasonable time
            assert processing_time < 120.0  # 2 minutes max
            assert X_train.shape[0] > 0
            
            # Memory usage should not be excessive
            memory_mb = X_train.nbytes / (1024 * 1024)
            assert memory_mb < 1000  # Less than 1GB
    
    def test_maximum_features(self):
        """Test performance with many features."""
        
        feature_counts = [10, 50, 100]
        n_samples = 5000
        
        for n_features in feature_counts:
            df = pd.DataFrame({
                f'sensor_{i}': np.random.normal(0, 1, n_samples) 
                for i in range(n_features)
            })
            
            start_time = time.time()
            
            preprocessor = DataPreprocessor(
                window_size=30,
                step=5,
                scaler_type="standard"
            )
            
            X_train, _ = preprocessor.fit_transform(df)
            processing_time = time.time() - start_time
            
            # Should handle many features
            assert processing_time < 60.0  # 1 minute max
            assert X_train.shape[2] == n_features
            
            # Performance should not degrade linearly with features
            samples_per_second = n_samples / processing_time
            assert samples_per_second > 100
    
    def test_stress_conditions(self, temp_dir):
        """Test system behavior under stress conditions."""
        
        # Simulate high-frequency data
        n_samples = 20000  # Simulating 20k samples
        n_features = 20
        
        # Generate data with challenging characteristics
        df = pd.DataFrame({
            f'sensor_{i}': np.random.normal(0, 1, n_samples) + 
                          np.random.exponential(0.1, n_samples) *  # Outliers
                          np.random.choice([-1, 1], n_samples)
            for i in range(n_features)
        })
        
        # Add missing values
        mask = np.random.random((n_samples, n_features)) < 0.05  # 5% missing
        df_with_missing = df.copy()
        df_with_missing[mask] = np.nan
        
        # Test preprocessing under stress
        start_time = time.time()
        
        preprocessor = DataPreprocessor(
            window_size=50,
            step=1,
            scaler_type="robust",  # Use robust scaler for outliers
            enable_validation=True,
            validation_level="moderate"
        )
        
        try:
            X_train, _ = preprocessor.fit_transform(df_with_missing)
            stress_test_passed = True
            processing_time = time.time() - start_time
        except Exception as e:
            stress_test_passed = False
            processing_time = float('inf')
        
        # System should handle stress conditions gracefully
        assert stress_test_passed
        assert processing_time < 300.0  # 5 minutes max for stress test
        assert X_train.shape[0] > 0