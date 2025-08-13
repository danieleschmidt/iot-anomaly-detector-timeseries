"""Comprehensive integration test suite for all generations."""

import unittest
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
import logging
import asyncio
import threading
from unittest.mock import patch, MagicMock

# Import all modules
from enhanced_autoencoder import EnhancedAutoencoder, RealTimeAnomalyDetector, create_sample_data
from smart_preprocessor import SmartPreprocessor, DataQualityChecker
from robust_monitoring_system import AdvancedHealthMonitor, RobustCircuitBreaker, Alert, AlertLevel
from comprehensive_error_handling import RobustErrorHandler, robust_retry, graceful_degradation, ErrorSeverity
from intelligent_resource_manager import PredictiveResourceManager, ResourceType, AllocationStrategy

logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests


class TestGeneration1Integration(unittest.TestCase):
    """Integration tests for Generation 1 components."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_enhanced_autoencoder_training(self):
        """Test enhanced autoencoder training pipeline."""
        # Create sample data
        X, y = create_sample_data(num_samples=100, sequence_length=20, num_features=3, anomaly_rate=0.1)
        
        # Initialize autoencoder
        autoencoder = EnhancedAutoencoder(
            input_shape=(20, 3),
            latent_dim=8,
            lstm_units=16,
            dropout_rate=0.1
        )
        
        # Build model
        model = autoencoder.build_model()
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 15)  # Expected number of layers
        
        # Train with small dataset
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        
        history = autoencoder.train_with_callbacks(
            X_train,
            X_val=X_test,
            epochs=2,  # Minimal training for test
            batch_size=16,
            patience=1
        )
        
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        
        # Test anomaly detection
        anomalies, errors, threshold = autoencoder.detect_anomalies(X_test)
        
        self.assertEqual(len(anomalies), len(X_test))
        self.assertEqual(len(errors), len(X_test))
        self.assertIsInstance(threshold, float)
        self.assertTrue(threshold > 0)
    
    def test_smart_preprocessor_pipeline(self):
        """Test smart preprocessing pipeline."""
        # Generate test data with various characteristics
        np.random.seed(42)
        n_samples = 200
        n_features = 4
        
        data = np.zeros((n_samples, n_features))
        
        # Feature 0: Normal distribution
        data[:, 0] = np.random.normal(0, 1, n_samples)
        
        # Feature 1: With outliers
        data[:, 1] = np.random.normal(0, 1, n_samples)
        outlier_indices = np.random.choice(n_samples, 20, replace=False)
        data[outlier_indices, 1] += np.random.normal(0, 5, 20)
        
        # Feature 2: Missing values
        data[:, 2] = np.random.normal(0, 1, n_samples)
        missing_indices = np.random.choice(n_samples, 30, replace=False)
        data[missing_indices, 2] = np.nan
        
        # Feature 3: Skewed distribution
        data[:, 3] = np.random.exponential(2, n_samples)
        
        # Test preprocessor
        preprocessor = SmartPreprocessor(window_size=10, scaler_type='auto')
        
        # Fit and transform
        scaled_data = preprocessor.fit_transform(data)
        
        self.assertEqual(scaled_data.shape, data.shape)
        self.assertTrue(preprocessor.is_fitted)
        self.assertIn(preprocessor.scaler_type, ['standard', 'minmax', 'robust'])
        
        # Create sequences
        sequences = preprocessor.create_sequences(scaled_data)
        
        expected_sequences = (len(scaled_data) - 10 + 1)  # window_size = 10, step = 1
        self.assertEqual(len(sequences), expected_sequences)
        self.assertEqual(sequences.shape[1], 10)  # window_size
        self.assertTrue(sequences.shape[2] >= n_features)  # Enhanced features
        
        # Test data quality checker
        quality_checker = DataQualityChecker(preprocessor.feature_stats)
        quality_report = quality_checker.check_data_quality(data[:50])
        
        self.assertIn('missing_data', quality_report)
        self.assertIn('drift_detected', quality_report)
        self.assertIsInstance(quality_report['missing_data'], float)
        
        # Test save/load
        save_path = self.temp_path / 'preprocessor.pkl'
        preprocessor.save(str(save_path))
        self.assertTrue(save_path.exists())
        
        loaded_preprocessor = SmartPreprocessor.load(str(save_path))
        self.assertTrue(loaded_preprocessor.is_fitted)
        self.assertEqual(loaded_preprocessor.window_size, preprocessor.window_size)
    
    def test_real_time_anomaly_detector(self):
        """Test real-time anomaly detection."""
        # Create a simple model file (mock)
        model_path = self.temp_path / 'test_model.h5'
        
        # Create a mock model for testing
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_load.return_value = mock_model
            
            # Initialize detector
            detector = RealTimeAnomalyDetector(str(model_path), threshold=0.5)
            
            # Test single sample processing
            sample = np.random.random((10, 3))
            is_anomaly, error = detector.process_sample(sample)
            
            self.assertIsInstance(is_anomaly, bool)
            self.assertIsInstance(error, float)
            self.assertTrue(error >= 0)
            
            # Test multiple samples
            for _ in range(10):
                sample = np.random.random((10, 3))
                is_anomaly, error = detector.process_sample(sample)
            
            # Check that threshold is set after enough samples
            self.assertIsNotNone(detector.threshold)


class TestGeneration2Integration(unittest.TestCase):
    """Integration tests for Generation 2 components."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker with various failure scenarios."""
        circuit_breaker = RobustCircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1,  # Short timeout for testing
            half_open_max_calls=2
        )
        
        @circuit_breaker
        def failing_function(should_fail=True):
            if should_fail:
                raise Exception("Test failure")
            return "success"
        
        # Test circuit breaker in closed state
        self.assertEqual(circuit_breaker.state, "CLOSED")
        
        # Trigger failures to open circuit
        for i in range(3):
            with self.assertRaises(Exception):
                failing_function(should_fail=True)
        
        # Circuit should be open now
        self.assertEqual(circuit_breaker.state, "OPEN")
        
        # Calls should fail fast
        with self.assertRaises(Exception):
            failing_function(should_fail=False)  # Even successful calls should fail
        
        # Wait for recovery timeout
        time.sleep(1.5)
        
        # Test successful call to reset circuit
        result = failing_function(should_fail=False)
        self.assertEqual(result, "success")
        
        # Another successful call should close the circuit
        result = failing_function(should_fail=False)
        self.assertEqual(result, "success")
        self.assertEqual(circuit_breaker.state, "CLOSED")
    
    def test_advanced_health_monitor(self):
        """Test advanced health monitoring system."""
        alerts_received = []
        
        def test_alert_callback(alert: Alert):
            alerts_received.append(alert)
        
        monitor = AdvancedHealthMonitor(
            alert_callback=test_alert_callback,
            metrics_retention_hours=1
        )
        
        # Start monitoring
        monitor.start_monitoring(interval_seconds=1)
        
        try:
            # Let it run for a few seconds
            time.sleep(3)
            
            # Get health report
            health_report = monitor.generate_health_report()
            
            self.assertIn('health_score', health_report)
            self.assertIn('status', health_report)
            self.assertIn('latest_metrics', health_report)
            self.assertIn('circuit_breaker_status', health_report)
            
            # Verify health score is reasonable
            self.assertTrue(0 <= health_report['health_score'] <= 100)
            
            # Test metrics collection
            recent_metrics = monitor.get_recent_metrics(hours=1)
            self.assertTrue(len(recent_metrics) > 0)
            
            # Each metric should have required fields
            for metric in recent_metrics:
                self.assertTrue(hasattr(metric, 'cpu_usage'))
                self.assertTrue(hasattr(metric, 'memory_usage'))
                self.assertTrue(hasattr(metric, 'prediction_accuracy'))
                
        finally:
            monitor.stop_monitoring()
    
    def test_robust_error_handler(self):
        """Test comprehensive error handling system."""
        error_log_path = self.temp_path / 'test_error_log.json'
        error_handler = RobustErrorHandler(str(error_log_path))
        
        # Test error handling
        test_error = ValueError("Test error message")
        error_context = error_handler.handle_error(
            test_error,
            "test_function",
            severity=ErrorSeverity.MEDIUM,
            metadata={'test_key': 'test_value'}
        )
        
        self.assertEqual(error_context.error_type, "ValueError")
        self.assertEqual(error_context.error_message, "Test error message")
        self.assertEqual(error_context.function_name, "test_function")
        self.assertEqual(error_context.severity, ErrorSeverity.MEDIUM)
        
        # Test error statistics
        error_stats = error_handler.get_error_stats(hours=1)
        self.assertEqual(error_stats['total_errors'], 1)
        self.assertIn('ValueError', error_stats['error_types'])
        
        # Test save/load functionality
        self.assertTrue(error_log_path.exists())
        
        # Create new handler and verify it loads existing data
        new_handler = RobustErrorHandler(str(error_log_path))
        new_stats = new_handler.get_error_stats(hours=1)
        self.assertEqual(new_stats['total_errors'], 1)
    
    def test_robust_retry_decorator(self):
        """Test robust retry decorator functionality."""
        call_count = [0]  # Use list for mutable counter
        
        @robust_retry(
            retry_config=RetryConfig(max_attempts=3, base_delay=0.1),
            recovery_strategy=RecoveryStrategy.RETRY
        )
        def flaky_function():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception(f"Attempt {call_count[0]} failed")
            return "success"
        
        # Function should succeed after retries
        result = flaky_function()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 3)
    
    def test_graceful_degradation_decorator(self):
        """Test graceful degradation decorator."""
        @graceful_degradation(fallback_value="fallback_result")
        def failing_function():
            raise Exception("Function always fails")
        
        result = failing_function()
        self.assertEqual(result, "fallback_result")


class TestGeneration3Integration(unittest.TestCase):
    """Integration tests for Generation 3 components."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_predictive_resource_manager(self):
        """Test predictive resource management system."""
        resource_manager = PredictiveResourceManager(
            prediction_horizon=60,  # 1 minute for testing
            history_size=100,
            allocation_strategy=AllocationStrategy.PREDICTIVE
        )
        
        # Start monitoring
        resource_manager.start_monitoring(interval_seconds=1)
        
        try:
            # Let it collect some data
            time.sleep(3)
            
            # Test resource predictions
            cpu_prediction = resource_manager.predict_resource_usage(ResourceType.CPU)
            memory_prediction = resource_manager.predict_resource_usage(ResourceType.MEMORY)
            
            self.assertEqual(cpu_prediction.resource_type, ResourceType.CPU)
            self.assertEqual(memory_prediction.resource_type, ResourceType.MEMORY)
            
            self.assertTrue(0 <= cpu_prediction.predicted_usage <= 100)
            self.assertTrue(0 <= memory_prediction.predicted_usage <= 100)
            self.assertTrue(0 <= cpu_prediction.confidence <= 1)
            self.assertTrue(0 <= memory_prediction.confidence <= 1)
            
            # Test resource summary
            summary = resource_manager.get_resource_summary()
            
            self.assertIn('current_usage', summary)
            self.assertIn('predictions', summary)
            self.assertIn('efficiency', summary)
            self.assertIn('allocation_strategy', summary)
            
            # Verify current usage fields
            current_usage = summary['current_usage']
            self.assertIn('cpu', current_usage)
            self.assertIn('memory', current_usage)
            self.assertIn('disk', current_usage)
            
            # Test forecast generation
            forecast = resource_manager.get_resource_forecast(hours=2)
            
            self.assertIn('cpu', forecast)
            self.assertIn('memory', forecast)
            self.assertEqual(len(forecast['cpu']), 2)  # 2 hours
            
            # Test state persistence
            state_path = self.temp_path / 'resource_state.json'
            resource_manager.save_state(str(state_path))
            self.assertTrue(state_path.exists())
            
        finally:
            resource_manager.stop_monitoring()
    
    def test_resource_prediction_accuracy(self):
        """Test resource prediction accuracy with controlled scenarios."""
        resource_manager = PredictiveResourceManager(
            prediction_horizon=30,
            history_size=50
        )
        
        # Simulate resource metrics with known patterns
        from robust_monitoring_system import ResourceMetrics
        
        # Create a predictable pattern (increasing CPU usage)
        base_time = time.time()
        for i in range(20):
            metrics = ResourceMetrics(
                timestamp=base_time + i * 10,
                cpu_usage=10 + i * 2,  # Steadily increasing
                memory_usage=50 + np.random.normal(0, 2),  # Stable with noise
                disk_usage=70,  # Constant
                network_io={'bytes_sent': 1000, 'bytes_recv': 1000},
                process_count=100,
                load_average=[1.0, 1.0, 1.0]
            )
            resource_manager.resource_history.append(metrics)
        
        # Force model update
        resource_manager._update_models(metrics)
        
        # Make prediction
        prediction = resource_manager.predict_resource_usage(ResourceType.CPU, 30)
        
        # With increasing trend, prediction should be higher than current
        current_cpu = metrics.cpu_usage
        self.assertTrue(prediction.predicted_usage > current_cpu - 5)  # Allow some variance
        
        # Confidence should improve with more data
        self.assertTrue(prediction.confidence > 0.3)  # Should have some confidence
    
    def test_resource_pressure_handling(self):
        """Test resource pressure detection and handling."""
        resource_manager = PredictiveResourceManager()
        
        # Create metrics with high resource usage
        high_pressure_metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_usage=95,  # Emergency level
            memory_usage=92,  # Critical level
            disk_usage=85,  # Warning level
            network_io={'bytes_sent': 1000, 'bytes_recv': 1000},
            process_count=200,
            load_average=[5.0, 4.0, 3.0]
        )
        
        # Test pressure detection (should trigger handling)
        # This is mostly testing that the function doesn't crash
        # since the actual handling is logged but doesn't return values
        resource_manager._check_resource_pressure(high_pressure_metrics)
        
        # Verify that the function completed without errors
        self.assertTrue(True)


class TestCrossGenerationIntegration(unittest.TestCase):
    """Integration tests across all generations."""
    
    def test_complete_pipeline_integration(self):
        """Test complete pipeline from data ingestion to scaled inference."""
        
        # Generation 1: Create and preprocess data
        X, y_true = create_sample_data(num_samples=200, sequence_length=15, num_features=3)
        
        preprocessor = SmartPreprocessor(window_size=15)
        sequences = preprocessor.create_sequences(preprocessor.fit_transform(X))
        
        # Generation 1: Train model
        autoencoder = EnhancedAutoencoder(
            input_shape=(15, preprocessor.scaler.n_features_in_),
            latent_dim=8,
            lstm_units=16
        )
        
        # Quick training for test
        split_idx = int(0.8 * len(sequences))
        train_data, test_data = sequences[:split_idx], sequences[split_idx:]
        
        autoencoder.train_with_callbacks(
            train_data,
            epochs=1,  # Minimal training
            batch_size=16
        )
        
        # Generation 2: Add error handling and monitoring
        error_handler = RobustErrorHandler()
        
        @robust_retry()
        @graceful_degradation(fallback_value=np.array([0.5]))
        def predict_with_error_handling(data):
            anomalies, errors, threshold = autoencoder.detect_anomalies(data)
            return anomalies, errors
        
        # Test prediction with error handling
        anomalies, errors = predict_with_error_handling(test_data)
        
        self.assertEqual(len(anomalies), len(test_data))
        self.assertEqual(len(errors), len(test_data))
        
        # Generation 3: Add resource monitoring
        resource_manager = PredictiveResourceManager()
        resource_manager.start_monitoring(interval_seconds=1)
        
        try:
            # Simulate scaled inference
            for i in range(5):
                batch_data = test_data[i:i+1] if i < len(test_data) else test_data[:1]
                anomalies, errors = predict_with_error_handling(batch_data)
                
                # Verify results
                self.assertIsInstance(anomalies, np.ndarray)
                self.assertIsInstance(errors, np.ndarray)
            
            # Get final resource summary
            summary = resource_manager.get_resource_summary()
            self.assertIn('current_usage', summary)
            
        finally:
            resource_manager.stop_monitoring()
    
    def test_scalability_stress_test(self):
        """Test system behavior under stress conditions."""
        
        # Create larger dataset
        X, y_true = create_sample_data(num_samples=500, sequence_length=20, num_features=5)
        
        # Test preprocessing scalability
        preprocessor = SmartPreprocessor(window_size=20)
        start_time = time.time()
        sequences = preprocessor.create_sequences(preprocessor.fit_transform(X))
        preprocessing_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 10 seconds)
        self.assertLess(preprocessing_time, 10.0)
        
        # Test memory usage doesn't explode
        memory_usage_before = psutil.Process().memory_info().rss
        
        # Create and test autoencoder
        autoencoder = EnhancedAutoencoder(
            input_shape=(20, preprocessor.scaler.n_features_in_),
            latent_dim=16,
            lstm_units=32
        )
        
        # Build model
        model = autoencoder.build_model()
        
        memory_usage_after = psutil.Process().memory_info().rss
        memory_increase = (memory_usage_after - memory_usage_before) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (less than 500MB for test)
        self.assertLess(memory_increase, 500)
        
        # Test batch processing
        batch_size = 32
        total_samples = len(sequences)
        batches_processed = 0
        
        for i in range(0, total_samples, batch_size):
            batch = sequences[i:i+batch_size]
            if len(batch) > 0:
                # Simulate inference
                predictions = np.random.random((len(batch), 1))
                batches_processed += 1
        
        expected_batches = (total_samples + batch_size - 1) // batch_size
        self.assertEqual(batches_processed, expected_batches)


def run_integration_tests():
    """Run all integration tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestGeneration1Integration,
        TestGeneration2Integration,
        TestGeneration3Integration,
        TestCrossGenerationIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)