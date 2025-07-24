import unittest
import numpy as np
import pandas as pd
import time
from datetime import datetime

from src.data_drift_detector import DataDriftDetector, DriftDetectionConfig, DriftResult, DriftAlert


class TestDriftDetectionConfig(unittest.TestCase):
    def test_config_initialization(self):
        """Test DriftDetectionConfig initialization with default values."""
        config = DriftDetectionConfig()
        
        self.assertEqual(config.ks_threshold, 0.05)
        self.assertEqual(config.psi_threshold, 0.25)
        self.assertEqual(config.wasserstein_threshold, 0.3)
        self.assertEqual(config.min_samples, 100)
        self.assertEqual(config.detection_window_size, 1000)
        self.assertTrue(config.enable_alerts)
        
    def test_config_custom_values(self):
        """Test DriftDetectionConfig with custom values."""
        config = DriftDetectionConfig(
            ks_threshold=0.01,
            psi_threshold=0.15,
            wasserstein_threshold=0.2,
            min_samples=50,
            detection_window_size=500,
            enable_alerts=False
        )
        
        self.assertEqual(config.ks_threshold, 0.01)
        self.assertEqual(config.psi_threshold, 0.15)
        self.assertEqual(config.wasserstein_threshold, 0.2)
        self.assertEqual(config.min_samples, 50)
        self.assertEqual(config.detection_window_size, 500)
        self.assertFalse(config.enable_alerts)
        
    def test_config_validation(self):
        """Test DriftDetectionConfig parameter validation."""
        # Test invalid threshold values
        with self.assertRaises(ValueError):
            DriftDetectionConfig(ks_threshold=-0.1)
            
        with self.assertRaises(ValueError):
            DriftDetectionConfig(ks_threshold=1.1)
            
        with self.assertRaises(ValueError):
            DriftDetectionConfig(min_samples=0)
            
        with self.assertRaises(ValueError):
            DriftDetectionConfig(detection_window_size=50)


class TestDriftResult(unittest.TestCase):
    def test_drift_result_creation(self):
        """Test DriftResult creation and properties."""
        result = DriftResult(
            timestamp=datetime.now(),
            ks_statistic=0.15,
            ks_p_value=0.02,
            psi_score=0.3,
            wasserstein_distance=0.25,
            drift_detected=True,
            feature_drifts={'feature1': True, 'feature2': False}
        )
        
        self.assertIsInstance(result.timestamp, datetime)
        self.assertEqual(result.ks_statistic, 0.15)
        self.assertEqual(result.ks_p_value, 0.02)
        self.assertEqual(result.psi_score, 0.3)
        self.assertEqual(result.wasserstein_distance, 0.25)
        self.assertTrue(result.drift_detected)
        self.assertEqual(result.feature_drifts['feature1'], True)
        self.assertEqual(result.feature_drifts['feature2'], False)
        
    def test_drift_result_serialization(self):
        """Test DriftResult JSON serialization."""
        result = DriftResult(
            timestamp=datetime.now(),
            ks_statistic=0.15,
            ks_p_value=0.02,
            psi_score=0.3,
            wasserstein_distance=0.25,
            drift_detected=True,
            feature_drifts={'feature1': True}
        )
        
        serialized = result.to_dict()
        self.assertIsInstance(serialized, dict)
        self.assertIn('timestamp', serialized)
        self.assertIn('drift_detected', serialized)
        self.assertIn('ks_statistic', serialized)


class TestDataDriftDetector(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = DriftDetectionConfig(min_samples=50, detection_window_size=200)
        self.detector = DataDriftDetector(config=self.config)
        
        # Create reference dataset
        np.random.seed(42)
        self.reference_data = np.random.normal(0, 1, (300, 5))
        self.reference_df = pd.DataFrame(
            self.reference_data,
            columns=[f'feature_{i}' for i in range(5)]
        )
        
    def test_detector_initialization(self):
        """Test DataDriftDetector initialization."""
        detector = DataDriftDetector()
        self.assertIsInstance(detector.config, DriftDetectionConfig)
        self.assertIsNone(detector.reference_data)
        self.assertEqual(len(detector.drift_history), 0)
        
    def test_set_reference_data(self):
        """Test setting reference/baseline data."""
        self.detector.set_reference_data(self.reference_df)
        
        self.assertIsNotNone(self.detector.reference_data)
        pd.testing.assert_frame_equal(self.detector.reference_data, self.reference_df)
        
    def test_set_reference_data_validation(self):
        """Test reference data validation."""
        # Test empty DataFrame
        with self.assertRaises(ValueError):
            self.detector.set_reference_data(pd.DataFrame())
            
        # Test DataFrame with insufficient samples
        small_df = pd.DataFrame(np.random.randn(10, 3))
        with self.assertRaises(ValueError):
            self.detector.set_reference_data(small_df)
        
        # Test DataFrame with non-numeric data
        mixed_df = pd.DataFrame({
            'numeric': [1, 2, 3] * 20,
            'categorical': ['A', 'B', 'C'] * 20
        })
        
        # Should work but only keep numeric columns
        self.detector.set_reference_data(mixed_df)
        self.assertIn('numeric', self.detector.reference_data.columns)
        self.assertNotIn('categorical', self.detector.reference_data.columns)
        
    def test_kolmogorov_smirnov_test(self):
        """Test Kolmogorov-Smirnov drift detection."""
        self.detector.set_reference_data(self.reference_df)
        
        # Test with similar data (no drift expected)
        np.random.seed(123)
        similar_data = np.random.normal(0, 1, (200, 5))
        similar_df = pd.DataFrame(similar_data, columns=[f'feature_{i}' for i in range(5)])
        
        ks_stats, p_values = self.detector._kolmogorov_smirnov_test(similar_df)
        
        self.assertEqual(len(ks_stats), 5)
        self.assertEqual(len(p_values), 5)
        self.assertTrue(all(p > 0.05 for p in p_values))  # No significant drift
        
        # Test with drifted data
        drifted_data = np.random.normal(2, 1.5, (200, 5))  # Different mean and std
        drifted_df = pd.DataFrame(drifted_data, columns=[f'feature_{i}' for i in range(5)])
        
        ks_stats_drift, p_values_drift = self.detector._kolmogorov_smirnov_test(drifted_df)
        
        self.assertTrue(any(p < 0.05 for p in p_values_drift))  # Significant drift expected
        
    def test_population_stability_index(self):
        """Test Population Stability Index calculation."""
        self.detector.set_reference_data(self.reference_df)
        
        # Test with similar data
        similar_data = np.random.normal(0, 1, (200, 5))
        similar_df = pd.DataFrame(similar_data, columns=[f'feature_{i}' for i in range(5)])
        
        psi_scores = self.detector._population_stability_index(similar_df)
        self.assertEqual(len(psi_scores), 5)
        self.assertTrue(all(score < 0.25 for score in psi_scores))  # Low PSI expected
        
        # Test with drifted data
        drifted_data = np.random.normal(3, 2, (200, 5))
        drifted_df = pd.DataFrame(drifted_data, columns=[f'feature_{i}' for i in range(5)])
        
        psi_scores_drift = self.detector._population_stability_index(drifted_df)
        self.assertTrue(any(score > 0.25 for score in psi_scores_drift))  # High PSI expected
        
    def test_wasserstein_distance(self):
        """Test Wasserstein distance calculation."""
        self.detector.set_reference_data(self.reference_df)
        
        # Test with similar data
        similar_data = np.random.normal(0, 1, (200, 5))
        similar_df = pd.DataFrame(similar_data, columns=[f'feature_{i}' for i in range(5)])
        
        wasserstein_distances = self.detector._wasserstein_distance(similar_df)
        self.assertEqual(len(wasserstein_distances), 5)
        self.assertTrue(all(dist < 0.3 for dist in wasserstein_distances))  # Low distance
        
        # Test with drifted data
        drifted_data = np.random.normal(5, 3, (200, 5))
        drifted_df = pd.DataFrame(drifted_data, columns=[f'feature_{i}' for i in range(5)])
        
        wasserstein_distances_drift = self.detector._wasserstein_distance(drifted_df)
        self.assertTrue(any(dist > 0.3 for dist in wasserstein_distances_drift))  # High distance
        
    def test_detect_drift_no_reference(self):
        """Test drift detection without reference data."""
        new_data = pd.DataFrame(np.random.randn(100, 3))
        
        with self.assertRaises(ValueError):
            self.detector.detect_drift(new_data)
            
    def test_detect_drift_insufficient_samples(self):
        """Test drift detection with insufficient samples."""
        self.detector.set_reference_data(self.reference_df)
        
        # Too few samples
        small_data = pd.DataFrame(np.random.randn(10, 5))
        
        with self.assertRaises(ValueError):
            self.detector.detect_drift(small_data)
            
    def test_detect_drift_comprehensive(self):
        """Test comprehensive drift detection."""
        self.detector.set_reference_data(self.reference_df)
        
        # Test with no drift
        no_drift_data = np.random.normal(0, 1, (150, 5))
        no_drift_df = pd.DataFrame(no_drift_data, columns=[f'feature_{i}' for i in range(5)])
        
        result = self.detector.detect_drift(no_drift_df)
        
        self.assertIsInstance(result, DriftResult)
        self.assertFalse(result.drift_detected or 
                        any(result.feature_drifts.values()))  # Might have some false positives
        
        # Test with drift
        drift_data = np.random.normal(3, 2, (150, 5))
        drift_df = pd.DataFrame(drift_data, columns=[f'feature_{i}' for i in range(5)])
        
        result_drift = self.detector.detect_drift(drift_df)
        
        self.assertTrue(result_drift.drift_detected)
        self.assertTrue(any(result_drift.feature_drifts.values()))
        
    def test_drift_history_tracking(self):
        """Test drift detection history tracking."""
        self.detector.set_reference_data(self.reference_df)
        
        # Perform multiple drift detections
        for i in range(3):
            test_data = np.random.normal(i, 1, (100, 5))
            test_df = pd.DataFrame(test_data, columns=[f'feature_{i}' for i in range(5)])
            
            self.detector.detect_drift(test_df)
        
        self.assertEqual(len(self.detector.drift_history), 3)
        self.assertTrue(all(isinstance(result, DriftResult) 
                          for result in self.detector.drift_history))
        
    def test_get_drift_summary(self):
        """Test drift detection summary generation."""
        self.detector.set_reference_data(self.reference_df)
        
        # Perform some drift detections
        for i in range(5):
            test_data = np.random.normal(i * 0.5, 1, (100, 5))
            test_df = pd.DataFrame(test_data, columns=[f'feature_{i}' for i in range(5)])
            
            self.detector.detect_drift(test_df)
        
        summary = self.detector.get_drift_summary()
        
        self.assertIn('total_detections', summary)
        self.assertIn('drift_rate', summary)
        self.assertIn('feature_drift_rates', summary)
        self.assertIn('recent_drift_trend', summary)
        
        self.assertEqual(summary['total_detections'], 5)
        self.assertIsInstance(summary['drift_rate'], float)
        
    def test_reset_detector(self):
        """Test detector reset functionality."""
        self.detector.set_reference_data(self.reference_df)
        
        # Perform some drift detections
        test_data = pd.DataFrame(np.random.randn(100, 5))
        self.detector.detect_drift(test_data)
        
        # Verify state before reset
        self.assertIsNotNone(self.detector.reference_data)
        self.assertEqual(len(self.detector.drift_history), 1)
        
        # Reset and verify
        self.detector.reset()
        
        self.assertIsNone(self.detector.reference_data)
        self.assertEqual(len(self.detector.drift_history), 0)


class TestDriftAlert(unittest.TestCase):
    def test_drift_alert_creation(self):
        """Test DriftAlert creation and properties."""
        alert = DriftAlert(
            severity='HIGH',
            message='Significant drift detected',
            feature_alerts={'feature1': 'High PSI score'},
            timestamp=datetime.now(),
            recommendations=['Retrain model', 'Investigate data source']
        )
        
        self.assertEqual(alert.severity, 'HIGH')
        self.assertEqual(alert.message, 'Significant drift detected')
        self.assertIn('feature1', alert.feature_alerts)
        self.assertIn('Retrain model', alert.recommendations)
        
    def test_alert_generation_from_drift_result(self):
        """Test alert generation from drift detection results."""
        drift_result = DriftResult(
            timestamp=datetime.now(),
            ks_statistic=0.8,
            ks_p_value=0.001,
            psi_score=0.4,
            wasserstein_distance=0.6,
            drift_detected=True,
            feature_drifts={'feature1': True, 'feature2': False}
        )
        
        alert = DriftAlert.from_drift_result(drift_result)
        
        self.assertIsInstance(alert, DriftAlert)
        self.assertIn('drift detected', alert.message.lower())
        self.assertTrue(len(alert.recommendations) > 0)


class TestDriftDetectionIntegration(unittest.TestCase):
    def test_drift_detection_workflow(self):
        """Test complete drift detection workflow."""
        # Initialize detector
        detector = DataDriftDetector()
        
        # Set reference data
        reference_data = np.random.normal(0, 1, (500, 4))
        reference_df = pd.DataFrame(reference_data, columns=['A', 'B', 'C', 'D'])
        detector.set_reference_data(reference_df)
        
        # Simulate monitoring over time
        results = []
        for week in range(4):
            # Gradually introduce drift
            drift_factor = week * 0.5
            new_data = np.random.normal(drift_factor, 1 + drift_factor * 0.2, (200, 4))
            new_df = pd.DataFrame(new_data, columns=['A', 'B', 'C', 'D'])
            
            result = detector.detect_drift(new_df)
            results.append(result)
        
        # Verify drift is detected as it increases
        drift_counts = [sum(r.feature_drifts.values()) for r in results]
        
        # Later weeks should generally have more drift
        self.assertTrue(drift_counts[-1] >= drift_counts[0])
        
        # Verify summary
        summary = detector.get_drift_summary()
        self.assertEqual(summary['total_detections'], 4)
        
    def test_performance_with_large_datasets(self):
        """Test drift detection performance with large datasets."""
        detector = DataDriftDetector()
        
        # Large reference dataset
        large_reference = np.random.normal(0, 1, (10000, 10))
        large_reference_df = pd.DataFrame(large_reference)
        
        start_time = time.time()
        detector.set_reference_data(large_reference_df)
        
        # Large test dataset
        large_test = np.random.normal(0.5, 1.2, (5000, 10))
        large_test_df = pd.DataFrame(large_test)
        
        result = detector.detect_drift(large_test_df)
        detection_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(detection_time, 30.0)  # 30 seconds max
        self.assertIsInstance(result, DriftResult)


if __name__ == "__main__":
    unittest.main()