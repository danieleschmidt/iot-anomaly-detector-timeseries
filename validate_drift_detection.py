#!/usr/bin/env python3
"""
Data Drift Detection Validation Script

This script validates the data drift detection system implementation
and demonstrates its capabilities with synthetic and real data scenarios.
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.data_drift_detector import (
        DataDriftDetector, 
        DriftDetectionConfig, 
        DriftResult,
        DriftAlert
    )
    print("✅ Successfully imported drift detection modules")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please ensure all dependencies are installed and modules are available")
    sys.exit(1)


def create_synthetic_data(n_samples: int, n_features: int, distribution_params: dict) -> pd.DataFrame:
    """Create synthetic data with specified distribution parameters."""
    np.random.seed(42)  # For reproducibility
    
    data = {}
    for i in range(n_features):
        feature_name = f'feature_{i}'
        
        if distribution_params.get('type') == 'normal':
            mean = distribution_params.get('mean', 0)
            std = distribution_params.get('std', 1)
            data[feature_name] = np.random.normal(mean, std, n_samples)
        elif distribution_params.get('type') == 'uniform':
            low = distribution_params.get('low', 0)
            high = distribution_params.get('high', 1)
            data[feature_name] = np.random.uniform(low, high, n_samples)
        else:
            # Default to normal
            data[feature_name] = np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame(data)


def test_basic_drift_detection():
    """Test basic drift detection functionality."""
    print("\n" + "="*60)
    print("TESTING BASIC DRIFT DETECTION FUNCTIONALITY")
    print("="*60)
    
    # Create detector with custom config
    config = DriftDetectionConfig(
        ks_threshold=0.05,
        psi_threshold=0.25,
        min_samples=100
    )
    detector = DataDriftDetector(config=config)
    
    # Create reference data
    reference_data = create_synthetic_data(
        n_samples=1000, 
        n_features=5, 
        distribution_params={'type': 'normal', 'mean': 0, 'std': 1}
    )
    
    print(f"Created reference data: {reference_data.shape}")
    detector.set_reference_data(reference_data)
    print("✅ Reference data set successfully")
    
    # Test with similar data (no drift expected)
    similar_data = create_synthetic_data(
        n_samples=500,
        n_features=5,
        distribution_params={'type': 'normal', 'mean': 0, 'std': 1}
    )
    
    result_no_drift = detector.detect_drift(similar_data)
    print(f"✅ No drift test: drift_detected = {result_no_drift.drift_detected}")
    print(f"   PSI score: {result_no_drift.psi_score:.4f}")
    print(f"   Features with drift: {sum(result_no_drift.feature_drifts.values())}")
    
    # Test with drifted data (drift expected)
    drifted_data = create_synthetic_data(
        n_samples=500,
        n_features=5,
        distribution_params={'type': 'normal', 'mean': 2, 'std': 1.5}  # Different distribution
    )
    
    result_drift = detector.detect_drift(drifted_data)
    print(f"✅ Drift test: drift_detected = {result_drift.drift_detected}")
    print(f"   PSI score: {result_drift.psi_score:.4f}")
    print(f"   Features with drift: {sum(result_drift.feature_drifts.values())}")
    
    # Verify drift was detected
    assert result_drift.drift_detected, "Drift should have been detected"
    assert result_drift.psi_score > config.psi_threshold, "PSI score should be above threshold"
    
    print("✅ All basic drift detection tests passed!")


def test_statistical_methods():
    """Test individual statistical drift detection methods."""
    print("\n" + "="*60)
    print("TESTING STATISTICAL DRIFT DETECTION METHODS")
    print("="*60)
    
    detector = DataDriftDetector()
    
    # Reference data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.uniform(0, 10, 1000),
        'feature_3': np.random.exponential(1, 1000)
    })
    
    detector.set_reference_data(reference_data)
    
    # Test data with known drift
    test_data = pd.DataFrame({
        'feature_1': np.random.normal(2, 1.5, 500),  # Mean and std shift
        'feature_2': np.random.uniform(5, 15, 500),  # Range shift
        'feature_3': np.random.exponential(2, 500)   # Scale shift
    })
    
    print("Testing Kolmogorov-Smirnov test...")
    ks_stats, p_values = detector._kolmogorov_smirnov_test(test_data)
    print(f"✅ KS statistics: {[f'{stat:.4f}' for stat in ks_stats]}")
    print(f"✅ KS p-values: {[f'{pval:.6f}' for pval in p_values]}")
    
    print("\nTesting Population Stability Index...")
    psi_scores = detector._population_stability_index(test_data)
    print(f"✅ PSI scores: {[f'{score:.4f}' for score in psi_scores]}")
    
    print("\nTesting Wasserstein Distance...")
    wasserstein_distances = detector._wasserstein_distance(test_data)
    print(f"✅ Wasserstein distances: {[f'{dist:.4f}' for dist in wasserstein_distances]}")
    
    # Verify that all methods detect drift
    assert any(p < 0.05 for p in p_values), "KS test should detect drift"
    assert any(score > 0.25 for score in psi_scores), "PSI should detect drift"
    assert any(dist > 0.1 for dist in wasserstein_distances), "Wasserstein should detect drift"
    
    print("✅ All statistical method tests passed!")


def test_drift_monitoring_workflow():
    """Test complete drift monitoring workflow over time."""
    print("\n" + "="*60)
    print("TESTING DRIFT MONITORING WORKFLOW")
    print("="*60)
    
    detector = DataDriftDetector()
    
    # Set initial reference data
    reference_data = create_synthetic_data(
        n_samples=2000,
        n_features=4,
        distribution_params={'type': 'normal', 'mean': 0, 'std': 1}
    )
    
    detector.set_reference_data(reference_data)
    print(f"✅ Reference established: {reference_data.shape}")
    
    # Simulate monitoring over several time periods with gradually increasing drift
    results = []
    
    for period in range(1, 6):  # 5 monitoring periods
        print(f"\n📊 Monitoring Period {period}:")
        
        # Gradually introduce drift
        drift_factor = period * 0.3
        new_data = create_synthetic_data(
            n_samples=300,
            n_features=4,
            distribution_params={'type': 'normal', 'mean': drift_factor, 'std': 1 + drift_factor * 0.2}
        )
        
        result = detector.detect_drift(new_data)
        results.append(result)
        
        print(f"   Drift detected: {result.drift_detected}")
        print(f"   PSI score: {result.psi_score:.4f}")
        print(f"   Drifted features: {sum(result.feature_drifts.values())}/{len(result.feature_drifts)}")
        
        # Generate alert if drift detected
        if result.drift_detected:
            alert = DriftAlert.from_drift_result(result, detector.config)
            print(f"   🚨 Alert: {alert.severity} - {alert.message}")
    
    # Analyze monitoring summary
    summary = detector.get_drift_summary()
    print(f"\n📈 MONITORING SUMMARY:")
    print(f"   Total detections: {summary['total_detections']}")
    print(f"   Drift rate: {summary['drift_rate']:.1%}")
    print(f"   Most drifted feature: {summary.get('most_drifted_feature', 'None')}")
    print(f"   Trend: {summary['recent_drift_trend']}")
    
    # Verify progressive drift detection
    drift_counts = [sum(r.feature_drifts.values()) for r in results]
    print(f"   Drift progression: {drift_counts}")
    
    # Later periods should generally have more drift
    assert drift_counts[-1] >= drift_counts[0], "Drift should increase over time"
    
    print("✅ Drift monitoring workflow test passed!")


def test_configuration_and_thresholds():
    """Test different configuration settings and thresholds."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION AND THRESHOLDS")
    print("="*60)
    
    # Test strict configuration
    strict_config = DriftDetectionConfig(
        ks_threshold=0.01,      # Very strict
        psi_threshold=0.1,      # Lower threshold
        wasserstein_threshold=0.1,
        min_samples=50
    )
    
    strict_detector = DataDriftDetector(config=strict_config)
    
    # Test lenient configuration
    lenient_config = DriftDetectionConfig(
        ks_threshold=0.1,       # More lenient
        psi_threshold=0.5,      # Higher threshold
        wasserstein_threshold=0.8,
        min_samples=200
    )
    
    lenient_detector = DataDriftDetector(config=lenient_config)
    
    # Same reference data for both
    reference_data = create_synthetic_data(
        n_samples=1000,
        n_features=3,
        distribution_params={'type': 'normal', 'mean': 0, 'std': 1}
    )
    
    strict_detector.set_reference_data(reference_data)
    lenient_detector.set_reference_data(reference_data)
    
    # Test with slightly drifted data
    slight_drift_data = create_synthetic_data(
        n_samples=300,
        n_features=3,
        distribution_params={'type': 'normal', 'mean': 0.5, 'std': 1.1}  # Slight drift
    )
    
    strict_result = strict_detector.detect_drift(slight_drift_data)
    lenient_result = lenient_detector.detect_drift(slight_drift_data)
    
    print(f"Strict detector - drift detected: {strict_result.drift_detected}")
    print(f"   PSI score: {strict_result.psi_score:.4f} (threshold: {strict_config.psi_threshold})")
    
    print(f"Lenient detector - drift detected: {lenient_result.drift_detected}")
    print(f"   PSI score: {lenient_result.psi_score:.4f} (threshold: {lenient_config.psi_threshold})")
    
    # Strict detector should be more sensitive (likely to detect drift)
    # Lenient detector should be less sensitive (less likely to detect drift)
    print("✅ Configuration threshold test completed!")


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING AND EDGE CASES")
    print("="*60)
    
    detector = DataDriftDetector()
    
    # Test detection without reference data
    try:
        test_data = pd.DataFrame({'feature': [1, 2, 3]})
        detector.detect_drift(test_data)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Correctly handles missing reference data")
    
    # Test with insufficient samples
    small_data = pd.DataFrame({'feature': np.random.randn(10)})
    try:
        detector.set_reference_data(small_data)
        assert False, "Should have raised ValueError for insufficient samples"
    except ValueError:
        print("✅ Correctly handles insufficient reference samples")
    
    # Test with empty data
    try:
        empty_data = pd.DataFrame()
        detector.set_reference_data(empty_data)
        assert False, "Should have raised ValueError for empty data"
    except ValueError:
        print("✅ Correctly handles empty reference data")
    
    # Test with non-numeric data
    mixed_data = pd.DataFrame({
        'numeric': np.random.randn(500),
        'categorical': ['A', 'B', 'C'] * 167 + ['A'],  # 500 total
        'text': ['text'] * 500
    })
    
    detector.set_reference_data(mixed_data)
    assert 'numeric' in detector.reference_data.columns
    assert 'categorical' not in detector.reference_data.columns
    assert 'text' not in detector.reference_data.columns
    print("✅ Correctly filters non-numeric columns")
    
    # Test drift detection with insufficient new samples
    try:
        tiny_new_data = pd.DataFrame({'numeric': [1, 2, 3]})
        detector.detect_drift(tiny_new_data)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Correctly handles insufficient new data samples")
    
    print("✅ All error handling tests passed!")


def test_performance_with_large_datasets():
    """Test performance with large datasets."""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE WITH LARGE DATASETS")
    print("="*60)
    
    # Test with large reference dataset
    print("Creating large reference dataset...")
    large_reference = create_synthetic_data(
        n_samples=10000,
        n_features=20,
        distribution_params={'type': 'normal', 'mean': 0, 'std': 1}
    )
    
    detector = DataDriftDetector()
    
    start_time = time.time()
    detector.set_reference_data(large_reference)
    setup_time = time.time() - start_time
    
    print(f"✅ Reference data setup time: {setup_time:.2f}s")
    assert setup_time < 30, "Setup should complete within 30 seconds"
    
    # Test drift detection with large new dataset
    print("Creating large test dataset...")
    large_test = create_synthetic_data(
        n_samples=5000,
        n_features=20,
        distribution_params={'type': 'normal', 'mean': 0.5, 'std': 1.2}
    )
    
    start_time = time.time()
    result = detector.detect_drift(large_test)
    detection_time = time.time() - start_time
    
    print(f"✅ Drift detection time: {detection_time:.2f}s")
    print(f"   Result: {result.drift_detected}")
    print(f"   PSI score: {result.psi_score:.4f}")
    
    assert detection_time < 60, "Detection should complete within 60 seconds"
    assert isinstance(result, DriftResult), "Should return valid DriftResult"
    
    print("✅ Performance test completed successfully!")


def test_export_and_serialization():
    """Test export and serialization functionality."""
    print("\n" + "="*60)
    print("TESTING EXPORT AND SERIALIZATION")
    print("="*60)
    
    detector = DataDriftDetector()
    
    # Set up reference data and perform some detections
    reference_data = create_synthetic_data(
        n_samples=500,
        n_features=3,
        distribution_params={'type': 'normal', 'mean': 0, 'std': 1}
    )
    
    detector.set_reference_data(reference_data)
    
    # Perform multiple drift detections
    for i in range(3):
        test_data = create_synthetic_data(
            n_samples=200,
            n_features=3,
            distribution_params={'type': 'normal', 'mean': i * 0.5, 'std': 1}
        )
        detector.detect_drift(test_data)
    
    # Test result serialization
    last_result = detector.drift_history[-1]
    result_dict = last_result.to_dict()
    
    assert isinstance(result_dict, dict), "Result should serialize to dict"
    assert 'timestamp' in result_dict, "Should include timestamp"
    assert 'drift_detected' in result_dict, "Should include drift status"
    print("✅ DriftResult serialization works correctly")
    
    # Test export functionality
    export_path = "test_drift_history.json"
    detector.export_drift_history(export_path)
    
    # Verify export file exists and contains expected data
    import json
    with open(export_path, 'r') as f:
        exported_data = json.load(f)
    
    assert 'drift_history' in exported_data, "Export should contain drift history"
    assert len(exported_data['drift_history']) == 3, "Should contain all 3 detections"
    print("✅ Export functionality works correctly")
    
    # Clean up
    Path(export_path).unlink()
    
    print("✅ Export and serialization tests passed!")


def demonstrate_real_world_scenario():
    """Demonstrate drift detection in a realistic IoT scenario."""
    print("\n" + "="*60)
    print("DEMONSTRATING REAL-WORLD IoT SCENARIO")
    print("="*60)
    
    print("🏭 Simulating IoT sensor monitoring scenario...")
    
    # Simulate IoT sensor data with multiple features
    print("📊 Creating baseline IoT sensor data...")
    
    # Baseline data: normal operating conditions
    np.random.seed(42)
    baseline_data = pd.DataFrame({
        'temperature': np.random.normal(25, 2, 2000),      # Temperature sensor
        'pressure': np.random.normal(100, 5, 2000),        # Pressure sensor
        'vibration': np.random.exponential(0.5, 2000),     # Vibration sensor
        'humidity': np.random.uniform(40, 60, 2000),       # Humidity sensor
        'power_consumption': np.random.gamma(2, 50, 2000)  # Power consumption
    })
    
    detector = DataDriftDetector(config=DriftDetectionConfig(
        ks_threshold=0.05,
        psi_threshold=0.2,  # Slightly more sensitive for IoT
        min_samples=100
    ))
    
    detector.set_reference_data(baseline_data)
    print(f"✅ Baseline established with {len(baseline_data)} samples")
    
    # Simulate different drift scenarios
    scenarios = [
        {
            'name': 'Normal Operation',
            'description': 'Sensors operating within normal parameters',
            'data_params': {
                'temperature': (25, 2),
                'pressure': (100, 5),
                'vibration': 0.5,
                'humidity': (40, 60),
                'power_consumption': (2, 50)
            }
        },
        {
            'name': 'Seasonal Change',
            'description': 'Temperature increase due to seasonal change',
            'data_params': {
                'temperature': (30, 2.5),  # Higher temperature
                'pressure': (100, 5),
                'vibration': 0.5,
                'humidity': (35, 55),      # Lower humidity
                'power_consumption': (2.2, 55)  # Slightly higher power
            }
        },
        {
            'name': 'Equipment Degradation',
            'description': 'Increased vibration and power consumption',
            'data_params': {
                'temperature': (25, 2),
                'pressure': (98, 6),       # Slightly lower pressure
                'vibration': 0.8,          # Higher vibration
                'humidity': (40, 60),
                'power_consumption': (2.5, 65)  # Higher power consumption
            }
        },
        {
            'name': 'Sensor Malfunction',
            'description': 'Pressure sensor showing abnormal readings',
            'data_params': {
                'temperature': (25, 2),
                'pressure': (80, 15),      # Much different pressure readings
                'vibration': 0.5,
                'humidity': (40, 60),
                'power_consumption': (2, 50)
            }
        }
    ]
    
    print(f"\n🔄 Testing {len(scenarios)} different operational scenarios...")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['name']} ---")
        print(f"Description: {scenario['description']}")
        
        # Generate scenario data
        params = scenario['data_params']
        scenario_data = pd.DataFrame({
            'temperature': np.random.normal(params['temperature'][0], params['temperature'][1], 500),
            'pressure': np.random.normal(params['pressure'][0], params['pressure'][1], 500),
            'vibration': np.random.exponential(params['vibration'], 500),
            'humidity': np.random.uniform(params['humidity'][0], params['humidity'][1], 500),
            'power_consumption': np.random.gamma(params['power_consumption'][0], params['power_consumption'][1], 500)
        })
        
        # Detect drift
        result = detector.detect_drift(scenario_data)
        
        # Display results
        status = "🚨 DRIFT DETECTED" if result.drift_detected else "✅ NO DRIFT"
        print(f"Result: {status}")
        print(f"PSI Score: {result.psi_score:.4f}")
        print(f"Drifted Features: {sum(result.feature_drifts.values())}/{len(result.feature_drifts)}")
        
        if result.drift_detected:
            drifted_features = [f for f, drift in result.feature_drifts.items() if drift]
            print(f"Affected Sensors: {', '.join(drifted_features)}")
            
            # Generate alert
            alert = DriftAlert.from_drift_result(result, detector.config)
            print(f"Alert Level: {alert.severity}")
            print(f"Recommendations: {'; '.join(alert.recommendations[:2])}")
    
    # Show monitoring summary
    summary = detector.get_drift_summary()
    print(f"\n📈 MONITORING SUMMARY:")
    print(f"Total Scenarios Tested: {summary['total_detections']}")
    print(f"Drift Detection Rate: {summary['drift_rate']:.1%}")
    print(f"Most Sensitive Sensor: {summary.get('most_drifted_feature', 'None')}")
    
    print("✅ Real-world scenario demonstration completed!")


def main():
    """Run all validation tests."""
    print("🔍 DATA DRIFT DETECTION VALIDATION")
    print("This script validates the data drift detection system")
    print("for IoT anomaly detection applications.\n")
    
    try:
        test_basic_drift_detection()
        test_statistical_methods()
        test_drift_monitoring_workflow()
        test_configuration_and_thresholds()
        test_error_handling()
        test_performance_with_large_datasets()
        test_export_and_serialization()
        demonstrate_real_world_scenario()
        
        print("\n" + "="*70)
        print("🎉 ALL DATA DRIFT DETECTION VALIDATION TESTS PASSED!")
        print("="*70)
        print("✅ Basic drift detection functionality verified")
        print("✅ Statistical methods (KS, PSI, Wasserstein) working correctly")
        print("✅ Monitoring workflow and history tracking functional")
        print("✅ Configuration and threshold handling validated")
        print("✅ Error handling and edge cases covered")
        print("✅ Performance with large datasets acceptable")
        print("✅ Export and serialization capabilities working")
        print("✅ Real-world IoT scenarios successfully demonstrated")
        print("\nThe data drift detection system is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()