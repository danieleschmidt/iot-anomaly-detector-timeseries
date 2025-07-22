#!/usr/bin/env python3
"""Simple validation script for batched inference functionality."""

import numpy as np
from unittest.mock import MagicMock, patch
import sys
import traceback

# Add src to path
sys.path.insert(0, 'src')

def test_batched_inference():
    """Test batched inference implementation."""
    try:
        from anomaly_detector import AnomalyDetector
        
        print("✓ Successfully imported AnomalyDetector")
        
        # Create test data
        test_sequences = np.random.randn(100, 30, 3)
        print(f"✓ Created test data: {test_sequences.shape}")
        
        # Mock the model and create detector
        with patch('anomaly_detector.load_model') as mock_load:
            mock_model = MagicMock()
            
            def mock_predict(batch, verbose=0):
                """Mock prediction that returns slightly noisy reconstruction."""
                return batch + np.random.randn(*batch.shape) * 0.1
            
            mock_model.predict.side_effect = mock_predict
            mock_load.return_value = mock_model
            
            # Create detector instance
            detector = AnomalyDetector("dummy_model.h5")
            print("✓ Created AnomalyDetector instance")
            
            # Test original score method
            original_scores = detector.score(test_sequences)
            print(f"✓ Original score method: {len(original_scores)} scores computed")
            print(f"  Score stats: mean={original_scores.mean():.4f}, std={original_scores.std():.4f}")
            
            # Reset mock for consistent comparison
            mock_model.reset_mock()
            mock_model.predict.side_effect = mock_predict
            
            # Test new batched score method
            batched_scores = detector.score_batched(test_sequences, batch_size=25)
            print(f"✓ Batched score method: {len(batched_scores)} scores computed")
            print(f"  Score stats: mean={batched_scores.mean():.4f}, std={batched_scores.std():.4f}")
            print(f"  Model called {mock_model.predict.call_count} times (expected: 4 batches)")
            
            # Verify results have same shape
            assert len(original_scores) == len(batched_scores), "Score arrays have different lengths"
            print("✓ Score array lengths match")
            
            # Test with different batch sizes
            for batch_size in [10, 33, 100, 200]:
                mock_model.reset_mock()
                mock_model.predict.side_effect = mock_predict
                
                scores = detector.score_batched(test_sequences, batch_size=batch_size)
                expected_calls = (len(test_sequences) + batch_size - 1) // batch_size
                actual_calls = mock_model.predict.call_count
                
                print(f"✓ Batch size {batch_size}: {len(scores)} scores, {actual_calls} model calls (expected: {expected_calls})")
                assert len(scores) == len(test_sequences), f"Wrong number of scores for batch_size={batch_size}"
                assert actual_calls == expected_calls, f"Wrong number of model calls for batch_size={batch_size}"
            
            # Test edge cases
            empty_sequences = np.array([]).reshape(0, 30, 3)
            empty_scores = detector.score_batched(empty_sequences, batch_size=10)
            assert len(empty_scores) == 0, "Empty input should return empty scores"
            print("✓ Empty input handled correctly")
            
            # Test single item
            single_sequence = test_sequences[:1]
            single_scores = detector.score_batched(single_sequence, batch_size=100)
            assert len(single_scores) == 1, "Single input should return single score"
            print("✓ Single sequence handled correctly")
            
            print("\n🎉 All batched inference tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

def test_predict_integration():
    """Test integration with predict method."""
    try:
        from anomaly_detector import AnomalyDetector
        from data_preprocessor import DataPreprocessor
        
        # Mock preprocessor and model
        with patch('anomaly_detector.load_model') as mock_load, \
             patch('anomaly_detector.DataPreprocessor') as mock_preprocessor_class:
            
            mock_model = MagicMock()
            mock_model.predict.return_value = np.random.randn(500, 30, 3) * 0.1  # Small errors
            mock_load.return_value = mock_model
            
            mock_preprocessor = MagicMock()
            # Large dataset to trigger automatic batching
            mock_preprocessor.load_and_preprocess.return_value = np.random.randn(1500, 30, 3)
            mock_preprocessor_class.return_value = mock_preprocessor
            
            detector = AnomalyDetector("dummy_model.h5")
            detector.preprocessor = mock_preprocessor
            
            # Test automatic batching for large datasets
            predictions = detector.predict(
                "dummy.csv", 
                window_size=30, 
                step=1, 
                threshold=0.5
            )
            
            print(f"✓ Predict method with auto-batching: {len(predictions)} predictions")
            assert len(predictions) == 1500, "Wrong number of predictions"
            assert predictions.dtype == bool, "Predictions should be boolean"
            
            # Test explicit batching
            mock_model.reset_mock()
            predictions_batched = detector.predict(
                "dummy.csv",
                window_size=30,
                step=1,
                threshold=0.5,
                batch_size=200,
                use_batched=True
            )
            
            print(f"✓ Predict method with explicit batching: {len(predictions_batched)} predictions")
            assert len(predictions_batched) == 1500, "Wrong number of predictions with explicit batching"
            
            print("✓ Predict integration tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing batched inference implementation...")
    print("=" * 50)
    
    success = True
    
    print("\n1. Testing batched inference core functionality")
    success &= test_batched_inference()
    
    print("\n2. Testing integration with predict method")
    success &= test_predict_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! Batched inference is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please review the implementation.")
        sys.exit(1)