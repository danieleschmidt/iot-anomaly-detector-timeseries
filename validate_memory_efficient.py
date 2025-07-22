#!/usr/bin/env python3
"""Validation script for memory-efficient window creation functionality."""

import sys
import numpy as np
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from data_preprocessor import DataPreprocessor
    print("✅ DataPreprocessor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import DataPreprocessor: {e}")
    sys.exit(1)

def test_window_generator():
    """Test basic window generator functionality."""
    print("\n🧪 Testing window generator...")
    
    preprocessor = DataPreprocessor()
    test_data = np.random.random((1000, 5))
    window_size = 50
    step = 10
    
    # Test generator creation
    generator = preprocessor.create_windows_generator(test_data, window_size, step)
    
    # Test that it's actually a generator
    assert hasattr(generator, '__iter__')
    assert hasattr(generator, '__next__')
    print("✅ Generator created successfully")
    
    # Test first few windows
    windows = []
    for i, window in enumerate(generator):
        windows.append(window)
        if i >= 2:
            break
    
    assert len(windows) == 3
    assert all(window.shape == (window_size, 5) for window in windows)
    print("✅ Generator yields correct window shapes")
    
    # Test window contents
    expected_start = 0
    for i, window in enumerate(windows):
        expected_window = test_data[expected_start:expected_start + window_size]
        np.testing.assert_array_equal(window, expected_window)
        expected_start += step
    print("✅ Window contents are correct")

def test_memory_efficiency():
    """Test memory efficiency comparison."""
    print("\n🧪 Testing memory efficiency...")
    
    preprocessor = DataPreprocessor()
    
    # Test memory estimation
    data_shape = (10000, 8)
    window_size = 100
    batch_size = 256
    
    memory_info = preprocessor.estimate_window_memory_usage(
        data_shape, window_size, batch_size
    )
    
    required_keys = [
        'traditional_memory_mb', 
        'generator_memory_mb', 
        'memory_savings_ratio',
        'n_windows'
    ]
    
    for key in required_keys:
        assert key in memory_info
    
    # Generator should use significantly less memory
    assert memory_info['generator_memory_mb'] < memory_info['traditional_memory_mb']
    assert memory_info['memory_savings_ratio'] > 1.0
    
    print(f"✅ Memory savings: {memory_info['memory_savings_ratio']:.1f}x")
    print(f"   Traditional: {memory_info['traditional_memory_mb']:.1f} MB")
    print(f"   Generator: {memory_info['generator_memory_mb']:.1f} MB")

def test_batched_processing():
    """Test batched window processing."""
    print("\n🧪 Testing batched processing...")
    
    preprocessor = DataPreprocessor()
    test_data = np.random.random((500, 4))
    window_size = 25
    batch_size = 32
    
    # Create generator
    generator = preprocessor.create_windows_generator(test_data, window_size, step=1)
    
    # Process in batches
    batch_count = 0
    total_windows = 0
    
    for batch in preprocessor.process_windows_batched(generator, batch_size):
        assert batch.shape[0] <= batch_size  # Batch size constraint
        assert batch.shape[1] == window_size  # Window size
        assert batch.shape[2] == 4  # Features
        
        batch_count += 1
        total_windows += batch.shape[0]
        
        if batch_count >= 5:  # Test first few batches
            break
    
    assert batch_count == 5
    assert total_windows > 0
    print(f"✅ Processed {total_windows} windows in {batch_count} batches")

def test_sliding_windows_generator():
    """Test sliding windows generator with preprocessing."""
    print("\n🧪 Testing sliding windows generator...")
    
    preprocessor = DataPreprocessor()
    # Create data that needs scaling
    test_data = np.random.random((200, 3)) * 100  # Unscaled data
    window_size = 20
    
    # Create generator
    generator = preprocessor.create_sliding_windows_generator(test_data, window_size)
    
    # Test first few windows
    windows = []
    for i, window in enumerate(generator):
        windows.append(window)
        if i >= 2:
            break
    
    assert len(windows) == 3
    assert all(window.shape == (window_size, 3) for window in windows)
    
    # Verify scaling was applied (MinMaxScaler -> [0, 1])
    for window in windows:
        assert window.min() >= 0.0
        assert window.max() <= 1.0
    
    print("✅ Sliding windows with preprocessing working")

def test_optimal_batch_size():
    """Test optimal batch size calculation."""
    print("\n🧪 Testing optimal batch size calculation...")
    
    preprocessor = DataPreprocessor()
    
    data_shape = (50000, 6)
    window_size = 75
    available_memory_mb = 512
    
    optimal_batch = preprocessor.calculate_optimal_batch_size(
        data_shape, window_size, available_memory_mb
    )
    
    assert isinstance(optimal_batch, int)
    assert optimal_batch > 0
    assert optimal_batch <= 2048  # Should be within reasonable bounds
    
    print(f"✅ Optimal batch size: {optimal_batch}")

def test_progress_tracking():
    """Test progress tracking functionality."""
    print("\n🧪 Testing progress tracking...")
    
    preprocessor = DataPreprocessor()
    test_data = np.random.random((300, 4))
    window_size = 30
    batch_size = 50
    
    # Track progress updates
    progress_updates = []
    
    def progress_callback(current, total, elapsed_time):
        progress_updates.append({
            'current': current,
            'total': total,
            'elapsed': elapsed_time
        })
    
    # Create generator and process with progress
    generator = preprocessor.create_windows_generator(test_data, window_size)
    total_windows = len(test_data) - window_size + 1
    
    processed = preprocessor.process_windows_with_progress(
        generator, batch_size, progress_callback, total_windows
    )
    
    assert processed == total_windows
    assert len(progress_updates) > 0
    
    # Verify progress updates are reasonable
    for update in progress_updates:
        assert update['current'] <= total_windows
        assert update['elapsed'] >= 0
    
    print(f"✅ Processed {processed} windows with progress tracking")

def test_large_dataset_simulation():
    """Test with simulated large dataset."""
    print("\n🧪 Testing large dataset simulation...")
    
    preprocessor = DataPreprocessor()
    
    # Simulate larger dataset processing
    large_data = np.random.random((5000, 8))
    window_size = 100
    batch_size = 128
    
    start_time = time.time()
    
    # Process using generator
    generator = preprocessor.create_windows_generator(large_data, window_size, step=5)
    
    processed_count = 0
    for batch in preprocessor.process_windows_batched(generator, batch_size):
        processed_count += batch.shape[0]
        
        # Verify batch properties
        assert batch.shape[1] == window_size
        assert batch.shape[2] == 8
        
        # Process only first few batches for test
        if processed_count >= 256:
            break
    
    elapsed = time.time() - start_time
    rate = processed_count / elapsed if elapsed > 0 else 0
    
    print(f"✅ Processed {processed_count} windows in {elapsed:.2f}s ({rate:.1f} windows/sec)")

def test_validation_and_errors():
    """Test input validation and error handling."""
    print("\n🧪 Testing validation and error handling...")
    
    preprocessor = DataPreprocessor()
    test_data = np.random.random((100, 3))
    
    # Test invalid window size
    try:
        list(preprocessor.create_windows_generator(test_data, 0))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test invalid step size
    try:
        list(preprocessor.create_windows_generator(test_data, 10, 0))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test window size larger than data
    try:
        list(preprocessor.create_windows_generator(test_data, 200))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test invalid batch size
    try:
        generator = preprocessor.create_windows_generator(test_data, 10)
        list(preprocessor.process_windows_batched(generator, 0))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✅ Input validation working correctly")

def main():
    """Run all validation tests."""
    print("🚀 Starting memory-efficient window creation validation...")
    
    try:
        test_window_generator()
        test_memory_efficiency()
        test_batched_processing()
        test_sliding_windows_generator()
        test_optimal_batch_size()
        test_progress_tracking()
        test_large_dataset_simulation()
        test_validation_and_errors()
        
        print("\n✅ All validation tests passed!")
        print("🎉 Memory-efficient window creation is working correctly")
        print("📋 Key features validated:")
        print("   • Generator-based window creation")
        print("   • Memory usage estimation and optimization")
        print("   • Batched processing for large datasets")
        print("   • Sliding windows with preprocessing")
        print("   • Optimal batch size calculation")
        print("   • Progress tracking for long operations")
        print("   • Large dataset processing capability")
        print("   • Comprehensive input validation")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()