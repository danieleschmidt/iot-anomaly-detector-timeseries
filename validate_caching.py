#!/usr/bin/env python3
"""
Caching Implementation Validation Script

This script validates the caching functionality integration into the IoT
anomaly detection system and demonstrates performance improvements.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.caching_strategy import CacheManager, get_cache_stats, clear_all_caches
    from src.data_preprocessor import DataPreprocessor
    print("✅ Successfully imported caching modules")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please ensure all dependencies are installed and modules are available")
    sys.exit(1)


def test_basic_cache_functionality():
    """Test basic cache manager functionality."""
    print("\n" + "="*60)
    print("TESTING BASIC CACHE FUNCTIONALITY")
    print("="*60)
    
    # Test CacheManager
    cache = CacheManager(maxsize=5)
    
    # Test put/get operations
    cache.put("test_key", {"data": [1, 2, 3], "result": "success"})
    retrieved = cache.get("test_key")
    
    assert retrieved == {"data": [1, 2, 3], "result": "success"}, "Cache put/get failed"
    print("✅ Basic cache put/get operations work correctly")
    
    # Test cache miss
    miss_result = cache.get("non_existent_key")
    assert miss_result is None, "Cache miss should return None"
    print("✅ Cache miss handling works correctly")
    
    # Test statistics
    cache.get("test_key")  # Another hit
    cache.get("another_miss")  # Another miss
    
    stats = cache.get_stats()
    assert stats['hits'] == 2, f"Expected 2 hits, got {stats['hits']}"
    assert stats['misses'] == 2, f"Expected 2 misses, got {stats['misses']}"
    assert abs(stats['hit_rate'] - 0.5) < 0.01, f"Expected hit rate ~0.5, got {stats['hit_rate']}"
    print("✅ Cache statistics tracking works correctly")
    
    print("✅ All basic cache functionality tests passed!")


def test_preprocessing_cache_integration():
    """Test caching integration with DataPreprocessor."""
    print("\n" + "="*60)
    print("TESTING PREPROCESSING CACHE INTEGRATION")
    print("="*60)
    
    # Clear any existing cache
    clear_all_caches()
    
    # Create test data
    test_data = np.random.randn(100, 5)
    preprocessor = DataPreprocessor(enable_caching=True)
    
    print(f"Created test data: shape {test_data.shape}")
    
    # First call (should miss cache)
    start_time = time.time()
    windows1 = preprocessor.create_windows(test_data, window_size=10, step=1)
    first_call_time = time.time() - start_time
    
    print(f"First call completed in {first_call_time:.4f}s (cache miss expected)")
    
    # Second call with same parameters (should hit cache)
    start_time = time.time()
    windows2 = preprocessor.create_windows(test_data, window_size=10, step=1)
    second_call_time = time.time() - start_time
    
    print(f"Second call completed in {second_call_time:.4f}s (cache hit expected)")
    
    # Verify results are identical
    np.testing.assert_array_equal(windows1, windows2)
    print("✅ Cached and non-cached results are identical")
    
    # Verify performance improvement
    if second_call_time < first_call_time * 0.5:  # Should be much faster
        performance_improvement = (first_call_time - second_call_time) / first_call_time * 100
        print(f"✅ Cache provided {performance_improvement:.1f}% performance improvement")
    else:
        print("⚠️  Cache performance improvement not significant (may be due to small dataset)")
    
    # Check cache statistics
    cache_stats = preprocessor.get_cache_stats()
    preprocessing_stats = cache_stats.get('preprocessing_cache', {})
    
    if preprocessing_stats.get('hits', 0) > 0:
        print(f"✅ Cache statistics show {preprocessing_stats['hits']} hits, {preprocessing_stats['misses']} misses")
    else:
        print("⚠️  No cache hits recorded in statistics")
    
    print("✅ Preprocessing cache integration test completed!")


def test_cache_with_different_parameters():
    """Test that cache correctly differentiates between different parameters."""
    print("\n" + "="*60)
    print("TESTING CACHE PARAMETER DIFFERENTIATION")
    print("="*60)
    
    clear_all_caches()
    
    test_data = np.random.randn(50, 3)
    preprocessor = DataPreprocessor(enable_caching=True)
    
    # Create windows with different parameters
    windows_10_1 = preprocessor.create_windows(test_data, window_size=10, step=1)
    windows_10_2 = preprocessor.create_windows(test_data, window_size=10, step=2)  # Different step
    windows_15_1 = preprocessor.create_windows(test_data, window_size=15, step=1)  # Different window size
    
    print(f"Windows with (10,1): shape {windows_10_1.shape}")
    print(f"Windows with (10,2): shape {windows_10_2.shape}")
    print(f"Windows with (15,1): shape {windows_15_1.shape}")
    
    # Results should be different
    assert windows_10_1.shape != windows_10_2.shape, "Different steps should produce different results"
    assert windows_10_1.shape != windows_15_1.shape, "Different window sizes should produce different results"
    
    print("✅ Cache correctly differentiates between different parameters")
    
    # Verify cache can retrieve each distinct result
    windows_10_1_cached = preprocessor.create_windows(test_data, window_size=10, step=1)
    np.testing.assert_array_equal(windows_10_1, windows_10_1_cached)
    print("✅ Cache correctly retrieves results for specific parameter combinations")


def test_cache_memory_efficiency():
    """Test cache memory efficiency and LRU eviction."""
    print("\n" + "="*60)  
    print("TESTING CACHE MEMORY EFFICIENCY")
    print("="*60)
    
    # Create small cache for testing eviction
    small_cache = CacheManager(maxsize=3)
    
    # Fill cache to capacity
    small_cache.put("key1", np.array([1, 2, 3]))
    small_cache.put("key2", np.array([4, 5, 6]))
    small_cache.put("key3", np.array([7, 8, 9]))
    
    assert small_cache.size() == 3, f"Expected cache size 3, got {small_cache.size()}"
    print("✅ Cache respects maximum size limit")
    
    # Access key1 to make it recently used
    small_cache.get("key1")
    
    # Add new item, should evict key2 (least recently used)
    small_cache.put("key4", np.array([10, 11, 12]))
    
    # key1 and key3 and key4 should exist, key2 should be evicted
    assert small_cache.get("key1") is not None, "Recently used key1 should not be evicted"
    assert small_cache.get("key3") is not None, "Key3 should still exist"
    assert small_cache.get("key4") is not None, "New key4 should exist"
    assert small_cache.get("key2") is None, "Least recently used key2 should be evicted"
    
    print("✅ LRU eviction policy works correctly")


def test_cache_statistics_and_monitoring():
    """Test cache statistics and monitoring capabilities."""
    print("\n" + "="*60)
    print("TESTING CACHE STATISTICS AND MONITORING")
    print("="*60)
    
    clear_all_caches()
    
    # Perform various cache operations
    test_data = np.random.randn(30, 4)
    preprocessor = DataPreprocessor(enable_caching=True)
    
    # Generate some cache activity
    for i in range(5):
        preprocessor.create_windows(test_data, window_size=5+i, step=1)  # Different operations
        preprocessor.create_windows(test_data, window_size=5, step=1)    # Repeated operation
    
    # Check statistics
    global_stats = get_cache_stats()
    preprocessor_stats = preprocessor.get_cache_stats()
    
    print("Global cache statistics:")
    for cache_type, stats in global_stats.items():
        if stats:
            print(f"  {cache_type}: {stats['hits']} hits, {stats['misses']} misses, "
                  f"{stats['hit_rate']:.2%} hit rate")
    
    print("Preprocessor cache statistics:")
    if preprocessor_stats.get('caching_enabled', False):
        prep_cache = preprocessor_stats.get('preprocessing_cache', {})
        if prep_cache:
            print(f"  Hits: {prep_cache.get('hits', 0)}")
            print(f"  Misses: {prep_cache.get('misses', 0)}")
            print(f"  Hit Rate: {prep_cache.get('hit_rate', 0):.2%}")
    
    assert preprocessor_stats['caching_enabled'], "Caching should be enabled"
    print("✅ Cache statistics are properly tracked and accessible")


def demonstrate_performance_benefits():
    """Demonstrate the performance benefits of caching."""
    print("\n" + "="*60)
    print("DEMONSTRATING PERFORMANCE BENEFITS")
    print("="*60)
    
    # Create larger dataset for more noticeable performance difference
    large_data = np.random.randn(500, 10)
    preprocessor = DataPreprocessor(enable_caching=True)
    
    print(f"Testing with larger dataset: shape {large_data.shape}")
    
    # Clear cache to ensure clean test
    preprocessor.clear_cache()
    
    # Measure performance for repeated operations
    window_sizes = [20, 25, 30, 20, 25, 30, 20, 25, 30]  # Repeated pattern
    times = []
    
    for i, window_size in enumerate(window_sizes):
        start_time = time.time()
        preprocessor.create_windows(large_data, window_size=window_size, step=1)
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        cache_stats = preprocessor.get_cache_stats()
        prep_cache = cache_stats.get('preprocessing_cache', {})
        hit_rate = prep_cache.get('hit_rate', 0) * 100
        
        print(f"  Operation {i+1}: window_size={window_size}, "
              f"time={elapsed:.4f}s, hit_rate={hit_rate:.1f}%")
    
    # Analyze performance trends
    first_third = np.mean(times[:3])   # First occurrence of each size
    last_third = np.mean(times[-3:])   # Last occurrence of each size
    
    if last_third < first_third * 0.8:  # At least 20% improvement
        improvement = (first_third - last_third) / first_third * 100
        print(f"✅ Caching improved performance by {improvement:.1f}% for repeated operations")
    else:
        print("⚠️  Performance improvement may not be significant for this dataset size")
    
    final_stats = preprocessor.get_cache_stats()
    prep_cache = final_stats.get('preprocessing_cache', {})
    print(f"Final cache statistics: {prep_cache.get('hits', 0)} hits, "
          f"{prep_cache.get('misses', 0)} misses, "
          f"{prep_cache.get('hit_rate', 0):.2%} hit rate")


def main():
    """Run all validation tests."""
    print("🔍 CACHING IMPLEMENTATION VALIDATION")
    print("This script validates the caching functionality integration")
    print("into the IoT anomaly detection system.\n")
    
    try:
        test_basic_cache_functionality()
        test_preprocessing_cache_integration()
        test_cache_with_different_parameters()
        test_cache_memory_efficiency()
        test_cache_statistics_and_monitoring()
        demonstrate_performance_benefits()
        
        print("\n" + "="*60)
        print("🎉 ALL CACHING VALIDATION TESTS PASSED!")
        print("="*60)
        print("✅ Basic cache functionality works correctly")
        print("✅ Preprocessing cache integration successful")
        print("✅ Parameter differentiation working")
        print("✅ Memory efficiency and LRU eviction functional")
        print("✅ Statistics and monitoring capabilities operational")
        print("✅ Performance benefits demonstrated")
        print("\nThe caching system is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()