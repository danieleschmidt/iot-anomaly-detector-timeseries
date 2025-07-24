import unittest
import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from caching_strategy import CacheManager, cache_result, generate_cache_key


class TestCacheManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.cache_manager = CacheManager(maxsize=10)
        
    def test_cache_manager_initialization(self):
        """Test CacheManager initialization with different parameters."""
        # Test default initialization
        default_cache = CacheManager()
        self.assertEqual(default_cache.maxsize, 128)
        
        # Test custom maxsize
        custom_cache = CacheManager(maxsize=50)
        self.assertEqual(custom_cache.maxsize, 50)
        
    def test_cache_put_and_get(self):
        """Test basic cache put and get operations."""
        key = "test_key"
        value = {"data": [1, 2, 3], "timestamp": time.time()}
        
        # Test put operation
        self.cache_manager.put(key, value)
        
        # Test get operation
        retrieved_value = self.cache_manager.get(key)
        self.assertEqual(retrieved_value, value)
        
    def test_cache_miss(self):
        """Test cache miss behavior."""
        non_existent_key = "non_existent"
        result = self.cache_manager.get(non_existent_key)
        self.assertIsNone(result)
        
    def test_cache_eviction_lru(self):
        """Test LRU eviction policy."""
        small_cache = CacheManager(maxsize=2)
        
        # Fill cache to capacity
        small_cache.put("key1", "value1")
        small_cache.put("key2", "value2")
        
        # Access key1 to make it recently used
        small_cache.get("key1")
        
        # Add new item, should evict key2 (least recently used)
        small_cache.put("key3", "value3")
        
        # key1 and key3 should exist, key2 should be evicted
        self.assertIsNotNone(small_cache.get("key1"))
        self.assertIsNotNone(small_cache.get("key3"))
        self.assertIsNone(small_cache.get("key2"))
        
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        # Initial stats should be zero
        stats = self.cache_manager.get_stats()
        self.assertEqual(stats['hits'], 0)
        self.assertEqual(stats['misses'], 0)
        self.assertEqual(stats['hit_rate'], 0.0)
        
        # Add item and test hit
        self.cache_manager.put("key1", "value1")
        self.cache_manager.get("key1")  # Hit
        self.cache_manager.get("key2")  # Miss
        
        stats = self.cache_manager.get_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['hit_rate'], 0.5)
        
    def test_cache_clear(self):
        """Test cache clearing functionality."""
        self.cache_manager.put("key1", "value1")
        self.cache_manager.put("key2", "value2")
        
        # Verify items exist
        self.assertIsNotNone(self.cache_manager.get("key1"))
        self.assertIsNotNone(self.cache_manager.get("key2"))
        
        # Clear cache
        self.cache_manager.clear()
        
        # Verify items are gone
        self.assertIsNone(self.cache_manager.get("key1"))
        self.assertIsNone(self.cache_manager.get("key2"))
        
    def test_cache_size_tracking(self):
        """Test cache size tracking."""
        self.assertEqual(self.cache_manager.size(), 0)
        
        self.cache_manager.put("key1", "value1")
        self.assertEqual(self.cache_manager.size(), 1)
        
        self.cache_manager.put("key2", "value2")
        self.assertEqual(self.cache_manager.size(), 2)
        
        self.cache_manager.clear()
        self.assertEqual(self.cache_manager.size(), 0)


class TestCacheKeyGeneration(unittest.TestCase):
    def test_generate_cache_key_arrays(self):
        """Test cache key generation for numpy arrays."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        arr3 = np.array([1, 2, 4])
        
        key1 = generate_cache_key(arr1, param1="test")
        key2 = generate_cache_key(arr2, param1="test")
        key3 = generate_cache_key(arr3, param1="test")
        
        # Same arrays should produce same keys
        self.assertEqual(key1, key2)
        # Different arrays should produce different keys
        self.assertNotEqual(key1, key3)
        
    def test_generate_cache_key_dataframes(self):
        """Test cache key generation for pandas DataFrames."""
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df3 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 7]})
        
        key1 = generate_cache_key(df1, param1="test")
        key2 = generate_cache_key(df2, param1="test")
        key3 = generate_cache_key(df3, param1="test")
        
        # Same DataFrames should produce same keys
        self.assertEqual(key1, key2)
        # Different DataFrames should produce different keys
        self.assertNotEqual(key1, key3)
        
    def test_generate_cache_key_mixed_parameters(self):
        """Test cache key generation with mixed parameter types."""
        data = np.array([1, 2, 3])
        
        key1 = generate_cache_key(data, window_size=10, normalize=True, method="default")
        key2 = generate_cache_key(data, window_size=10, normalize=True, method="default")
        key3 = generate_cache_key(data, window_size=20, normalize=True, method="default")
        
        # Same parameters should produce same keys
        self.assertEqual(key1, key2)
        # Different parameters should produce different keys
        self.assertNotEqual(key1, key3)


class TestCacheDecorator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.cache_manager = CacheManager(maxsize=10)
        
    def test_cache_decorator_basic(self):
        """Test basic cache decorator functionality."""
        call_count = 0
        
        @cache_result(self.cache_manager)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = expensive_function(1, 2)
        self.assertEqual(result1, 3)
        self.assertEqual(call_count, 1)
        
        # Second call with same parameters should use cache
        result2 = expensive_function(1, 2)
        self.assertEqual(result2, 3)
        self.assertEqual(call_count, 1)  # Function not called again
        
        # Call with different parameters should execute function
        result3 = expensive_function(2, 3)
        self.assertEqual(result3, 5)
        self.assertEqual(call_count, 2)
        
    def test_cache_decorator_with_numpy_arrays(self):
        """Test cache decorator with numpy array parameters."""
        call_count = 0
        
        @cache_result(self.cache_manager)
        def process_array(arr, multiplier=1):
            nonlocal call_count
            call_count += 1
            return arr * multiplier
        
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])  # Same values
        arr3 = np.array([4, 5, 6])  # Different values
        
        # First call
        result1 = process_array(arr1, multiplier=2)
        np.testing.assert_array_equal(result1, [2, 4, 6])
        self.assertEqual(call_count, 1)
        
        # Second call with equivalent array should use cache
        result2 = process_array(arr2, multiplier=2)
        np.testing.assert_array_equal(result2, [2, 4, 6])
        self.assertEqual(call_count, 1)
        
        # Call with different array should execute function
        result3 = process_array(arr3, multiplier=2)
        np.testing.assert_array_equal(result3, [8, 10, 12])
        self.assertEqual(call_count, 2)
        
    def test_cache_decorator_performance(self):
        """Test that caching provides performance improvement."""
        @cache_result(self.cache_manager)
        def slow_function(x):
            time.sleep(0.01)  # Simulate slow operation
            return x * 2
        
        # First call (uncached)
        start_time = time.time()
        result1 = slow_function(5)
        first_duration = time.time() - start_time
        
        # Second call (cached)
        start_time = time.time()
        result2 = slow_function(5)
        second_duration = time.time() - start_time
        
        self.assertEqual(result1, result2)
        # Cached call should be significantly faster
        self.assertLess(second_duration, first_duration / 2)


class TestCacheIntegration(unittest.TestCase):
    def test_cache_with_data_preprocessing(self):
        """Test cache integration with data preprocessing operations."""
        cache_manager = CacheManager(maxsize=5)
        
        @cache_result(cache_manager)
        def preprocess_data(data, window_size, normalize=True):
            # Simulate preprocessing logic
            if normalize:
                data = (data - data.mean()) / data.std()
            
            # Create windows
            windows = []
            for i in range(len(data) - window_size + 1):
                windows.append(data[i:i + window_size])
            
            return np.array(windows)
        
        # Test data
        test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # First call
        result1 = preprocess_data(test_data, window_size=3, normalize=True)
        
        # Second call with same parameters should use cache
        result2 = preprocess_data(test_data, window_size=3, normalize=True)
        
        np.testing.assert_array_equal(result1, result2)
        
        # Verify cache statistics
        stats = cache_manager.get_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        
    def test_cache_memory_efficiency(self):
        """Test that cache doesn't consume excessive memory."""
        large_cache = CacheManager(maxsize=1000)
        
        # Add many items to test memory management
        for i in range(1500):
            key = f"key_{i}"
            value = np.random.rand(100)  # Moderately large arrays
            large_cache.put(key, value)
        
        # Cache should not exceed maxsize
        self.assertLessEqual(large_cache.size(), 1000)
        
        # Should still be able to retrieve recent items
        recent_value = large_cache.get("key_1499")
        self.assertIsNotNone(recent_value)
        
        # Older items should be evicted
        old_value = large_cache.get("key_0")
        self.assertIsNone(old_value)


if __name__ == "__main__":
    unittest.main()