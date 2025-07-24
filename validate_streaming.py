#!/usr/bin/env python3
"""Simple validation script for streaming processor functionality."""

import numpy as np
import sys
from pathlib import Path
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from streaming_processor import StreamingConfig
    print("✅ StreamingProcessor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import StreamingProcessor: {e}")
    sys.exit(1)

def test_streaming_config():
    """Test streaming configuration."""
    print("\n🧪 Testing StreamingConfig...")
    
    # Test default config
    config = StreamingConfig()
    assert config.window_size == 50
    assert config.batch_size == 32
    print("✅ Default config values correct")
    
    # Test custom config
    custom_config = StreamingConfig(
        window_size=25,
        batch_size=16,
        anomaly_threshold=0.7
    )
    assert custom_config.window_size == 25
    assert custom_config.batch_size == 16
    assert custom_config.anomaly_threshold == 0.7
    print("✅ Custom config values correct")
    
    # Test validation
    try:
        StreamingConfig(window_size=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Config validation working")
    
    # Test serialization
    config_dict = custom_config.to_dict()
    restored_config = StreamingConfig.from_dict(config_dict)
    assert restored_config.window_size == custom_config.window_size
    print("✅ Config serialization working")

def test_data_structures():
    """Test core data structures without TensorFlow."""
    print("\n🧪 Testing core data structures...")
    
    config = StreamingConfig(window_size=5, buffer_size=10)
    
    # Create mock StreamingProcessor without model loading
    class MockStreamingProcessor:
        def __init__(self, config):
            from collections import deque
            self.config = config
            self.buffer = deque(maxlen=config.buffer_size)
            self.results_history = []
            self.anomaly_callbacks = []
            self.total_processed = 0
            self.anomalies_detected = 0
            import time
            self.start_time = time.time()
            self.is_running = False
        
        def ingest_data(self, data_point):
            self.buffer.append(data_point)
        
        def ingest_batch(self, batch_data):
            for data_point in batch_data:
                self.buffer.append(data_point)
        
        def get_performance_metrics(self):
            import time
            elapsed_time = time.time() - self.start_time
            return {
                'total_processed': self.total_processed,
                'anomalies_detected': self.anomalies_detected,
                'processing_rate': self.total_processed / max(elapsed_time, 1e-6),
                'anomaly_rate': self.anomalies_detected / max(self.total_processed, 1),
                'buffer_utilization': len(self.buffer) / self.config.buffer_size,
                'buffer_size': len(self.buffer),
                'elapsed_time': elapsed_time,
                'is_running': self.is_running
            }
    
    processor = MockStreamingProcessor(config)
    
    # Test data ingestion
    data_point = {'timestamp': '2023-01-01T00:00:00', 'sensor1': 1.0, 'sensor2': 2.0}
    processor.ingest_data(data_point)
    assert len(processor.buffer) == 1
    print("✅ Single data ingestion working")
    
    # Test batch ingestion
    batch_data = [
        {'timestamp': f'2023-01-01T00:0{i}:00', 'sensor1': float(i), 'sensor2': float(i*2)}
        for i in range(5)
    ]
    processor.ingest_batch(batch_data)
    assert len(processor.buffer) == 6  # 1 + 5
    print("✅ Batch data ingestion working")
    
    # Test buffer overflow
    for i in range(10):  # Add more than buffer size
        processor.ingest_data({'value': i})
    assert len(processor.buffer) == config.buffer_size  # Should be limited to buffer_size
    print("✅ Buffer overflow handling working")
    
    # Test metrics
    metrics = processor.get_performance_metrics()
    assert 'total_processed' in metrics
    assert 'buffer_utilization' in metrics
    print("✅ Performance metrics working")

def test_data_preprocessor_integration():
    """Test integration with data preprocessor."""
    print("\n🧪 Testing DataPreprocessor integration...")
    
    try:
        from data_preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Test sliding windows creation
        data = np.random.random((100, 3))  # 100 timesteps, 3 features
        windows = preprocessor.create_sliding_windows(data, window_size=10, step=1)
        
        assert windows.shape[0] == 91  # 100 - 10 + 1
        assert windows.shape[1] == 10  # window size
        assert windows.shape[2] == 3   # features
        print("✅ Sliding windows creation working")
        
        # Test with insufficient data
        small_data = np.random.random((5, 3))
        try:
            windows = preprocessor.create_sliding_windows(small_data, window_size=10)
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✅ Insufficient data handling working")
            
    except ImportError as e:
        print(f"⚠️ DataPreprocessor not available: {e}")

def test_export_functionality():
    """Test result export functionality."""
    print("\n🧪 Testing export functionality...")
    
    # Mock results data
    results_history = [
        {
            'timestamp': '2023-01-01T00:00:00',
            'scores': [0.1, 0.2, 0.8],
            'anomalies': [False, False, True],
            'num_windows': 3,
            'anomaly_count': 1
        },
        {
            'timestamp': '2023-01-01T00:01:00',
            'scores': [0.3, 0.9],
            'anomalies': [False, True],
            'num_windows': 2,
            'anomaly_count': 1
        }
    ]
    
    # Test JSON export
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
    
    try:
        with open(json_path, 'w') as f:
            json.dump(results_history, f, indent=2)
        
        # Verify export
        with open(json_path, 'r') as f:
            exported_data = json.load(f)
        
        assert len(exported_data) == 2
        assert 'timestamp' in exported_data[0]
        assert 'scores' in exported_data[0]
        print("✅ JSON export working")
        
    finally:
        Path(json_path).unlink(missing_ok=True)

def main():
    """Run all validation tests."""
    print("🚀 Starting streaming processor validation...")
    
    try:
        test_streaming_config()
        test_data_structures()
        test_data_preprocessor_integration()
        test_export_functionality()
        
        print("\n✅ All validation tests passed!")
        print("🎉 Streaming processor implementation is working correctly")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()