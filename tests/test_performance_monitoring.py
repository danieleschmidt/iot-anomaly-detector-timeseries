"""Test performance monitoring functionality."""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch
import json

from src.logging_config import (
    PerformanceMetrics, 
    get_performance_metrics, 
    performance_monitor,
    PerformanceMonitor
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = PerformanceMetrics(buffer_size=100)
    
    def test_initialization(self):
        """Test PerformanceMetrics initialization."""
        assert self.metrics.buffer_size == 100
        assert len(self.metrics.metrics) == 0
        assert len(self.metrics.counters) == 0
        assert self.metrics.start_time <= time.time()
    
    def test_record_timing(self):
        """Test timing metric recording."""
        operation = "test_operation"
        duration = 2.5
        
        self.metrics.record_timing(operation, duration, status="success")
        
        # Check that timing was recorded
        timing_metrics = list(self.metrics.metrics['timing'])
        assert len(timing_metrics) == 1
        
        recorded = timing_metrics[0]
        assert recorded['operation'] == operation
        assert recorded['duration'] == duration
        assert recorded['metadata']['status'] == "success"
        
        # Check counter was incremented
        assert self.metrics.counters[f'{operation}_count'] == 1
    
    def test_record_memory_usage(self):
        """Test memory usage recording."""
        with patch('psutil.Process') as mock_process_class:
            # Mock psutil Process
            mock_process = Mock()
            mock_memory_info = Mock()
            mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
            mock_memory_info.vms = 200 * 1024 * 1024  # 200 MB
            
            mock_process.memory_info.return_value = mock_memory_info
            mock_process.memory_percent.return_value = 15.5
            mock_process_class.return_value = mock_process
            
            # Create new metrics instance with mocked process
            metrics = PerformanceMetrics()
            metrics.process = mock_process
            
            operation = "memory_test"
            metrics.record_memory_usage(operation, context="testing")
            
            # Check memory metrics
            memory_metrics = list(metrics.metrics['memory'])
            assert len(memory_metrics) == 1
            
            recorded = memory_metrics[0]
            assert recorded['operation'] == operation
            assert recorded['rss_mb'] == 100.0
            assert recorded['vms_mb'] == 200.0
            assert recorded['memory_percent'] == 15.5
            assert recorded['metadata']['context'] == "testing"
    
    def test_record_gpu_usage_no_gpu(self):
        """Test GPU recording when GPU is not available."""
        # GPU should not be available in test environment
        self.metrics.record_gpu_usage("test_op")
        
        # Should not record anything
        gpu_metrics = list(self.metrics.metrics['gpu'])
        assert len(gpu_metrics) == 0
    
    def test_record_custom_metric(self):
        """Test custom metric recording."""
        metric_type = "cache_hit_rate"
        value = 0.85
        operation = "cache_test"
        
        self.metrics.record_custom_metric(
            metric_type, value, operation, 
            cache_size=1000, threshold=0.8
        )
        
        # Check custom metric
        custom_metrics = list(self.metrics.metrics[metric_type])
        assert len(custom_metrics) == 1
        
        recorded = custom_metrics[0]
        assert recorded['metric_type'] == metric_type
        assert recorded['value'] == value
        assert recorded['operation'] == operation
        assert recorded['metadata']['cache_size'] == 1000
    
    def test_increment_counter(self):
        """Test counter incrementing."""
        counter_name = "test_counter"
        
        # Initial increment
        self.metrics.increment_counter(counter_name, 5)
        assert self.metrics.counters[counter_name] == 5
        
        # Second increment
        self.metrics.increment_counter(counter_name, 3)
        assert self.metrics.counters[counter_name] == 8
        
        # Default increment
        self.metrics.increment_counter(counter_name)
        assert self.metrics.counters[counter_name] == 9
    
    def test_get_summary_stats(self):
        """Test summary statistics generation."""
        # Add some test data
        self.metrics.record_timing("op1", 1.5, status="success")
        self.metrics.record_timing("op2", 0.8, status="success")
        self.metrics.record_timing("op1", 2.1, status="error")
        self.metrics.increment_counter("test_count", 10)
        
        stats = self.metrics.get_summary_stats()
        
        # Check basic structure
        assert 'counters' in stats
        assert 'uptime_seconds' in stats
        assert 'timing' in stats
        
        # Check timing stats
        timing = stats['timing']
        assert timing['count'] == 3
        assert timing['min'] == 0.8
        assert timing['max'] == 2.1
        assert abs(timing['avg'] - 1.4667) < 0.001  # (1.5 + 0.8 + 2.1) / 3
        assert timing['total'] == 4.4
        
        # Check counters
        assert stats['counters']['test_count'] == 10
        assert stats['counters']['op1_count'] == 2
        assert stats['counters']['op2_count'] == 1
    
    def test_get_summary_stats_filtered(self):
        """Test filtered summary statistics."""
        # Add test data for different operations
        self.metrics.record_timing("op1", 1.0)
        self.metrics.record_timing("op2", 2.0)
        self.metrics.record_timing("op1", 1.5)
        
        # Filter by operation
        stats = self.metrics.get_summary_stats(operation="op1")
        timing = stats['timing']
        
        assert timing['count'] == 2
        assert timing['min'] == 1.0
        assert timing['max'] == 1.5
        assert timing['avg'] == 1.25
    
    def test_get_summary_stats_last_n(self):
        """Test last N entries filtering."""
        # Add test data
        for i in range(10):
            self.metrics.record_timing("test_op", float(i))
        
        # Get last 3 entries
        stats = self.metrics.get_summary_stats(last_n=3)
        timing = stats['timing']
        
        assert timing['count'] == 3
        assert timing['min'] == 7.0  # Last 3 are 7, 8, 9
        assert timing['max'] == 9.0
        assert timing['avg'] == 8.0
    
    def test_export_metrics_json(self):
        """Test JSON metrics export."""
        # Add test data
        self.metrics.record_timing("test_op", 1.5, status="success")
        self.metrics.increment_counter("test_counter", 5)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            self.metrics.export_metrics(json_path, 'json')
            
            # Verify export
            with open(json_path, 'r') as f:
                exported_data = json.load(f)
            
            assert 'counters' in exported_data
            assert 'metrics' in exported_data
            assert 'export_timestamp' in exported_data
            
            assert exported_data['counters']['test_counter'] == 5
            assert len(exported_data['metrics']['timing']) == 1
            
        finally:
            Path(json_path).unlink(missing_ok=True)
    
    def test_buffer_size_limit(self):
        """Test that buffer respects size limits."""
        small_metrics = PerformanceMetrics(buffer_size=3)
        
        # Add more entries than buffer size
        for i in range(5):
            small_metrics.record_timing("test_op", float(i))
        
        # Should only keep last 3 entries
        timing_metrics = list(small_metrics.metrics['timing'])
        assert len(timing_metrics) == 3
        
        # Check that we have the last 3 entries (2, 3, 4)
        durations = [m['duration'] for m in timing_metrics]
        assert durations == [2.0, 3.0, 4.0]


class TestPerformanceDecorator:
    """Test performance monitoring decorator."""
    
    def test_performance_monitor_decorator_basic(self):
        """Test basic decorator functionality."""
        metrics = PerformanceMetrics()
        
        @performance_monitor("test_function", threshold=0.1)
        def test_function(x, y):
            time.sleep(0.05)  # Short sleep
            return x + y
        
        # Mock the global metrics
        with patch('logging_config.get_performance_metrics', return_value=metrics):
            result = test_function(2, 3)
        
        assert result == 5
        
        # Check that timing was recorded
        timing_metrics = list(metrics.metrics['timing'])
        assert len(timing_metrics) == 1
        
        recorded = timing_metrics[0]
        assert recorded['operation'] == 'test_function'
        assert recorded['duration'] >= 0.05
        assert recorded['metadata']['status'] == 'success'
    
    def test_performance_monitor_decorator_error(self):
        """Test decorator with function that raises error."""
        metrics = PerformanceMetrics()
        
        @performance_monitor("error_function")
        def error_function():
            raise ValueError("Test error")
        
        with patch('logging_config.get_performance_metrics', return_value=metrics):
            with pytest.raises(ValueError):
                error_function()
        
        # Check error was recorded
        timing_metrics = list(metrics.metrics['timing'])
        assert len(timing_metrics) == 1
        
        recorded = timing_metrics[0]
        assert recorded['operation'] == 'error_function'
        assert recorded['metadata']['status'] == 'error'
        assert recorded['metadata']['error_type'] == 'ValueError'
        
        # Check error counter
        assert metrics.counters['error_function_errors'] == 1
    
    def test_performance_monitor_with_memory_tracking(self):
        """Test decorator with memory tracking."""
        metrics = PerformanceMetrics()
        
        with patch('psutil.Process'):
            mock_process = Mock()
            mock_memory_info = Mock()
            mock_memory_info.rss = 50 * 1024 * 1024
            mock_memory_info.vms = 100 * 1024 * 1024
            mock_process.memory_info.return_value = mock_memory_info
            mock_process.memory_percent.return_value = 10.0
            metrics.process = mock_process
            
            @performance_monitor("memory_function", track_memory=True)
            def memory_function():
                return "done"
            
            with patch('logging_config.get_performance_metrics', return_value=metrics):
                result = memory_function()
            
            assert result == "done"
            
            # Check memory recordings (start and end)
            memory_metrics = list(metrics.metrics['memory'])
            assert len(memory_metrics) == 2  # start and end


class TestPerformanceMonitorContext:
    """Test PerformanceMonitor context manager."""
    
    def test_context_manager_success(self):
        """Test context manager with successful operation."""
        metrics = PerformanceMetrics()
        
        with patch('logging_config.get_performance_metrics', return_value=metrics):
            with PerformanceMonitor("context_test") as monitor:
                time.sleep(0.01)
                assert monitor.operation == "context_test"
        
        # Check timing was recorded
        timing_metrics = list(metrics.metrics['timing'])
        assert len(timing_metrics) == 1
        
        recorded = timing_metrics[0]
        assert recorded['operation'] == 'context_test'
        assert recorded['duration'] >= 0.01
        assert recorded['metadata']['status'] == 'success'
    
    def test_context_manager_error(self):
        """Test context manager with error."""
        metrics = PerformanceMetrics()
        
        with patch('logging_config.get_performance_metrics', return_value=metrics):
            with pytest.raises(RuntimeError):
                with PerformanceMonitor("error_context"):
                    raise RuntimeError("Test error")
        
        # Check error was recorded
        timing_metrics = list(metrics.metrics['timing'])
        assert len(timing_metrics) == 1
        
        recorded = timing_metrics[0]
        assert recorded['operation'] == 'error_context'
        assert recorded['metadata']['status'] == 'error'
        assert recorded['metadata']['error_type'] == 'RuntimeError'
    
    def test_context_manager_with_memory_tracking(self):
        """Test context manager with memory tracking."""
        metrics = PerformanceMetrics()
        
        with patch('psutil.Process'):
            mock_process = Mock()
            mock_memory_info = Mock()
            mock_memory_info.rss = 75 * 1024 * 1024
            mock_memory_info.vms = 150 * 1024 * 1024
            mock_process.memory_info.return_value = mock_memory_info
            mock_process.memory_percent.return_value = 12.5
            metrics.process = mock_process
            
            with patch('logging_config.get_performance_metrics', return_value=metrics):
                with PerformanceMonitor("memory_context", track_memory=True):
                    pass
            
            # Check memory recordings
            memory_metrics = list(metrics.metrics['memory'])
            assert len(memory_metrics) == 2  # start and end


class TestGlobalMetrics:
    """Test global metrics functionality."""
    
    def test_get_performance_metrics(self):
        """Test global metrics instance."""
        metrics1 = get_performance_metrics()
        metrics2 = get_performance_metrics()
        
        # Should return the same instance
        assert metrics1 is metrics2
        assert isinstance(metrics1, PerformanceMetrics)
    
    def test_global_metrics_persistence(self):
        """Test that global metrics persist across calls."""
        metrics = get_performance_metrics()
        
        # Add some data
        metrics.record_timing("global_test", 1.0)
        metrics.increment_counter("global_counter", 5)
        
        # Get metrics again
        metrics2 = get_performance_metrics()
        
        # Should have the same data
        timing_metrics = list(metrics2.metrics['timing'])
        assert len(timing_metrics) >= 1  # Might have data from other tests
        
        assert metrics2.counters['global_counter'] == 5


class TestPerformanceIntegration:
    """Test integration scenarios."""
    
    def test_multiple_operations(self):
        """Test tracking multiple operations."""
        metrics = PerformanceMetrics()
        
        # Simulate various operations
        metrics.record_timing("data_load", 0.5, rows=1000)
        metrics.record_timing("preprocessing", 1.2, method="scaling")
        metrics.record_timing("model_inference", 2.1, batch_size=32)
        metrics.record_timing("data_load", 0.6, rows=1200)
        
        metrics.increment_counter("total_predictions", 64)
        metrics.record_custom_metric("cache_hit_rate", 0.87, "cache_check")
        
        # Get comprehensive stats
        stats = metrics.get_summary_stats()
        
        # Check timing stats
        assert stats['timing']['count'] == 4
        assert 'data_load_count' in stats['counters']
        assert stats['counters']['data_load_count'] == 2
        assert stats['counters']['total_predictions'] == 64
        
        # Check operation-specific stats
        data_load_stats = metrics.get_summary_stats(operation="data_load")
        assert data_load_stats['timing']['count'] == 2
        assert data_load_stats['timing']['avg'] == 0.55  # (0.5 + 0.6) / 2
    
    def test_performance_monitoring_workflow(self):
        """Test a realistic performance monitoring workflow."""
        metrics = PerformanceMetrics()
        
        # Simulate a machine learning workflow
        with patch('logging_config.get_performance_metrics', return_value=metrics):
            
            # Data loading with monitoring
            @performance_monitor("data_loading", threshold=1.0)
            def load_data():
                time.sleep(0.02)  # Simulate data loading
                return {"samples": 1000, "features": 10}
            
            # Preprocessing with context manager
            def preprocess_data(data):
                with PerformanceMonitor("preprocessing"):
                    time.sleep(0.01)  # Simulate preprocessing
                    return data
            
            # Model inference
            @performance_monitor("model_inference", track_memory=True)
            def run_inference(data):
                time.sleep(0.03)  # Simulate inference
                return {"predictions": [0.1, 0.9, 0.3]}
            
            # Execute workflow
            data = load_data()
            processed_data = preprocess_data(data)
            run_inference(processed_data)
            
            # Add custom metrics
            metrics.record_custom_metric("accuracy", 0.95, "evaluation")
            metrics.increment_counter("workflow_runs")
        
        # Analyze results
        stats = metrics.get_summary_stats()
        
        assert stats['timing']['count'] == 3  # Three timed operations
        assert 'workflow_runs' in stats['counters']
        assert stats['counters']['workflow_runs'] == 1
        
        # Check that all operations were recorded
        timing_metrics = list(metrics.metrics['timing'])
        operations = {m['operation'] for m in timing_metrics}
        expected_ops = {'data_loading', 'preprocessing', 'model_inference'}
        assert operations == expected_ops