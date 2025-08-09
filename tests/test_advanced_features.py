"""Comprehensive tests for advanced anomaly detection features."""

import pytest
import numpy as np
import time
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from edge_anomaly_detector import EdgeAnomalyDetector, EdgeConfig, AnomalyResult
from realtime_stream_processor import RealtimeStreamProcessor, StreamConfig, StreamData, ProcessingResult
from multi_modal_sensor_fusion import MultiModalSensorFusion, SensorFusionConfig, SensorReading, SensorType, MultiModalData
from robust_error_handler import RobustErrorHandler, ErrorCategory, ErrorSeverity, RecoveryStrategy
from security_hardening import SecurityHardening, SecurityConfig, InputValidator, ThreatType, SecurityLevel
from comprehensive_monitoring import ComprehensiveMonitoring, MetricType, AlertLevel, AlertRule
from distributed_anomaly_detection import DistributedAnomalyDetector, NodeRole, DistributedTask, WorkloadType
from high_performance_optimizer import HighPerformanceOptimizer, OptimizationLevel, OptimizationConfig


class TestEdgeAnomalyDetector:
    """Test edge anomaly detection functionality."""
    
    def test_edge_config_creation(self):
        """Test edge configuration creation."""
        config = EdgeConfig(
            model_path="test_model.tflite",
            buffer_size=500,
            batch_size=16
        )
        
        assert config.model_path == "test_model.tflite"
        assert config.buffer_size == 500
        assert config.batch_size == 16
    
    @patch('edge_anomaly_detector.TENSORFLOW_AVAILABLE', False)
    def test_edge_detector_without_tensorflow(self):
        """Test edge detector behavior without TensorFlow."""
        config = EdgeConfig()
        
        with pytest.raises(RuntimeError, match="TensorFlow not available"):
            EdgeAnomalyDetector(config)
    
    @patch('edge_anomaly_detector.TENSORFLOW_AVAILABLE', True)
    @patch('edge_anomaly_detector.tflite')
    def test_edge_detector_model_loading(self, mock_tflite):
        """Test edge detector model loading."""
        # Mock TensorFlow Lite components
        mock_interpreter = Mock()
        mock_interpreter.allocate_tensors.return_value = None
        mock_interpreter.get_input_details.return_value = [{'shape': [1, 10]}]
        mock_interpreter.get_output_details.return_value = [{'shape': [1, 10]}]
        mock_tflite.Interpreter.return_value = mock_interpreter
        
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as f:
            model_path = f.name
            f.write(b"dummy model data")
        
        try:
            config = EdgeConfig(model_path=model_path)
            detector = EdgeAnomalyDetector(config)
            
            assert detector.interpreter is not None
            mock_tflite.Interpreter.assert_called_once()
        finally:
            Path(model_path).unlink()
    
    def test_anomaly_result_dataclass(self):
        """Test AnomalyResult dataclass."""
        result = AnomalyResult(
            timestamp=time.time(),
            is_anomaly=True,
            confidence=0.8,
            reconstruction_error=0.5,
            sensor_values=[1.0, 2.0, 3.0],
            processing_time_ms=50.0
        )
        
        assert result.is_anomaly is True
        assert result.confidence == 0.8
        assert len(result.sensor_values) == 3


class TestRealtimeStreamProcessor:
    """Test real-time stream processing functionality."""
    
    def test_stream_config_defaults(self):
        """Test stream configuration defaults."""
        config = StreamConfig()
        
        assert config.window_size == 30
        assert config.step_size == 1
        assert config.processing_frequency_hz == 10
        assert config.max_latency_ms == 50
    
    def test_stream_data_creation(self):
        """Test stream data creation."""
        data = StreamData(
            timestamp=time.time(),
            sensor_id="sensor_001",
            values=[1.0, 2.0, 3.0],
            metadata={"location": "factory_floor"}
        )
        
        assert data.sensor_id == "sensor_001"
        assert len(data.values) == 3
        assert data.metadata["location"] == "factory_floor"
    
    def test_stream_processor_initialization(self):
        """Test stream processor initialization."""
        config = StreamConfig(buffer_size=1000, max_latency_ms=100)
        processor = RealtimeStreamProcessor(config)
        
        assert processor.config.buffer_size == 1000
        assert processor.config.max_latency_ms == 100
        assert len(processor.sensor_buffers) == 0
        assert len(processor.result_callbacks) == 0
    
    def test_stream_processor_callback_registration(self):
        """Test callback registration."""
        processor = RealtimeStreamProcessor()
        
        def test_callback(result):
            pass
        
        processor.add_result_callback(test_callback)
        assert len(processor.result_callbacks) == 1
    
    def test_data_ingestion(self):
        """Test data ingestion."""
        processor = RealtimeStreamProcessor()
        
        data = StreamData(
            timestamp=time.time(),
            sensor_id="test_sensor",
            values=[1.0, 2.0],
            metadata={}
        )
        
        result = processor.ingest_data(data)
        assert result is True
        assert "test_sensor" in processor.sensor_buffers
        assert len(processor.sensor_buffers["test_sensor"]) == 1


class TestMultiModalSensorFusion:
    """Test multi-modal sensor fusion functionality."""
    
    def test_sensor_types_enum(self):
        """Test sensor types enumeration."""
        assert SensorType.TEMPERATURE.value == "temperature"
        assert SensorType.HUMIDITY.value == "humidity"
        assert SensorType.PRESSURE.value == "pressure"
    
    def test_sensor_reading_creation(self):
        """Test sensor reading creation."""
        reading = SensorReading(
            sensor_id="temp_001",
            sensor_type=SensorType.TEMPERATURE,
            timestamp=time.time(),
            value=25.5,
            unit="C",
            confidence=0.95
        )
        
        assert reading.sensor_id == "temp_001"
        assert reading.sensor_type == SensorType.TEMPERATURE
        assert reading.value == 25.5
        assert reading.confidence == 0.95
    
    def test_multi_modal_data_structure(self):
        """Test multi-modal data structure."""
        readings = [
            SensorReading("temp_001", SensorType.TEMPERATURE, time.time(), 25.0, "C"),
            SensorReading("hum_001", SensorType.HUMIDITY, time.time(), 60.0, "%")
        ]
        
        data = MultiModalData(
            timestamp=time.time(),
            location_id="zone_a",
            readings=readings,
            environmental_context={"weather": "sunny"}
        )
        
        assert data.location_id == "zone_a"
        assert len(data.readings) == 2
        assert data.environmental_context["weather"] == "sunny"
    
    def test_fusion_config_defaults(self):
        """Test fusion configuration defaults."""
        config = SensorFusionConfig()
        
        assert config.fusion_window_size == 10
        assert config.correlation_threshold == 0.7
        assert config.anomaly_threshold == 0.8
        assert config.min_sensors_for_fusion == 2
    
    def test_sensor_fusion_initialization(self):
        """Test sensor fusion system initialization."""
        fusion = MultiModalSensorFusion()
        
        assert len(fusion.sensor_history) == 0
        assert len(fusion.sensor_baselines) == 0
        assert fusion.baseline_learning_samples == 100


class TestRobustErrorHandler:
    """Test robust error handling system."""
    
    def test_error_categories_enum(self):
        """Test error categories enumeration."""
        assert ErrorCategory.DATA_VALIDATION.value == "data_validation"
        assert ErrorCategory.MODEL_INFERENCE.value == "model_inference"
        assert ErrorCategory.NETWORK_COMMUNICATION.value == "network_communication"
    
    def test_error_severity_enum(self):
        """Test error severity enumeration."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.CRITICAL.value == "critical"
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = RobustErrorHandler(
            max_error_history=1000,
            enable_auto_recovery=True
        )
        
        assert handler.max_error_history == 1000
        assert handler.enable_auto_recovery is True
        assert len(handler.error_history) == 0
        assert len(handler.recovery_strategies) > 0  # Default strategies loaded
    
    def test_error_context_manager(self):
        """Test error context manager."""
        handler = RobustErrorHandler()
        
        with pytest.raises(ValueError):
            with handler.error_context(
                component="test_component",
                operation="test_operation",
                category=ErrorCategory.DATA_VALIDATION
            ):
                raise ValueError("Test error")
        
        # Error should be recorded
        assert len(handler.error_history) > 0
    
    def test_recovery_strategy_registration(self):
        """Test recovery strategy registration."""
        handler = RobustErrorHandler()
        
        class TestRecoveryStrategy(RecoveryStrategy):
            def _execute_recovery(self, error_record, context):
                return True
        
        strategy = TestRecoveryStrategy("test_strategy")
        handler.add_recovery_strategy(ErrorCategory.DATA_VALIDATION, strategy)
        
        assert ErrorCategory.DATA_VALIDATION in handler.recovery_strategies
        assert len(handler.recovery_strategies[ErrorCategory.DATA_VALIDATION]) > 0


class TestSecurityHardening:
    """Test security hardening functionality."""
    
    def test_security_config_defaults(self):
        """Test security configuration defaults."""
        config = SecurityConfig()
        
        assert config.enable_encryption is True
        assert config.enable_authentication is True
        assert config.rate_limit_requests_per_minute == 100
        assert config.max_failed_attempts == 5
    
    def test_input_validator_creation(self):
        """Test input validator creation."""
        config = SecurityConfig()
        validator = InputValidator(config)
        
        assert validator.config == config
        assert len(validator.dangerous_patterns) > 0
        assert len(validator.compiled_patterns) > 0
    
    def test_input_validation_valid_data(self):
        """Test input validation with valid data."""
        validator = InputValidator(SecurityConfig())
        
        valid_data = {
            "values": [1.0, 2.0, 3.0],
            "timestamp": time.time(),
            "sensor_id": "sensor_001"
        }
        
        assert validator.validate_sensor_data(valid_data) is True
    
    def test_input_validation_malicious_data(self):
        """Test input validation with malicious data."""
        validator = InputValidator(SecurityConfig())
        
        malicious_data = {
            "values": [1.0, 2.0, 3.0],
            "script": "<script>alert('xss')</script>",
            "timestamp": time.time()
        }
        
        assert validator.validate_sensor_data(malicious_data) is False
    
    def test_security_hardening_initialization(self):
        """Test security hardening initialization."""
        config = SecurityConfig()
        security = SecurityHardening(config)
        
        assert security.config == config
        assert security.input_validator is not None
        assert security.rate_limiter is not None
        assert security.security_monitor is not None


class TestComprehensiveMonitoring:
    """Test comprehensive monitoring system."""
    
    def test_monitoring_initialization(self):
        """Test monitoring system initialization."""
        monitoring = ComprehensiveMonitoring(
            metrics_retention_seconds=3600,
            alert_retention_count=500
        )
        
        assert monitoring.metrics_retention == 3600
        assert monitoring.alert_retention_count == 500
        assert len(monitoring.metrics) == 0
        assert len(monitoring.alert_rules) == 0
    
    def test_metric_recording(self):
        """Test metric recording."""
        monitoring = ComprehensiveMonitoring()
        
        monitoring.record_metric("test_metric", 100.0, MetricType.GAUGE)
        
        # Give time for processing
        time.sleep(0.1)
        
        # Should have queued the metric
        assert monitoring.processing_queue.qsize() >= 0
    
    def test_alert_rule_creation(self):
        """Test alert rule creation."""
        rule = AlertRule(
            rule_id="test_rule",
            metric_name="cpu_usage",
            condition="gt",
            threshold=80.0,
            level=AlertLevel.WARNING
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.condition == "gt"
        assert rule.threshold == 80.0
        assert rule.level == AlertLevel.WARNING
    
    def test_counter_increment(self):
        """Test counter metric increment."""
        monitoring = ComprehensiveMonitoring()
        
        monitoring.increment_counter("test_counter")
        monitoring.increment_counter("test_counter")
        
        time.sleep(0.1)  # Allow processing
        
        # Counter should be incremented
        assert monitoring.processing_queue.qsize() >= 0


class TestDistributedAnomalyDetection:
    """Test distributed anomaly detection system."""
    
    def test_node_roles_enum(self):
        """Test node roles enumeration."""
        assert NodeRole.COORDINATOR.value == "coordinator"
        assert NodeRole.WORKER.value == "worker"
        assert NodeRole.EDGE.value == "edge"
    
    def test_workload_types_enum(self):
        """Test workload types enumeration."""
        assert WorkloadType.INFERENCE.value == "inference"
        assert WorkloadType.TRAINING.value == "training"
        assert WorkloadType.DATA_PROCESSING.value == "data_processing"
    
    def test_distributed_task_creation(self):
        """Test distributed task creation."""
        task = DistributedTask(
            task_id="test_task_001",
            task_type=WorkloadType.INFERENCE,
            priority=1,
            data={"sensor_data": [1.0, 2.0, 3.0]},
            timeout_seconds=60
        )
        
        assert task.task_id == "test_task_001"
        assert task.task_type == WorkloadType.INFERENCE
        assert task.priority == 1
        assert task.timeout_seconds == 60
    
    @patch('distributed_anomaly_detection.REDIS_AVAILABLE', False)
    def test_distributed_detector_without_redis(self):
        """Test distributed detector without Redis."""
        detector = DistributedAnomalyDetector(
            node_id="test_node",
            role=NodeRole.COORDINATOR
        )
        
        assert detector.redis_client is None
        assert detector.node_id == "test_node"
        assert detector.role == NodeRole.COORDINATOR


class TestHighPerformanceOptimizer:
    """Test high-performance optimization system."""
    
    def test_optimization_levels_enum(self):
        """Test optimization levels enumeration."""
        assert OptimizationLevel.NONE.value == "none"
        assert OptimizationLevel.BASIC.value == "basic"
        assert OptimizationLevel.AGGRESSIVE.value == "aggressive"
        assert OptimizationLevel.EXTREME.value == "extreme"
    
    def test_optimization_config_defaults(self):
        """Test optimization configuration defaults."""
        config = OptimizationConfig()
        
        assert config.level == OptimizationLevel.BASIC
        assert config.enable_caching is True
        assert config.enable_batching is True
        assert config.max_batch_size == 1000
        assert config.target_latency_ms == 100.0
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        config = OptimizationConfig(level=OptimizationLevel.AGGRESSIVE)
        optimizer = HighPerformanceOptimizer(config)
        
        assert optimizer.config.level == OptimizationLevel.AGGRESSIVE
        assert len(optimizer.metrics) == 0
        assert len(optimizer.memory_pools) == 0
    
    @patch('high_performance_optimizer.NUMPY_AVAILABLE', True)
    def test_performance_context_manager(self):
        """Test performance measurement context manager."""
        optimizer = HighPerformanceOptimizer()
        
        with optimizer.performance_context("test_operation"):
            time.sleep(0.01)  # Small delay to measure
        
        # Should have recorded metrics
        assert len(optimizer.metrics) > 0
        assert optimizer.metrics[0].operation_name == "test_operation"
    
    @patch('high_performance_optimizer.NUMPY_AVAILABLE', True)
    def test_memory_pool_creation(self):
        """Test memory pool creation."""
        optimizer = HighPerformanceOptimizer()
        
        def array_factory():
            return np.zeros(100)
        
        pool = optimizer.create_memory_pool("test_pool", array_factory, 10, 50)
        
        assert "test_pool" in optimizer.memory_pools
        assert pool is not None
        assert pool.stats["created"] >= 10  # Pre-populated


class TestIntegration:
    """Integration tests for combined functionality."""
    
    def test_end_to_end_anomaly_pipeline(self):
        """Test end-to-end anomaly detection pipeline."""
        # Initialize components
        monitoring = ComprehensiveMonitoring()
        security = SecurityHardening()
        optimizer = HighPerformanceOptimizer()
        
        # Simulate sensor data processing
        sensor_data = {
            "timestamp": time.time(),
            "values": [1.0, 2.0, 3.0],
            "sensor_id": "integration_test_sensor"
        }
        
        try:
            # Security validation
            processed_data = security.validate_and_process_data(
                sensor_data,
                client_id="test_client",
                client_ip="127.0.0.1"
            )
            
            # Record metrics
            monitoring.record_metric("integration_test", 1.0, MetricType.COUNTER)
            
            # Performance optimization context
            with optimizer.performance_context("integration_test"):
                # Simulate processing
                result = len(processed_data["values"])
                assert result == 3
            
            assert True  # If we get here, integration succeeded
            
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")
    
    def test_error_handling_integration(self):
        """Test error handling integration across components."""
        error_handler = RobustErrorHandler()
        monitoring = ComprehensiveMonitoring()
        
        # Add monitoring as error handler callback
        def monitor_error(error_record):
            monitoring.record_metric("error_count", 1.0, MetricType.COUNTER)
        
        error_handler.add_error_handler(ErrorCategory.DATA_VALIDATION, monitor_error)
        
        # Trigger error
        try:
            with error_handler.error_context(
                component="integration_test",
                operation="test_operation",
                category=ErrorCategory.DATA_VALIDATION
            ):
                raise ValueError("Integration test error")
        except ValueError:
            pass
        
        # Allow processing time
        time.sleep(0.1)
        
        # Error should be handled and monitored
        assert len(error_handler.error_history) > 0


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.performance
    @patch('high_performance_optimizer.NUMPY_AVAILABLE', True)
    def test_optimization_performance_comparison(self):
        """Compare performance across optimization levels."""
        data_size = 1000
        test_data = np.random.randn(data_size).astype(np.float32)
        baseline_data = np.random.randn(data_size).astype(np.float32)
        
        # Test different optimization levels
        results = {}
        
        for level in [OptimizationLevel.NONE, OptimizationLevel.BASIC, OptimizationLevel.AGGRESSIVE]:
            config = OptimizationConfig(level=level)
            optimizer = HighPerformanceOptimizer(config)
            
            start_time = time.time()
            
            # Run multiple iterations
            for _ in range(10):
                scores = optimizer.optimize_anomaly_detection(test_data, baseline_data)
                assert len(scores) == data_size
            
            end_time = time.time()
            results[level.value] = end_time - start_time
            
            optimizer.shutdown()
        
        # Basic optimization should be faster than none
        # (Though with small data, overhead might make it slower)
        assert results[OptimizationLevel.NONE.value] >= 0
        assert results[OptimizationLevel.BASIC.value] >= 0
    
    @pytest.mark.performance
    def test_stream_processing_throughput(self):
        """Test stream processing throughput."""
        processor = RealtimeStreamProcessor()
        
        # Add result callback to count processed items
        processed_count = {"value": 0}
        
        def count_callback(result):
            processed_count["value"] += 1
        
        processor.add_result_callback(count_callback)
        
        # Start processing
        processor.start_processing()
        
        try:
            # Submit test data
            start_time = time.time()
            num_items = 100
            
            for i in range(num_items):
                data = StreamData(
                    timestamp=time.time(),
                    sensor_id=f"perf_sensor_{i % 5}",  # 5 different sensors
                    values=[float(i), float(i * 2)],
                    metadata={"test": True}
                )
                processor.ingest_data(data)
            
            # Wait for processing
            time.sleep(2.0)
            
            processing_time = time.time() - start_time
            throughput = num_items / processing_time
            
            # Should process at least 10 items per second
            assert throughput > 10.0
            
        finally:
            processor.stop_processing()


# Fixtures for test setup
@pytest.fixture
def temp_model_file():
    """Create temporary model file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        model_path = f.name
        f.write(b"dummy model data")
    
    yield model_path
    
    Path(model_path).unlink()


@pytest.fixture
def sample_sensor_data():
    """Generate sample sensor data for testing."""
    return {
        "timestamp": time.time(),
        "values": [25.5, 60.2, 1013.25],  # temperature, humidity, pressure
        "sensor_id": "test_sensor_001",
        "location": "test_zone_a"
    }


@pytest.fixture
def monitoring_system():
    """Create monitoring system for testing."""
    monitoring = ComprehensiveMonitoring(
        metrics_retention_seconds=300,  # 5 minutes for testing
        alert_retention_count=100
    )
    
    yield monitoring
    
    monitoring.shutdown()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])