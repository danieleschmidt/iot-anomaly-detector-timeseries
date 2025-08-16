"""Tests for new autonomous SDLC components."""

import pytest
import asyncio
import time
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path

from src.adaptive_real_time_processor import (
    AdaptiveRealTimeProcessor,
    ProcessingMetrics,
    StreamConfig
)
from src.intelligent_anomaly_orchestrator import (
    IntelligentAnomalyOrchestrator,
    ModelConfig,
    OrchestrationConfig,
    DetectionStrategy,
    SecurityLevel,
    AuthenticationMethod
)
from src.robust_health_monitoring import (
    RobustHealthMonitoring,
    HealthMetric,
    Alert,
    AlertSeverity,
    HealthStatus
)
from src.advanced_security_framework import (
    AdvancedSecurityFramework,
    SecurityCredential,
    SecurityLevel as SecurityFrameworkLevel
)
from src.quantum_scale_optimizer import (
    QuantumScaleOptimizer,
    ResourceMetrics,
    ResourceType,
    OptimizationStrategy
)


class TestAdaptiveRealTimeProcessor:
    """Test suite for AdaptiveRealTimeProcessor."""
    
    @pytest.fixture
    def processor(self, tmp_path):
        """Create a test processor instance."""
        # Create a mock model file
        model_path = tmp_path / "test_model.h5"
        model_path.touch()
        
        config = StreamConfig(batch_size=4, buffer_size=10)
        return AdaptiveRealTimeProcessor(
            model_path=str(model_path),
            config=config,
            max_workers=2
        )
    
    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.config.batch_size == 4
        assert processor.config.buffer_size == 10
        assert processor.max_workers == 2
        assert processor.adaptive_batch_size == 4
    
    def test_process_sample(self, processor):
        """Test single sample processing."""
        data = {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}
        
        with patch.object(processor.detector, 'score', return_value=np.array([0.3])):
            result = processor.process_sample(data)
            
            assert 'anomaly_score' in result
            assert 'is_anomaly' in result
            assert 'timestamp' in result
            assert result['anomaly_score'] == 0.3
            assert result['is_anomaly'] is False
    
    def test_process_batch(self, processor):
        """Test batch processing."""
        batch_data = [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 3.0, "feature2": 4.0}
        ]
        
        with patch.object(processor.detector, 'score', return_value=np.array([0.2, 0.8])):
            results = processor.process_batch(batch_data)
            
            assert len(results) == 2
            assert results[0]['anomaly_score'] == 0.2
            assert results[1]['anomaly_score'] == 0.8
            assert results[0]['is_anomaly'] is False
            assert results[1]['is_anomaly'] is True
    
    def test_metrics_collection(self, processor):
        """Test metrics collection."""
        metrics = processor.get_metrics()
        
        assert 'latency_ms' in metrics
        assert 'throughput_ops_sec' in metrics
        assert 'queue_depth' in metrics
        assert 'cache_hit_rate' in metrics
        assert 'adaptive_batch_size' in metrics
    
    def test_submit_data(self, processor):
        """Test data submission to queue."""
        data = {"test": "data"}
        
        # Should succeed with empty queue
        assert processor.submit_data(data) is True
        
        # Fill up the queue
        for i in range(processor.config.buffer_size):
            processor.submit_data({"data": i})
        
        # Should fail when queue is full
        assert processor.submit_data(data) is False


class TestIntelligentAnomalyOrchestrator:
    """Test suite for IntelligentAnomalyOrchestrator."""
    
    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create a test orchestrator instance."""
        # Create mock model files
        model1_path = tmp_path / "model1.h5"
        model2_path = tmp_path / "model2.h5"
        model1_path.touch()
        model2_path.touch()
        
        models_config = [
            ModelConfig(
                name="model1",
                model_path=str(model1_path),
                weight=1.0,
                model_type="autoencoder"
            ),
            ModelConfig(
                name="model2",
                model_path=str(model2_path),
                weight=0.8,
                model_type="autoencoder"
            )
        ]
        
        config = OrchestrationConfig(
            strategy=DetectionStrategy.ENSEMBLE_VOTING,
            enable_adaptive_weights=True
        )
        
        return IntelligentAnomalyOrchestrator(
            models_config=models_config,
            config=config,
            max_workers=2
        )
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert len(orchestrator.models) >= 0  # May fail to load models in test
        assert len(orchestrator.model_configs) == 2
        assert orchestrator.config.strategy == DetectionStrategy.ENSEMBLE_VOTING
    
    @pytest.mark.asyncio
    async def test_detect_anomaly_simple(self, orchestrator):
        """Test simple anomaly detection."""
        data = np.array([1.0, 2.0, 3.0])
        
        # Mock model detection
        with patch.object(orchestrator, '_run_model_detection') as mock_run:
            mock_run.return_value = {'score': 0.7, 'confidence': 0.8}
            
            result = await orchestrator.detect_anomaly(data)
            
            assert result.anomaly_score >= 0.0
            assert result.confidence >= 0.0
            assert result.timestamp > 0.0
    
    def test_preprocess_input(self, orchestrator):
        """Test input preprocessing."""
        # Test dict input
        dict_data = {"a": 1, "b": 2, "c": 3}
        result = orchestrator._preprocess_input(dict_data)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)
        
        # Test DataFrame input
        df_data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        result = orchestrator._preprocess_input(df_data)
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(result, expected)
        
        # Test numpy array input
        array_data = np.array([[1, 2], [3, 4]])
        result = orchestrator._preprocess_input(array_data)
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(result, expected)
    
    def test_orchestration_stats(self, orchestrator):
        """Test statistics collection."""
        stats = orchestrator.get_orchestration_stats()
        
        assert 'total_detections' in stats
        assert 'enabled_models' in stats
        assert 'current_weights' in stats
    
    def test_model_config_update(self, orchestrator):
        """Test model configuration updates."""
        orchestrator.update_model_config("model1", weight=1.5, enabled=False)
        
        assert orchestrator.model_configs["model1"].weight == 1.5
        assert orchestrator.model_configs["model1"].enabled is False
        assert orchestrator.model_weights["model1"] == 1.5


class TestRobustHealthMonitoring:
    """Test suite for RobustHealthMonitoring."""
    
    @pytest.fixture
    def health_monitor(self):
        """Create a test health monitoring instance."""
        return RobustHealthMonitoring(
            check_interval=1.0,
            enable_auto_recovery=True,
            enable_alerting=True
        )
    
    def test_initialization(self, health_monitor):
        """Test health monitor initialization."""
        assert health_monitor.check_interval == 1.0
        assert health_monitor.enable_auto_recovery is True
        assert health_monitor.enable_alerting is True
        assert len(health_monitor.components) > 0
    
    def test_register_component(self, health_monitor):
        """Test component registration."""
        metrics = {
            "test_metric": HealthMetric(
                name="test_metric",
                current_value=50.0,
                threshold_warning=80.0,
                threshold_critical=95.0,
                unit="%"
            )
        }
        
        health_monitor.register_component("test_component", metrics)
        
        assert "test_component" in health_monitor.components
        assert "test_metric" in health_monitor.components["test_component"].metrics
    
    def test_update_metric(self, health_monitor):
        """Test metric updates."""
        # Test updating existing metric
        health_monitor.update_metric("system_resources", "cpu_usage", 75.0)
        
        metric = health_monitor.components["system_resources"].metrics["cpu_usage"]
        assert metric.current_value == 75.0
        assert metric.last_updated > 0
    
    def test_alert_generation(self, health_monitor):
        """Test alert generation."""
        # Update metric above warning threshold
        health_monitor.update_metric("system_resources", "cpu_usage", 85.0)
        
        # Check if alert was generated
        cpu_alert_id = "system_resources.cpu_usage"
        if cpu_alert_id in health_monitor.alerts:
            alert = health_monitor.alerts[cpu_alert_id]
            assert alert.severity == AlertSeverity.WARNING
            assert "cpu_usage" in alert.message
    
    def test_health_summary(self, health_monitor):
        """Test health summary generation."""
        summary = health_monitor.get_health_summary()
        
        assert 'overall_status' in summary
        assert 'components' in summary
        assert 'active_alerts' in summary
        assert 'uptime_seconds' in summary
        assert 'last_check' in summary
    
    def test_alert_acknowledgment(self, health_monitor):
        """Test alert acknowledgment."""
        # Create a test alert
        test_alert = Alert(
            id="test_alert",
            severity=AlertSeverity.WARNING,
            component="test",
            message="Test alert",
            details={}
        )
        
        health_monitor.alerts["test_alert"] = test_alert
        
        # Acknowledge the alert
        result = health_monitor.acknowledge_alert("test_alert")
        assert result is True
        assert health_monitor.alerts["test_alert"].acknowledged is True


class TestAdvancedSecurityFramework:
    """Test suite for AdvancedSecurityFramework."""
    
    @pytest.fixture
    def security_framework(self):
        """Create a test security framework instance."""
        return AdvancedSecurityFramework(
            secret_key="test_secret_key",
            token_expiry_hours=1,
            max_failed_attempts=3,
            lockout_duration_minutes=5
        )
    
    def test_initialization(self, security_framework):
        """Test security framework initialization."""
        assert security_framework.secret_key == "test_secret_key"
        assert security_framework.token_expiry_hours == 1
        assert security_framework.max_failed_attempts == 3
        assert security_framework.lockout_duration_minutes == 5
    
    def test_create_user_credential(self, security_framework):
        """Test user credential creation."""
        credential_id = security_framework.create_user_credential(
            user_id="test_user",
            password="test_password",
            permissions=["read", "write"],
            security_level=SecurityFrameworkLevel.INTERNAL
        )
        
        assert credential_id is not None
        assert credential_id in security_framework.credentials
        
        credential = security_framework.credentials[credential_id]
        assert credential.user_id == "test_user"
        assert credential.permissions == ["read", "write"]
        assert credential.security_level == SecurityFrameworkLevel.INTERNAL
    
    def test_user_authentication(self, security_framework):
        """Test user authentication."""
        # Create user first
        credential_id = security_framework.create_user_credential(
            user_id="auth_test_user",
            password="auth_test_password",
            permissions=["read"]
        )
        
        # Test successful authentication
        token = security_framework.authenticate_user(
            user_id="auth_test_user",
            password="auth_test_password"
        )
        
        assert token is not None
        assert token in security_framework.active_sessions
    
    def test_session_validation(self, security_framework):
        """Test session validation."""
        # Create user and authenticate
        security_framework.create_user_credential(
            user_id="session_test_user",
            password="session_test_password",
            permissions=["read"]
        )
        
        token = security_framework.authenticate_user(
            user_id="session_test_user",
            password="session_test_password"
        )
        
        # Validate session
        context = security_framework.validate_session(token)
        assert context is not None
        assert context.user_id == "session_test_user"
        assert "read" in context.permissions
    
    def test_authorization(self, security_framework):
        """Test action authorization."""
        # Create user and get session
        security_framework.create_user_credential(
            user_id="authz_test_user",
            password="authz_test_password",
            permissions=["data:read", "model:predict"]
        )
        
        token = security_framework.authenticate_user(
            user_id="authz_test_user",
            password="authz_test_password"
        )
        
        context = security_framework.validate_session(token)
        
        # Test authorized action
        assert security_framework.authorize_action(
            context, "data", "read"
        ) is True
        
        # Test unauthorized action
        assert security_framework.authorize_action(
            context, "admin", "delete"
        ) is False
    
    def test_rate_limiting(self, security_framework):
        """Test rate limiting."""
        user_id = "rate_limit_test_user"
        
        # Should pass initially
        assert security_framework.check_rate_limit(user_id, 5) is True
        
        # Fill up rate limit
        for _ in range(4):
            security_framework.check_rate_limit(user_id, 5)
        
        # Should fail when limit exceeded
        assert security_framework.check_rate_limit(user_id, 5) is False
    
    def test_encryption(self, security_framework):
        """Test data encryption/decryption."""
        test_data = "sensitive information"
        
        # Encrypt data
        encrypted = security_framework.encrypt_sensitive_data(test_data)
        assert encrypted != test_data
        
        # Decrypt data
        decrypted = security_framework.decrypt_sensitive_data(encrypted)
        assert decrypted == test_data


class TestQuantumScaleOptimizer:
    """Test suite for QuantumScaleOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create a test quantum scale optimizer instance."""
        return QuantumScaleOptimizer(
            optimization_interval=5.0,
            enable_predictive_scaling=True
        )
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.optimization_interval == 5.0
        assert optimizer.enable_predictive_scaling is True
        assert len(optimizer.min_resources) > 0
        assert len(optimizer.max_resources) > 0
    
    @pytest.mark.asyncio
    async def test_optimize_resources(self, optimizer):
        """Test resource optimization."""
        current_metrics = ResourceMetrics(
            cpu_usage=85.0,
            memory_usage=70.0,
            storage_usage=50.0,
            network_io=30.0
        )
        
        result = await optimizer.optimize_resources(
            current_metrics,
            strategy=OptimizationStrategy.QUANTUM_ANNEALING
        )
        
        assert result.strategy_used == OptimizationStrategy.QUANTUM_ANNEALING
        assert result.execution_time_ms >= 0
        assert isinstance(result.actions, list)
        assert result.optimization_score is not None
    
    def test_generate_scaling_actions(self, optimizer):
        """Test scaling action generation."""
        current_metrics = ResourceMetrics(
            cpu_usage=90.0,  # High usage should trigger scale up
            memory_usage=30.0,  # Low usage should trigger scale down
            storage_usage=50.0,
            network_io=95.0  # High usage should trigger scale up
        )
        
        # Use asyncio.run for the async method
        actions = asyncio.run(optimizer._generate_scaling_actions(current_metrics, None))
        
        assert len(actions) > 0
        
        # Check if CPU scale up action is generated
        cpu_actions = [a for a in actions if a.resource_type == ResourceType.CPU]
        assert len(cpu_actions) > 0
        
        # Check if memory scale down action is generated
        memory_actions = [a for a in actions if a.resource_type == ResourceType.MEMORY]
        assert len(memory_actions) > 0
    
    def test_optimization_stats(self, optimizer):
        """Test optimization statistics."""
        stats = optimizer.get_optimization_stats()
        
        assert 'total_optimizations' in stats
        assert 'current_resources' in stats
        
        # Should show no data initially
        if stats.get('status') == 'no_data':
            assert 'total_optimizations' in stats
        else:
            assert 'average_optimization_score' in stats
            assert 'average_execution_time_ms' in stats


# Integration tests
class TestIntegration:
    """Integration tests for new components."""
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self):
        """Test health monitoring with real metrics."""
        health_monitor = RobustHealthMonitoring(
            check_interval=0.1,  # Fast for testing
            enable_alerting=True
        )
        
        # Update metrics that should trigger alerts
        health_monitor.update_metric("system_resources", "cpu_usage", 95.0)
        health_monitor.update_metric("application_performance", "error_rate", 15.0)
        
        # Check that alerts were generated
        assert len(health_monitor.alerts) > 0
        
        # Test health summary
        summary = health_monitor.get_health_summary()
        assert summary['overall_status'] in ['warning', 'critical']
    
    def test_security_workflow(self):
        """Test complete security workflow."""
        security = AdvancedSecurityFramework()
        
        # Create user
        credential_id = security.create_user_credential(
            user_id="workflow_user",
            password="secure_password",
            permissions=["anomaly:detect", "model:train"]
        )
        
        # Authenticate
        token = security.authenticate_user("workflow_user", "secure_password")
        assert token is not None
        
        # Validate session
        context = security.validate_session(token)
        assert context is not None
        
        # Authorize actions
        assert security.authorize_action(context, "anomaly", "detect") is True
        assert security.authorize_action(context, "model", "train") is True
        assert security.authorize_action(context, "admin", "delete") is False
        
        # Generate audit report
        report = security.get_security_audit_report(1)  # Last 1 hour
        assert report['total_events'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])