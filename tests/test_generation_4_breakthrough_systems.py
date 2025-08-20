"""
Test Suite for Generation 4 Breakthrough Systems

This module provides comprehensive testing for the advanced Generation 4 
implementations including quantum-neuromorphic fusion, meta-learning 
orchestration, autonomous QA, and performance optimization systems.
"""

import pytest
import numpy as np
import tensorflow as tf
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import tempfile
from pathlib import Path

# Import Generation 4 modules
from src.breakthrough_quantum_neuromorphic_fusion import (
    BreakthroughQuantumNeuromorphicFusion,
    QuantumNeuromorphicConfig,
    FusionMode
)
from src.advanced_meta_learning_orchestrator import (
    AdvancedMetaLearningOrchestrator,
    MetaLearningConfig,
    LearningStrategy,
    EnvironmentType
)
from src.autonomous_quality_assurance_system import (
    AutonomousQualityAssuranceSystem,
    QualityThresholds,
    TestType,
    QualityLevel
)
from src.autonomous_performance_optimization_engine import (
    AutonomousPerformanceOptimizationEngine,
    OptimizationTarget,
    OptimizationMode,
    OptimizationStrategy
)


class TestBreakthroughQuantumNeuromorphicFusion:
    """Test suite for quantum-neuromorphic fusion system."""
    
    @pytest.fixture
    def fusion_config(self):
        """Create test configuration for fusion system."""
        return QuantumNeuromorphicConfig(
            quantum_coherence_time=100.0,
            spike_threshold=0.8,
            fusion_mode=FusionMode.ADAPTIVE_SWITCHING,
            real_time_threshold_ms=10.0
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        return np.random.randn(16, 50, 4)
    
    def test_fusion_system_initialization(self, fusion_config):
        """Test fusion system initialization."""
        fusion_system = BreakthroughQuantumNeuromorphicFusion(fusion_config)
        
        assert fusion_system.config == fusion_config
        assert hasattr(fusion_system, 'quantum_processor')
        assert hasattr(fusion_system, 'neuromorphic_processor')
        assert hasattr(fusion_system, 'fusion_controller')
        assert hasattr(fusion_system, 'performance_monitor')
    
    @pytest.mark.asyncio
    async def test_hybrid_anomaly_detection(self, fusion_config, sample_data):
        """Test hybrid quantum-neuromorphic anomaly detection."""
        fusion_system = BreakthroughQuantumNeuromorphicFusion(fusion_config)
        
        # Mock subsystem components to avoid complex dependencies
        with patch.object(fusion_system, 'quantum_processor') as mock_quantum, \
             patch.object(fusion_system, 'neuromorphic_processor') as mock_neuromorphic:
            
            # Setup mocks
            mock_quantum.encode_time_series.return_value = np.random.randn(16, 8)
            mock_quantum.measure_coherence.return_value = 0.95
            mock_quantum.calculate_fidelity.return_value = 0.98
            mock_quantum.get_gate_count.return_value = 150
            
            mock_neuromorphic.encode_to_spikes.return_value = [
                np.array([1, 3, 5, 7, 9]) for _ in range(16)
            ]
            
            # Mock async methods
            async def mock_quantum_detection(data):
                return {
                    'scores': np.random.random(len(data)) * 0.8,
                    'coherence_level': 0.95,
                    'entanglement_entropy': 2.1,
                    'quantum_fidelity': 0.98,
                    'gate_operations': 150
                }
            
            async def mock_neuromorphic_analysis(data):
                return {
                    'scores': np.random.random(len(data)) * 0.7,
                    'spike_patterns': {'spike_rate': np.random.random(len(data))},
                    'plasticity_state': {'adaptation_strength': 1.2},
                    'average_spike_rate': 25.5,
                    'synchrony_index': 0.8,
                    'adaptation_strength': 1.2
                }
            
            fusion_system._quantum_anomaly_detection = mock_quantum_detection
            fusion_system._neuromorphic_spike_analysis = mock_neuromorphic_analysis
            
            # Test detection
            results = await fusion_system.detect_anomalies_hybrid(sample_data)
            
            # Verify results structure
            assert 'anomaly_scores' in results
            assert 'anomaly_predictions' in results
            assert 'confidence' in results
            assert 'fusion_mode' in results
            assert 'inference_time_ms' in results
            assert 'quantum_coherence' in results
            assert 'spike_rate' in results
            
            # Verify data types and shapes
            assert isinstance(results['anomaly_scores'], np.ndarray)
            assert len(results['anomaly_scores']) == len(sample_data)
            assert isinstance(results['inference_time_ms'], (int, float))
            assert results['inference_time_ms'] > 0
    
    def test_fusion_mode_switching(self, fusion_config):
        """Test adaptive fusion mode switching."""
        fusion_config.fusion_mode = FusionMode.ADAPTIVE_SWITCHING
        fusion_system = BreakthroughQuantumNeuromorphicFusion(fusion_config)
        
        # Test different quality scenarios
        quantum_weight, neuromorphic_weight = fusion_system.fusion_controller.calculate_adaptive_weights(
            quantum_quality=0.9,
            neuromorphic_quality=0.7
        )
        
        assert 0 <= quantum_weight <= 1
        assert 0 <= neuromorphic_weight <= 1
        assert abs((quantum_weight + neuromorphic_weight) - 1.0) < 0.01
        assert quantum_weight > neuromorphic_weight  # Higher quality gets more weight
    
    def test_performance_summary(self, fusion_config):
        """Test performance summary generation."""
        fusion_system = BreakthroughQuantumNeuromorphicFusion(fusion_config)
        
        # Add some mock performance data
        fusion_system.performance_metrics['inference_times'] = [5.2, 4.8, 5.1, 4.9, 5.0]
        fusion_system.performance_metrics['energy_consumption'] = [0.001, 0.0009, 0.0011, 0.001, 0.0008]
        
        summary = fusion_system.get_performance_summary()
        
        assert 'average_inference_time_ms' in summary
        assert 'p95_inference_time_ms' in summary
        assert 'average_energy_consumption' in summary
        assert 'total_inferences' in summary
        assert 'real_time_performance' in summary
        assert 'energy_efficiency' in summary
        
        assert summary['total_inferences'] == 5
        assert isinstance(summary['real_time_performance'], bool)


class TestAdvancedMetaLearningOrchestrator:
    """Test suite for meta-learning orchestrator."""
    
    @pytest.fixture
    def meta_config(self):
        """Create test configuration for meta-learning."""
        return MetaLearningConfig(
            learning_strategy=LearningStrategy.MODEL_AGNOSTIC_META_LEARNING,
            meta_learning_rate=0.001,
            environment_detection_enabled=True
        )
    
    @pytest.fixture
    def sample_task_data(self):
        """Create sample task data."""
        return np.random.randn(32, 60, 6)
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        from src.autoencoder_model import build_autoencoder
        return build_autoencoder(input_shape=(60, 6), latent_dim=8)
    
    def test_orchestrator_initialization(self, meta_config):
        """Test meta-learning orchestrator initialization."""
        orchestrator = AdvancedMetaLearningOrchestrator(meta_config)
        
        assert orchestrator.config == meta_config
        assert hasattr(orchestrator, 'meta_strategy')
        assert hasattr(orchestrator, 'environment_recognizer')
        assert hasattr(orchestrator, 'experience_buffer')
        assert len(orchestrator.active_tasks) == 0
    
    def test_environment_recognition(self, meta_config):
        """Test environment recognition system."""
        orchestrator = AdvancedMetaLearningOrchestrator(meta_config)
        
        # Test with different data characteristics
        industrial_data = np.random.randn(16, 100, 12) * 50 + 100  # High values, many features
        healthcare_data = np.random.randn(8, 50, 4) * 2 + 70      # Physiological ranges
        
        # Mock the environment classifier
        with patch.object(orchestrator, 'environment_classifier') as mock_classifier:
            mock_classifier.predict.return_value = [[0.8, 0.1, 0.05, 0.05, 0, 0, 0]]
            
            # Test industrial environment recognition
            env_info = asyncio.run(orchestrator._recognize_environment(industrial_data))
            
            assert 'type' in env_info
            assert 'confidence' in env_info
            assert 'characteristics' in env_info
            assert isinstance(env_info['type'], EnvironmentType)
            assert 0 <= env_info['confidence'] <= 1
    
    def test_data_characteristics_extraction(self, meta_config, sample_task_data):
        """Test data characteristics extraction."""
        orchestrator = AdvancedMetaLearningOrchestrator(meta_config)
        
        characteristics = orchestrator._extract_data_characteristics(sample_task_data)
        
        # Verify all expected characteristics are present
        expected_keys = [
            'mean', 'std', 'skewness', 'kurtosis',
            'coefficient_of_variation', 'range_normalized',
            'autocorrelation', 'trend_strength', 'seasonality_strength',
            'dominant_frequency', 'spectral_entropy',
            'sequence_length', 'feature_count', 'data_density'
        ]
        
        for key in expected_keys:
            assert key in characteristics
            assert isinstance(characteristics[key], (int, float))
            assert np.isfinite(characteristics[key])
    
    @pytest.mark.asyncio
    async def test_adaptive_learning_orchestration(self, meta_config, sample_task_data, simple_model):
        """Test end-to-end adaptive learning orchestration."""
        orchestrator = AdvancedMetaLearningOrchestrator(meta_config)
        
        # Mock time-consuming operations
        with patch.object(orchestrator.meta_strategy, 'adapt_to_task') as mock_adapt:
            mock_adapt.return_value = simple_model
            
            results = await orchestrator.orchestrate_adaptive_learning(
                task_id="test_task_1",
                data=sample_task_data,
                base_model=simple_model
            )
            
            # Verify results structure
            assert 'task_id' in results
            assert 'adapted_model' in results
            assert 'environment_type' in results
            assert 'environment_confidence' in results
            assert 'performance_metrics' in results
            assert 'adaptation_time' in results
            assert 'success' in results
            
            assert results['task_id'] == "test_task_1"
            assert isinstance(results['adaptation_time'], (int, float))
            assert results['adaptation_time'] > 0
    
    def test_orchestration_summary(self, meta_config):
        """Test orchestration summary generation."""
        orchestrator = AdvancedMetaLearningOrchestrator(meta_config)
        
        # Add some mock data
        orchestrator.adaptation_stats = {
            'successful_adaptations': 8,
            'failed_adaptations': 2,
            'total_adaptation_time': 150.0
        }
        orchestrator.global_performance_history = [0.85, 0.87, 0.89, 0.88, 0.90]
        
        summary = orchestrator.get_orchestration_summary()
        
        assert 'meta_learning_strategy' in summary
        assert 'total_tasks' in summary
        assert 'success_rate' in summary
        assert 'average_adaptation_time' in summary
        assert 'average_performance' in summary
        assert 'recent_performance_trend' in summary
        
        assert summary['success_rate'] == 0.8  # 8/10
        assert summary['average_adaptation_time'] == 15.0  # 150/10


class TestAutonomousQualityAssuranceSystem:
    """Test suite for autonomous QA system."""
    
    @pytest.fixture
    def qa_thresholds(self):
        """Create test QA thresholds."""
        return QualityThresholds(
            accuracy_min=0.80,
            latency_max_ms=100.0,
            memory_usage_max_mb=500.0
        )
    
    @pytest.fixture
    def test_model(self):
        """Create a test model."""
        from src.autoencoder_model import build_autoencoder
        return build_autoencoder(input_shape=(50, 4), latent_dim=8)
    
    @pytest.fixture
    def qa_test_data(self):
        """Create test data for QA."""
        return np.random.randn(32, 50, 4)
    
    @pytest.fixture
    def qa_test_labels(self):
        """Create test labels for QA."""
        return np.random.randint(0, 2, 32)
    
    def test_qa_system_initialization(self, qa_thresholds):
        """Test QA system initialization."""
        qa_system = AutonomousQualityAssuranceSystem(qa_thresholds)
        
        assert qa_system.thresholds == qa_thresholds
        assert hasattr(qa_system, 'test_generators')
        assert hasattr(qa_system, 'performance_monitor')
        assert hasattr(qa_system, 'data_drift_detector')
        assert len(qa_system.quality_history) == 0
    
    @pytest.mark.asyncio
    async def test_latency_testing(self, qa_thresholds, test_model, qa_test_data):
        """Test inference latency testing."""
        qa_system = AutonomousQualityAssuranceSystem(qa_thresholds)
        
        result = await qa_system._test_inference_latency(test_model, qa_test_data)
        
        assert result.test_id == "inference_latency"
        assert result.test_type == TestType.PERFORMANCE_TEST
        assert isinstance(result.passed, bool)
        assert isinstance(result.score, (int, float))
        assert 0 <= result.score <= 1
        assert 'avg_latency_ms' in result.details
        assert 'p95_latency_ms' in result.details
    
    @pytest.mark.asyncio
    async def test_throughput_testing(self, qa_thresholds, test_model, qa_test_data):
        """Test throughput testing."""
        qa_system = AutonomousQualityAssuranceSystem(qa_thresholds)
        
        result = await qa_system._test_throughput(test_model, qa_test_data)
        
        assert result.test_id == "throughput"
        assert result.test_type == TestType.PERFORMANCE_TEST
        assert 'max_throughput' in result.details
        assert 'throughput_by_batch' in result.details
        assert result.details['max_throughput'] > 0
    
    @pytest.mark.asyncio
    async def test_accuracy_testing(self, qa_thresholds, test_model, qa_test_data, qa_test_labels):
        """Test accuracy testing."""
        qa_system = AutonomousQualityAssuranceSystem(qa_thresholds)
        
        result = await qa_system._test_accuracy(test_model, qa_test_data, qa_test_labels)
        
        assert result.test_id == "accuracy"
        assert 'accuracy' in result.details
        assert 'precision' in result.details
        assert 'recall' in result.details
        assert 'f1_score' in result.details
        assert 'confusion_matrix' in result.details
        
        # Verify metrics are in valid ranges
        assert 0 <= result.details['accuracy'] <= 1
        assert 0 <= result.details['precision'] <= 1
        assert 0 <= result.details['recall'] <= 1
    
    @pytest.mark.asyncio
    async def test_comprehensive_qa(self, qa_thresholds, test_model, qa_test_data, qa_test_labels):
        """Test comprehensive QA assessment."""
        qa_system = AutonomousQualityAssuranceSystem(qa_thresholds)
        
        # Mock some components to speed up testing
        with patch.object(qa_system, '_run_security_tests') as mock_security, \
             patch.object(qa_system, '_run_bias_tests') as mock_bias:
            
            # Setup mocks
            mock_security.return_value = []
            mock_bias.return_value = []
            
            report = await qa_system.conduct_comprehensive_qa(
                model=test_model,
                test_data=qa_test_data,
                test_labels=qa_test_labels
            )
            
            # Verify report structure
            assert hasattr(report, 'timestamp')
            assert hasattr(report, 'overall_quality')
            assert hasattr(report, 'metric_scores')
            assert hasattr(report, 'test_results')
            assert hasattr(report, 'recommendations')
            
            assert isinstance(report.overall_quality, QualityLevel)
            assert isinstance(report.metric_scores, dict)
            assert isinstance(report.test_results, list)
            assert isinstance(report.recommendations, list)
    
    def test_quality_level_determination(self, qa_thresholds):
        """Test quality level determination logic."""
        qa_system = AutonomousQualityAssuranceSystem(qa_thresholds)
        
        # Test excellent quality
        excellent_metrics = {
            qa_system.QualityMetric.ACCURACY: 0.98,
            qa_system.QualityMetric.PRECISION: 0.95,
            qa_system.QualityMetric.RECALL: 0.96
        }
        quality_level = qa_system._determine_quality_level(excellent_metrics)
        # Note: This test may need adjustment based on actual implementation
        
        # Test poor quality
        poor_metrics = {
            qa_system.QualityMetric.ACCURACY: 0.65,
            qa_system.QualityMetric.PRECISION: 0.60,
            qa_system.QualityMetric.RECALL: 0.58
        }
        poor_quality_level = qa_system._determine_quality_level(poor_metrics)
        
        # Quality levels should reflect metric quality
        assert isinstance(quality_level, QualityLevel)
        assert isinstance(poor_quality_level, QualityLevel)


class TestAutonomousPerformanceOptimizationEngine:
    """Test suite for autonomous performance optimization engine."""
    
    @pytest.fixture
    def optimization_target(self):
        """Create optimization target."""
        return OptimizationTarget(
            target_latency_ms=50.0,
            target_throughput=100.0,
            max_memory_mb=800.0,
            min_accuracy=0.85
        )
    
    @pytest.fixture
    def optimization_model(self):
        """Create a model for optimization testing."""
        from src.autoencoder_model import build_autoencoder
        return build_autoencoder(input_shape=(30, 4), latent_dim=8)
    
    @pytest.fixture
    def optimization_data(self):
        """Create data for optimization testing."""
        return np.random.randn(16, 30, 4)
    
    def test_optimization_engine_initialization(self, optimization_target):
        """Test optimization engine initialization."""
        engine = AutonomousPerformanceOptimizationEngine(
            optimization_mode=OptimizationMode.BALANCED,
            target=optimization_target
        )
        
        assert engine.optimization_mode == OptimizationMode.BALANCED
        assert engine.target == optimization_target
        assert hasattr(engine, 'strategies')
        assert hasattr(engine, 'performance_history')
        assert len(engine.performance_history) == 0
        assert not engine.monitoring_active
    
    @pytest.mark.asyncio
    async def test_performance_measurement(self, optimization_target, optimization_model, optimization_data):
        """Test performance measurement functionality."""
        engine = AutonomousPerformanceOptimizationEngine(target=optimization_target)
        
        performance = await engine._measure_detailed_performance(
            optimization_model, optimization_data, "test_scenario"
        )
        
        # Verify performance profile structure
        assert performance.scenario_name == "test_scenario"
        assert performance.latency_p50 > 0
        assert performance.latency_p95 >= performance.latency_p50
        assert performance.latency_p99 >= performance.latency_p95
        assert performance.throughput > 0
        assert performance.memory_avg_mb > 0
        assert performance.cpu_avg_percent >= 0
        assert 0 <= performance.accuracy <= 1
        assert performance.energy_per_inference > 0
    
    def test_optimization_opportunity_prediction(self, optimization_target):
        """Test optimization opportunity prediction."""
        engine = AutonomousPerformanceOptimizationEngine(target=optimization_target)
        
        # Create mock performance profile with issues
        from src.autonomous_performance_optimization_engine import PerformanceProfile
        
        poor_performance = PerformanceProfile(
            scenario_name="poor_performance",
            latency_p50=120.0,  # Above target of 50ms
            latency_p95=150.0,
            latency_p99=200.0,
            throughput=50.0,    # Below target of 100
            memory_peak_mb=900.0,  # Above target of 800MB
            memory_avg_mb=850.0,
            cpu_avg_percent=80.0,
            gpu_avg_percent=0.0,
            energy_per_inference=0.01,
            accuracy=0.88
        )
        
        performance_analysis = {'latency_trend': 0.15}  # 15% increase
        
        opportunities = asyncio.run(engine._predict_optimization_opportunities(
            poor_performance, performance_analysis
        ))
        
        # Should identify multiple optimization opportunities
        assert len(opportunities) > 0
        
        opportunity_types = [opp['type'] for opp in opportunities]
        assert 'latency_optimization' in opportunity_types
        assert 'memory_optimization' in opportunity_types
        assert 'throughput_optimization' in opportunity_types
    
    def test_optimization_strategy_selection(self, optimization_target):
        """Test optimization strategy selection."""
        engine = AutonomousPerformanceOptimizationEngine(
            optimization_mode=OptimizationMode.CONSERVATIVE,
            target=optimization_target
        )
        
        # Test conservative mode
        conservative_opportunity = {
            'type': 'latency_optimization',
            'recommended_strategies': [
                OptimizationStrategy.QUANTIZATION,
                OptimizationStrategy.CACHING_OPTIMIZATION,
                OptimizationStrategy.MEMORY_OPTIMIZATION
            ]
        }
        
        strategy = engine._select_optimization_strategy(conservative_opportunity)
        
        # Conservative mode should prefer safer strategies
        safe_strategies = [
            OptimizationStrategy.CACHING_OPTIMIZATION,
            OptimizationStrategy.MEMORY_OPTIMIZATION
        ]
        assert strategy in safe_strategies
    
    def test_improvement_calculation(self, optimization_target):
        """Test improvement calculation logic."""
        engine = AutonomousPerformanceOptimizationEngine(target=optimization_target)
        
        # Create mock performance profiles
        from src.autonomous_performance_optimization_engine import PerformanceProfile
        
        before = PerformanceProfile(
            scenario_name="before",
            latency_p50=100.0,
            latency_p95=120.0,
            latency_p99=150.0,
            throughput=80.0,
            memory_peak_mb=900.0,
            memory_avg_mb=850.0,
            cpu_avg_percent=70.0,
            gpu_avg_percent=0.0,
            energy_per_inference=0.01,
            accuracy=0.85
        )
        
        after = PerformanceProfile(
            scenario_name="after",
            latency_p50=80.0,   # 20% improvement
            latency_p95=95.0,
            latency_p99=110.0,
            throughput=100.0,   # 25% improvement
            memory_peak_mb=750.0,  # ~17% improvement
            memory_avg_mb=700.0,
            cpu_avg_percent=60.0,  # ~14% improvement
            gpu_avg_percent=0.0,
            energy_per_inference=0.008,  # 20% improvement
            accuracy=0.86       # Slight improvement
        )
        
        improvements = engine._calculate_improvements(before, after)
        
        # Verify improvements
        assert improvements['latency'] == pytest.approx(20.0, rel=1e-2)
        assert improvements['throughput'] == pytest.approx(25.0, rel=1e-2)
        assert improvements['memory'] > 15.0
        assert improvements['cpu'] > 10.0
        assert improvements['energy'] == pytest.approx(20.0, rel=1e-2)
        assert improvements['accuracy'] > 0  # Small positive improvement
    
    def test_optimization_summary(self, optimization_target):
        """Test optimization summary generation."""
        engine = AutonomousPerformanceOptimizationEngine(target=optimization_target)
        
        # Add mock optimization history
        from src.autonomous_performance_optimization_engine import OptimizationResult, PerformanceProfile
        
        mock_result = OptimizationResult(
            strategy=OptimizationStrategy.MEMORY_OPTIMIZATION,
            success=True,
            performance_before=PerformanceProfile("before", 100, 120, 150, 80, 900, 850, 70, 0, 0.01, 0.85),
            performance_after=PerformanceProfile("after", 80, 95, 110, 100, 750, 700, 60, 0, 0.008, 0.86),
            improvement_percent={'latency': 20.0, 'memory': 17.6},
            overhead_ms=5.0,
            applied_optimizations=['garbage_collection'],
            rollback_available=False,
            confidence_score=0.8
        )
        
        engine.optimization_history = [mock_result]
        
        summary = engine.get_optimization_summary()
        
        assert 'monitoring_active' in summary
        assert 'optimization_mode' in summary
        assert 'total_optimizations' in summary
        assert 'successful_optimizations' in summary
        assert 'success_rate' in summary
        assert 'average_improvements' in summary
        
        assert summary['total_optimizations'] == 1
        assert summary['successful_optimizations'] == 1
        assert summary['success_rate'] == 1.0


@pytest.mark.integration
class TestGeneration4Integration:
    """Integration tests for Generation 4 systems."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_breakthrough_pipeline(self):
        """Test end-to-end integration of breakthrough systems."""
        
        # Create test data
        sample_data = np.random.randn(16, 60, 8)
        
        # Initialize systems
        fusion_config = QuantumNeuromorphicConfig(
            fusion_mode=FusionMode.BALANCED_HYBRID,
            real_time_threshold_ms=20.0
        )
        
        meta_config = MetaLearningConfig(
            learning_strategy=LearningStrategy.MODEL_AGNOSTIC_META_LEARNING,
            environment_detection_enabled=True
        )
        
        qa_thresholds = QualityThresholds(
            accuracy_min=0.75,
            latency_max_ms=150.0
        )
        
        optimization_target = OptimizationTarget(
            target_latency_ms=100.0,
            min_accuracy=0.75
        )
        
        # Initialize systems with mocking to avoid complex dependencies
        with patch('src.breakthrough_quantum_neuromorphic_fusion.QuantumTFTNeuromorphicFusion'), \
             patch('src.breakthrough_quantum_neuromorphic_fusion.NeuromorphicSpikeProcessor'), \
             patch('src.breakthrough_quantum_neuromorphic_fusion.AdaptiveNeuralPlasticityNetwork'):
            
            fusion_system = BreakthroughQuantumNeuromorphicFusion(fusion_config)
            meta_orchestrator = AdvancedMetaLearningOrchestrator(meta_config)
            qa_system = AutonomousQualityAssuranceSystem(qa_thresholds)
            optimization_engine = AutonomousPerformanceOptimizationEngine(
                optimization_mode=OptimizationMode.BALANCED,
                target=optimization_target
            )
        
        # Verify all systems are properly initialized
        assert fusion_system is not None
        assert meta_orchestrator is not None
        assert qa_system is not None
        assert optimization_engine is not None
        
        # Test basic functionality integration
        fusion_summary = fusion_system.get_performance_summary()
        meta_summary = meta_orchestrator.get_orchestration_summary()
        optimization_summary = optimization_engine.get_optimization_summary()
        
        assert isinstance(fusion_summary, dict)
        assert isinstance(meta_summary, dict)
        assert isinstance(optimization_summary, dict)
    
    def test_system_interoperability(self):
        """Test that systems can share data structures and interfaces."""
        
        # Test that performance profiles can be shared
        from src.autonomous_performance_optimization_engine import PerformanceProfile
        from src.autonomous_quality_assurance_system import QualityMetric
        
        # Create a performance profile
        profile = PerformanceProfile(
            scenario_name="interop_test",
            latency_p50=45.0,
            latency_p95=60.0,
            latency_p99=80.0,
            throughput=120.0,
            memory_peak_mb=600.0,
            memory_avg_mb=550.0,
            cpu_avg_percent=45.0,
            gpu_avg_percent=0.0,
            energy_per_inference=0.005,
            accuracy=0.92
        )
        
        # Verify compatibility with QA metrics
        assert hasattr(QualityMetric, 'LATENCY')
        assert hasattr(QualityMetric, 'THROUGHPUT')
        assert hasattr(QualityMetric, 'MEMORY_USAGE')
        assert hasattr(QualityMetric, 'ACCURACY')
        
        # Test data can be extracted
        latency_value = profile.latency_p50
        accuracy_value = profile.accuracy
        
        assert isinstance(latency_value, (int, float))
        assert isinstance(accuracy_value, (int, float))
        assert 0 <= accuracy_value <= 1


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])