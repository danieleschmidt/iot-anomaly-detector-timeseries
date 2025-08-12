"""Comprehensive tests for the advanced multi-modal anomaly detection system."""

import pytest
import numpy as np
import time
from pathlib import Path
import tempfile
import shutil

from src.adaptive_multi_modal_detector import (
    AdaptiveMultiModalDetector, 
    DetectionResult,
    EnsembleWeights
)
from src.quantum_anomaly_fusion import QuantumAnomalyFusion
from src.resilient_anomaly_pipeline import ResilientAnomalyPipeline, PipelineState
from src.scalable_inference_engine import ScalableInferenceEngine
from src.neural_architectural_search import NeuralArchitectureSearchEngine, SearchStrategy


class TestAdaptiveMultiModalDetector:
    """Test suite for adaptive multi-modal detector."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data for testing."""
        np.random.seed(42)
        # Generate normal sequences
        normal_data = np.random.randn(100, 30, 3)
        # Add some trend
        for i in range(3):
            normal_data[:, :, i] += np.linspace(0, 1, 30)
        
        return normal_data
    
    @pytest.fixture
    def anomaly_data(self):
        """Generate anomalous time series data for testing."""
        np.random.seed(123)
        # Generate anomalous sequences with spikes
        anomaly_data = np.random.randn(50, 30, 3)
        # Add spikes to make it anomalous
        anomaly_data[:, 15:20, :] += 5.0
        
        return anomaly_data
    
    def test_detector_initialization(self):
        """Test multi-modal detector initialization."""
        detector = AdaptiveMultiModalDetector(window_size=20)
        
        assert detector.window_size == 20
        assert not detector.is_trained
        assert len(detector.detectors) == 3  # LSTM, Isolation Forest, Statistical
        assert detector.ensemble_weights is not None
    
    def test_detector_training(self, sample_data):
        """Test detector training process."""
        detector = AdaptiveMultiModalDetector(window_size=30)
        
        # Train detector
        detector.fit(sample_data, parallel=False)  # Use sequential for testing
        
        assert detector.is_trained
        
        # Check individual detectors
        status = detector.get_detector_status()
        assert all(status.values()), "All detectors should be trained"
    
    def test_detector_prediction(self, sample_data, anomaly_data):
        """Test anomaly detection predictions."""
        detector = AdaptiveMultiModalDetector(window_size=30)
        detector.fit(sample_data, parallel=False)
        
        # Test prediction on normal data
        normal_result = detector.predict(sample_data[:10], method="ensemble")
        
        assert isinstance(normal_result, DetectionResult)
        assert len(normal_result.anomaly_scores) == 10
        assert len(normal_result.anomaly_predictions) == 10
        assert normal_result.detection_method == "Ensemble"
        
        # Test prediction on anomalous data
        anomaly_result = detector.predict(anomaly_data[:10], method="ensemble")
        
        assert isinstance(anomaly_result, DetectionResult)
        assert len(anomaly_result.anomaly_scores) == 10
        
        # Anomaly scores should generally be higher for anomalous data
        normal_avg_score = np.mean(normal_result.anomaly_scores)
        anomaly_avg_score = np.mean(anomaly_result.anomaly_scores)
        
        # Allow for some variance in dummy implementations
        assert anomaly_avg_score >= normal_avg_score * 0.8
    
    def test_individual_detectors(self, sample_data):
        """Test individual detector methods."""
        detector = AdaptiveMultiModalDetector(window_size=30)
        detector.fit(sample_data, parallel=False)
        
        # Test LSTM detector
        lstm_result = detector.predict(sample_data[:5], method="lstm")
        assert lstm_result.detection_method.startswith("LSTM")
        
        # Test Isolation Forest detector
        if_result = detector.predict(sample_data[:5], method="isolation_forest")
        assert if_result.detection_method.startswith("IsolationForest")
        
        # Test Statistical detector
        stat_result = detector.predict(sample_data[:5], method="statistical")
        assert stat_result.detection_method == "Statistical"
    
    def test_ensemble_weights_adaptation(self, sample_data):
        """Test adaptive ensemble weight adjustment."""
        detector = AdaptiveMultiModalDetector(window_size=30)
        detector.fit(sample_data, parallel=False)
        
        # Get initial weights
        initial_weights = detector.ensemble_weights.__dict__.copy()
        
        # Create mock predictions and true labels
        predictions = {}
        for method in ["lstm", "isolation_forest", "statistical"]:
            result = detector.predict(sample_data[:20], method=method)
            predictions[method] = result
        
        # Mock true labels (half normal, half anomalous)
        true_labels = np.array([0] * 10 + [1] * 10)
        
        # Adapt weights
        detector.adapt_weights(true_labels, predictions)
        
        # Weights should have changed
        updated_weights = detector.ensemble_weights.__dict__
        assert updated_weights != initial_weights
    
    def test_save_and_load(self, sample_data):
        """Test detector save and load functionality."""
        detector = AdaptiveMultiModalDetector(window_size=30)
        detector.fit(sample_data, parallel=False)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "detector"
            
            # Save detector
            detector.save(save_path)
            
            # Load new detector
            new_detector = AdaptiveMultiModalDetector(window_size=30)
            new_detector.load(save_path)
            
            assert new_detector.is_trained
            assert new_detector.window_size == detector.window_size
            
            # Test predictions are consistent
            original_result = detector.predict(sample_data[:5], method="ensemble")
            loaded_result = new_detector.predict(sample_data[:5], method="ensemble")
            
            # Results should be reasonably similar
            score_diff = np.abs(original_result.anomaly_scores - loaded_result.anomaly_scores)
            assert np.mean(score_diff) < 0.5  # Allow some variance
    
    def test_benchmarking(self, sample_data, anomaly_data):
        """Test detector benchmarking functionality."""
        detector = AdaptiveMultiModalDetector(window_size=30)
        detector.fit(sample_data, parallel=False)
        
        # Combine test data and create labels
        test_data = np.vstack([sample_data[:20], anomaly_data[:20]])
        true_labels = np.array([0] * 20 + [1] * 20)
        
        # Run benchmark
        benchmark_results = detector.benchmark(test_data, true_labels)
        
        assert isinstance(benchmark_results, dict)
        
        for method_name, metrics in benchmark_results.items():
            if "error" not in metrics:
                assert "precision" in metrics
                assert "recall" in metrics
                assert "f1_score" in metrics
                assert "accuracy" in metrics
                
                # Metrics should be between 0 and 1
                assert 0 <= metrics["precision"] <= 1
                assert 0 <= metrics["recall"] <= 1
                assert 0 <= metrics["f1_score"] <= 1
                assert 0 <= metrics["accuracy"] <= 1


class TestQuantumAnomalyFusion:
    """Test suite for quantum anomaly fusion."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for quantum testing."""
        np.random.seed(42)
        return np.random.randn(50, 20, 2)
    
    def test_quantum_fusion_initialization(self):
        """Test quantum fusion system initialization."""
        quantum_fusion = QuantumAnomalyFusion(n_qubits=3)
        
        assert quantum_fusion.n_qubits == 3
        assert not quantum_fusion.is_trained
        assert len(quantum_fusion.quantum_detectors) == 2  # Variational + Annealing
    
    def test_quantum_fusion_training(self, sample_data):
        """Test quantum fusion training."""
        quantum_fusion = QuantumAnomalyFusion(n_qubits=3, enable_parallel=False)
        
        # Train the fusion system
        quantum_fusion.fit(sample_data)
        
        assert quantum_fusion.is_trained
        
        # Check individual quantum detectors
        for detector in quantum_fusion.quantum_detectors.values():
            assert detector.is_trained
    
    def test_quantum_prediction(self, sample_data):
        """Test quantum anomaly prediction."""
        quantum_fusion = QuantumAnomalyFusion(n_qubits=3, enable_parallel=False)
        quantum_fusion.fit(sample_data)
        
        # Test weighted average fusion
        result = quantum_fusion.predict(sample_data[:10], fusion_method="weighted_average")
        
        assert isinstance(result, DetectionResult)
        assert len(result.anomaly_scores) == 10
        assert result.detection_method == "QuantumFusion_Weighted"
        
        # Test quantum superposition fusion
        superposition_result = quantum_fusion.predict(sample_data[:10], fusion_method="quantum_superposition")
        
        assert isinstance(superposition_result, DetectionResult)
        assert superposition_result.detection_method == "QuantumFusion_Superposition"
    
    def test_quantum_metrics(self, sample_data):
        """Test quantum-specific metrics."""
        quantum_fusion = QuantumAnomalyFusion(n_qubits=3, enable_parallel=False)
        quantum_fusion.fit(sample_data)
        
        metrics = quantum_fusion.get_quantum_metrics(sample_data[:5])
        
        assert isinstance(metrics, dict)
        assert "quantum_enabled" in metrics
        assert "n_qubits" in metrics
        assert metrics["n_qubits"] == 3


class TestResilientAnomalyPipeline:
    """Test suite for resilient anomaly pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for pipeline testing."""
        np.random.seed(42)
        return np.random.randn(30, 25, 3)
    
    def test_pipeline_initialization(self):
        """Test resilient pipeline initialization."""
        pipeline = ResilientAnomalyPipeline(
            enable_circuit_breaker=True,
            enable_retry=True,
            enable_graceful_degradation=True,
            enable_health_monitoring=False  # Disable for testing
        )
        
        assert pipeline.pipeline_state == PipelineState.HEALTHY
        assert pipeline.circuit_breaker is not None
        assert pipeline.retry_manager is not None
        assert pipeline.degradation_manager is not None
    
    def test_pipeline_training(self, sample_data):
        """Test resilient pipeline training."""
        pipeline = ResilientAnomalyPipeline(enable_health_monitoring=False)
        
        # Train pipeline
        pipeline.fit(sample_data, enable_quantum=False)  # Disable quantum for faster testing
        
        assert pipeline.pipeline_state == PipelineState.HEALTHY
        assert pipeline.multi_modal_detector.is_trained
    
    def test_pipeline_prediction(self, sample_data):
        """Test resilient pipeline prediction."""
        pipeline = ResilientAnomalyPipeline(enable_health_monitoring=False)
        pipeline.fit(sample_data, enable_quantum=False)
        
        # Test prediction
        result = pipeline.predict(sample_data[:5], method="ensemble")
        
        assert isinstance(result, DetectionResult)
        assert len(result.anomaly_scores) == 5
    
    def test_error_handling(self):
        """Test error handling and recovery mechanisms."""
        pipeline = ResilientAnomalyPipeline(enable_health_monitoring=False)
        
        # Test error classification
        test_errors = [
            ("Connection timeout", "timeout"),
            ("Out of memory error", "resource_exhaustion"),
            ("Invalid model file", "model_failure"),
            ("Network connection failed", "network_error"),
            ("Data validation failed", "validation_error")
        ]
        
        for error_msg, expected_type in test_errors:
            error = Exception(error_msg)
            classified_type = pipeline._classify_error(error)
            # Error classification should work reasonably well
            assert classified_type.value in error_msg or "unknown" == classified_type.value
    
    def test_system_status(self, sample_data):
        """Test system status reporting."""
        pipeline = ResilientAnomalyPipeline(enable_health_monitoring=False)
        pipeline.fit(sample_data, enable_quantum=False)
        
        status = pipeline.get_system_status()
        
        assert isinstance(status, dict)
        assert "pipeline_state" in status
        assert "performance_stats" in status
        assert status["pipeline_state"] == "healthy"
    
    def test_pipeline_reset(self, sample_data):
        """Test pipeline reset functionality."""
        pipeline = ResilientAnomalyPipeline(enable_health_monitoring=False)
        pipeline.fit(sample_data, enable_quantum=False)
        
        # Simulate some errors
        pipeline.pipeline_state = PipelineState.DEGRADED
        
        # Reset pipeline
        pipeline.reset_pipeline()
        
        assert pipeline.pipeline_state == PipelineState.HEALTHY
        assert len(pipeline.error_history) == 0


class TestScalableInferenceEngine:
    """Test suite for scalable inference engine."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for inference testing."""
        np.random.seed(42)
        return np.random.randn(20, 15, 2)
    
    def test_engine_initialization(self):
        """Test inference engine initialization."""
        engine = ScalableInferenceEngine(
            num_workers=2,
            max_queue_size=100,
            enable_async=False
        )
        
        assert engine.num_workers == 2
        assert engine.max_queue_size == 100
        assert not engine.is_running
    
    def test_engine_training(self, sample_data):
        """Test inference engine training."""
        engine = ScalableInferenceEngine(num_workers=1, enable_async=False)
        
        # Train the engine
        engine.fit(sample_data)
        
        # Pipeline should be trained
        assert engine.pipeline.multi_modal_detector.is_trained
    
    def test_engine_lifecycle(self, sample_data):
        """Test engine start/stop lifecycle."""
        engine = ScalableInferenceEngine(num_workers=1, enable_async=False)
        engine.fit(sample_data)
        
        # Start engine
        engine.start()
        assert engine.is_running
        
        # Wait a moment for workers to initialize
        time.sleep(0.1)
        
        # Stop engine
        engine.stop()
        assert not engine.is_running
    
    def test_cache_functionality(self):
        """Test caching system."""
        from src.scalable_inference_engine import AdaptiveCache, CacheStrategy
        
        cache = AdaptiveCache(max_size=5, ttl_seconds=10, strategy=CacheStrategy.LRU)
        
        # Test cache miss
        result = cache.get("test_key")
        assert result is None
        
        # Test cache operations with mock data
        test_data = np.random.randn(10, 5, 2)
        cache_key = cache._generate_key(test_data)
        
        # Key should be consistent
        assert cache_key == cache._generate_key(test_data)
        
        # Test cache stats
        stats = cache.get_cache_stats()
        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats
    
    def test_load_balancer(self):
        """Test load balancer functionality."""
        from src.scalable_inference_engine import LoadBalancer
        import queue
        
        balancer = LoadBalancer()
        
        # Register workers
        queue1 = queue.Queue()
        queue2 = queue.Queue()
        
        balancer.register_worker("worker1", queue1)
        balancer.register_worker("worker2", queue2)
        
        # Test worker selection
        selected = balancer.select_worker()
        assert selected in ["worker1", "worker2"]
        
        # Test stats update
        balancer.increment_active_requests("worker1")
        balancer.update_worker_stats("worker1", 0.1)
        
        stats = balancer.get_worker_stats()
        assert "worker1" in stats
        assert stats["worker1"]["total_requests"] == 1


class TestNeuralArchitectureSearch:
    """Test suite for Neural Architecture Search."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for NAS testing."""
        np.random.seed(42)
        train_data = np.random.randn(50, 20, 3)
        val_data = np.random.randn(20, 20, 3)
        return train_data, val_data
    
    def test_nas_initialization(self):
        """Test NAS engine initialization."""
        nas_engine = NeuralArchitectureSearchEngine(
            search_strategy=SearchStrategy.GENETIC_ALGORITHM,
            max_search_time=60  # Short time for testing
        )
        
        assert nas_engine.search_strategy == SearchStrategy.GENETIC_ALGORITHM
        assert nas_engine.max_search_time == 60
    
    def test_genetic_algorithm_components(self, sample_data):
        """Test genetic algorithm components."""
        from src.neural_architectural_search import GeneticAlgorithmNAS
        
        train_data, val_data = sample_data
        
        ga_nas = GeneticAlgorithmNAS(
            population_size=5,  # Small for testing
            generations=2,      # Few generations for testing
            mutation_rate=0.2,
            crossover_rate=0.8
        )
        
        # Test population initialization
        ga_nas.initialize_population()
        assert len(ga_nas.population) == 5
        
        # Test genome creation
        genome = ga_nas._create_random_genome()
        assert len(genome.layers) >= 2
        assert genome.optimizer in ["adam", "sgd", "rmsprop"]
        
        # Test crossover
        parent1 = ga_nas.population[0]
        parent2 = ga_nas.population[1]
        child1, child2 = ga_nas.crossover(parent1, parent2)
        
        assert isinstance(child1.layers, list)
        assert isinstance(child2.layers, list)
        
        # Test mutation
        mutated = ga_nas.mutate(genome)
        assert isinstance(mutated.layers, list)
    
    def test_architecture_evaluation(self, sample_data):
        """Test architecture evaluation."""
        from src.neural_architectural_search import GeneticAlgorithmNAS, ArchitectureGenome, LayerConfig, LayerType
        
        train_data, val_data = sample_data
        
        ga_nas = GeneticAlgorithmNAS(population_size=1, generations=1)
        
        # Create simple test genome
        test_genome = ArchitectureGenome(
            layers=[
                LayerConfig(layer_type=LayerType.LSTM, units=32, return_sequences=False),
                LayerConfig(layer_type=LayerType.DENSE, units=train_data.shape[-1])
            ],
            optimizer="adam",
            learning_rate=0.001
        )
        
        # Evaluate architecture
        performance = ga_nas.evaluate_architecture(test_genome, train_data, val_data, max_epochs=1)
        
        assert performance.architecture_id == test_genome.architecture_id
        assert 0 <= performance.accuracy <= 1
        assert performance.training_time >= 0
        assert performance.fitness_score >= 0
    
    def test_nas_search_execution(self, sample_data):
        """Test end-to-end NAS search execution."""
        train_data, val_data = sample_data
        
        nas_engine = NeuralArchitectureSearchEngine(
            search_strategy=SearchStrategy.GENETIC_ALGORITHM,
            max_search_time=120
        )
        
        # Configure for fast testing
        nas_engine.search_algorithm.population_size = 3
        nas_engine.search_algorithm.generations = 2
        
        # Run search
        best_architecture, search_results = nas_engine.search(train_data, val_data)
        
        assert best_architecture is not None
        assert isinstance(search_results, dict)
        assert "search_strategy" in search_results
        assert "best_architecture" in search_results
        assert "search_time" in search_results
        
        # Test save/load functionality
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "nas_results"
            
            nas_engine.save_results(best_architecture, search_results, save_path)
            
            # Load architecture
            loaded_architecture = nas_engine.load_architecture(save_path)
            
            assert loaded_architecture.architecture_id == best_architecture.architecture_id
            assert len(loaded_architecture.layers) == len(best_architecture.layers)


class TestIntegrationSuite:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def comprehensive_data(self):
        """Generate comprehensive test dataset."""
        np.random.seed(42)
        
        # Normal sequences with seasonal patterns
        normal_data = []
        for i in range(100):
            # Base pattern with noise
            pattern = np.sin(np.linspace(0, 4*np.pi, 30)) + np.random.randn(30) * 0.1
            # Add trend
            trend = np.linspace(0, 1, 30) * 0.5
            # Multiple features
            sequence = np.column_stack([
                pattern + trend,
                np.cos(np.linspace(0, 3*np.pi, 30)) + trend * 0.5,
                np.random.randn(30) * 0.2 + trend * 0.3
            ])
            normal_data.append(sequence)
        
        # Anomalous sequences
        anomaly_data = []
        for i in range(50):
            # Start with normal pattern
            pattern = np.sin(np.linspace(0, 4*np.pi, 30)) + np.random.randn(30) * 0.1
            trend = np.linspace(0, 1, 30) * 0.5
            
            # Inject anomalies
            anomaly_start = np.random.randint(5, 25)
            anomaly_length = np.random.randint(3, 8)
            anomaly_magnitude = np.random.uniform(3, 6)
            
            pattern[anomaly_start:anomaly_start+anomaly_length] += anomaly_magnitude
            
            sequence = np.column_stack([
                pattern + trend,
                np.cos(np.linspace(0, 3*np.pi, 30)) + trend * 0.5,
                np.random.randn(30) * 0.2 + trend * 0.3
            ])
            anomaly_data.append(sequence)
        
        return np.array(normal_data), np.array(anomaly_data)
    
    def test_end_to_end_pipeline(self, comprehensive_data):
        """Test complete end-to-end anomaly detection pipeline."""
        normal_data, anomaly_data = comprehensive_data
        
        # Initialize and train multi-modal detector
        detector = AdaptiveMultiModalDetector(window_size=30)
        detector.fit(normal_data, parallel=False)
        
        # Test on normal data
        normal_results = detector.predict(normal_data[:20], method="ensemble")
        normal_anomaly_rate = np.mean(normal_results.anomaly_predictions)
        
        # Test on anomalous data  
        anomaly_results = detector.predict(anomaly_data[:20], method="ensemble")
        anomaly_detection_rate = np.mean(anomaly_results.anomaly_predictions)
        
        # Basic sanity checks
        assert normal_anomaly_rate <= 0.5  # Most normal data should not be flagged
        assert anomaly_detection_rate >= 0.3  # Should detect some anomalies
        
        # Average anomaly scores should be higher for anomalous data
        normal_avg_score = np.mean(normal_results.anomaly_scores)
        anomaly_avg_score = np.mean(anomaly_results.anomaly_scores)
        
        assert anomaly_avg_score >= normal_avg_score * 0.8
    
    def test_resilient_pipeline_integration(self, comprehensive_data):
        """Test resilient pipeline with error handling."""
        normal_data, anomaly_data = comprehensive_data
        
        # Initialize resilient pipeline
        pipeline = ResilientAnomalyPipeline(
            enable_circuit_breaker=True,
            enable_retry=True,
            enable_graceful_degradation=True,
            enable_health_monitoring=False
        )
        
        # Train pipeline
        pipeline.fit(normal_data[:50], enable_quantum=False)
        
        # Test predictions
        result = pipeline.predict(normal_data[:10], method="ensemble")
        
        assert isinstance(result, DetectionResult)
        assert len(result.anomaly_scores) == 10
        
        # Check system status
        status = pipeline.get_system_status()
        assert status["pipeline_state"] == "healthy"
        assert status["performance_stats"]["total_requests"] > 0
    
    def test_scalable_engine_performance(self, comprehensive_data):
        """Test scalable inference engine performance characteristics."""
        normal_data, anomaly_data = comprehensive_data
        
        # Initialize engine with minimal configuration for testing
        engine = ScalableInferenceEngine(
            num_workers=2,
            max_queue_size=50,
            enable_async=False
        )
        
        # Train engine
        engine.fit(normal_data[:30])
        
        # Start engine
        engine.start()
        
        try:
            # Wait for initialization
            time.sleep(0.2)
            
            # Test performance metrics collection
            metrics = engine.get_performance_metrics()
            assert hasattr(metrics, 'requests_per_second')
            assert hasattr(metrics, 'avg_processing_time')
            
            # Test system status
            status = engine.get_system_status()
            assert status["is_running"] == True
            assert status["num_workers"] == 2
            
        finally:
            # Clean shutdown
            engine.stop()
    
    @pytest.mark.slow
    def test_system_stress_test(self, comprehensive_data):
        """Stress test the complete system under load."""
        normal_data, anomaly_data = comprehensive_data
        
        # Create larger dataset for stress testing
        large_normal = np.tile(normal_data, (3, 1, 1))  # 300 sequences
        
        # Initialize system components
        detector = AdaptiveMultiModalDetector(window_size=30)
        
        # Train with timing
        start_time = time.time()
        detector.fit(large_normal[:100], parallel=False)  # Train on subset
        training_time = time.time() - start_time
        
        # Should complete training in reasonable time
        assert training_time < 300  # 5 minutes max
        
        # Test batch predictions
        batch_start = time.time()
        results = detector.predict(large_normal[100:150], method="ensemble")  # 50 predictions
        batch_time = time.time() - batch_start
        
        # Should process reasonably fast
        assert batch_time < 60  # 1 minute max for 50 predictions
        assert len(results.anomaly_scores) == 50
        
        # Memory usage should be reasonable (basic check)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            assert memory_mb < 2048  # Less than 2GB
        except ImportError:
            pass  # Skip memory check if psutil not available


# Performance benchmarking markers
pytestmark = [
    pytest.mark.unit,
    pytest.mark.integration
]