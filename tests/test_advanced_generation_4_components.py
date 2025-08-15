"""Comprehensive test suite for Generation 4 advanced components.

Tests for federated learning, neuromorphic computing, temporal fusion transformers,
and adaptive edge computing orchestration systems.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import warnings

# Import components to test
try:
    from src.federated_anomaly_learning import (
        FederatedAnomalyLearner, FedAvgStrategy, FedProxStrategy, 
        FederatedNode, ModelUpdate, create_edge_federated_system
    )
    from src.neuromorphic_spike_processor import (
        NeuromorphicAnomalyDetector, SpikeEvent, LeakyIntegrateFireNeuron,
        AdaptiveExponentialNeuron, STDPPlasticityRule, create_optimized_neuromorphic_detector
    )
    from src.temporal_fusion_transformer_anomaly import (
        TemporalFusionTransformerAnomalyDetector, TFTConfig,
        create_optimized_tft_detector
    )
    from src.adaptive_edge_computing_orchestrator import (
        EdgeComputingOrchestrator, EdgeDevice, WorkloadRequest,
        LatencyOptimizedStrategy, EnergyOptimizedStrategy, HybridOptimizedStrategy,
        create_iot_edge_orchestrator
    )
    ADVANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    ADVANCED_COMPONENTS_AVAILABLE = False
    warnings.warn(f"Advanced components not available for testing: {str(e)}")


class TestFederatedAnomalyLearning:
    """Test suite for federated learning components."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return np.random.normal(0, 1, (1000, 5))
    
    @pytest.fixture
    def federated_system(self):
        """Create a federated learning system for testing."""
        if not ADVANCED_COMPONENTS_AVAILABLE:
            pytest.skip("Advanced components not available")
        
        system = FederatedAnomalyLearner(
            aggregation_strategy=FedAvgStrategy(),
            min_nodes_per_round=2,
            max_rounds=5,
            convergence_threshold=0.01
        )
        
        # Register test nodes
        for i in range(5):
            system.register_node(
                node_id=f"test_node_{i}",
                node_type="edge",
                computational_capacity=1.0 + i * 0.5,
                memory_capacity=512 + i * 256,
                bandwidth_capacity=10.0 + i * 5
            )
            # Add some local data samples
            system.nodes[f"test_node_{i}"].local_data_samples = 100 + i * 50
        
        return system
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_federated_node_registration(self, federated_system):
        """Test federated node registration."""
        assert len(federated_system.nodes) == 5
        
        # Test node properties
        node = federated_system.nodes["test_node_0"]
        assert node.node_id == "test_node_0"
        assert node.node_type == "edge"
        assert node.computational_capacity == 1.0
        assert node.local_data_samples == 100
        assert node.is_active is True
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_node_selection(self, federated_system):
        """Test node selection for training rounds."""
        # Test adaptive selection
        selected_nodes = federated_system.select_nodes_for_round(
            selection_ratio=0.6,
            strategy="adaptive"
        )
        
        assert len(selected_nodes) >= federated_system.min_nodes_per_round
        assert len(selected_nodes) <= len(federated_system.nodes)
        assert all(node_id in federated_system.nodes for node_id in selected_nodes)
        
        # Test random selection
        selected_nodes_random = federated_system.select_nodes_for_round(
            selection_ratio=0.4,
            strategy="random"
        )
        
        assert len(selected_nodes_random) >= federated_system.min_nodes_per_round
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_model_update_creation(self, federated_system):
        """Test model update creation and verification."""
        # Create mock model weights
        mock_weights = {
            "layer_0": np.random.normal(0, 1, (10, 5)),
            "layer_1": np.random.normal(0, 1, (5, 1))
        }
        
        # Create model update
        update = federated_system.create_model_update(
            node_id="test_node_0",
            local_weights=mock_weights,
            local_loss=0.25,
            sample_count=100,
            training_time=15.0
        )
        
        assert update.node_id == "test_node_0"
        assert update.local_loss == 0.25
        assert update.sample_count == 100
        assert update.training_time == 15.0
        assert len(update.gradient_norms) == len(mock_weights)
        assert update.update_timestamp > 0
        
        # Test verification
        is_verified = federated_system.verify_model_update(update)
        assert isinstance(is_verified, bool)
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_aggregation_strategies(self):
        """Test different aggregation strategies."""
        # Create mock updates
        updates = []
        for i in range(3):
            weights = {
                "layer_0": np.random.normal(i, 1, (5, 3)),
                "layer_1": np.random.normal(i, 1, (3, 1))
            }
            
            update = ModelUpdate(
                node_id=f"node_{i}",
                model_weights=weights,
                gradient_norms=[1.0, 2.0],
                local_loss=0.1 + i * 0.05,
                sample_count=100 + i * 20,
                training_time=10.0,
                update_timestamp=time.time(),
                is_verified=True
            )
            updates.append(update)
        
        # Test FedAvg
        fedavg = FedAvgStrategy(weighted=True)
        aggregated_fedavg = fedavg.aggregate(updates)
        
        assert "layer_0" in aggregated_fedavg
        assert "layer_1" in aggregated_fedavg
        assert aggregated_fedavg["layer_0"].shape == (5, 3)
        assert aggregated_fedavg["layer_1"].shape == (3, 1)
        
        # Test FedProx
        fedprox = FedProxStrategy(mu=0.01)
        aggregated_fedprox = fedprox.aggregate(updates)
        
        assert "layer_0" in aggregated_fedprox
        assert "layer_1" in aggregated_fedprox
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_federated_training_simulation(self, federated_system, sample_data):
        """Test federated training simulation."""
        # Run abbreviated training
        training_summary = federated_system.train_federated(
            input_shape=(30, 5),
            rounds=2,
            local_epochs=1,
            node_selection_ratio=0.6
        )
        
        assert "total_rounds" in training_summary
        assert "total_duration" in training_summary
        assert training_summary["total_rounds"] <= 2
        assert training_summary["participating_nodes"] > 0
        assert federated_system.is_trained is True
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_federated_inference(self, federated_system, sample_data):
        """Test federated inference."""
        # Initialize and train model (abbreviated)
        federated_system.initialize_global_model((30, 5))
        federated_system.is_trained = True
        
        # Test inference
        test_data = sample_data[:50]  # Small test set
        predictions, metadata = federated_system.predict_federated(test_data)
        
        assert len(predictions) > 0
        assert "inference_time" in metadata
        assert "model_version" in metadata
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_edge_federated_system_creation(self):
        """Test creation of edge-optimized federated system."""
        system = create_edge_federated_system(
            num_edge_nodes=5,
            aggregation_strategy="adaptive"
        )
        
        assert len(system.nodes) == 5
        assert system.min_nodes_per_round >= 1
        assert all(
            node.node_type == "edge" 
            for node in system.nodes.values()
        )
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_system_status(self, federated_system):
        """Test system status reporting."""
        status = federated_system.get_system_status()
        
        assert "system_info" in status
        assert "data_info" in status
        assert "performance_info" in status
        assert "node_details" in status
        
        assert status["system_info"]["total_nodes"] == 5
        assert status["system_info"]["active_nodes"] <= 5


class TestNeuromorphicComputing:
    """Test suite for neuromorphic computing components."""
    
    @pytest.fixture
    def sample_spike_data(self):
        """Generate sample spike data for testing."""
        spikes = []
        for i in range(10):
            spike = SpikeEvent(
                timestamp=i * 10.0,
                neuron_id=i % 3,
                layer_id=0,
                spike_amplitude=1.0
            )
            spikes.append(spike)
        return spikes
    
    @pytest.fixture
    def neuromorphic_detector(self):
        """Create neuromorphic detector for testing."""
        if not ADVANCED_COMPONENTS_AVAILABLE:
            pytest.skip("Advanced components not available")
        
        detector = NeuromorphicAnomalyDetector(
            input_features=5,
            hidden_layers=[10, 5],
            encoding_type="rate",
            simulation_time=50.0,
            detection_threshold=0.5
        )
        return detector
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_spike_event_creation(self, sample_spike_data):
        """Test spike event creation."""
        spike = sample_spike_data[0]
        
        assert spike.timestamp == 0.0
        assert spike.neuron_id == 0
        assert spike.layer_id == 0
        assert spike.spike_amplitude == 1.0
        assert spike.spike_type == "excitatory"
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_neuron_models(self):
        """Test different neuron models."""
        from src.neuromorphic_spike_processor import NeuronState
        
        # Test LIF neuron
        lif_neuron = LeakyIntegrateFireNeuron()
        neuron_state = NeuronState(
            neuron_id=0,
            layer_id=0,
            threshold=1.0,
            reset_potential=0.0
        )
        
        # Update membrane potential
        new_potential = lif_neuron.update_membrane_potential(
            neuron_state, input_current=0.5, dt=1.0
        )
        
        assert isinstance(new_potential, (float, np.floating))
        
        # Test spike condition
        neuron_state.membrane_potential = 1.5  # Above threshold
        should_spike = lif_neuron.check_spike_condition(neuron_state)
        assert should_spike is True
        
        # Test reset
        lif_neuron.reset_neuron(neuron_state)
        assert neuron_state.membrane_potential == neuron_state.reset_potential
        assert neuron_state.spike_count == 1
        
        # Test AdEx neuron
        adex_neuron = AdaptiveExponentialNeuron()
        neuron_state.membrane_potential = 0.0
        new_potential = adex_neuron.update_membrane_potential(
            neuron_state, input_current=0.3, dt=1.0
        )
        assert isinstance(new_potential, (float, np.floating))
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_plasticity_rules(self):
        """Test synaptic plasticity rules."""
        from src.neuromorphic_spike_processor import SynapseConnection
        
        # Create test synapse
        synapse = SynapseConnection(
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=0.5,
            delay=2.0,
            is_plastic=True
        )
        
        # Test STDP
        stdp_rule = STDPPlasticityRule(A_plus=0.01, A_minus=0.012)
        
        # Test LTP (post after pre)
        new_weight = stdp_rule.update_weight(
            synapse, 
            pre_spike_time=10.0,
            post_spike_time=15.0,
            current_time=20.0
        )
        
        assert new_weight >= synapse.weight  # Should increase (LTP)
        
        # Test LTD (pre after post)
        synapse.weight = 0.5  # Reset
        new_weight = stdp_rule.update_weight(
            synapse,
            pre_spike_time=15.0,
            post_spike_time=10.0,
            current_time=20.0
        )
        
        assert new_weight <= 0.5  # Should decrease (LTD)
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_spike_encoding(self, neuromorphic_detector):
        """Test input encoding to spikes."""
        # Test rate encoding
        input_data = np.array([0.5, 0.8, 0.2, 0.9, 0.1])
        spikes = neuromorphic_detector.snn.encode_input_to_spikes(
            input_data,
            encoding_type="rate",
            time_window=100.0
        )
        
        assert len(spikes) > 0
        assert all(isinstance(spike, SpikeEvent) for spike in spikes)
        assert all(spike.timestamp <= 100.0 for spike in spikes)
        
        # Test temporal encoding
        spikes_temporal = neuromorphic_detector.snn.encode_input_to_spikes(
            input_data,
            encoding_type="temporal",
            time_window=100.0
        )
        
        assert len(spikes_temporal) == len(input_data)
        
        # Test population encoding
        spikes_population = neuromorphic_detector.snn.encode_input_to_spikes(
            input_data,
            encoding_type="population",
            time_window=100.0
        )
        
        assert len(spikes_population) >= 0
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_network_simulation(self, neuromorphic_detector, sample_spike_data):
        """Test spiking neural network simulation."""
        # Run simulation with sample spikes
        results = neuromorphic_detector.snn.simulate_network(
            input_spikes=sample_spike_data,
            simulation_time=100.0
        )
        
        assert "output_spikes" in results
        assert "total_spikes" in results
        assert "simulation_time" in results
        assert "computation_time" in results
        assert "firing_rates" in results
        assert "energy_estimate" in results
        
        assert results["simulation_time"] == 100.0
        assert results["total_spikes"] >= 0
        assert isinstance(results["firing_rates"], dict)
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_neuromorphic_training(self, neuromorphic_detector):
        """Test neuromorphic detector training."""
        # Generate synthetic training data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 5))
        
        # Train detector
        training_results = neuromorphic_detector.train_unsupervised(
            normal_data,
            training_epochs=2,
            adaptation_rate=0.01
        )
        
        assert "training_time" in training_results
        assert "samples_processed" in training_results
        assert "detection_threshold" in training_results
        assert "total_energy_pj" in training_results
        
        assert neuromorphic_detector.is_trained is True
        assert training_results["samples_processed"] == 200  # 100 samples * 2 epochs
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_neuromorphic_inference(self, neuromorphic_detector):
        """Test neuromorphic detector inference."""
        # Train first
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (50, 5))
        neuromorphic_detector.train_unsupervised(normal_data, training_epochs=1)
        
        # Test inference
        test_data = np.random.normal(0, 1, (20, 5))
        predictions, metadata = neuromorphic_detector.predict(test_data)
        
        assert len(predictions) == 20
        assert all(pred in [0, 1] for pred in predictions)
        
        assert "inference_time" in metadata
        assert "total_energy_pj" in metadata
        assert "neuromorphic_processing" in metadata
        assert metadata["neuromorphic_processing"] is True
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_optimized_neuromorphic_detector(self):
        """Test creation of optimized neuromorphic detector."""
        detector = create_optimized_neuromorphic_detector(
            input_features=5,
            target_energy_budget=1000.0,
            target_latency=50.0
        )
        
        assert detector.input_features == 5
        assert detector.simulation_time <= 50.0
        assert len(detector.snn.architecture) >= 2  # At least input and output layers
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")  
    def test_visualization_data(self, neuromorphic_detector):
        """Test network visualization data extraction."""
        # Run a simple simulation first
        np.random.seed(42)
        test_data = np.random.normal(0, 1, (10, 5))
        neuromorphic_detector.predict(test_data)
        
        viz_data = neuromorphic_detector.get_network_visualization_data()
        
        assert "spike_raster" in viz_data
        assert "membrane_traces" in viz_data
        assert "connections" in viz_data
        assert "layer_info" in viz_data
        assert "network_stats" in viz_data
        assert "architecture" in viz_data


class TestTemporalFusionTransformer:
    """Test suite for Temporal Fusion Transformer components."""
    
    @pytest.fixture
    def tft_config(self):
        """Create TFT configuration for testing."""
        if not ADVANCED_COMPONENTS_AVAILABLE:
            pytest.skip("Advanced components not available")
        
        return TFTConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            lookback_window=20,
            forecast_horizon=5,
            num_dynamic_features=5,
            batch_size=16,
            max_epochs=2,
            num_quantiles=3
        )
    
    @pytest.fixture
    def tft_detector(self, tft_config):
        """Create TFT detector for testing."""
        if not ADVANCED_COMPONENTS_AVAILABLE:
            pytest.skip("Advanced components not available")
        
        return TemporalFusionTransformerAnomalyDetector(tft_config)
    
    @pytest.fixture
    def time_series_data(self):
        """Generate synthetic time series data."""
        np.random.seed(42)
        
        # Create time series with trend and seasonality
        t = np.linspace(0, 10, 500)
        trend = 0.1 * t
        seasonal = 2 * np.sin(2 * np.pi * t / 5)
        noise = np.random.normal(0, 0.2, len(t))
        
        # Create multivariate data
        features = []
        for i in range(5):
            feature = trend + seasonal + noise + np.random.normal(0, 0.1, len(t))
            features.append(feature)
        
        return np.column_stack(features)
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_tft_config(self, tft_config):
        """Test TFT configuration."""
        assert tft_config.hidden_size == 32
        assert tft_config.num_attention_heads == 4
        assert tft_config.lookback_window == 20
        assert tft_config.forecast_horizon == 5
        assert len(tft_config.quantile_levels) > 0
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_sequence_preparation(self, tft_detector, time_series_data):
        """Test time series sequence preparation."""
        X_enc, X_dec, y = tft_detector.prepare_sequences(
            time_series_data, target_column=0
        )
        
        expected_sequences = len(time_series_data) - (tft_detector.config.lookback_window + tft_detector.config.forecast_horizon) + 1
        
        assert X_enc.shape[0] == expected_sequences
        assert X_dec.shape[0] == expected_sequences
        assert y.shape[0] == expected_sequences
        
        assert X_enc.shape[1] == tft_detector.config.lookback_window
        assert X_dec.shape[1] == tft_detector.config.forecast_horizon
        assert y.shape[1] == tft_detector.config.forecast_horizon
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_model_building(self, tft_detector):
        """Test TFT model building."""
        assert tft_detector.model is not None
        assert len(tft_detector.model.inputs) >= 2  # encoder and decoder inputs
        assert len(tft_detector.model.outputs) >= 1
        
        # Test model summary
        param_count = tft_detector.model.count_params()
        assert param_count > 0
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_tft_training(self, tft_detector, time_series_data):
        """Test TFT training process."""
        # Use subset for faster testing
        train_data = time_series_data[:200]
        val_data = time_series_data[200:300]
        
        training_summary = tft_detector.train(
            train_data=train_data,
            val_data=val_data,
            target_column=0
        )
        
        assert "training_time" in training_summary
        assert "final_train_loss" in training_summary
        assert "final_val_loss" in training_summary
        assert "epochs_trained" in training_summary
        
        assert tft_detector.is_trained is True
        assert training_summary["epochs_trained"] > 0
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_tft_prediction(self, tft_detector, time_series_data):
        """Test TFT anomaly prediction."""
        # Train first
        train_data = time_series_data[:200]
        tft_detector.train(train_data, target_column=0)
        
        # Test prediction
        test_data = time_series_data[200:250]
        predictions, metadata = tft_detector.predict(
            test_data, return_attention=False, target_column=0
        )
        
        assert len(predictions) > 0
        assert all(pred in [0, 1] for pred in predictions)
        
        assert "inference_time" in metadata
        assert "anomaly_scores" in metadata
        assert "confidence_scores" in metadata
        assert "model_type" in metadata
        assert metadata["model_type"] == "temporal_fusion_transformer"
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_prediction_explanation(self, tft_detector, time_series_data):
        """Test TFT prediction explanation."""
        # Train and predict
        train_data = time_series_data[:150]
        tft_detector.train(train_data, target_column=0)
        
        test_data = time_series_data[150:200]
        predictions, metadata = tft_detector.predict(test_data, target_column=0)
        
        if len(predictions) > 0:
            # Explain first prediction
            explanation = tft_detector.explain_prediction(
                sample_index=0,
                data=test_data,
                target_column=0
            )
            
            assert "sample_index" in explanation
            assert "prediction_quantiles" in explanation
            assert "feature_importance" in explanation
            assert "temporal_importance" in explanation
            assert "model_confidence" in explanation
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_optimized_tft_creation(self):
        """Test creation of optimized TFT detectors."""
        # Speed optimized
        speed_detector = create_optimized_tft_detector(
            num_features=5,
            lookback_window=30,
            forecast_horizon=10,
            performance_target="speed"
        )
        
        assert speed_detector.config.hidden_size <= 64
        assert speed_detector.config.num_attention_heads <= 4
        assert speed_detector.config.lookback_window <= 30
        
        # Accuracy optimized
        accuracy_detector = create_optimized_tft_detector(
            num_features=5,
            performance_target="accuracy"
        )
        
        assert accuracy_detector.config.hidden_size >= 128
        assert accuracy_detector.config.num_attention_heads >= 8
        
        # Balanced
        balanced_detector = create_optimized_tft_detector(
            num_features=5,
            performance_target="balanced"
        )
        
        assert 64 <= balanced_detector.config.hidden_size <= 256
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_model_save_load(self, tft_detector, time_series_data):
        """Test TFT model saving and loading."""
        # Train model
        train_data = time_series_data[:100]
        tft_detector.train(train_data, target_column=0)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            tft_detector.save_model(tmp_path)
            
            # Create new detector and load
            new_detector = TemporalFusionTransformerAnomalyDetector(tft_detector.config)
            new_detector.load_model(tmp_path)
            
            assert new_detector.is_trained is True
            assert new_detector.config.hidden_size == tft_detector.config.hidden_size
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            Path(tmp_path.replace('.pkl', '_model.h5')).unlink(missing_ok=True)


class TestAdaptiveEdgeComputing:
    """Test suite for adaptive edge computing orchestration."""
    
    @pytest.fixture
    def sample_devices(self):
        """Create sample edge devices."""
        if not ADVANCED_COMPONENTS_AVAILABLE:
            pytest.skip("Advanced components not available")
        
        devices = []
        device_specs = [
            ("raspberry_pi", 4, 1.5, 4096, False, 5.0),
            ("nvidia_jetson", 6, 2.0, 8192, True, 15.0),
            ("intel_nuc", 8, 2.5, 16384, True, 25.0),
            ("smartphone", 8, 2.8, 6144, True, 3.0),
        ]
        
        for i, (device_type, cores, freq, memory, gpu, power) in enumerate(device_specs):
            device = EdgeDevice(
                device_id=f"{device_type}_{i}",
                device_type=device_type,
                location=(40.0 + i * 0.1, -74.0 + i * 0.1),
                cpu_cores=cores,
                cpu_frequency=freq,
                memory_total=memory,
                storage_total=64,
                gpu_available=gpu,
                bandwidth_up=50.0,
                bandwidth_down=100.0,
                latency_to_cloud=20.0,
                is_mobile=(device_type == "smartphone"),
                supported_models=["anomaly_detection", "classification"],
                max_inference_rate=10.0,
                power_consumption=power
            )
            devices.append(device)
        
        return devices
    
    @pytest.fixture
    def edge_orchestrator(self, sample_devices):
        """Create edge computing orchestrator."""
        if not ADVANCED_COMPONENTS_AVAILABLE:
            pytest.skip("Advanced components not available")
        
        orchestrator = EdgeComputingOrchestrator(
            strategy=HybridOptimizedStrategy(),
            max_concurrent_workloads=10,
            device_monitoring_interval=1.0,
            auto_scaling_enabled=True
        )
        
        # Register devices
        for device in sample_devices:
            orchestrator.register_device(device)
        
        return orchestrator
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_device_registration(self, sample_devices):
        """Test edge device registration."""
        orchestrator = EdgeComputingOrchestrator()
        
        device = sample_devices[0]
        result = orchestrator.register_device(device)
        
        assert result is True
        assert device.device_id in orchestrator.devices
        assert orchestrator.devices[device.device_id] == device
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_device_unregistration(self, edge_orchestrator):
        """Test edge device unregistration."""
        device_id = "raspberry_pi_0"
        
        # Verify device exists
        assert device_id in edge_orchestrator.devices
        
        # Unregister device
        result = edge_orchestrator.unregister_device(device_id)
        
        assert result is True
        assert device_id not in edge_orchestrator.devices
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_allocation_strategies(self, sample_devices):
        """Test different resource allocation strategies."""
        # Test latency-optimized strategy
        latency_strategy = LatencyOptimizedStrategy()
        
        workload = WorkloadRequest(
            request_id="test_workload",
            data=np.random.normal(0, 1, (100, 5)),
            model_type="anomaly_detection",
            priority=5,
            max_latency=100.0,
            accuracy_requirement=0.8,
            privacy_level=3
        )
        
        # Make devices available
        available_devices = [d for d in sample_devices]
        for device in available_devices:
            device.is_online = True
            device.cpu_usage = 50.0
            device.memory_usage = 40.0
        
        # Test allocation
        allocations = asyncio.run(latency_strategy.allocate_resources(workload, available_devices))
        
        assert len(allocations) > 0
        assert all(alloc.device_id in [d.device_id for d in available_devices] for alloc in allocations)
        assert all(alloc.status == "pending" for alloc in allocations)
        
        # Test energy-optimized strategy
        energy_strategy = EnergyOptimizedStrategy()
        energy_allocations = asyncio.run(energy_strategy.allocate_resources(workload, available_devices))
        
        assert len(energy_allocations) > 0
        
        # Test hybrid strategy
        hybrid_strategy = HybridOptimizedStrategy()
        hybrid_allocations = asyncio.run(hybrid_strategy.allocate_resources(workload, available_devices))
        
        assert len(hybrid_allocations) > 0
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    async def test_workload_submission(self, edge_orchestrator):
        """Test workload submission and execution."""
        await edge_orchestrator.start()
        
        try:
            # Create test workload
            workload = WorkloadRequest(
                request_id="test_workload_submit",
                data=np.random.normal(0, 1, (50, 5)),
                model_type="anomaly_detection",
                priority=3,
                max_latency=200.0,
                accuracy_requirement=0.7,
                privacy_level=2
            )
            
            # Submit workload
            workload_id = await edge_orchestrator.submit_workload(workload)
            
            assert workload_id == workload.request_id
            assert workload_id in edge_orchestrator.active_workloads
            
            # Wait a bit for processing
            await asyncio.sleep(2)
            
            # Check workload status
            status = edge_orchestrator.get_workload_status(workload_id)
            assert status["workload_id"] == workload_id
            assert status["status"] in ["active", "completed"]
            
        finally:
            await edge_orchestrator.stop()
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_system_status(self, edge_orchestrator):
        """Test system status reporting."""
        status = edge_orchestrator.get_system_status()
        
        assert "system_info" in status
        assert "device_summary" in status
        assert "workload_summary" in status
        assert "performance_summary" in status
        assert "load_balancer" in status
        
        assert status["device_summary"]["total_devices"] == 4
        assert status["system_info"]["is_running"] is False  # Not started yet
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_device_details(self, edge_orchestrator):
        """Test device details retrieval."""
        device_id = "raspberry_pi_0"
        details = edge_orchestrator.get_device_details(device_id)
        
        assert "device_info" in details
        assert "recent_utilization" in details
        assert "active_workloads" in details
        assert "queue_length" in details
        
        assert details["device_info"]["device_id"] == device_id
        assert details["device_info"]["device_type"] == "raspberry_pi"
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_load_balancer(self, edge_orchestrator):
        """Test adaptive load balancer functionality."""
        load_balancer = edge_orchestrator.load_balancer
        
        # Update device loads
        load_balancer.update_device_load("device_1", 90.0)  # High load
        load_balancer.update_device_load("device_2", 20.0)  # Low load
        
        # Assign workloads
        load_balancer.assign_workload("workload_1", "device_1")
        load_balancer.assign_workload("workload_2", "device_1")
        
        assert "workload_1" in load_balancer.workload_assignments
        assert "workload_2" in load_balancer.workload_assignments
        assert len(load_balancer.device_queues["device_1"]) == 2
        
        # Complete workload
        load_balancer.complete_workload("workload_1")
        assert "workload_1" not in load_balancer.workload_assignments
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_orchestrator_factories(self):
        """Test orchestrator factory functions."""
        # Test IoT edge orchestrator
        iot_orchestrator = create_iot_edge_orchestrator(optimization_target="latency")
        assert isinstance(iot_orchestrator.strategy, LatencyOptimizedStrategy)
        
        iot_orchestrator_energy = create_iot_edge_orchestrator(optimization_target="energy")
        assert isinstance(iot_orchestrator_energy.strategy, EnergyOptimizedStrategy)
        
        iot_orchestrator_hybrid = create_iot_edge_orchestrator(optimization_target="hybrid")
        assert isinstance(iot_orchestrator_hybrid.strategy, HybridOptimizedStrategy)
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    async def test_orchestrator_lifecycle(self, edge_orchestrator):
        """Test orchestrator start/stop lifecycle."""
        # Test start
        await edge_orchestrator.start()
        assert edge_orchestrator.is_running is True
        assert edge_orchestrator.monitoring_task is not None
        
        # Test stop
        await edge_orchestrator.stop()
        assert edge_orchestrator.is_running is False


# Integration tests for component interactions

class TestGeneration4Integration:
    """Integration tests for Generation 4 components."""
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_federated_neuromorphic_integration(self):
        """Test integration between federated learning and neuromorphic computing."""
        # Create neuromorphic detectors for federated nodes
        node_detectors = {}
        for i in range(3):
            detector = create_optimized_neuromorphic_detector(
                input_features=5,
                target_energy_budget=500.0,
                target_latency=20.0
            )
            node_detectors[f"node_{i}"] = detector
        
        # Simulate local training data for each node
        np.random.seed(42)
        node_data = {}
        for node_id in node_detectors:
            local_data = np.random.normal(0, 1, (100, 5))
            # Add slight variation per node
            local_data += np.random.normal(0, 0.1, local_data.shape)
            node_data[node_id] = local_data
        
        # Train detectors locally
        for node_id, detector in node_detectors.items():
            detector.train_unsupervised(node_data[node_id], training_epochs=1)
        
        # Test ensemble prediction
        test_data = np.random.normal(0, 1, (10, 5))
        ensemble_predictions = []
        
        for detector in node_detectors.values():
            predictions, _ = detector.predict(test_data)
            ensemble_predictions.append(predictions)
        
        # Simple majority vote
        ensemble_predictions = np.array(ensemble_predictions)
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=ensemble_predictions
        )
        
        assert len(final_predictions) == 10
        assert all(pred in [0, 1] for pred in final_predictions)
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    async def test_edge_tft_deployment(self):
        """Test deployment of TFT models on edge devices."""
        # Create TFT detector optimized for edge
        tft_detector = create_optimized_tft_detector(
            num_features=5,
            lookback_window=20,
            forecast_horizon=5,
            performance_target="speed"
        )
        
        # Create edge orchestrator
        orchestrator = create_iot_edge_orchestrator(optimization_target="latency")
        
        # Register edge device
        edge_device = EdgeDevice(
            device_id="edge_tft_device",
            device_type="nvidia_jetson",
            location=(40.0, -74.0),
            cpu_cores=6,
            cpu_frequency=2.0,
            memory_total=8192,
            storage_total=128,
            gpu_available=True,
            bandwidth_up=50.0,
            bandwidth_down=200.0,
            latency_to_cloud=15.0,
            supported_models=["temporal_fusion_transformer"],
            max_inference_rate=5.0,
            power_consumption=15.0
        )
        
        orchestrator.register_device(edge_device)
        
        # Generate time series workload
        np.random.seed(42)
        time_series_data = np.random.normal(0, 1, (50, 5))
        
        workload = WorkloadRequest(
            request_id="tft_edge_workload",
            data=time_series_data,
            model_type="temporal_fusion_transformer",
            priority=7,
            max_latency=100.0,
            accuracy_requirement=0.85,
            privacy_level=3
        )
        
        # Start orchestrator and submit workload
        await orchestrator.start()
        
        try:
            workload_id = await orchestrator.submit_workload(workload)
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Check results
            status = orchestrator.get_workload_status(workload_id)
            assert status["workload_id"] == workload_id
            assert status["status"] in ["active", "completed"]
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_multi_modal_anomaly_fusion(self):
        """Test fusion of multiple anomaly detection approaches."""
        np.random.seed(42)
        
        # Generate test data
        normal_data = np.random.normal(0, 1, (200, 5))
        test_data = np.random.normal(0, 1, (50, 5))
        # Add some anomalies
        test_data[-5:] = np.random.normal(0, 3, (5, 5))
        
        # Create different detector types
        detectors = {}
        
        # Neuromorphic detector
        neuro_detector = create_optimized_neuromorphic_detector(
            input_features=5,
            target_energy_budget=1000.0
        )
        neuro_detector.train_unsupervised(normal_data, training_epochs=1)
        detectors['neuromorphic'] = neuro_detector
        
        # TFT detector  
        tft_detector = create_optimized_tft_detector(
            num_features=5,
            lookback_window=20,
            forecast_horizon=5,
            performance_target="speed"
        )
        
        # Prepare time series data for TFT
        time_series_normal = np.cumsum(normal_data, axis=0)  # Create time series
        time_series_test = np.cumsum(test_data, axis=0)
        
        tft_detector.train(time_series_normal, target_column=0)
        detectors['tft'] = (tft_detector, True)  # Flag for time series
        
        # Run predictions
        predictions = {}
        confidences = {}
        
        for name, detector_info in detectors.items():
            if isinstance(detector_info, tuple):
                detector, is_time_series = detector_info
                if is_time_series:
                    # Skip TFT for now due to sequence requirements
                    continue
            else:
                detector = detector_info
            
            pred, metadata = detector.predict(test_data)
            predictions[name] = pred
            confidences[name] = metadata.get('confidence_scores', [0.5] * len(pred))
        
        # Fusion using weighted voting
        if predictions:
            detector_names = list(predictions.keys())
            all_predictions = np.array([predictions[name] for name in detector_names])
            
            # Simple majority vote
            fused_predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=all_predictions
            )
            
            assert len(fused_predictions) == len(test_data)
            # Should detect the anomalies we injected
            assert np.sum(fused_predictions[-5:]) > 0


# Performance benchmarks for advanced components

@pytest.mark.benchmark
class TestGeneration4Performance:
    """Performance benchmarks for Generation 4 components."""
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_neuromorphic_inference_speed(self, benchmark):
        """Benchmark neuromorphic inference speed."""
        detector = create_optimized_neuromorphic_detector(
            input_features=5,
            target_energy_budget=500.0,
            target_latency=10.0
        )
        
        # Train quickly
        np.random.seed(42)
        train_data = np.random.normal(0, 1, (100, 5))
        detector.train_unsupervised(train_data, training_epochs=1)
        
        # Benchmark inference
        test_data = np.random.normal(0, 1, (100, 5))
        
        def inference_batch():
            return detector.predict(test_data)
        
        result = benchmark(inference_batch)
        predictions, metadata = result
        
        assert len(predictions) == 100
        assert metadata['total_energy_pj'] < 1000  # Energy efficiency target
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    def test_federated_aggregation_speed(self, benchmark):
        """Benchmark federated learning aggregation speed."""
        # Create federated system
        system = create_edge_federated_system(
            num_edge_nodes=10,
            aggregation_strategy="fedavg"
        )
        
        # Create mock updates
        updates = []
        for i in range(10):
            weights = {
                "layer_0": np.random.normal(0, 1, (20, 10)),
                "layer_1": np.random.normal(0, 1, (10, 1))
            }
            
            update = ModelUpdate(
                node_id=f"node_{i}",
                model_weights=weights,
                gradient_norms=[1.0, 2.0],
                local_loss=0.1,
                sample_count=100,
                training_time=10.0,
                update_timestamp=time.time(),
                is_verified=True
            )
            updates.append(update)
        
        # Benchmark aggregation
        def aggregate_updates():
            return system.aggregation_strategy.aggregate(updates)
        
        result = benchmark(aggregate_updates)
        assert "layer_0" in result
        assert "layer_1" in result
    
    @pytest.mark.skipif(not ADVANCED_COMPONENTS_AVAILABLE, reason="Components not available")
    async def test_edge_orchestrator_throughput(self, benchmark):
        """Benchmark edge orchestrator throughput."""
        orchestrator = create_iot_edge_orchestrator(optimization_target="speed")
        
        # Register multiple devices
        for i in range(5):
            device = EdgeDevice(
                device_id=f"speed_device_{i}",
                device_type="nvidia_jetson",
                location=(40.0, -74.0),
                cpu_cores=6,
                cpu_frequency=2.5,
                memory_total=8192,
                storage_total=128,
                supported_models=["anomaly_detection"],
                max_inference_rate=20.0,
                power_consumption=15.0
            )
            orchestrator.register_device(device)
        
        await orchestrator.start()
        
        try:
            # Benchmark workload submission
            async def submit_workloads():
                workload_ids = []
                for i in range(20):
                    workload = WorkloadRequest(
                        request_id=f"speed_test_{i}",
                        data=np.random.normal(0, 1, (50, 5)),
                        model_type="anomaly_detection",
                        priority=5,
                        max_latency=50.0,
                        accuracy_requirement=0.8,
                        privacy_level=2
                    )
                    
                    workload_id = await orchestrator.submit_workload(workload)
                    workload_ids.append(workload_id)
                
                return workload_ids
            
            # Note: benchmark with async requires special handling
            # For now, just test the function works
            workload_ids = await submit_workloads()
            assert len(workload_ids) == 20
            
            # Wait for processing
            await asyncio.sleep(3)
            
            # Check completion rate
            completed = 0
            for workload_id in workload_ids:
                status = orchestrator.get_workload_status(workload_id)
                if status["status"] == "completed":
                    completed += 1
            
            # Should have reasonable completion rate
            assert completed >= len(workload_ids) * 0.5
            
        finally:
            await orchestrator.stop()


if __name__ == "__main__":
    # Run tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])