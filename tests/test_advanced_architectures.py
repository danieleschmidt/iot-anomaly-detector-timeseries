"""Comprehensive tests for advanced autoencoder architectures.

Test suite for Transformer, Variational, and Quantum-Hybrid autoencoders
introduced in Generation 3+ enhancements.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import json
from pathlib import Path

# Test imports - handle TensorFlow availability
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Import modules under test
from src.transformer_autoencoder import (
    TransformerAutoencoder,
    TransformerAutoencoderBuilder,
    create_transformer_autoencoder,
    get_transformer_presets
)

from src.variational_autoencoder import (
    VariationalAutoencoder,
    VAEAutoencoderBuilder,
    create_variational_autoencoder,
    get_vae_presets
)

from src.quantum_hybrid_autoencoder import (
    QuantumHybridAutoencoder,
    QuantumHybridAutoencoderBuilder,
    create_quantum_hybrid_autoencoder,
    get_quantum_hybrid_presets,
    QuantumFeatureMap
)

from src.advanced_autoencoder_cli import AdvancedAutoencoderManager


@pytest.fixture
def sample_time_series_data():
    """Generate sample time series data for testing."""
    np.random.seed(42)
    # Generate synthetic IoT sensor data
    n_samples, sequence_length, n_features = 100, 30, 5
    
    # Create normal patterns
    t = np.linspace(0, 4 * np.pi, sequence_length)
    data = np.zeros((n_samples, sequence_length, n_features))
    
    for i in range(n_samples):
        for j in range(n_features):
            # Sinusoidal patterns with noise
            amplitude = np.random.uniform(0.5, 2.0)
            frequency = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            noise = np.random.normal(0, 0.1, sequence_length)
            
            data[i, :, j] = amplitude * np.sin(frequency * t + phase) + noise
    
    return data


@pytest.fixture
def input_shape():
    """Standard input shape for testing."""
    return (30, 5)  # 30 time steps, 5 features


class TestTransformerAutoencoder:
    """Test suite for Transformer-based Autoencoder."""
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_transformer_builder_creation(self, input_shape):
        """Test transformer builder instantiation and configuration."""
        builder = TransformerAutoencoderBuilder(input_shape)
        
        assert builder.input_shape == input_shape
        assert builder.config['d_model'] == 256
        assert builder.config['num_heads'] == 8
        assert builder.config['latent_dim'] == 64
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_transformer_builder_configuration(self, input_shape):
        """Test transformer builder configuration methods."""
        builder = TransformerAutoencoderBuilder(input_shape)
        
        # Test model dimensions configuration
        builder.set_model_dimensions(d_model=128, latent_dim=32)
        assert builder.config['d_model'] == 128
        assert builder.config['latent_dim'] == 32
        
        # Test attention configuration
        builder.set_attention_config(num_heads=4, dff=256)
        assert builder.config['num_heads'] == 4
        assert builder.config['dff'] == 256
        
        # Test architecture depth
        builder.set_architecture_depth(num_encoder_layers=2, num_decoder_layers=2)
        assert builder.config['num_encoder_layers'] == 2
        assert builder.config['num_decoder_layers'] == 2
        
        # Test regularization
        builder.set_regularization(dropout_rate=0.2, use_positional_encoding=False)
        assert builder.config['dropout_rate'] == 0.2
        assert builder.config['use_positional_encoding'] == False
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_transformer_model_creation(self, input_shape):
        """Test transformer model creation and basic functionality."""
        builder = TransformerAutoencoderBuilder(input_shape)
        builder.set_model_dimensions(d_model=64, latent_dim=16)  # Smaller for testing
        builder.set_architecture_depth(num_encoder_layers=1, num_decoder_layers=1)
        
        model = builder.build()
        
        assert model is not None
        assert isinstance(model, TransformerAutoencoder)
        assert model.input_shape_val == input_shape
        assert model.d_model == 64
        assert model.latent_dim == 16
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_transformer_forward_pass(self, input_shape, sample_time_series_data):
        """Test transformer forward pass with sample data."""
        builder = TransformerAutoencoderBuilder(input_shape)
        builder.set_model_dimensions(d_model=32, latent_dim=8)  # Very small for testing
        builder.set_architecture_depth(num_encoder_layers=1, num_decoder_layers=1)
        
        model = builder.build()
        
        # Test forward pass
        batch_size = 10
        test_input = sample_time_series_data[:batch_size]
        
        output = model(test_input)
        
        assert output.shape == test_input.shape
        assert not np.isnan(output.numpy()).any()
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_transformer_presets(self, input_shape):
        """Test transformer preset configurations."""
        presets = get_transformer_presets()
        
        assert 'lightweight_transformer' in presets
        assert 'standard_transformer' in presets
        assert 'deep_transformer' in presets
        
        # Test preset creation
        model = create_transformer_autoencoder(input_shape, 'lightweight_transformer')
        assert model is not None
        assert isinstance(model, TransformerAutoencoder)


class TestVariationalAutoencoder:
    """Test suite for Variational Autoencoder."""
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_vae_builder_creation(self, input_shape):
        """Test VAE builder instantiation and configuration."""
        builder = VAEAutoencoderBuilder(input_shape)
        
        assert builder.input_shape == input_shape
        assert builder.config['latent_dim'] == 32
        assert builder.config['architecture_type'] == 'lstm'
        assert builder.config['beta'] == 1.0
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_vae_builder_configuration(self, input_shape):
        """Test VAE builder configuration methods."""
        builder = VAEAutoencoderBuilder(input_shape)
        
        # Test latent configuration
        builder.set_latent_config(latent_dim=16, beta=2.0)
        assert builder.config['latent_dim'] == 16
        assert builder.config['beta'] == 2.0
        
        # Test architecture configuration
        builder.set_architecture(architecture_type='gru', hidden_units=[64, 32])
        assert builder.config['architecture_type'] == 'gru'
        assert builder.config['hidden_units'] == [64, 32]
        
        # Test regularization
        builder.set_regularization(dropout_rate=0.15, use_batch_norm=False)
        assert builder.config['dropout_rate'] == 0.15
        assert builder.config['use_batch_norm'] == False
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_vae_model_creation(self, input_shape):
        """Test VAE model creation and basic functionality."""
        builder = VAEAutoencoderBuilder(input_shape)
        builder.set_latent_config(latent_dim=8)  # Smaller for testing
        builder.set_architecture(hidden_units=[32, 16])  # Smaller architecture
        
        model = builder.build()
        
        assert model is not None
        assert isinstance(model, VariationalAutoencoder)
        assert model.input_shape_val == input_shape
        assert model.latent_dim == 8
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_vae_forward_pass(self, input_shape, sample_time_series_data):
        """Test VAE forward pass with sample data."""
        builder = VAEAutoencoderBuilder(input_shape)
        builder.set_latent_config(latent_dim=4)  # Very small for testing
        builder.set_architecture(hidden_units=[16, 8])  # Very small architecture
        
        model = builder.build()
        
        # Test forward pass
        batch_size = 10
        test_input = sample_time_series_data[:batch_size]
        
        output = model(test_input)
        
        assert output.shape == test_input.shape
        assert not np.isnan(output.numpy()).any()
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_vae_encoding_components(self, input_shape, sample_time_series_data):
        """Test VAE encoding and latent space operations."""
        builder = VAEAutoencoderBuilder(input_shape)
        builder.set_latent_config(latent_dim=4)
        builder.set_architecture(hidden_units=[16, 8])
        
        model = builder.build()
        
        batch_size = 5
        test_input = sample_time_series_data[:batch_size]
        
        # Test encoding
        z_mean, z_log_var, z = model.encode(test_input)
        
        assert z_mean.shape == (batch_size, 4)
        assert z_log_var.shape == (batch_size, 4)
        assert z.shape == (batch_size, 4)
        
        # Test reconstruction error calculation
        if hasattr(model, 'get_reconstruction_error'):
            error_dict = model.get_reconstruction_error(test_input)
            assert 'mse_error' in error_dict
            assert 'uncertainty' in error_dict
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_vae_presets(self, input_shape):
        """Test VAE preset configurations."""
        presets = get_vae_presets()
        
        assert 'lightweight_vae' in presets
        assert 'standard_vae' in presets
        assert 'beta_vae' in presets
        
        # Test preset creation
        model = create_variational_autoencoder(input_shape, 'lightweight_vae')
        assert model is not None
        assert isinstance(model, VariationalAutoencoder)


class TestQuantumHybridAutoencoder:
    """Test suite for Quantum-Classical Hybrid Autoencoder."""
    
    def test_quantum_feature_map_creation(self):
        """Test quantum feature map configuration."""
        config = QuantumFeatureMap(
            encoding_type="amplitude",
            num_qubits=4,
            entanglement_pattern="linear",
            use_variational_form=True
        )
        
        assert config.encoding_type == "amplitude"
        assert config.num_qubits == 4
        assert config.entanglement_pattern == "linear"
        assert config.use_variational_form == True
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_quantum_builder_creation(self, input_shape):
        """Test quantum hybrid builder instantiation."""
        builder = QuantumHybridAutoencoderBuilder(input_shape)
        
        assert builder.input_shape == input_shape
        assert builder.classical_latent_dim == 64
        assert builder.quantum_latent_dim == 32
        assert builder.use_quantum_attention == True
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_quantum_builder_configuration(self, input_shape):
        """Test quantum hybrid builder configuration methods."""
        builder = QuantumHybridAutoencoderBuilder(input_shape)
        
        # Test quantum configuration
        builder.set_quantum_config(
            encoding_type="angle",
            num_qubits=6,
            entanglement_pattern="circular"
        )
        assert builder.quantum_config.encoding_type == "angle"
        assert builder.quantum_config.num_qubits == 6
        assert builder.quantum_config.entanglement_pattern == "circular"
        
        # Test latent dimensions
        builder.set_latent_dimensions(classical_dim=32, quantum_dim=16)
        assert builder.classical_latent_dim == 32
        assert builder.quantum_latent_dim == 16
        
        # Test hybrid options
        builder.set_hybrid_options(use_quantum_attention=False, fusion_method="attention")
        assert builder.use_quantum_attention == False
        assert builder.hybrid_fusion_method == "attention"
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_quantum_model_creation(self, input_shape):
        """Test quantum hybrid model creation."""
        builder = QuantumHybridAutoencoderBuilder(input_shape)
        builder.set_latent_dimensions(classical_dim=16, quantum_dim=8)  # Smaller for testing
        builder.set_quantum_config(num_qubits=4)  # Smaller quantum register
        
        model = builder.build()
        
        assert model is not None
        assert isinstance(model, QuantumHybridAutoencoder)
        assert model.input_shape_val == input_shape
        assert model.classical_latent_dim == 16
        assert model.quantum_latent_dim == 8
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_quantum_forward_pass(self, input_shape, sample_time_series_data):
        """Test quantum hybrid forward pass."""
        builder = QuantumHybridAutoencoderBuilder(input_shape)
        builder.set_latent_dimensions(classical_dim=8, quantum_dim=4)  # Very small
        builder.set_quantum_config(num_qubits=2)  # Minimal quantum register
        builder.set_hybrid_options(use_quantum_attention=False)  # Disable for simplicity
        
        model = builder.build()
        
        batch_size = 5
        test_input = sample_time_series_data[:batch_size]
        
        output = model(test_input)
        
        assert output.shape == test_input.shape
        assert not np.isnan(output.numpy()).any()
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_quantum_latent_representations(self, input_shape, sample_time_series_data):
        """Test quantum hybrid latent representations."""
        builder = QuantumHybridAutoencoderBuilder(input_shape)
        builder.set_latent_dimensions(classical_dim=8, quantum_dim=4)
        builder.set_quantum_config(num_qubits=2)
        builder.set_hybrid_options(use_quantum_attention=False)
        
        model = builder.build()
        
        batch_size = 5
        test_input = sample_time_series_data[:batch_size]
        
        latent_dict = model.get_latent_representations(test_input)
        
        assert 'classical' in latent_dict
        assert 'quantum' in latent_dict
        assert 'fused' in latent_dict
        
        assert latent_dict['classical'].shape[0] == batch_size
        assert latent_dict['quantum'].shape[0] == batch_size
        assert latent_dict['fused'].shape[0] == batch_size
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_quantum_presets(self, input_shape):
        """Test quantum hybrid preset configurations."""
        presets = get_quantum_hybrid_presets()
        
        assert 'lightweight_quantum' in presets
        assert 'standard_quantum' in presets
        assert 'advanced_quantum' in presets
        
        # Test preset creation
        model = create_quantum_hybrid_autoencoder(input_shape, 'lightweight_quantum')
        assert model is not None
        assert isinstance(model, QuantumHybridAutoencoder)


class TestAdvancedAutoencoderManager:
    """Test suite for Advanced Autoencoder Manager."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = AdvancedAutoencoderManager()
        
        assert 'transformer' in manager.supported_architectures
        assert 'variational' in manager.supported_architectures
        assert 'quantum_hybrid' in manager.supported_architectures
        assert 'classical' in manager.supported_architectures
    
    def test_get_available_presets(self):
        """Test getting available presets for different architectures."""
        manager = AdvancedAutoencoderManager()
        
        # Test each architecture type
        for arch_type in ['transformer', 'variational', 'quantum_hybrid', 'classical']:
            presets = manager.get_available_presets(arch_type)
            assert isinstance(presets, dict)
            assert len(presets) > 0
    
    def test_get_available_presets_invalid(self):
        """Test getting presets for invalid architecture type."""
        manager = AdvancedAutoencoderManager()
        
        with pytest.raises(ValueError, match="Unsupported architecture type"):
            manager.get_available_presets('invalid_architecture')
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_create_model_with_presets(self, input_shape):
        """Test model creation with presets."""
        manager = AdvancedAutoencoderManager()
        
        # Test transformer with preset
        model = manager.create_model('transformer', input_shape, 'lightweight_transformer')
        assert model is not None
        assert isinstance(model, TransformerAutoencoder)
        
        # Test VAE with preset
        model = manager.create_model('variational', input_shape, 'lightweight_vae')
        assert model is not None
        assert isinstance(model, VariationalAutoencoder)
        
        # Test quantum hybrid with preset
        model = manager.create_model('quantum_hybrid', input_shape, 'lightweight_quantum')
        assert model is not None
        assert isinstance(model, QuantumHybridAutoencoder)
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_create_model_with_custom_config(self, input_shape):
        """Test model creation with custom configuration."""
        manager = AdvancedAutoencoderManager()
        
        # Test transformer with custom config
        custom_config = {
            'd_model': 64,
            'latent_dim': 16,
            'num_heads': 4,
            'num_encoder_layers': 1
        }
        
        model = manager.create_model('transformer', input_shape, 
                                   custom_config=custom_config)
        assert model is not None
        assert model.d_model == 64
        assert model.latent_dim == 16
        assert model.num_heads == 4
    
    def test_create_model_without_tensorflow(self, input_shape):
        """Test model creation error handling when TensorFlow is not available."""
        manager = AdvancedAutoencoderManager()
        
        with patch('src.advanced_autoencoder_cli.TENSORFLOW_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="TensorFlow is required"):
                manager.create_model('transformer', input_shape, 'standard_transformer')


class TestAdvancedAutoencoderCLI:
    """Test suite for Advanced Autoencoder CLI functionality."""
    
    def test_cli_imports(self):
        """Test that CLI module imports correctly."""
        from src.advanced_autoencoder_cli import main, AdvancedAutoencoderManager
        
        assert callable(main)
        assert AdvancedAutoencoderManager is not None
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_training_integration(self, input_shape, sample_time_series_data):
        """Test training integration with sample data."""
        manager = AdvancedAutoencoderManager()
        
        # Create a simple model for testing
        model = manager.create_model('transformer', input_shape, 'lightweight_transformer')
        
        # Prepare training data
        train_data = sample_time_series_data[:80]  # 80% for training
        val_data = sample_time_series_data[80:]    # 20% for validation
        
        # Test training with minimal epochs
        history = manager.train_model(
            model, train_data, val_data,
            epochs=2, batch_size=16, verbose=0  # Minimal for testing
        )
        
        assert history is not None
        assert 'loss' in history.history
        assert len(history.history['loss']) <= 2  # May stop early due to callbacks
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_evaluation_integration(self, input_shape, sample_time_series_data):
        """Test evaluation integration."""
        manager = AdvancedAutoencoderManager()
        
        # Create and train a minimal model
        model = manager.create_model('transformer', input_shape, 'lightweight_transformer')
        
        # Use sample data for evaluation
        test_data = sample_time_series_data[:20]
        
        # Evaluate model
        results = manager.evaluate_model(model, test_data, 'transformer')
        
        assert 'mse_error' in results
        assert 'mae_error' in results
        assert 'num_parameters' in results
        assert isinstance(results['mse_error'], float)
        assert isinstance(results['mae_error'], float)
        assert isinstance(results['num_parameters'], int)


@pytest.mark.integration
class TestArchitecturesIntegration:
    """Integration tests for all advanced architectures."""
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_all_architectures_training_pipeline(self, input_shape, sample_time_series_data):
        """Test complete training pipeline for all architectures."""
        manager = AdvancedAutoencoderManager()
        
        architectures = ['transformer', 'variational', 'quantum_hybrid']
        presets = ['lightweight_transformer', 'lightweight_vae', 'lightweight_quantum']
        
        for arch, preset in zip(architectures, presets):
            # Create model
            model = manager.create_model(arch, input_shape, preset)
            assert model is not None
            
            # Train briefly
            train_data = sample_time_series_data[:50]
            history = manager.train_model(
                model, train_data, epochs=1, batch_size=10, verbose=0
            )
            assert history is not None
            
            # Evaluate
            test_data = sample_time_series_data[50:60]
            results = manager.evaluate_model(model, test_data, arch)
            assert 'mse_error' in results
            assert 'mae_error' in results
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available") 
    def test_model_saving_loading(self, input_shape, sample_time_series_data):
        """Test model saving and loading functionality."""
        manager = AdvancedAutoencoderManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.h5"
            
            # Create and train a model
            model = manager.create_model('transformer', input_shape, 'lightweight_transformer')
            train_data = sample_time_series_data[:30]
            
            # Train briefly
            manager.train_model(model, train_data, epochs=1, batch_size=10, verbose=0)
            
            # Save model
            model.save(str(model_path))
            assert model_path.exists()
            
            # Load model
            loaded_model = tf.keras.models.load_model(str(model_path), compile=False)
            
            # Test that loaded model works
            test_input = sample_time_series_data[:5]
            original_output = model.predict(test_input)
            loaded_output = loaded_model.predict(test_input)
            
            # Outputs should be very similar (allowing for small numerical differences)
            np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])