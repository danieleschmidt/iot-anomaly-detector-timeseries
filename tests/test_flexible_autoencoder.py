"""Tests for flexible autoencoder architecture system."""

import pytest
from unittest.mock import MagicMock, patch
import json

pytest.importorskip("tensorflow")

from src.flexible_autoencoder import (
    FlexibleAutoencoderBuilder,
    create_autoencoder_from_config,
    validate_architecture_config,
    get_predefined_architectures
)


class TestFlexibleAutoencoderBuilder:
    """Test the flexible autoencoder builder."""
    
    def test_builder_initialization(self):
        """Test builder initialization with default values."""
        builder = FlexibleAutoencoderBuilder(input_shape=(30, 3))
        
        assert builder.input_shape == (30, 3)
        assert builder.encoder_layers == []
        assert builder.decoder_layers == []
        assert builder.latent_dim == 16
        assert builder.optimizer == 'adam'
        assert builder.loss == 'mse'
    
    def test_add_lstm_encoder_layer(self):
        """Test adding LSTM encoder layers."""
        builder = FlexibleAutoencoderBuilder(input_shape=(30, 3))
        
        # Add LSTM layer
        builder.add_encoder_layer('lstm', units=64, return_sequences=True, dropout=0.2)
        
        assert len(builder.encoder_layers) == 1
        layer_config = builder.encoder_layers[0]
        assert layer_config['type'] == 'lstm'
        assert layer_config['units'] == 64
        assert layer_config['return_sequences'] == True
        assert layer_config['dropout'] == 0.2
    
    def test_add_gru_encoder_layer(self):
        """Test adding GRU encoder layers."""
        builder = FlexibleAutoencoderBuilder(input_shape=(30, 3))
        
        builder.add_encoder_layer('gru', units=32, return_sequences=False, recurrent_dropout=0.1)
        
        layer_config = builder.encoder_layers[0]
        assert layer_config['type'] == 'gru'
        assert layer_config['units'] == 32
        assert layer_config['return_sequences'] == False
        assert layer_config['recurrent_dropout'] == 0.1
    
    def test_add_dense_layer(self):
        """Test adding dense layers."""
        builder = FlexibleAutoencoderBuilder(input_shape=(30, 3))
        
        builder.add_encoder_layer('dense', units=128, activation='relu', use_bias=True)
        
        layer_config = builder.encoder_layers[0]
        assert layer_config['type'] == 'dense'
        assert layer_config['units'] == 128
        assert layer_config['activation'] == 'relu'
        assert layer_config['use_bias'] == True
    
    def test_add_conv1d_layer(self):
        """Test adding Conv1D layers."""
        builder = FlexibleAutoencoderBuilder(input_shape=(30, 3))
        
        builder.add_encoder_layer('conv1d', filters=32, kernel_size=3, activation='tanh', padding='same')
        
        layer_config = builder.encoder_layers[0]
        assert layer_config['type'] == 'conv1d'
        assert layer_config['filters'] == 32
        assert layer_config['kernel_size'] == 3
        assert layer_config['activation'] == 'tanh'
        assert layer_config['padding'] == 'same'
    
    def test_add_normalization_layer(self):
        """Test adding normalization layers."""
        builder = FlexibleAutoencoderBuilder(input_shape=(30, 3))
        
        builder.add_encoder_layer('batch_norm')
        builder.add_encoder_layer('layer_norm')
        
        assert len(builder.encoder_layers) == 2
        assert builder.encoder_layers[0]['type'] == 'batch_norm'
        assert builder.encoder_layers[1]['type'] == 'layer_norm'
    
    def test_add_dropout_layer(self):
        """Test adding dropout layers."""
        builder = FlexibleAutoencoderBuilder(input_shape=(30, 3))
        
        builder.add_encoder_layer('dropout', rate=0.3)
        
        layer_config = builder.encoder_layers[0]
        assert layer_config['type'] == 'dropout'
        assert layer_config['rate'] == 0.3
    
    def test_set_latent_configuration(self):
        """Test setting latent space configuration."""
        builder = FlexibleAutoencoderBuilder(input_shape=(30, 3))
        
        builder.set_latent_config(dim=32, activation='tanh', regularization='l2')
        
        assert builder.latent_dim == 32
        assert builder.latent_activation == 'tanh'
        assert builder.latent_regularization == 'l2'
    
    def test_set_compilation_options(self):
        """Test setting model compilation options."""
        builder = FlexibleAutoencoderBuilder(input_shape=(30, 3))
        
        builder.set_compilation(optimizer='rmsprop', loss='mae', metrics=['mse', 'mae'])
        
        assert builder.optimizer == 'rmsprop'
        assert builder.loss == 'mae'
        assert builder.metrics == ['mse', 'mae']
    
    def test_invalid_layer_type_error(self):
        """Test error handling for invalid layer types."""
        builder = FlexibleAutoencoderBuilder(input_shape=(30, 3))
        
        with pytest.raises(ValueError, match="Unsupported layer type"):
            builder.add_encoder_layer('invalid_layer_type')
    
    @patch('src.flexible_autoencoder.layers')
    @patch('src.flexible_autoencoder.models')
    def test_build_model_basic_architecture(self, mock_models, mock_layers, tmp_path):
        """Test building a basic flexible model."""
        builder = FlexibleAutoencoderBuilder(input_shape=(30, 3))
        
        # Add simple architecture
        builder.add_encoder_layer('lstm', units=64, return_sequences=True)
        builder.add_encoder_layer('lstm', units=32, return_sequences=False)
        builder.set_latent_config(dim=16)
        
        # Mock TensorFlow components
        mock_input = MagicMock()
        mock_layers.Input.return_value = mock_input
        mock_layers.LSTM.return_value = MagicMock()
        mock_layers.Dense.return_value = MagicMock()
        mock_layers.RepeatVector.return_value = MagicMock()
        mock_layers.TimeDistributed.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_models.Model.return_value = mock_model
        
        # Build model
        model = builder.build()
        
        # Verify calls
        mock_layers.Input.assert_called_once_with(shape=(30, 3))
        assert mock_layers.LSTM.call_count >= 2  # Encoder + decoder LSTM layers
        mock_models.Model.assert_called_once()
        mock_model.compile.assert_called_once()
    
    def test_symmetric_decoder_generation(self):
        """Test automatic symmetric decoder generation."""
        builder = FlexibleAutoencoderBuilder(input_shape=(30, 3))
        
        # Add encoder layers
        builder.add_encoder_layer('lstm', units=64, return_sequences=True)
        builder.add_encoder_layer('dropout', rate=0.2)
        builder.add_encoder_layer('lstm', units=32, return_sequences=False)
        
        # Generate symmetric decoder
        builder.generate_symmetric_decoder()
        
        # Should have 3 decoder layers (reverse of encoder, excluding final LSTM)
        assert len(builder.decoder_layers) == 2  # LSTM + dropout (reversed, excluding final LSTM)
        
        # Check layer types are reversed
        assert builder.decoder_layers[0]['type'] == 'lstm'
        assert builder.decoder_layers[1]['type'] == 'dropout'
    
    def test_architecture_validation(self):
        """Test architecture validation."""
        builder = FlexibleAutoencoderBuilder(input_shape=(30, 3))
        
        # Valid architecture
        builder.add_encoder_layer('lstm', units=64, return_sequences=True)
        builder.add_encoder_layer('lstm', units=32, return_sequences=False)
        assert builder.validate_architecture() == True
        
        # Invalid: all LSTM layers have return_sequences=False
        builder_invalid = FlexibleAutoencoderBuilder(input_shape=(30, 3))
        builder_invalid.add_encoder_layer('lstm', units=64, return_sequences=False)
        builder_invalid.add_encoder_layer('lstm', units=32, return_sequences=False)
        assert builder_invalid.validate_architecture() == False


class TestArchitectureConfig:
    """Test architecture configuration and validation."""
    
    def test_validate_architecture_config_valid(self):
        """Test validation of valid architecture config."""
        config = {
            "input_shape": [30, 3],
            "encoder_layers": [
                {"type": "lstm", "units": 64, "return_sequences": True},
                {"type": "dropout", "rate": 0.2},
                {"type": "lstm", "units": 32, "return_sequences": False}
            ],
            "latent_config": {"dim": 16, "activation": "linear"},
            "compilation": {
                "optimizer": "adam",
                "loss": "mse",
                "metrics": ["mae"]
            }
        }
        
        assert validate_architecture_config(config) == True
    
    def test_validate_architecture_config_invalid(self):
        """Test validation of invalid architecture config."""
        # Missing required fields
        config_missing = {
            "encoder_layers": [
                {"type": "lstm", "units": 64}
            ]
        }
        
        assert validate_architecture_config(config_missing) == False
        
        # Invalid layer type
        config_invalid_layer = {
            "input_shape": [30, 3],
            "encoder_layers": [
                {"type": "invalid_layer", "units": 64}
            ],
            "latent_config": {"dim": 16},
            "compilation": {"optimizer": "adam", "loss": "mse"}
        }
        
        assert validate_architecture_config(config_invalid_layer) == False
    
    @patch('src.flexible_autoencoder.FlexibleAutoencoderBuilder')
    def test_create_autoencoder_from_config(self, mock_builder_class):
        """Test creating autoencoder from configuration."""
        # Mock builder
        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_model = MagicMock()
        mock_builder.build.return_value = mock_model
        
        config = {
            "input_shape": [30, 3],
            "encoder_layers": [
                {"type": "lstm", "units": 64, "return_sequences": True},
                {"type": "lstm", "units": 32, "return_sequences": False}
            ],
            "latent_config": {"dim": 16},
            "compilation": {"optimizer": "adam", "loss": "mse"}
        }
        
        result = create_autoencoder_from_config(config)
        
        # Verify builder was created and configured
        mock_builder_class.assert_called_once_with(input_shape=(30, 3))
        assert mock_builder.add_encoder_layer.call_count == 2
        mock_builder.set_latent_config.assert_called_once_with(dim=16)
        mock_builder.set_compilation.assert_called_once_with(optimizer="adam", loss="mse")
        mock_builder.build.assert_called_once()
        assert result == mock_model
    
    def test_get_predefined_architectures(self):
        """Test getting predefined architecture configurations."""
        architectures = get_predefined_architectures()
        
        # Should have standard architectures
        assert 'simple_lstm' in architectures
        assert 'deep_lstm' in architectures
        assert 'hybrid_conv_lstm' in architectures
        assert 'gru_based' in architectures
        
        # Each architecture should have required fields
        for name, config in architectures.items():
            assert 'input_shape' in config
            assert 'encoder_layers' in config
            assert 'latent_config' in config
            assert 'compilation' in config
            assert validate_architecture_config(config) == True


class TestArchitectureIntegration:
    """Integration tests for flexible architecture system."""
    
    def test_architecture_config_serialization(self, tmp_path):
        """Test saving and loading architecture configurations."""
        config = {
            "name": "test_architecture",
            "description": "Test architecture for validation",
            "input_shape": [30, 3],
            "encoder_layers": [
                {"type": "lstm", "units": 64, "return_sequences": True, "dropout": 0.1},
                {"type": "batch_norm"},
                {"type": "lstm", "units": 32, "return_sequences": False}
            ],
            "latent_config": {"dim": 16, "activation": "linear"},
            "compilation": {
                "optimizer": "adam",
                "loss": "mse",
                "metrics": ["mae"]
            }
        }
        
        # Save configuration
        config_path = tmp_path / "test_architecture.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Load and validate
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config == config
        assert validate_architecture_config(loaded_config) == True
    
    @patch('src.flexible_autoencoder.layers')
    @patch('src.flexible_autoencoder.models')
    def test_complex_architecture_building(self, mock_models, mock_layers):
        """Test building a complex architecture with multiple layer types."""
        # Mock TensorFlow components
        mock_layers.Input.return_value = MagicMock()
        mock_layers.LSTM.return_value = MagicMock()
        mock_layers.GRU.return_value = MagicMock()
        mock_layers.Dense.return_value = MagicMock()
        mock_layers.Conv1D.return_value = MagicMock()
        mock_layers.BatchNormalization.return_value = MagicMock()
        mock_layers.Dropout.return_value = MagicMock()
        mock_layers.RepeatVector.return_value = MagicMock()
        mock_layers.TimeDistributed.return_value = MagicMock()
        
        mock_model = MagicMock()
        mock_models.Model.return_value = mock_model
        
        # Build complex architecture
        builder = FlexibleAutoencoderBuilder(input_shape=(50, 5))
        builder.add_encoder_layer('conv1d', filters=64, kernel_size=3, activation='relu')
        builder.add_encoder_layer('batch_norm')
        builder.add_encoder_layer('lstm', units=128, return_sequences=True, dropout=0.2)
        builder.add_encoder_layer('gru', units=64, return_sequences=False, recurrent_dropout=0.1)
        builder.add_encoder_layer('dense', units=32, activation='tanh')
        builder.set_latent_config(dim=16, activation='linear')
        builder.set_compilation(optimizer='rmsprop', loss='mae', metrics=['mse'])
        
        model = builder.build()
        
        # Verify complex architecture was built
        assert mock_layers.Conv1D.called
        assert mock_layers.BatchNormalization.called
        assert mock_layers.LSTM.called
        assert mock_layers.GRU.called
        assert mock_layers.Dense.called
        mock_models.Model.assert_called_once()
        mock_model.compile.assert_called_once()


def test_flexible_autoencoder_imports():
    """Test that flexible autoencoder module can be imported."""
    # This verifies the module structure is correct
    import src.flexible_autoencoder
    
    # Check key classes and functions exist
    assert hasattr(src.flexible_autoencoder, 'FlexibleAutoencoderBuilder')
    assert hasattr(src.flexible_autoencoder, 'create_autoencoder_from_config')
    assert hasattr(src.flexible_autoencoder, 'validate_architecture_config')
    assert hasattr(src.flexible_autoencoder, 'get_predefined_architectures')