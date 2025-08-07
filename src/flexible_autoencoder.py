"""Flexible autoencoder architecture system for IoT Anomaly Detection.

This module provides a flexible system for creating autoencoder architectures
with configurable layers, activation functions, and compilation options.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

try:
    from tensorflow.keras import layers, models, losses, optimizers, regularizers
    from tensorflow.keras.models import Model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Create mock Model for when TensorFlow is not available
    class Model:
        def __init__(self, *args, **kwargs):
            pass

from .logging_config import get_logger


class FlexibleAutoencoderBuilder:
    """Builder class for creating flexible autoencoder architectures."""
    
    def __init__(self, input_shape: Tuple[int, int]):
        """Initialize the autoencoder builder.
        
        Parameters
        ----------
        input_shape : Tuple[int, int]
            Shape of the input data (time_steps, features)
        """
        self.input_shape = input_shape
        self.encoder_layers: List[Dict[str, Any]] = []
        self.decoder_layers: List[Dict[str, Any]] = []
        
        # Latent space configuration
        self.latent_dim = 16
        self.latent_activation = 'linear'
        self.latent_regularization = None
        
        # Compilation configuration
        self.optimizer = 'adam'
        self.loss = 'mse'
        self.metrics: Optional[List[str]] = None
        
        self.logger = get_logger(__name__)
        
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available. Model building will be mocked.")
    
    def add_encoder_layer(self, layer_type: str, **kwargs) -> 'FlexibleAutoencoderBuilder':
        """Add a layer to the encoder.
        
        Parameters
        ----------
        layer_type : str
            Type of layer ('lstm', 'gru', 'dense', 'conv1d', 'batch_norm', 'layer_norm', 'dropout')
        **kwargs
            Layer-specific parameters
            
        Returns
        -------
        FlexibleAutoencoderBuilder
            Self for method chaining
        """
        supported_layers = [
            'lstm', 'gru', 'dense', 'conv1d', 'batch_norm', 'layer_norm', 'dropout'
        ]
        
        if layer_type not in supported_layers:
            raise ValueError(f"Unsupported layer type: {layer_type}. "
                           f"Supported types: {supported_layers}")
        
        layer_config = {'type': layer_type, **kwargs}
        self.encoder_layers.append(layer_config)
        
        self.logger.debug(f"Added encoder layer: {layer_type} with config: {kwargs}")
        return self
    
    def add_decoder_layer(self, layer_type: str, **kwargs) -> 'FlexibleAutoencoderBuilder':
        """Add a layer to the decoder.
        
        Parameters
        ----------
        layer_type : str
            Type of layer
        **kwargs
            Layer-specific parameters
            
        Returns
        -------
        FlexibleAutoencoderBuilder
            Self for method chaining
        """
        layer_config = {'type': layer_type, **kwargs}
        self.decoder_layers.append(layer_config)
        
        self.logger.debug(f"Added decoder layer: {layer_type} with config: {kwargs}")
        return self
    
    def set_latent_config(self, dim: int, activation: str = 'linear', 
                         regularization: Optional[str] = None) -> 'FlexibleAutoencoderBuilder':
        """Configure the latent space.
        
        Parameters
        ----------
        dim : int
            Latent space dimensionality
        activation : str
            Activation function for latent layer
        regularization : str, optional
            Regularization type ('l1', 'l2', 'l1_l2')
            
        Returns
        -------
        FlexibleAutoencoderBuilder
            Self for method chaining
        """
        self.latent_dim = dim
        self.latent_activation = activation
        self.latent_regularization = regularization
        
        self.logger.debug(f"Set latent config: dim={dim}, activation={activation}, "
                         f"regularization={regularization}")
        return self
    
    def set_compilation(self, optimizer: str = 'adam', loss: str = 'mse', 
                       metrics: Optional[List[str]] = None) -> 'FlexibleAutoencoderBuilder':
        """Configure model compilation options.
        
        Parameters
        ----------
        optimizer : str
            Optimizer name
        loss : str
            Loss function name
        metrics : List[str], optional
            List of metrics to track
            
        Returns
        -------
        FlexibleAutoencoderBuilder
            Self for method chaining
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        
        self.logger.debug(f"Set compilation: optimizer={optimizer}, loss={loss}, "
                         f"metrics={metrics}")
        return self
    
    def generate_symmetric_decoder(self) -> 'FlexibleAutoencoderBuilder':
        """Generate a symmetric decoder from the encoder layers.
        
        Returns
        -------
        FlexibleAutoencoderBuilder
            Self for method chaining
        """
        self.decoder_layers.clear()
        
        # Reverse encoder layers (excluding the final layer which becomes latent)
        encoder_for_decoder = self.encoder_layers[:-1] if self.encoder_layers else []
        
        for layer_config in reversed(encoder_for_decoder):
            decoder_config = layer_config.copy()
            
            # Adjust layer configuration for decoder
            if decoder_config['type'] in ['lstm', 'gru']:
                decoder_config['return_sequences'] = True
            
            self.decoder_layers.append(decoder_config)
        
        self.logger.debug(f"Generated symmetric decoder with {len(self.decoder_layers)} layers")
        return self
    
    def validate_architecture(self) -> bool:
        """Validate the current architecture configuration.
        
        Returns
        -------
        bool
            True if architecture is valid
        """
        if not self.encoder_layers:
            self.logger.warning("No encoder layers defined")
            return False
        
        # Check LSTM/GRU layer configurations
        rnn_layers = [layer for layer in self.encoder_layers 
                     if layer['type'] in ['lstm', 'gru']]
        
        if len(rnn_layers) > 1:
            # All but last RNN layer should have return_sequences=True
            for layer in rnn_layers[:-1]:
                if not layer.get('return_sequences', False):
                    self.logger.warning(f"RNN layer {layer} should have return_sequences=True")
                    return False
        
        # Check latent dimension is positive
        if self.latent_dim <= 0:
            self.logger.warning(f"Invalid latent dimension: {self.latent_dim}")
            return False
        
        return True
    
    def build(self) -> Optional[Model]:
        """Build the autoencoder model.
        
        Returns
        -------
        Model or None
            The built Keras model, or None if TensorFlow is not available
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available, returning None")
            return None
        
        if not self.validate_architecture():
            raise ValueError("Invalid architecture configuration")
        
        self.logger.info(f"Building autoencoder with input shape {self.input_shape}")
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        x = inputs
        
        # Encoder layers
        for layer_config in self.encoder_layers:
            x = self._create_layer(layer_config, x)
        
        # Latent layer
        regularizer = None
        if self.latent_regularization:
            if self.latent_regularization == 'l1':
                regularizer = regularizers.l1(0.01)
            elif self.latent_regularization == 'l2':
                regularizer = regularizers.l2(0.01)
            elif self.latent_regularization == 'l1_l2':
                regularizer = regularizers.l1_l2(l1=0.01, l2=0.01)
        
        encoded = layers.Dense(self.latent_dim, 
                              activation=self.latent_activation,
                              activity_regularizer=regularizer,
                              name='latent_layer')(x)
        
        # Decoder preparation
        x = layers.RepeatVector(self.input_shape[0])(encoded)
        # Decoder layers
        if self.decoder_layers:
            for layer_config in self.decoder_layers:
                x = self._create_layer(layer_config, x)
        else:
            # Default symmetric decoder
            self.generate_symmetric_decoder()
            for layer_config in self.decoder_layers:
                x = self._create_layer(layer_config, x)
        
        # Output layer
        decoded = layers.TimeDistributed(
            layers.Dense(self.input_shape[1], activation='linear'),
            name='output_layer'
        )(x)
        
        # Create model
        model = models.Model(inputs, decoded, name='flexible_autoencoder')
        
        # Compile model
        loss_fn = self._get_loss_function(self.loss)
        optimizer_fn = self._get_optimizer(self.optimizer)
        
        model.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=self.metrics)
        
        self.logger.info(f"Built autoencoder with {model.count_params()} parameters")
        return model
    
    def _create_layer(self, layer_config: Dict[str, Any], input_tensor):
        """Create a Keras layer from configuration.
        
        Parameters
        ----------
        layer_config : Dict[str, Any]
            Layer configuration
        input_tensor
            Input tensor for the layer
            
        Returns
        -------
        Tensor
            Output tensor from the layer
        """
        layer_type = layer_config['type']
        config = {k: v for k, v in layer_config.items() if k != 'type'}
        
        if layer_type == 'lstm':
            return layers.LSTM(**config)(input_tensor)
        elif layer_type == 'gru':
            return layers.GRU(**config)(input_tensor)
        elif layer_type == 'dense':
            return layers.Dense(**config)(input_tensor)
        elif layer_type == 'conv1d':
            return layers.Conv1D(**config)(input_tensor)
        elif layer_type == 'batch_norm':
            return layers.BatchNormalization(**config)(input_tensor)
        elif layer_type == 'layer_norm':
            return layers.LayerNormalization(**config)(input_tensor)
        elif layer_type == 'dropout':
            return layers.Dropout(**config)(input_tensor)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
    
    def _get_loss_function(self, loss_name: str):
        """Get loss function from name."""
        loss_map = {
            'mse': losses.MeanSquaredError(),
            'mae': losses.MeanAbsoluteError(),
            'huber': losses.Huber(),
            'logcosh': losses.LogCosh()
        }
        return loss_map.get(loss_name, loss_name)
    
    def _get_optimizer(self, optimizer_name: str):
        """Get optimizer from name."""
        optimizer_map = {
            'adam': optimizers.Adam(),
            'rmsprop': optimizers.RMSprop(),
            'sgd': optimizers.SGD(),
            'adagrad': optimizers.Adagrad()
        }
        return optimizer_map.get(optimizer_name, optimizer_name)


def validate_architecture_config(config: Dict[str, Any]) -> bool:
    """Validate an architecture configuration dictionary.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Architecture configuration
        
    Returns
    -------
    bool
        True if configuration is valid
    """
    logger = get_logger(__name__)
    
    # Required fields
    required_fields = ['input_shape', 'encoder_layers', 'latent_config', 'compilation']
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required field: {field}")
            return False
    
    # Validate input shape
    input_shape = config['input_shape']
    if not isinstance(input_shape, list) or len(input_shape) != 2:
        logger.error(f"Invalid input_shape: {input_shape}")
        return False
    
    # Validate encoder layers
    encoder_layers = config['encoder_layers']
    if not isinstance(encoder_layers, list) or not encoder_layers:
        logger.error("encoder_layers must be a non-empty list")
        return False
    
    supported_layer_types = ['lstm', 'gru', 'dense', 'conv1d', 'batch_norm', 'layer_norm', 'dropout']
    for layer in encoder_layers:
        if 'type' not in layer:
            logger.error(f"Layer missing 'type' field: {layer}")
            return False
        if layer['type'] not in supported_layer_types:
            logger.error(f"Unsupported layer type: {layer['type']}")
            return False
    
    # Validate latent config
    latent_config = config['latent_config']
    if 'dim' not in latent_config:
        logger.error("latent_config missing 'dim' field")
        return False
    if not isinstance(latent_config['dim'], int) or latent_config['dim'] <= 0:
        logger.error(f"Invalid latent dimension: {latent_config['dim']}")
        return False
    
    # Validate compilation config
    compilation = config['compilation']
    required_compilation_fields = ['optimizer', 'loss']
    for field in required_compilation_fields:
        if field not in compilation:
            logger.error(f"compilation missing required field: {field}")
            return False
    
    return True


def create_autoencoder_from_config(config: Dict[str, Any]) -> Optional[Model]:
    """Create an autoencoder from a configuration dictionary.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Architecture configuration
        
    Returns
    -------
    Model or None
        Built autoencoder model
    """
    logger = get_logger(__name__)
    
    if not validate_architecture_config(config):
        raise ValueError("Invalid architecture configuration")
    
    # Create builder
    input_shape = tuple(config['input_shape'])
    builder = FlexibleAutoencoderBuilder(input_shape)
    
    # Add encoder layers
    for layer_config in config['encoder_layers']:
        layer_type = layer_config['type']
        layer_params = {k: v for k, v in layer_config.items() if k != 'type'}
        builder.add_encoder_layer(layer_type, **layer_params)
    
    # Configure latent space
    latent_config = config['latent_config']
    builder.set_latent_config(**latent_config)
    
    # Set compilation options
    compilation = config['compilation']
    builder.set_compilation(**compilation)
    
    # Add decoder layers if specified
    if 'decoder_layers' in config:
        for layer_config in config['decoder_layers']:
            layer_type = layer_config['type']
            layer_params = {k: v for k, v in layer_config.items() if k != 'type'}
            builder.add_decoder_layer(layer_type, **layer_params)
    else:
        # Generate symmetric decoder
        builder.generate_symmetric_decoder()
    
    logger.info(f"Creating autoencoder from config: {config.get('name', 'unnamed')}")
    return builder.build()


def get_predefined_architectures() -> Dict[str, Dict[str, Any]]:
    """Get predefined architecture configurations.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary of predefined architectures
    """
    return {
        'simple_lstm': {
            'name': 'Simple LSTM Autoencoder',
            'description': 'Basic LSTM autoencoder with single encoder/decoder layers',
            'input_shape': [30, 3],
            'encoder_layers': [
                {'type': 'lstm', 'units': 64, 'return_sequences': True},
                {'type': 'lstm', 'units': 32, 'return_sequences': False}
            ],
            'latent_config': {'dim': 16, 'activation': 'linear'},
            'compilation': {'optimizer': 'adam', 'loss': 'mse', 'metrics': ['mae']}
        },
        
        'deep_lstm': {
            'name': 'Deep LSTM Autoencoder',
            'description': 'Multi-layer LSTM autoencoder with dropout and batch normalization',
            'input_shape': [50, 5],
            'encoder_layers': [
                {'type': 'lstm', 'units': 128, 'return_sequences': True, 'dropout': 0.1},
                {'type': 'batch_norm'},
                {'type': 'lstm', 'units': 64, 'return_sequences': True, 'dropout': 0.1},
                {'type': 'batch_norm'},
                {'type': 'lstm', 'units': 32, 'return_sequences': False, 'dropout': 0.1}
            ],
            'latent_config': {'dim': 16, 'activation': 'tanh', 'regularization': 'l2'},
            'compilation': {'optimizer': 'adam', 'loss': 'mse', 'metrics': ['mae']}
        },
        
        'hybrid_conv_lstm': {
            'name': 'Hybrid Conv1D-LSTM Autoencoder',
            'description': 'Combination of convolutional and LSTM layers',
            'input_shape': [60, 8],
            'encoder_layers': [
                {'type': 'conv1d', 'filters': 64, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'},
                {'type': 'batch_norm'},
                {'type': 'conv1d', 'filters': 32, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'},
                {'type': 'lstm', 'units': 64, 'return_sequences': True, 'dropout': 0.2},
                {'type': 'lstm', 'units': 32, 'return_sequences': False}
            ],
            'latent_config': {'dim': 16, 'activation': 'linear'},
            'compilation': {'optimizer': 'rmsprop', 'loss': 'mae', 'metrics': ['mse']}
        },
        
        'gru_based': {
            'name': 'GRU-based Autoencoder',
            'description': 'GRU-based autoencoder with dense layers',
            'input_shape': [40, 4],
            'encoder_layers': [
                {'type': 'gru', 'units': 64, 'return_sequences': True, 'recurrent_dropout': 0.1},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'gru', 'units': 32, 'return_sequences': False, 'recurrent_dropout': 0.1},
                {'type': 'dense', 'units': 24, 'activation': 'relu'}
            ],
            'latent_config': {'dim': 12, 'activation': 'tanh'},
            'compilation': {'optimizer': 'adam', 'loss': 'huber', 'metrics': ['mse', 'mae']}
        },
        
        'lightweight': {
            'name': 'Lightweight Autoencoder',
            'description': 'Minimal architecture for resource-constrained environments',
            'input_shape': [20, 3],
            'encoder_layers': [
                {'type': 'lstm', 'units': 32, 'return_sequences': False}
            ],
            'latent_config': {'dim': 8, 'activation': 'linear'},
            'compilation': {'optimizer': 'adam', 'loss': 'mse'}
        }
    }


def save_architecture_config(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save architecture configuration to JSON file.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Architecture configuration
    filepath : str or Path
        Path to save the configuration
    """
    logger = get_logger(__name__)
    
    if not validate_architecture_config(config):
        raise ValueError("Invalid architecture configuration")
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Architecture configuration saved to {filepath}")


def load_architecture_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load architecture configuration from JSON file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the configuration file
        
    Returns
    -------
    Dict[str, Any]
        Architecture configuration
    """
    logger = get_logger(__name__)
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    if not validate_architecture_config(config):
        raise ValueError(f"Invalid architecture configuration in {filepath}")
    
    logger.info(f"Architecture configuration loaded from {filepath}")
    return config