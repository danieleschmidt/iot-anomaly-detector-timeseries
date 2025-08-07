"""Variational Autoencoder for Probabilistic Anomaly Detection in IoT Time Series.

This module implements a Variational Autoencoder (VAE) with probabilistic latent
representations for enhanced anomaly detection with uncertainty quantification.
Part of Generation 3+ research enhancements.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import json

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, losses, optimizers, regularizers
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend as K
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Create mock objects for when TensorFlow is not available
    class _MockLayer:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return None
    
    class _MockLayers:
        Layer = _MockLayer
        Dense = _MockLayer
        Dropout = _MockLayer
        BatchNormalization = _MockLayer
        GlobalAveragePooling1D = _MockLayer
        Lambda = _MockLayer
        Flatten = _MockLayer
        LSTM = _MockLayer
        GRU = _MockLayer
        Conv1D = _MockLayer
        RepeatVector = _MockLayer
        TimeDistributed = _MockLayer
        UpSampling1D = _MockLayer
        Reshape = _MockLayer
    
    class _MockModel:
        def __init__(self, *args, **kwargs):
            pass
    
    layers = _MockLayers()
    Model = _MockModel

from .logging_config import get_logger


class Sampling(layers.Layer):
    """Sampling layer using reparameterization trick."""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalEncoder(layers.Layer):
    """Variational encoder with probabilistic latent representation."""
    
    def __init__(self, latent_dim: int, architecture_type: str = 'lstm',
                 hidden_units: List[int] = None, dropout_rate: float = 0.1,
                 use_batch_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.architecture_type = architecture_type
        self.hidden_units = hidden_units or [128, 64]
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        self.encoder_layers = []
        self._build_encoder()
        
        # Latent distribution parameters
        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")
        self.sampling = Sampling()
        
    def _build_encoder(self):
        """Build encoder layers based on architecture type."""
        if self.architecture_type == 'lstm':
            for i, units in enumerate(self.hidden_units):
                return_sequences = i < len(self.hidden_units) - 1
                self.encoder_layers.append(
                    layers.LSTM(
                        units, 
                        return_sequences=return_sequences,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.dropout_rate
                    )
                )
                if self.use_batch_norm:
                    self.encoder_layers.append(layers.BatchNormalization())
                    
        elif self.architecture_type == 'gru':
            for i, units in enumerate(self.hidden_units):
                return_sequences = i < len(self.hidden_units) - 1
                self.encoder_layers.append(
                    layers.GRU(
                        units,
                        return_sequences=return_sequences,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.dropout_rate
                    )
                )
                if self.use_batch_norm:
                    self.encoder_layers.append(layers.BatchNormalization())
                    
        elif self.architecture_type == 'conv1d':
            for units in self.hidden_units:
                self.encoder_layers.extend([
                    layers.Conv1D(units, 3, activation='relu', padding='same'),
                    layers.BatchNormalization() if self.use_batch_norm else layers.Lambda(lambda x: x),
                    layers.Dropout(self.dropout_rate)
                ])
            self.encoder_layers.append(layers.GlobalAveragePooling1D())
            
        elif self.architecture_type == 'dense':
            self.encoder_layers.append(layers.Flatten())
            for units in self.hidden_units:
                self.encoder_layers.extend([
                    layers.Dense(units, activation='relu'),
                    layers.BatchNormalization() if self.use_batch_norm else layers.Lambda(lambda x: x),
                    layers.Dropout(self.dropout_rate)
                ])
    
    def call(self, inputs, training=None):
        x = inputs
        for layer in self.encoder_layers:
            if isinstance(layer, layers.Lambda) and layer.function(x) is x:
                continue  # Skip identity lambda layers
            x = layer(x, training=training) if hasattr(layer, 'training') else layer(x)
        
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling([z_mean, z_log_var])
        
        return z_mean, z_log_var, z


class VariationalDecoder(layers.Layer):
    """Variational decoder for reconstructing sequences from latent space."""
    
    def __init__(self, original_shape: Tuple[int, int], architecture_type: str = 'lstm',
                 hidden_units: List[int] = None, dropout_rate: float = 0.1,
                 use_batch_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        
        self.original_shape = original_shape
        self.sequence_length, self.n_features = original_shape
        self.architecture_type = architecture_type
        self.hidden_units = hidden_units or [64, 128]
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        self.decoder_layers = []
        self._build_decoder()
        
        # Output layer
        if architecture_type in ['lstm', 'gru']:
            self.output_layer = layers.TimeDistributed(
                layers.Dense(self.n_features, activation='linear')
            )
        else:
            self.output_layer = layers.Dense(
                self.sequence_length * self.n_features, 
                activation='linear'
            )
    
    def _build_decoder(self):
        """Build decoder layers based on architecture type."""
        if self.architecture_type == 'lstm':
            self.decoder_layers.append(layers.RepeatVector(self.sequence_length))
            for units in self.hidden_units:
                self.decoder_layers.append(
                    layers.LSTM(
                        units,
                        return_sequences=True,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.dropout_rate
                    )
                )
                if self.use_batch_norm:
                    self.decoder_layers.append(layers.BatchNormalization())
                    
        elif self.architecture_type == 'gru':
            self.decoder_layers.append(layers.RepeatVector(self.sequence_length))
            for units in self.hidden_units:
                self.decoder_layers.append(
                    layers.GRU(
                        units,
                        return_sequences=True,
                        dropout=self.dropout_rate,
                        recurrent_dropout=self.dropout_rate
                    )
                )
                if self.use_batch_norm:
                    self.decoder_layers.append(layers.BatchNormalization())
                    
        elif self.architecture_type == 'conv1d':
            # Upsample and apply transposed convolutions
            self.decoder_layers.extend([
                layers.Dense(self.sequence_length // 4 * self.hidden_units[0]),
                layers.Reshape((self.sequence_length // 4, self.hidden_units[0])),
                layers.UpSampling1D(2),
                layers.Conv1D(self.hidden_units[1], 3, activation='relu', padding='same'),
                layers.BatchNormalization() if self.use_batch_norm else layers.Lambda(lambda x: x),
                layers.UpSampling1D(2),
                layers.Conv1D(self.n_features, 3, activation='linear', padding='same')
            ])
            
        elif self.architecture_type == 'dense':
            for units in self.hidden_units:
                self.decoder_layers.extend([
                    layers.Dense(units, activation='relu'),
                    layers.BatchNormalization() if self.use_batch_norm else layers.Lambda(lambda x: x),
                    layers.Dropout(self.dropout_rate)
                ])
    
    def call(self, inputs, training=None):
        x = inputs
        for layer in self.decoder_layers:
            if isinstance(layer, layers.Lambda) and layer.function(x) is x:
                continue  # Skip identity lambda layers
            x = layer(x, training=training) if hasattr(layer, 'training') else layer(x)
        
        output = self.output_layer(x)
        
        # Reshape output if necessary
        if self.architecture_type == 'dense':
            output = tf.reshape(output, (-1, self.sequence_length, self.n_features))
        elif self.architecture_type == 'conv1d':
            # Ensure correct output shape
            current_shape = tf.shape(output)
            if current_shape[1] != self.sequence_length:
                # Crop or pad to match expected sequence length
                if current_shape[1] > self.sequence_length:
                    output = output[:, :self.sequence_length, :]
                else:
                    padding = self.sequence_length - current_shape[1]
                    output = tf.pad(output, [[0, 0], [0, padding], [0, 0]])
        
        return output


class VariationalAutoencoder(Model):
    """Variational Autoencoder for probabilistic anomaly detection."""
    
    def __init__(self, input_shape: Tuple[int, int], latent_dim: int = 32,
                 architecture_type: str = 'lstm', hidden_units: List[int] = None,
                 dropout_rate: float = 0.1, use_batch_norm: bool = True,
                 beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        
        self.input_shape_val = input_shape
        self.latent_dim = latent_dim
        self.architecture_type = architecture_type
        self.hidden_units = hidden_units or [128, 64]
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.beta = beta  # Beta-VAE parameter for controlling regularization
        
        # Build encoder and decoder
        self.encoder = VariationalEncoder(
            latent_dim=latent_dim,
            architecture_type=architecture_type,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        decoder_hidden = list(reversed(hidden_units))
        self.decoder = VariationalDecoder(
            original_shape=input_shape,
            architecture_type=architecture_type,
            hidden_units=decoder_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        
        self.logger = get_logger(__name__)
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def call(self, inputs, training=None):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        
        # Add KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        self.add_loss(self.beta * kl_loss)
        
        return reconstruction
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mse(data, reconstruction), axis=(1, 2)
                )
            )
            
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            total_loss = reconstruction_loss + self.beta * kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)
        
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.mse(data, reconstruction), axis=(1, 2)
            )
        )
        
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        
        total_loss = reconstruction_loss + self.beta * kl_loss
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def encode(self, x, training=None):
        """Encode input to latent space distribution parameters."""
        z_mean, z_log_var, z = self.encoder(x, training=training)
        return z_mean, z_log_var, z
    
    def decode(self, z, training=None):
        """Decode from latent space to reconstruction."""
        return self.decoder(z, training=training)
    
    def sample(self, eps=None):
        """Generate samples from the learned distribution."""
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, training=False)
    
    def get_reconstruction_error(self, x, training=None):
        """Get reconstruction error with uncertainty estimation."""
        z_mean, z_log_var, z = self.encode(x, training=training)
        reconstruction = self.decode(z, training=training)
        
        # Calculate reconstruction error
        mse_error = tf.reduce_mean(tf.square(x - reconstruction), axis=(1, 2))
        mae_error = tf.reduce_mean(tf.abs(x - reconstruction), axis=(1, 2))
        
        # Uncertainty estimation from variance
        uncertainty = tf.reduce_mean(tf.exp(0.5 * z_log_var), axis=1)
        
        return {
            'mse_error': mse_error,
            'mae_error': mae_error,
            'uncertainty': uncertainty,
            'z_mean': z_mean,
            'z_log_var': z_log_var
        }


class VAEAutoencoderBuilder:
    """Builder class for creating variational autoencoder configurations."""
    
    def __init__(self, input_shape: Tuple[int, int]):
        """Initialize the VAE autoencoder builder."""
        self.input_shape = input_shape
        self.config = {
            'latent_dim': 32,
            'architecture_type': 'lstm',
            'hidden_units': [128, 64],
            'dropout_rate': 0.1,
            'use_batch_norm': True,
            'beta': 1.0
        }
        
        self.optimizer = 'adam'
        self.learning_rate = 1e-3
        
        self.logger = get_logger(__name__)
        
    def set_latent_config(self, latent_dim: int = 32, beta: float = 1.0) -> 'VAEAutoencoderBuilder':
        """Set latent space configuration."""
        self.config['latent_dim'] = latent_dim
        self.config['beta'] = beta
        return self
    
    def set_architecture(self, architecture_type: str = 'lstm', 
                        hidden_units: List[int] = None) -> 'VAEAutoencoderBuilder':
        """Set architecture type and hidden units."""
        self.config['architecture_type'] = architecture_type
        if hidden_units is not None:
            self.config['hidden_units'] = hidden_units
        return self
    
    def set_regularization(self, dropout_rate: float = 0.1, 
                         use_batch_norm: bool = True) -> 'VAEAutoencoderBuilder':
        """Set regularization options."""
        self.config['dropout_rate'] = dropout_rate
        self.config['use_batch_norm'] = use_batch_norm
        return self
    
    def set_optimizer(self, optimizer: str = 'adam', 
                     learning_rate: float = 1e-3) -> 'VAEAutoencoderBuilder':
        """Set optimizer configuration."""
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        return self
    
    def build(self) -> Optional[VariationalAutoencoder]:
        """Build the variational autoencoder model."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available, returning None")
            return None
            
        self.logger.info(f"Building Variational Autoencoder with config: {self.config}")
        
        model = VariationalAutoencoder(
            input_shape=self.input_shape,
            **self.config
        )
        
        # Configure optimizer
        if self.optimizer == 'adam':
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            optimizer = optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(optimizer=optimizer)
        
        # Build the model by calling it once
        dummy_input = tf.random.normal((1,) + self.input_shape)
        _ = model(dummy_input)
        
        self.logger.info(f"Built Variational Autoencoder with {model.count_params()} parameters")
        return model


def get_vae_presets() -> Dict[str, Dict[str, Any]]:
    """Get predefined VAE configurations."""
    return {
        'lightweight_vae': {
            'name': 'Lightweight VAE',
            'description': 'Minimal VAE for resource-constrained environments',
            'latent_dim': 16,
            'architecture_type': 'lstm',
            'hidden_units': [64, 32],
            'dropout_rate': 0.1,
            'beta': 1.0
        },
        
        'standard_vae': {
            'name': 'Standard VAE',
            'description': 'Balanced VAE for general-purpose anomaly detection',
            'latent_dim': 32,
            'architecture_type': 'lstm',
            'hidden_units': [128, 64],
            'dropout_rate': 0.1,
            'beta': 1.0
        },
        
        'deep_vae': {
            'name': 'Deep VAE',
            'description': 'Deep VAE for complex temporal patterns',
            'latent_dim': 64,
            'architecture_type': 'lstm',
            'hidden_units': [256, 128, 64],
            'dropout_rate': 0.1,
            'beta': 1.0
        },
        
        'convolutional_vae': {
            'name': 'Convolutional VAE',
            'description': 'CNN-based VAE for local pattern detection',
            'latent_dim': 32,
            'architecture_type': 'conv1d',
            'hidden_units': [64, 128, 64],
            'dropout_rate': 0.1,
            'beta': 1.0
        },
        
        'beta_vae': {
            'name': 'Beta-VAE',
            'description': 'VAE with enhanced disentanglement (beta > 1)',
            'latent_dim': 32,
            'architecture_type': 'lstm',
            'hidden_units': [128, 64],
            'dropout_rate': 0.1,
            'beta': 4.0
        }
    }


def create_variational_autoencoder(input_shape: Tuple[int, int],
                                 preset: str = 'standard_vae',
                                 **overrides) -> Optional[VariationalAutoencoder]:
    """Create a variational autoencoder with preset configuration."""
    logger = get_logger(__name__)
    
    presets = get_vae_presets()
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    config = presets[preset].copy()
    config.update(overrides)
    
    # Remove non-config items
    config.pop('name', None)
    config.pop('description', None)
    
    builder = VAEAutoencoderBuilder(input_shape)
    
    builder.set_latent_config(config['latent_dim'], config['beta'])
    builder.set_architecture(config['architecture_type'], config['hidden_units'])
    builder.set_regularization(config['dropout_rate'])
    
    logger.info(f"Creating variational autoencoder with preset: {preset}")
    return builder.build()