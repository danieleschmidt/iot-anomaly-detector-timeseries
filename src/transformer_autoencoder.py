"""Transformer-based Autoencoder for Advanced IoT Time Series Anomaly Detection.

This module implements a novel transformer-based autoencoder architecture with
multi-head attention mechanisms for superior temporal pattern learning in IoT
sensor data. Part of Generation 3+ enhancements.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, losses, optimizers, regularizers
    from tensorflow.keras.models import Model
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
        LayerNormalization = _MockLayer
    
    class _MockModel:
        def __init__(self, *args, **kwargs):
            pass
    
    layers = _MockLayers()
    Model = _MockModel

from .logging_config import get_logger


class MultiHeadAttention(layers.Layer):
    """Multi-head attention layer for transformer architecture."""
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model, use_bias=False)
        self.wk = layers.Dense(d_model, use_bias=False)  
        self.wv = layers.Dense(d_model, use_bias=False)
        
        self.dense = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate attention weights and output."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Add mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
            
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
        
    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]
        
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        return output


class PositionalEncoding(layers.Layer):
    """Positional encoding for transformer architecture."""
    
    def __init__(self, d_model: int, max_seq_len: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Create positional encoding matrix
        pe = np.zeros((max_seq_len, d_model))
        position = np.arange(0, max_seq_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * 
                         -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.Variable(
            pe[np.newaxis, :, :], 
            trainable=False, 
            name="positional_encoding"
        )
        
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]


class TransformerEncoderLayer(layers.Layer):
    """Single transformer encoder layer with multi-head attention."""
    
    def __init__(self, d_model: int, num_heads: int, dff: int, 
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=None, mask=None):
        attn_output = self.mha(inputs, mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerDecoderLayer(layers.Layer):
    """Single transformer decoder layer with multi-head attention."""
    
    def __init__(self, d_model: int, num_heads: int, dff: int,
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.mha1 = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.mha2 = MultiHeadAttention(d_model, num_heads, dropout_rate)
        
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, enc_output, training=None, 
             look_ahead_mask=None, padding_mask=None):
        
        attn1 = self.mha1(inputs, mask=look_ahead_mask, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + inputs)
        
        attn2 = self.mha2(out1, enc_output, mask=padding_mask, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(ffn_output + out2)


class TransformerAutoencoder(Model):
    """Transformer-based Autoencoder for time series anomaly detection."""
    
    def __init__(self, input_shape: Tuple[int, int], d_model: int = 256,
                 num_heads: int = 8, num_encoder_layers: int = 4, 
                 num_decoder_layers: int = 4, dff: int = 512,
                 latent_dim: int = 64, dropout_rate: float = 0.1, 
                 use_positional_encoding: bool = True, **kwargs):
        super().__init__(**kwargs)
        
        self.input_shape_val = input_shape
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dff = dff
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.use_positional_encoding = use_positional_encoding
        
        # Input projection
        self.input_projection = layers.Dense(d_model)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, input_shape[0])
        
        # Encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_encoder_layers)
        ]
        
        # Latent bottleneck
        self.latent_projection = layers.Dense(latent_dim, activation='tanh')
        self.latent_expansion = layers.Dense(d_model)
        
        # Decoder layers
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_decoder_layers)
        ]
        
        # Output projection
        self.output_projection = layers.Dense(input_shape[1])
        
        self.logger = get_logger(__name__)
        
    def encode(self, x, training=None, mask=None):
        """Encode input sequence to latent representation."""
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        
        # Apply encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training, mask=mask)
        
        # Global average pooling to create fixed-size representation
        encoded = tf.reduce_mean(x, axis=1)  # [batch_size, d_model]
        
        # Project to latent space
        latent = self.latent_projection(encoded)  # [batch_size, latent_dim]
        
        return latent, x  # Return both latent and encoder output
    
    def decode(self, latent, encoder_output, training=None, mask=None):
        """Decode latent representation back to sequence."""
        # Expand latent to sequence
        x = self.latent_expansion(latent)  # [batch_size, d_model]
        x = tf.expand_dims(x, axis=1)  # [batch_size, 1, d_model]
        x = tf.tile(x, [1, self.input_shape_val[0], 1])  # [batch_size, seq_len, d_model]
        
        # Apply decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(
                x, encoder_output, 
                training=training, 
                padding_mask=mask
            )
        
        # Output projection
        output = self.output_projection(x)
        
        return output
    
    def call(self, inputs, training=None, mask=None):
        """Forward pass through the transformer autoencoder."""
        latent, encoder_output = self.encode(inputs, training=training, mask=mask)
        decoded = self.decode(latent, encoder_output, training=training, mask=mask)
        return decoded
    
    def get_latent_representation(self, inputs, training=None):
        """Get latent representation for analysis."""
        latent, _ = self.encode(inputs, training=training)
        return latent


class TransformerAutoencoderBuilder:
    """Builder class for creating transformer autoencoder configurations."""
    
    def __init__(self, input_shape: Tuple[int, int]):
        """Initialize the transformer autoencoder builder."""
        self.input_shape = input_shape
        self.config = {
            'd_model': 256,
            'num_heads': 8,
            'num_encoder_layers': 4,
            'num_decoder_layers': 4,
            'dff': 512,
            'latent_dim': 64,
            'dropout_rate': 0.1,
            'use_positional_encoding': True
        }
        
        self.optimizer = 'adam'
        self.loss = 'mse'
        self.metrics = None
        
        self.logger = get_logger(__name__)
        
    def set_model_dimensions(self, d_model: int = 256, latent_dim: int = 64) -> 'TransformerAutoencoderBuilder':
        """Set model dimensions."""
        self.config['d_model'] = d_model
        self.config['latent_dim'] = latent_dim
        return self
    
    def set_attention_config(self, num_heads: int = 8, 
                           dff: int = 512) -> 'TransformerAutoencoderBuilder':
        """Set attention mechanism configuration."""
        self.config['num_heads'] = num_heads
        self.config['dff'] = dff
        return self
    
    def set_architecture_depth(self, num_encoder_layers: int = 4, 
                             num_decoder_layers: int = 4) -> 'TransformerAutoencoderBuilder':
        """Set number of encoder and decoder layers."""
        self.config['num_encoder_layers'] = num_encoder_layers
        self.config['num_decoder_layers'] = num_decoder_layers
        return self
    
    def set_regularization(self, dropout_rate: float = 0.1, 
                         use_positional_encoding: bool = True) -> 'TransformerAutoencoderBuilder':
        """Set regularization options."""
        self.config['dropout_rate'] = dropout_rate
        self.config['use_positional_encoding'] = use_positional_encoding
        return self
    
    def set_compilation(self, optimizer: str = 'adam', loss: str = 'mse', 
                       metrics: Optional[List[str]] = None) -> 'TransformerAutoencoderBuilder':
        """Set compilation options."""
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        return self
    
    def build(self) -> Optional[TransformerAutoencoder]:
        """Build the transformer autoencoder model."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available, returning None")
            return None
            
        self.logger.info(f"Building Transformer Autoencoder with config: {self.config}")
        
        model = TransformerAutoencoder(
            input_shape=self.input_shape,
            **self.config
        )
        
        # Compile the model
        optimizer_fn = self._get_optimizer(self.optimizer)
        loss_fn = self._get_loss_function(self.loss)
        
        model.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=self.metrics)
        
        # Build the model by calling it once
        dummy_input = tf.random.normal((1,) + self.input_shape)
        _ = model(dummy_input)
        
        self.logger.info(f"Built Transformer Autoencoder with {model.count_params()} parameters")
        return model
    
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
            'adam': optimizers.Adam(learning_rate=1e-4),
            'rmsprop': optimizers.RMSprop(learning_rate=1e-4),
            'sgd': optimizers.SGD(learning_rate=1e-3),
            'adagrad': optimizers.Adagrad(learning_rate=1e-3)
        }
        return optimizer_map.get(optimizer_name, optimizer_name)


def get_transformer_presets() -> Dict[str, Dict[str, Any]]:
    """Get predefined transformer autoencoder configurations."""
    return {
        'lightweight_transformer': {
            'name': 'Lightweight Transformer Autoencoder',
            'description': 'Minimal transformer for resource-constrained environments',
            'd_model': 128,
            'num_heads': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dff': 256,
            'latent_dim': 32,
            'dropout_rate': 0.1
        },
        
        'standard_transformer': {
            'name': 'Standard Transformer Autoencoder',
            'description': 'Balanced transformer for general-purpose anomaly detection',
            'd_model': 256,
            'num_heads': 8,
            'num_encoder_layers': 4,
            'num_decoder_layers': 4,
            'dff': 512,
            'latent_dim': 64,
            'dropout_rate': 0.1
        },
        
        'deep_transformer': {
            'name': 'Deep Transformer Autoencoder',  
            'description': 'Deep transformer for complex temporal patterns',
            'd_model': 512,
            'num_heads': 16,
            'num_encoder_layers': 8,
            'num_decoder_layers': 8,
            'dff': 1024,
            'latent_dim': 128,
            'dropout_rate': 0.1
        },
        
        'high_resolution_transformer': {
            'name': 'High Resolution Transformer',
            'description': 'Large transformer for high-dimensional sensor data',
            'd_model': 768,
            'num_heads': 12,
            'num_encoder_layers': 12,
            'num_decoder_layers': 12,
            'dff': 2048,
            'latent_dim': 256,
            'dropout_rate': 0.1
        }
    }


def create_transformer_autoencoder(input_shape: Tuple[int, int], 
                                 preset: str = 'standard_transformer',
                                 **overrides) -> Optional[TransformerAutoencoder]:
    """Create a transformer autoencoder with preset configuration."""
    logger = get_logger(__name__)
    
    presets = get_transformer_presets()
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    config = presets[preset].copy()
    config.update(overrides)
    
    # Remove non-config items
    config.pop('name', None)
    config.pop('description', None)
    
    builder = TransformerAutoencoderBuilder(input_shape)
    
    builder.set_model_dimensions(config['d_model'], config['latent_dim'])
    builder.set_attention_config(config['num_heads'], config['dff'])
    builder.set_architecture_depth(config['num_encoder_layers'], config['num_decoder_layers'])
    builder.set_regularization(config['dropout_rate'])
    
    logger.info(f"Creating transformer autoencoder with preset: {preset}")
    return builder.build()