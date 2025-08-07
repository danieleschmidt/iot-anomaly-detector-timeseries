"""Quantum-Classical Hybrid Autoencoder for Enhanced IoT Anomaly Detection.

This module implements a novel hybrid architecture that combines quantum-inspired
optimization with classical neural networks for superior anomaly detection
in IoT time series data. Part of Generation 3+ research enhancements.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
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
        LSTM = _MockLayer
        RepeatVector = _MockLayer
        MultiHeadAttention = _MockLayer
        Add = _MockLayer
    
    class _MockModel:
        def __init__(self, *args, **kwargs):
            pass
    
    layers = _MockLayers()
    Model = _MockModel

from .quantum_inspired.quantum_utils import (
    QuantumRegister, 
    quantum_superposition, 
    quantum_entanglement,
    quantum_amplitude_amplification
)
from .quantum_inspired.quantum_optimization_base import QuantumOptimizationAlgorithm
from .logging_config import get_logger


@dataclass
class QuantumFeatureMap:
    """Configuration for quantum feature encoding."""
    encoding_type: str = "amplitude"  # amplitude, angle, basis
    num_qubits: int = 8
    entanglement_pattern: str = "linear"  # linear, circular, full
    use_variational_form: bool = True


class QuantumFeatureEncoder(layers.Layer):
    """Quantum-inspired feature encoding layer."""
    
    def __init__(self, quantum_config: QuantumFeatureMap, **kwargs):
        super().__init__(**kwargs)
        self.quantum_config = quantum_config
        self.num_qubits = quantum_config.num_qubits
        
        # Classical preprocessing for quantum encoding
        self.feature_preprocessor = layers.Dense(
            self.num_qubits, 
            activation='tanh',
            name='quantum_feature_prep'
        )
        
        # Variational quantum circuit parameters (simulated classically)
        if quantum_config.use_variational_form:
            self.variational_params = self.add_weight(
                name='variational_params',
                shape=(self.num_qubits, 3),  # RX, RY, RZ angles for each qubit
                initializer='random_uniform',
                trainable=True
            )
        
        self.logger = get_logger(__name__)
        
    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """
        Encode classical features into quantum-inspired representation.
        
        Args:
            inputs: Classical feature tensor [batch_size, sequence_length, features]
            
        Returns:
            Quantum-encoded features with enhanced correlation information
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        # Preprocess features for quantum encoding
        preprocessed = self.feature_preprocessor(inputs)  # [batch, seq, num_qubits]
        
        # Simulate quantum encoding process
        if self.quantum_config.encoding_type == "amplitude":
            # Amplitude encoding: normalize features to represent quantum amplitudes
            normalized = tf.nn.l2_normalize(preprocessed, axis=-1)
            # Add quantum interference patterns
            interference = self._simulate_quantum_interference(normalized)
            encoded = normalized + 0.1 * interference
            
        elif self.quantum_config.encoding_type == "angle":
            # Angle encoding: map features to rotation angles
            angles = preprocessed * np.pi  # Scale to [0, π]
            # Apply simulated quantum rotations
            encoded = tf.stack([
                tf.cos(angles),
                tf.sin(angles)
            ], axis=-1)  # [batch, seq, num_qubits, 2]
            encoded = tf.reshape(encoded, [batch_size, seq_length, -1])
            
        else:  # basis encoding
            # Basis encoding: discrete representation
            discrete = tf.round(tf.sigmoid(preprocessed))
            encoded = discrete
        
        # Add quantum entanglement simulation
        if self.quantum_config.entanglement_pattern != "none":
            encoded = self._simulate_entanglement(encoded)
        
        # Apply variational quantum circuit if enabled
        if self.quantum_config.use_variational_form:
            encoded = self._apply_variational_circuit(encoded)
        
        return encoded
    
    def _simulate_quantum_interference(self, features):
        """Simulate quantum interference patterns in feature space."""
        # Create interference patterns using trigonometric functions
        phase_shifts = tf.range(self.num_qubits, dtype=tf.float32) * 2 * np.pi / self.num_qubits
        
        interference = tf.sin(features + phase_shifts) * tf.cos(features * 2 + phase_shifts)
        return interference * 0.1  # Small interference effect
    
    def _simulate_entanglement(self, features):
        """Simulate quantum entanglement correlations."""
        if self.quantum_config.entanglement_pattern == "linear":
            # Linear entanglement: each qubit coupled to next
            entangled = tf.roll(features, shift=1, axis=-1) * features
            
        elif self.quantum_config.entanglement_pattern == "circular":
            # Circular entanglement: last qubit coupled to first
            rolled = tf.roll(features, shift=1, axis=-1)
            entangled = rolled * features
            
        elif self.quantum_config.entanglement_pattern == "full":
            # All-to-all entanglement (simplified)
            mean_feature = tf.reduce_mean(features, axis=-1, keepdims=True)
            entangled = features * mean_feature
            
        else:
            entangled = features
        
        return features + 0.05 * entangled  # Add small entanglement effect
    
    def _apply_variational_circuit(self, features):
        """Apply variational quantum circuit operations."""
        # Simulate parameterized quantum gates
        rx_rotations = tf.cos(self.variational_params[:, 0]) * features
        ry_rotations = tf.sin(self.variational_params[:, 1]) * features  
        rz_rotations = tf.exp(1j * self.variational_params[:, 2]) * tf.cast(features, tf.complex64)
        
        # Combine rotations (taking real part for classical compatibility)
        variational_output = rx_rotations + ry_rotations + tf.math.real(rz_rotations)
        
        return variational_output


class QuantumMeasurementLayer(layers.Layer):
    """Quantum measurement simulation layer."""
    
    def __init__(self, measurement_type: str = "pauli_z", **kwargs):
        super().__init__(**kwargs)
        self.measurement_type = measurement_type
        
    def call(self, inputs, training=None):
        """
        Simulate quantum measurements on encoded features.
        
        Args:
            inputs: Quantum-encoded features
            
        Returns:
            Classical measurement outcomes
        """
        if self.measurement_type == "pauli_z":
            # Z-measurement (computational basis)
            measurements = tf.sign(inputs) * tf.square(inputs)
            
        elif self.measurement_type == "pauli_x":
            # X-measurement (Hadamard basis)
            hadamard_transform = tf.nn.l2_normalize(inputs + tf.roll(inputs, shift=1, axis=-1), axis=-1)
            measurements = tf.sign(hadamard_transform) * tf.square(hadamard_transform)
            
        elif self.measurement_type == "pauli_y":
            # Y-measurement
            y_transform = tf.nn.l2_normalize(inputs - tf.roll(inputs, shift=1, axis=-1), axis=-1)
            measurements = tf.sign(y_transform) * tf.square(y_transform)
            
        else:  # expectation value
            # Compute expectation values
            measurements = tf.square(inputs)
            
        return measurements


class QuantumInspiredAttention(layers.Layer):
    """Quantum-inspired attention mechanism with entanglement simulation."""
    
    def __init__(self, d_model: int, num_heads: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        # Quantum-inspired parameters
        self.quantum_register = None
        self.entanglement_strength = self.add_weight(
            name='entanglement_strength',
            shape=(num_heads,),
            initializer='ones',
            trainable=True
        )
        
        # Standard attention components
        self.wq = layers.Dense(d_model, use_bias=False)
        self.wk = layers.Dense(d_model, use_bias=False)
        self.wv = layers.Dense(d_model, use_bias=False)
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def quantum_enhanced_attention(self, q, k, v):
        """Apply quantum-inspired enhancements to attention mechanism."""
        # Standard scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Quantum enhancement: simulate entanglement between attention heads
        batch_size = tf.shape(scaled_attention_logits)[0]
        seq_len = tf.shape(scaled_attention_logits)[-1]
        
        # Create quantum entanglement patterns between heads
        entanglement_matrix = tf.einsum('h,hij->hij', 
                                      self.entanglement_strength,
                                      tf.ones((self.num_heads, seq_len, seq_len)))
        
        # Apply quantum interference to attention weights
        quantum_interference = tf.sin(scaled_attention_logits) * tf.cos(entanglement_matrix)
        enhanced_logits = scaled_attention_logits + 0.1 * quantum_interference
        
        attention_weights = tf.nn.softmax(enhanced_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.quantum_enhanced_attention(q, k, v)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        return output


class QuantumHybridAutoencoder(Model):
    """Quantum-Classical Hybrid Autoencoder for anomaly detection."""
    
    def __init__(self, input_shape: Tuple[int, int], 
                 quantum_config: QuantumFeatureMap = None,
                 classical_latent_dim: int = 64,
                 quantum_latent_dim: int = 32,
                 use_quantum_attention: bool = True,
                 hybrid_fusion_method: str = "concatenate",
                 **kwargs):
        super().__init__(**kwargs)
        
        self.input_shape_val = input_shape
        self.quantum_config = quantum_config or QuantumFeatureMap()
        self.classical_latent_dim = classical_latent_dim
        self.quantum_latent_dim = quantum_latent_dim
        self.use_quantum_attention = use_quantum_attention
        self.hybrid_fusion_method = hybrid_fusion_method
        
        # Classical processing branch
        self.classical_encoder = self._build_classical_encoder()
        self.classical_decoder = self._build_classical_decoder()
        
        # Quantum processing branch
        self.quantum_encoder = QuantumFeatureEncoder(self.quantum_config)
        self.quantum_measurement = QuantumMeasurementLayer()
        self.quantum_projection = layers.Dense(quantum_latent_dim, activation='tanh')
        
        # Quantum-inspired attention (optional)
        if use_quantum_attention:
            self.quantum_attention = QuantumInspiredAttention(
                d_model=classical_latent_dim + quantum_latent_dim
            )
        
        # Hybrid fusion layer
        self.fusion_layer = self._build_fusion_layer()
        
        # Output reconstruction
        self.output_projection = layers.Dense(input_shape[1])
        
        self.logger = get_logger(__name__)
        
    def _build_classical_encoder(self):
        """Build classical encoder branch."""
        return tf.keras.Sequential([
            layers.LSTM(128, return_sequences=True, dropout=0.1),
            layers.LSTM(64, return_sequences=False, dropout=0.1),
            layers.Dense(self.classical_latent_dim, activation='tanh')
        ])
    
    def _build_classical_decoder(self):
        """Build classical decoder branch."""
        return tf.keras.Sequential([
            layers.RepeatVector(self.input_shape_val[0]),
            layers.LSTM(64, return_sequences=True, dropout=0.1),
            layers.LSTM(128, return_sequences=True, dropout=0.1)
        ])
    
    def _build_fusion_layer(self):
        """Build hybrid fusion layer."""
        if self.hybrid_fusion_method == "concatenate":
            return layers.Dense(self.classical_latent_dim + self.quantum_latent_dim)
        elif self.hybrid_fusion_method == "attention":
            return layers.MultiHeadAttention(num_heads=4, key_dim=32)
        elif self.hybrid_fusion_method == "gated":
            return tf.keras.Sequential([
                layers.Dense(128, activation='relu'),
                layers.Dense(self.classical_latent_dim + self.quantum_latent_dim, activation='sigmoid')
            ])
        else:
            return layers.Add()
    
    def encode(self, inputs, training=None):
        """Hybrid encoding using both classical and quantum branches."""
        # Classical encoding
        classical_features = self.classical_encoder(inputs, training=training)
        
        # Quantum encoding
        quantum_features = self.quantum_encoder(inputs, training=training)
        quantum_measured = self.quantum_measurement(quantum_features, training=training)
        quantum_latent = self.quantum_projection(quantum_measured, training=training)
        
        # Hybrid fusion
        if self.hybrid_fusion_method == "concatenate":
            fused_features = tf.concat([classical_features, quantum_latent], axis=-1)
            fused_latent = self.fusion_layer(fused_features, training=training)
        elif self.hybrid_fusion_method == "attention":
            # Use classical as query, quantum as key-value
            fused_latent = self.fusion_layer(
                classical_features, quantum_latent, training=training
            )
        elif self.hybrid_fusion_method == "gated":
            gate = self.fusion_layer(
                tf.concat([classical_features, quantum_latent], axis=-1),
                training=training
            )
            fused_latent = gate * classical_features + (1 - gate) * quantum_latent
        else:  # additive
            # Ensure dimensions match
            if classical_features.shape[-1] != quantum_latent.shape[-1]:
                quantum_latent = layers.Dense(classical_features.shape[-1])(quantum_latent)
            fused_latent = self.fusion_layer([classical_features, quantum_latent])
        
        # Apply quantum-inspired attention if enabled
        if self.use_quantum_attention:
            # Expand dims for attention
            fused_expanded = tf.expand_dims(fused_latent, axis=1)
            fused_latent = self.quantum_attention(fused_expanded, training=training)
            fused_latent = tf.squeeze(fused_latent, axis=1)
        
        return fused_latent, classical_features, quantum_latent
    
    def decode(self, latent_representation, training=None):
        """Decode fused latent representation to original space."""
        # Classical decoding
        decoded_sequence = self.classical_decoder(latent_representation, training=training)
        
        # Final output projection
        output = self.output_projection(decoded_sequence)
        
        return output
    
    def call(self, inputs, training=None):
        """Forward pass through hybrid autoencoder."""
        fused_latent, classical_latent, quantum_latent = self.encode(inputs, training=training)
        reconstruction = self.decode(fused_latent, training=training)
        
        # Store components for analysis
        self._last_classical_latent = classical_latent
        self._last_quantum_latent = quantum_latent
        self._last_fused_latent = fused_latent
        
        return reconstruction
    
    def get_latent_representations(self, inputs, training=None):
        """Get all latent representations for analysis."""
        fused_latent, classical_latent, quantum_latent = self.encode(inputs, training=training)
        
        return {
            'classical': classical_latent,
            'quantum': quantum_latent,
            'fused': fused_latent
        }
    
    def compute_quantum_advantage_metric(self, inputs, training=None):
        """Compute metric quantifying quantum contribution to anomaly detection."""
        latents = self.get_latent_representations(inputs, training=training)
        
        # Compare variance captured by quantum vs classical components
        classical_variance = tf.reduce_mean(tf.math.reduce_variance(latents['classical'], axis=0))
        quantum_variance = tf.reduce_mean(tf.math.reduce_variance(latents['quantum'], axis=0))
        
        # Quantum advantage metric
        quantum_advantage = quantum_variance / (classical_variance + quantum_variance + 1e-8)
        
        return {
            'quantum_advantage': quantum_advantage,
            'classical_variance': classical_variance,
            'quantum_variance': quantum_variance
        }


class QuantumHybridAutoencoderBuilder:
    """Builder for creating quantum-classical hybrid autoencoders."""
    
    def __init__(self, input_shape: Tuple[int, int]):
        """Initialize builder with input shape."""
        self.input_shape = input_shape
        self.quantum_config = QuantumFeatureMap()
        self.classical_latent_dim = 64
        self.quantum_latent_dim = 32
        self.use_quantum_attention = True
        self.hybrid_fusion_method = "concatenate"
        self.optimizer = "adam"
        self.learning_rate = 1e-4
        self.loss = "mse"
        
        self.logger = get_logger(__name__)
    
    def set_quantum_config(self, encoding_type: str = "amplitude", 
                          num_qubits: int = 8,
                          entanglement_pattern: str = "linear",
                          use_variational_form: bool = True) -> 'QuantumHybridAutoencoderBuilder':
        """Configure quantum feature encoding."""
        self.quantum_config = QuantumFeatureMap(
            encoding_type=encoding_type,
            num_qubits=num_qubits,
            entanglement_pattern=entanglement_pattern,
            use_variational_form=use_variational_form
        )
        return self
    
    def set_latent_dimensions(self, classical_dim: int = 64, 
                            quantum_dim: int = 32) -> 'QuantumHybridAutoencoderBuilder':
        """Set latent space dimensions."""
        self.classical_latent_dim = classical_dim
        self.quantum_latent_dim = quantum_dim
        return self
    
    def set_hybrid_options(self, use_quantum_attention: bool = True,
                          fusion_method: str = "concatenate") -> 'QuantumHybridAutoencoderBuilder':
        """Configure hybrid processing options."""
        self.use_quantum_attention = use_quantum_attention
        self.hybrid_fusion_method = fusion_method
        return self
    
    def set_training_config(self, optimizer: str = "adam", 
                           learning_rate: float = 1e-4,
                           loss: str = "mse") -> 'QuantumHybridAutoencoderBuilder':
        """Configure training parameters."""
        self.optimizer = optimizer
        self.learning_rate = learning_rate  
        self.loss = loss
        return self
    
    def build(self) -> Optional[QuantumHybridAutoencoder]:
        """Build the quantum-classical hybrid autoencoder."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available, returning None")
            return None
            
        self.logger.info("Building Quantum-Classical Hybrid Autoencoder")
        self.logger.info(f"Quantum config: {self.quantum_config}")
        self.logger.info(f"Classical latent dim: {self.classical_latent_dim}")
        self.logger.info(f"Quantum latent dim: {self.quantum_latent_dim}")
        
        model = QuantumHybridAutoencoder(
            input_shape=self.input_shape,
            quantum_config=self.quantum_config,
            classical_latent_dim=self.classical_latent_dim,
            quantum_latent_dim=self.quantum_latent_dim,
            use_quantum_attention=self.use_quantum_attention,
            hybrid_fusion_method=self.hybrid_fusion_method
        )
        
        # Configure optimizer
        if self.optimizer == "adam":
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        # Configure loss
        if self.loss == "mse":
            loss_fn = losses.MeanSquaredError()
        elif self.loss == "mae":
            loss_fn = losses.MeanAbsoluteError()
        else:
            loss_fn = losses.MeanSquaredError()
        
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])
        
        # Build model by calling it once
        dummy_input = tf.random.normal((1,) + self.input_shape)
        _ = model(dummy_input)
        
        self.logger.info(f"Built Quantum-Classical Hybrid Autoencoder with {model.count_params()} parameters")
        return model


def get_quantum_hybrid_presets() -> Dict[str, Dict[str, Any]]:
    """Get predefined quantum-hybrid configurations."""
    return {
        'lightweight_quantum': {
            'name': 'Lightweight Quantum Hybrid',
            'description': 'Minimal quantum enhancement for edge deployment',
            'quantum_config': {
                'encoding_type': 'amplitude',
                'num_qubits': 4,
                'entanglement_pattern': 'linear',
                'use_variational_form': False
            },
            'classical_latent_dim': 32,
            'quantum_latent_dim': 16,
            'use_quantum_attention': False,
            'fusion_method': 'concatenate'
        },
        
        'standard_quantum': {
            'name': 'Standard Quantum Hybrid',
            'description': 'Balanced quantum-classical hybrid',
            'quantum_config': {
                'encoding_type': 'amplitude',
                'num_qubits': 8,
                'entanglement_pattern': 'linear',
                'use_variational_form': True
            },
            'classical_latent_dim': 64,
            'quantum_latent_dim': 32,
            'use_quantum_attention': True,
            'fusion_method': 'concatenate'
        },
        
        'advanced_quantum': {
            'name': 'Advanced Quantum Hybrid',
            'description': 'Full quantum enhancement with attention',
            'quantum_config': {
                'encoding_type': 'angle',
                'num_qubits': 16,
                'entanglement_pattern': 'circular',
                'use_variational_form': True
            },
            'classical_latent_dim': 128,
            'quantum_latent_dim': 64,
            'use_quantum_attention': True,
            'fusion_method': 'attention'
        }
    }


def create_quantum_hybrid_autoencoder(input_shape: Tuple[int, int],
                                    preset: str = 'standard_quantum',
                                    **overrides) -> Optional[QuantumHybridAutoencoder]:
    """Create quantum-classical hybrid autoencoder with preset configuration."""
    logger = get_logger(__name__)
    
    presets = get_quantum_hybrid_presets()
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    config = presets[preset].copy()
    config.update(overrides)
    
    # Remove non-config items
    config.pop('name', None)
    config.pop('description', None)
    
    builder = QuantumHybridAutoencoderBuilder(input_shape)
    
    quantum_config = config.get('quantum_config', {})
    builder.set_quantum_config(**quantum_config)
    
    builder.set_latent_dimensions(
        config.get('classical_latent_dim', 64),
        config.get('quantum_latent_dim', 32)
    )
    
    builder.set_hybrid_options(
        config.get('use_quantum_attention', True),
        config.get('fusion_method', 'concatenate')
    )
    
    logger.info(f"Creating quantum-classical hybrid autoencoder with preset: {preset}")
    return builder.build()