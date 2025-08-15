"""Advanced Temporal Fusion Transformer for IoT Anomaly Detection.

State-of-the-art implementation combining Temporal Fusion Transformers (TFT)
with specialized IoT anomaly detection capabilities. Features multi-horizon
forecasting, interpretable attention mechanisms, and adaptive anomaly scoring.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import pickle
import json
import time
import warnings
from collections import defaultdict, deque

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.regularizers import l1_l2
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import matplotlib.pyplot as plt
    import seaborn as sns
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    warnings.warn("TFT dependencies not available. Using simplified implementations.")

from .logging_config import get_logger
from .data_preprocessor import DataPreprocessor


@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer."""
    
    # Model architecture
    hidden_size: int = 128
    num_attention_heads: int = 8
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dropout_rate: float = 0.1
    
    # Time series parameters
    lookback_window: int = 60
    forecast_horizon: int = 12
    num_quantiles: int = 7  # For quantile regression
    
    # Feature dimensions
    num_static_features: int = 0
    num_dynamic_features: int = 5
    num_categorical_features: int = 0
    categorical_vocab_sizes: List[int] = field(default_factory=list)
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 0.001
    max_epochs: int = 100
    patience: int = 10
    
    # Anomaly detection parameters
    quantile_levels: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    anomaly_threshold_quantile: float = 0.95
    multi_horizon_weights: List[float] = field(default_factory=list)


@dataclass
class AttentionWeights:
    """Container for attention weights from TFT."""
    
    temporal_attention: np.ndarray
    static_attention: np.ndarray
    variable_attention: np.ndarray
    decoder_attention: np.ndarray
    timestamps: List[float]
    feature_names: List[str]


class GatedLinearUnit(layers.Layer):
    """Gated Linear Unit activation function."""
    
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.dense = layers.Dense(hidden_size * 2)
        
    def call(self, inputs):
        x = self.dense(inputs)
        linear, gate = tf.split(x, 2, axis=-1)
        return linear * tf.sigmoid(gate)
    
    def get_config(self):
        config = super().get_config()
        config.update({'hidden_size': self.hidden_size})
        return config


class VariableSelectionNetwork(layers.Layer):
    """Variable Selection Network for feature importance."""
    
    def __init__(self, hidden_size: int, num_features: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.dropout_rate = dropout_rate
        
        # Feature selection layers
        self.feature_selection = layers.Dense(num_features, activation='softmax')
        self.feature_transform = layers.Dense(hidden_size)
        self.dropout = layers.Dropout(dropout_rate)
        self.glu = GatedLinearUnit(hidden_size)
        
    def call(self, inputs, training=None):
        # inputs: [batch, time, features]
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        # Flatten for feature selection
        flat_inputs = tf.reshape(inputs, [-1, self.num_features])
        
        # Feature selection weights
        selection_weights = self.feature_selection(flat_inputs)
        
        # Apply selection weights
        selected_features = flat_inputs * selection_weights
        
        # Transform selected features
        transformed = self.feature_transform(selected_features)
        transformed = self.dropout(transformed, training=training)
        transformed = self.glu(transformed)
        
        # Reshape back to original shape
        output = tf.reshape(transformed, [batch_size, time_steps, self.hidden_size])
        
        # Return both transformed features and selection weights
        selection_weights = tf.reshape(selection_weights, [batch_size, time_steps, self.num_features])
        
        return output, selection_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_features': self.num_features,
            'dropout_rate': self.dropout_rate
        })
        return config


class MultiHeadAttention(layers.Layer):
    """Multi-head attention mechanism with interpretability."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        
        # Attention layers
        self.query_dense = layers.Dense(hidden_size)
        self.key_dense = layers.Dense(hidden_size)
        self.value_dense = layers.Dense(hidden_size)
        self.output_dense = layers.Dense(hidden_size)
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, query, key, value, mask=None, training=None):
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[1]
        
        # Linear transformations
        q = self.query_dense(query)
        k = self.key_dense(key)
        v = self.value_dense(value)
        
        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        
        # Transpose for attention computation
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        
        if mask is not None:
            attention_scores += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention
        attended = tf.matmul(attention_weights, v)
        
        # Concatenate heads
        attended = tf.transpose(attended, [0, 2, 1, 3])
        attended = tf.reshape(attended, [batch_size, seq_len, self.hidden_size])
        
        # Final linear transformation
        output = self.output_dense(attended)
        
        return output, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config


class TemporalFusionTransformerBlock(layers.Layer):
    """Complete TFT block with encoder and decoder."""
    
    def __init__(self, config: TFTConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Variable selection networks
        self.encoder_vsn = VariableSelectionNetwork(
            config.hidden_size, 
            config.num_dynamic_features,
            config.dropout_rate
        )
        
        self.decoder_vsn = VariableSelectionNetwork(
            config.hidden_size,
            config.num_dynamic_features, 
            config.dropout_rate
        )
        
        # Encoder stack
        self.encoder_layers = []
        for _ in range(config.num_encoder_layers):
            self.encoder_layers.append(
                MultiHeadAttention(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.dropout_rate
                )
            )
        
        # Decoder stack
        self.decoder_layers = []
        for _ in range(config.num_decoder_layers):
            self.decoder_layers.append(
                MultiHeadAttention(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.dropout_rate
                )
            )
        
        # Layer normalization
        self.encoder_norms = [layers.LayerNormalization() for _ in range(config.num_encoder_layers)]
        self.decoder_norms = [layers.LayerNormalization() for _ in range(config.num_decoder_layers)]
        
        # Skip connections
        self.skip_dense = layers.Dense(config.hidden_size)
        
        # Output layers for quantile regression
        self.quantile_heads = []
        for _ in range(config.num_quantiles):
            self.quantile_heads.append(
                layers.Dense(config.forecast_horizon, name=f'quantile_{_}')
            )
    
    def call(self, encoder_inputs, decoder_inputs, training=None):
        # Variable selection for encoder
        encoder_features, encoder_selection = self.encoder_vsn(encoder_inputs, training=training)
        
        # Encoder processing
        encoder_output = encoder_features
        encoder_attentions = []
        
        for layer, norm in zip(self.encoder_layers, self.encoder_norms):
            attended_output, attention_weights = layer(
                encoder_output, encoder_output, encoder_output, training=training
            )
            encoder_output = norm(encoder_output + attended_output)
            encoder_attentions.append(attention_weights)
        
        # Variable selection for decoder
        decoder_features, decoder_selection = self.decoder_vsn(decoder_inputs, training=training)
        
        # Decoder processing with cross-attention to encoder
        decoder_output = decoder_features
        decoder_attentions = []
        
        for layer, norm in zip(self.decoder_layers, self.decoder_norms):
            # Self-attention in decoder
            self_attended, self_attention = layer(
                decoder_output, decoder_output, decoder_output, training=training
            )
            decoder_output = norm(decoder_output + self_attended)
            
            # Cross-attention to encoder
            cross_attended, cross_attention = layer(
                decoder_output, encoder_output, encoder_output, training=training
            )
            decoder_output = norm(decoder_output + cross_attended)
            
            decoder_attentions.append((self_attention, cross_attention))
        
        # Skip connection from encoder
        skip_connection = self.skip_dense(encoder_output)
        decoder_output = decoder_output + skip_connection[:, -tf.shape(decoder_output)[1]:, :]
        
        # Multi-quantile outputs
        quantile_outputs = []
        for quantile_head in self.quantile_heads:
            quantile_pred = quantile_head(decoder_output[:, -1, :])  # Use last decoder state
            quantile_outputs.append(quantile_pred)
        
        # Stack quantile predictions
        quantile_predictions = tf.stack(quantile_outputs, axis=-1)
        
        # Return predictions and attention weights
        attention_info = {
            'encoder_attention': encoder_attentions,
            'decoder_attention': decoder_attentions,
            'encoder_selection': encoder_selection,
            'decoder_selection': decoder_selection
        }
        
        return quantile_predictions, attention_info
    
    def get_config(self):
        config = super().get_config()
        config.update({'config': self.config.__dict__})
        return config


class QuantileLoss(tf.keras.losses.Loss):
    """Quantile loss for multi-quantile regression."""
    
    def __init__(self, quantiles: List[float], **kwargs):
        super().__init__(**kwargs)
        self.quantiles = quantiles
    
    def call(self, y_true, y_pred):
        """Compute quantile loss."""
        # y_true: [batch, horizon]
        # y_pred: [batch, horizon, num_quantiles]
        
        losses = []
        
        for i, q in enumerate(self.quantiles):
            error = y_true - y_pred[:, :, i]
            loss = tf.maximum(q * error, (q - 1) * error)
            losses.append(tf.reduce_mean(loss))
        
        return tf.reduce_mean(losses)


class TemporalFusionTransformerAnomalyDetector:
    """TFT-based anomaly detector for IoT time series."""
    
    def __init__(self, config: TFTConfig):
        """Initialize TFT anomaly detector.
        
        Args:
            config: TFT configuration object
        """
        self.config = config
        self.model: Optional[tf.keras.Model] = None
        self.preprocessor = DataPreprocessor()
        self.is_trained = False
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'quantile_coverage': [],
            'attention_evolution': []
        }
        
        # Anomaly detection components
        self.anomaly_thresholds = {}
        self.attention_baselines = {}
        self.forecast_baselines = {}
        
        self.logger = get_logger(__name__)
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the TFT model architecture."""
        try:
            # Input layers
            encoder_input = layers.Input(
                shape=(self.config.lookback_window, self.config.num_dynamic_features),
                name='encoder_input'
            )
            
            decoder_input = layers.Input(
                shape=(self.config.forecast_horizon, self.config.num_dynamic_features),
                name='decoder_input'
            )
            
            # Static features (if any)
            if self.config.num_static_features > 0:
                static_input = layers.Input(
                    shape=(self.config.num_static_features,),
                    name='static_input'
                )
            
            # TFT block
            tft_block = TemporalFusionTransformerBlock(self.config)
            
            # Forward pass
            quantile_predictions, attention_info = tft_block(
                encoder_input, decoder_input
            )
            
            # Build model
            inputs = [encoder_input, decoder_input]
            if self.config.num_static_features > 0:
                inputs.append(static_input)
            
            self.model = tf.keras.Model(
                inputs=inputs,
                outputs=[quantile_predictions],
                name='temporal_fusion_transformer'
            )
            
            # Compile with quantile loss
            quantile_loss = QuantileLoss(self.config.quantile_levels)
            
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
                loss=quantile_loss,
                metrics=['mae']
            )
            
            # Store TFT block for attention extraction
            self.tft_block = tft_block
            
            self.logger.info(f"Built TFT model with {self.model.count_params()} parameters")
            
        except Exception as e:
            self.logger.error(f"Failed to build TFT model: {str(e)}")
            raise
    
    def prepare_sequences(
        self,
        data: np.ndarray,
        target_column: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sequences for TFT training."""
        try:
            sequences = []
            targets = []
            
            total_window = self.config.lookback_window + self.config.forecast_horizon
            
            for i in range(len(data) - total_window + 1):
                # Historical sequence for encoder
                encoder_seq = data[i:i + self.config.lookback_window]
                
                # Future sequence for decoder (known future inputs)
                decoder_seq = data[
                    i + self.config.lookback_window:
                    i + total_window,
                    1:  # Exclude target variable for decoder input
                ]
                
                # Add zero padding if decoder input has fewer features
                if decoder_seq.shape[1] == 0:
                    decoder_seq = np.zeros((self.config.forecast_horizon, self.config.num_dynamic_features))
                else:
                    # Pad with zeros to match expected dimensions
                    padding_needed = self.config.num_dynamic_features - decoder_seq.shape[1]
                    if padding_needed > 0:
                        decoder_padding = np.zeros((decoder_seq.shape[0], padding_needed))
                        decoder_seq = np.concatenate([decoder_seq, decoder_padding], axis=1)
                
                # Target values (future values of target variable)
                target_seq = data[
                    i + self.config.lookback_window:
                    i + total_window,
                    target_column
                ]
                
                sequences.append((encoder_seq, decoder_seq))
                targets.append(target_seq)
            
            # Convert to arrays
            encoder_sequences = np.array([seq[0] for seq in sequences])
            decoder_sequences = np.array([seq[1] for seq in sequences])
            targets = np.array(targets)
            
            self.logger.info(
                f"Prepared {len(sequences)} sequences: "
                f"encoder{encoder_sequences.shape}, "
                f"decoder{decoder_sequences.shape}, "
                f"targets{targets.shape}"
            )
            
            return encoder_sequences, decoder_sequences, targets
            
        except Exception as e:
            self.logger.error(f"Failed to prepare sequences: {str(e)}")
            raise
    
    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        target_column: int = 0
    ) -> Dict[str, Any]:
        """Train the TFT model."""
        try:
            start_time = time.time()
            
            # Preprocess data
            train_data_scaled = self.preprocessor.fit_transform(train_data)
            
            if val_data is not None:
                val_data_scaled = self.preprocessor.transform(val_data)
            else:
                # Use last 20% of training data for validation
                split_idx = int(len(train_data_scaled) * 0.8)
                val_data_scaled = train_data_scaled[split_idx:]
                train_data_scaled = train_data_scaled[:split_idx]
            
            # Prepare sequences
            X_train_enc, X_train_dec, y_train = self.prepare_sequences(train_data_scaled, target_column)
            X_val_enc, X_val_dec, y_val = self.prepare_sequences(val_data_scaled, target_column)
            
            # Training callbacks
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                restore_best_weights=True
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
            
            # Custom callback for attention tracking
            attention_callback = AttentionTrackingCallback(self)
            
            # Train model
            self.logger.info(f"Starting TFT training: {len(X_train_enc)} samples")
            
            history = self.model.fit(
                x=[X_train_enc, X_train_dec],
                y=y_train,
                validation_data=([X_val_enc, X_val_dec], y_val),
                epochs=self.config.max_epochs,
                batch_size=self.config.batch_size,
                callbacks=[early_stopping, reduce_lr, attention_callback],
                verbose=1
            )
            
            # Store training history
            self.training_history['train_loss'] = history.history['loss']
            self.training_history['val_loss'] = history.history['val_loss']
            
            # Calculate baseline thresholds on validation data
            self._calculate_anomaly_thresholds(X_val_enc, X_val_dec, y_val)
            
            training_time = time.time() - start_time
            
            training_summary = {
                'training_time': training_time,
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'epochs_trained': len(history.history['loss']),
                'model_parameters': self.model.count_params(),
                'quantile_levels': self.config.quantile_levels,
                'lookback_window': self.config.lookback_window,
                'forecast_horizon': self.config.forecast_horizon
            }
            
            self.is_trained = True
            self.logger.info(f"TFT training completed: {training_summary}")
            
            return training_summary
            
        except Exception as e:
            self.logger.error(f"TFT training failed: {str(e)}")
            raise
    
    def _calculate_anomaly_thresholds(
        self,
        X_enc: np.ndarray,
        X_dec: np.ndarray, 
        y_true: np.ndarray
    ) -> None:
        """Calculate anomaly detection thresholds from validation data."""
        try:
            # Make predictions on validation data
            predictions = self.model.predict([X_enc, X_dec], verbose=0)
            
            # Calculate prediction intervals and anomaly scores
            anomaly_scores = []
            coverage_violations = []
            
            for i in range(len(predictions)):
                pred_quantiles = predictions[i]  # [horizon, num_quantiles]
                true_values = y_true[i]  # [horizon]
                
                # Calculate quantile-based anomaly scores
                median_idx = len(self.config.quantile_levels) // 2
                median_pred = pred_quantiles[:, median_idx]
                
                # Normalized prediction error
                pred_error = np.abs(true_values - median_pred)
                
                # Check coverage violations (true value outside prediction interval)
                lower_quantile = pred_quantiles[:, 1]  # 25th percentile
                upper_quantile = pred_quantiles[:, -2]  # 75th percentile
                
                outside_interval = (true_values < lower_quantile) | (true_values > upper_quantile)
                coverage_violations.append(np.mean(outside_interval))
                
                # Multi-quantile consistency check
                quantile_consistency = np.mean(np.diff(pred_quantiles, axis=1) >= 0)
                
                # Combined anomaly score
                sample_score = np.mean(pred_error) * (1 + np.sum(outside_interval)) * (2 - quantile_consistency)
                anomaly_scores.append(sample_score)
            
            # Set thresholds based on score distribution
            self.anomaly_thresholds = {
                'prediction_error': np.percentile(anomaly_scores, 95),
                'coverage_violation': np.percentile(coverage_violations, 90),
                'combined_score': np.percentile(anomaly_scores, self.config.anomaly_threshold_quantile * 100)
            }
            
            # Store baseline statistics
            self.forecast_baselines = {
                'mean_prediction_error': np.mean(anomaly_scores),
                'std_prediction_error': np.std(anomaly_scores),
                'mean_coverage_violation': np.mean(coverage_violations)
            }
            
            self.logger.info(f"Calculated anomaly thresholds: {self.anomaly_thresholds}")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate thresholds: {str(e)}")
    
    def predict(
        self,
        data: np.ndarray,
        return_attention: bool = False,
        target_column: int = 0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect anomalies using TFT."""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained. Call train() first.")
            
            start_time = time.time()
            
            # Preprocess data
            data_scaled = self.preprocessor.transform(data)
            
            # Prepare sequences
            X_enc, X_dec, y_true = self.prepare_sequences(data_scaled, target_column)
            
            if len(X_enc) == 0:
                return np.array([]), {'error': 'Insufficient data for sequence generation'}
            
            # Make predictions
            predictions = self.model.predict([X_enc, X_dec], verbose=0)
            
            # Calculate anomaly scores
            anomaly_scores = []
            anomalies = []
            confidence_scores = []
            attention_anomalies = []
            
            # Extract attention weights if requested
            attention_weights = None
            if return_attention:
                attention_weights = self._extract_attention_weights(X_enc, X_dec)
            
            for i in range(len(predictions)):
                pred_quantiles = predictions[i]  # [horizon, num_quantiles]
                true_values = y_true[i]  # [horizon]
                
                # Multi-horizon anomaly detection
                horizon_scores = []
                
                for h in range(self.config.forecast_horizon):
                    # Quantile-based anomaly detection
                    quantile_preds = pred_quantiles[h, :]
                    true_val = true_values[h]
                    
                    # Check if true value falls outside prediction intervals
                    lower_bound = quantile_preds[1]  # 25th percentile
                    upper_bound = quantile_preds[-2]  # 75th percentile
                    median_pred = quantile_preds[len(quantile_preds) // 2]
                    
                    # Prediction error
                    pred_error = abs(true_val - median_pred)
                    
                    # Interval violation
                    interval_violation = (true_val < lower_bound) or (true_val > upper_bound)
                    
                    # Quantile consistency (predictions should be monotonic)
                    quantile_consistency = np.all(np.diff(quantile_preds) >= -1e-6)
                    
                    # Combined horizon score
                    horizon_weight = self.config.multi_horizon_weights[h] if len(self.config.multi_horizon_weights) > h else 1.0
                    
                    horizon_score = horizon_weight * (
                        pred_error / (upper_bound - lower_bound + 1e-6) +  # Normalized error
                        (2.0 if interval_violation else 0.0) +  # Interval penalty
                        (1.0 if not quantile_consistency else 0.0)  # Consistency penalty
                    )
                    
                    horizon_scores.append(horizon_score)
                
                # Aggregate horizon scores
                sample_score = np.mean(horizon_scores)
                anomaly_scores.append(sample_score)
                
                # Determine anomaly based on threshold
                is_anomaly = sample_score > self.anomaly_thresholds.get('combined_score', 1.0)
                anomalies.append(int(is_anomaly))
                
                # Confidence based on score magnitude
                max_expected_score = self.anomaly_thresholds.get('combined_score', 1.0) * 2
                confidence = min(sample_score / max_expected_score, 1.0)
                confidence_scores.append(confidence)
                
                # Attention-based anomaly detection (if enabled)
                if return_attention and attention_weights:
                    attention_anomaly = self._detect_attention_anomalies(
                        attention_weights[i] if i < len(attention_weights) else None
                    )
                    attention_anomalies.append(attention_anomaly)
            
            inference_time = time.time() - start_time
            
            # Compile metadata
            metadata = {
                'inference_time': inference_time,
                'samples_processed': len(anomalies),
                'anomaly_scores': anomaly_scores,
                'confidence_scores': confidence_scores,
                'anomaly_thresholds': self.anomaly_thresholds,
                'forecast_horizon': self.config.forecast_horizon,
                'quantile_levels': self.config.quantile_levels,
                'model_type': 'temporal_fusion_transformer'
            }
            
            if return_attention:
                metadata['attention_weights'] = attention_weights
                metadata['attention_anomalies'] = attention_anomalies
            
            self.logger.info(
                f"TFT inference completed: {np.sum(anomalies)} anomalies detected "
                f"from {len(anomalies)} samples"
            )
            
            return np.array(anomalies), metadata
            
        except Exception as e:
            self.logger.error(f"TFT prediction failed: {str(e)}")
            raise
    
    def _extract_attention_weights(
        self,
        X_enc: np.ndarray,
        X_dec: np.ndarray
    ) -> List[AttentionWeights]:
        """Extract attention weights for interpretability."""
        try:
            # Create a model that outputs attention weights
            attention_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=self.model.outputs + [
                    # Add attention outputs here
                ]
            )
            
            # For now, return empty list (would need model modification)
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to extract attention weights: {str(e)}")
            return []
    
    def _detect_attention_anomalies(
        self,
        attention_weights: Optional[AttentionWeights]
    ) -> bool:
        """Detect anomalies based on attention patterns."""
        if not attention_weights:
            return False
        
        try:
            # Compare attention patterns to baseline
            # This would analyze unusual attention distributions
            # For now, return False (placeholder)
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to detect attention anomalies: {str(e)}")
            return False
    
    def explain_prediction(
        self,
        sample_index: int,
        data: np.ndarray,
        target_column: int = 0
    ) -> Dict[str, Any]:
        """Provide explanation for a specific prediction."""
        try:
            # Preprocess and prepare sequences
            data_scaled = self.preprocessor.transform(data)
            X_enc, X_dec, y_true = self.prepare_sequences(data_scaled, target_column)
            
            if sample_index >= len(X_enc):
                raise ValueError(f"Sample index {sample_index} out of range")
            
            # Get prediction for specific sample
            sample_enc = X_enc[sample_index:sample_index+1]
            sample_dec = X_dec[sample_index:sample_index+1]
            prediction = self.model.predict([sample_enc, sample_dec], verbose=0)
            
            # Calculate feature importance (simplified)
            feature_importance = self._calculate_feature_importance(sample_enc, sample_dec)
            
            # Temporal importance
            temporal_importance = self._calculate_temporal_importance(sample_enc)
            
            # Quantile analysis
            quantile_analysis = self._analyze_quantile_predictions(prediction[0])
            
            explanation = {
                'sample_index': sample_index,
                'prediction_quantiles': prediction[0].tolist(),
                'true_values': y_true[sample_index].tolist(),
                'feature_importance': feature_importance,
                'temporal_importance': temporal_importance,
                'quantile_analysis': quantile_analysis,
                'model_confidence': self._calculate_prediction_confidence(prediction[0]),
                'anomaly_factors': self._identify_anomaly_factors(
                    sample_enc, sample_dec, prediction[0], y_true[sample_index]
                )
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Failed to explain prediction: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_feature_importance(
        self,
        X_enc: np.ndarray,
        X_dec: np.ndarray
    ) -> List[float]:
        """Calculate feature importance for prediction."""
        try:
            # Use gradient-based importance (simplified)
            baseline_pred = self.model.predict([X_enc, X_dec], verbose=0)
            
            importance_scores = []
            
            for feature_idx in range(X_enc.shape[-1]):
                # Perturb feature
                X_enc_perturbed = X_enc.copy()
                X_enc_perturbed[:, :, feature_idx] = 0
                
                perturbed_pred = self.model.predict([X_enc_perturbed, X_dec], verbose=0)
                
                # Calculate importance as prediction change
                importance = np.mean(np.abs(baseline_pred - perturbed_pred))
                importance_scores.append(importance)
            
            # Normalize importance scores
            total_importance = sum(importance_scores) + 1e-8
            normalized_importance = [score / total_importance for score in importance_scores]
            
            return normalized_importance
            
        except Exception as e:
            self.logger.error(f"Failed to calculate feature importance: {str(e)}")
            return [1.0 / X_enc.shape[-1]] * X_enc.shape[-1]
    
    def _calculate_temporal_importance(self, X_enc: np.ndarray) -> List[float]:
        """Calculate temporal importance for each time step."""
        try:
            baseline_pred = self.model.predict([X_enc, np.zeros_like(X_enc[:, :self.config.forecast_horizon])], verbose=0)
            
            importance_scores = []
            
            for time_idx in range(X_enc.shape[1]):
                # Mask time step
                X_enc_masked = X_enc.copy()
                X_enc_masked[:, time_idx, :] = 0
                
                masked_pred = self.model.predict([X_enc_masked, np.zeros_like(X_enc[:, :self.config.forecast_horizon])], verbose=0)
                
                # Calculate importance
                importance = np.mean(np.abs(baseline_pred - masked_pred))
                importance_scores.append(importance)
            
            # Normalize
            total_importance = sum(importance_scores) + 1e-8
            normalized_importance = [score / total_importance for score in importance_scores]
            
            return normalized_importance
            
        except Exception as e:
            self.logger.error(f"Failed to calculate temporal importance: {str(e)}")
            return [1.0 / X_enc.shape[1]] * X_enc.shape[1]
    
    def _analyze_quantile_predictions(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze quantile predictions for insights."""
        try:
            # predictions: [horizon, num_quantiles]
            
            analysis = {
                'prediction_intervals': [],
                'uncertainty_levels': [],
                'forecast_drift': [],
                'quantile_spread': []
            }
            
            for h in range(predictions.shape[0]):
                quantiles = predictions[h, :]
                
                # Prediction intervals
                intervals = {
                    '50%': (quantiles[1], quantiles[-2]),  # IQR
                    '80%': (quantiles[0], quantiles[-1]),  # Full range
                    'median': quantiles[len(quantiles) // 2]
                }
                analysis['prediction_intervals'].append(intervals)
                
                # Uncertainty (interval width)
                uncertainty = quantiles[-1] - quantiles[0]
                analysis['uncertainty_levels'].append(uncertainty)
                
                # Quantile spread analysis
                spread = np.std(quantiles)
                analysis['quantile_spread'].append(spread)
            
            # Forecast drift (how predictions change over horizon)
            medians = [interval['median'] for interval in analysis['prediction_intervals']]
            drift = np.diff(medians) if len(medians) > 1 else [0.0]
            analysis['forecast_drift'] = drift.tolist()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze quantile predictions: {str(e)}")
            return {}
    
    def _calculate_prediction_confidence(self, predictions: np.ndarray) -> float:
        """Calculate overall prediction confidence."""
        try:
            # Based on quantile consistency and uncertainty levels
            
            # Quantile consistency check
            consistency_scores = []
            for h in range(predictions.shape[0]):
                quantiles = predictions[h, :]
                is_monotonic = np.all(np.diff(quantiles) >= -1e-6)
                consistency_scores.append(1.0 if is_monotonic else 0.0)
            
            consistency = np.mean(consistency_scores)
            
            # Uncertainty levels (lower uncertainty = higher confidence)
            uncertainties = []
            for h in range(predictions.shape[0]):
                quantiles = predictions[h, :]
                uncertainty = quantiles[-1] - quantiles[0]
                uncertainties.append(uncertainty)
            
            mean_uncertainty = np.mean(uncertainties)
            uncertainty_confidence = 1.0 / (1.0 + mean_uncertainty)
            
            # Combined confidence
            overall_confidence = (consistency + uncertainty_confidence) / 2.0
            
            return overall_confidence
            
        except Exception as e:
            self.logger.error(f"Failed to calculate prediction confidence: {str(e)}")
            return 0.5
    
    def _identify_anomaly_factors(
        self,
        X_enc: np.ndarray,
        X_dec: np.ndarray,
        predictions: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, Any]:
        """Identify factors contributing to anomaly detection."""
        try:
            factors = {
                'prediction_error': [],
                'interval_violations': [],
                'uncertainty_spikes': [],
                'trend_deviations': []
            }
            
            for h in range(len(y_true)):
                quantiles = predictions[h, :]
                true_val = y_true[h]
                
                # Prediction error
                median_pred = quantiles[len(quantiles) // 2]
                pred_error = abs(true_val - median_pred)
                factors['prediction_error'].append(pred_error)
                
                # Interval violations
                lower_bound = quantiles[1]
                upper_bound = quantiles[-2]
                violation = (true_val < lower_bound) or (true_val > upper_bound)
                factors['interval_violations'].append(violation)
                
                # Uncertainty spikes
                uncertainty = quantiles[-1] - quantiles[0]
                factors['uncertainty_spikes'].append(uncertainty)
            
            # Trend deviations (simplified)
            if len(factors['prediction_error']) > 1:
                trend_dev = np.std(np.diff(factors['prediction_error']))
                factors['trend_deviations'] = [trend_dev] * len(factors['prediction_error'])
            else:
                factors['trend_deviations'] = [0.0]
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Failed to identify anomaly factors: {str(e)}")
            return {}
    
    def save_model(self, filepath: str) -> None:
        """Save TFT model and configuration."""
        try:
            # Save model
            model_path = filepath.replace('.pkl', '_model.h5')
            self.model.save(model_path)
            
            # Save configuration and state
            state = {
                'config': self.config.__dict__,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'anomaly_thresholds': self.anomaly_thresholds,
                'forecast_baselines': self.forecast_baselines,
                'preprocessor_state': {
                    'mean_': getattr(self.preprocessor.scaler, 'mean_', None),
                    'scale_': getattr(self.preprocessor.scaler, 'scale_', None)
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            self.logger.info(f"Saved TFT model to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save TFT model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load TFT model and configuration."""
        try:
            # Load state
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore configuration
            self.config = TFTConfig(**state['config'])
            self.is_trained = state['is_trained']
            self.training_history = state.get('training_history', {})
            self.anomaly_thresholds = state.get('anomaly_thresholds', {})
            self.forecast_baselines = state.get('forecast_baselines', {})
            
            # Restore preprocessor
            preprocessor_state = state.get('preprocessor_state', {})
            if preprocessor_state.get('mean_') is not None:
                self.preprocessor.scaler.mean_ = preprocessor_state['mean_']
                self.preprocessor.scaler.scale_ = preprocessor_state['scale_']
            
            # Load model
            model_path = filepath.replace('.pkl', '_model.h5')
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'QuantileLoss': QuantileLoss,
                    'GatedLinearUnit': GatedLinearUnit,
                    'VariableSelectionNetwork': VariableSelectionNetwork,
                    'MultiHeadAttention': MultiHeadAttention,
                    'TemporalFusionTransformerBlock': TemporalFusionTransformerBlock
                }
            )
            
            # Rebuild TFT block for attention extraction
            self._build_model()
            
            self.logger.info(f"Loaded TFT model from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load TFT model: {str(e)}")
            raise


class AttentionTrackingCallback(tf.keras.callbacks.Callback):
    """Callback to track attention evolution during training."""
    
    def __init__(self, tft_detector: TemporalFusionTransformerAnomalyDetector):
        super().__init__()
        self.tft_detector = tft_detector
        self.attention_snapshots = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Save attention weights at end of epoch."""
        try:
            # This would extract and store attention weights
            # For now, just store epoch info
            snapshot = {
                'epoch': epoch,
                'train_loss': logs.get('loss', 0),
                'val_loss': logs.get('val_loss', 0)
            }
            self.attention_snapshots.append(snapshot)
            
            # Store in detector
            self.tft_detector.training_history['attention_evolution'] = self.attention_snapshots
            
        except Exception as e:
            pass  # Don't fail training for logging issues


def create_optimized_tft_detector(
    num_features: int,
    lookback_window: int = 60,
    forecast_horizon: int = 12,
    performance_target: str = "balanced"  # "speed", "accuracy", "balanced"
) -> TemporalFusionTransformerAnomalyDetector:
    """Create optimized TFT detector based on performance requirements."""
    
    if performance_target == "speed":
        # Optimized for inference speed
        config = TFTConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            lookback_window=min(lookback_window, 30),
            forecast_horizon=min(forecast_horizon, 6),
            num_dynamic_features=num_features,
            batch_size=128,
            num_quantiles=3
        )
    
    elif performance_target == "accuracy":
        # Optimized for detection accuracy
        config = TFTConfig(
            hidden_size=256,
            num_attention_heads=16,
            num_encoder_layers=4,
            num_decoder_layers=4,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            num_dynamic_features=num_features,
            batch_size=32,
            num_quantiles=9,
            dropout_rate=0.2
        )
    
    else:  # balanced
        config = TFTConfig(
            hidden_size=128,
            num_attention_heads=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            num_dynamic_features=num_features,
            batch_size=64,
            num_quantiles=7
        )
    
    # Set multi-horizon weights (closer horizons more important)
    weights = [1.0 / (1 + 0.1 * h) for h in range(forecast_horizon)]
    config.multi_horizon_weights = [w / sum(weights) for w in weights]
    
    return TemporalFusionTransformerAnomalyDetector(config)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Temporal Fusion Transformer Anomaly Detection")
    parser.add_argument("--num-features", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--performance", choices=["speed", "accuracy", "balanced"], default="balanced")
    parser.add_argument("--output", type=str, default="tft_model.pkl")
    
    args = parser.parse_args()
    
    # Create TFT detector
    detector = create_optimized_tft_detector(
        num_features=args.num_features,
        lookback_window=args.lookback,
        forecast_horizon=args.horizon,
        performance_target=args.performance
    )
    
    # Generate synthetic time series data
    np.random.seed(42)
    
    # Normal pattern with trend and seasonality
    t = np.linspace(0, 100, 2000)
    trend = 0.01 * t
    seasonal = 2 * np.sin(2 * np.pi * t / 20) + np.sin(2 * np.pi * t / 5)
    noise = np.random.normal(0, 0.5, len(t))
    
    # Create multivariate data
    base_signal = trend + seasonal + noise
    features = []
    for i in range(args.num_features):
        feature = base_signal + np.random.normal(0, 0.2, len(t))
        # Add some anomalies
        if i == 0:  # Target variable
            anomaly_indices = np.random.choice(len(feature), 50, replace=False)
            feature[anomaly_indices] += np.random.normal(0, 3, 50)
        features.append(feature)
    
    train_data = np.column_stack(features)
    
    print(f"Training TFT detector on {len(train_data)} samples...")
    training_results = detector.train(train_data, target_column=0)
    
    print(f"\nTraining Results:")
    for key, value in training_results.items():
        print(f"  {key}: {value}")
    
    # Test anomaly detection
    test_data = train_data[-200:]  # Use last 200 samples for testing
    
    print(f"\nRunning TFT anomaly detection...")
    predictions, metadata = detector.predict(test_data, return_attention=True)
    
    print(f"\nDetection Results:")
    print(f"  Samples processed: {metadata['samples_processed']}")
    print(f"  Anomalies detected: {np.sum(predictions)}")
    print(f"  Detection rate: {np.mean(predictions) * 100:.1f}%")
    print(f"  Inference time: {metadata['inference_time']:.3f}s")
    print(f"  Mean anomaly score: {np.mean(metadata['anomaly_scores']):.3f}")
    
    # Save model
    detector.save_model(args.output)
    
    # Explain a specific prediction
    if len(predictions) > 0:
        anomaly_idx = np.where(predictions == 1)[0]
        if len(anomaly_idx) > 0:
            explanation = detector.explain_prediction(anomaly_idx[0], test_data)
            print(f"\nExample Anomaly Explanation:")
            print(f"  Sample index: {explanation['sample_index']}")
            print(f"  Model confidence: {explanation['model_confidence']:.3f}")
            print(f"  Top anomaly factors: {list(explanation['anomaly_factors'].keys())}")
    
    print(f"\nTFT model saved to {args.output}")