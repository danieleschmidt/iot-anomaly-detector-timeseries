"""Enhanced autoencoder architecture with advanced features for Generation 1."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, regularizers
from tensorflow.keras.models import Model
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedAutoencoder:
    """Enhanced LSTM autoencoder with attention mechanisms and improved architecture."""
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        latent_dim: int = 32,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-4
    ):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        
    def build_model(self) -> Model:
        """Build enhanced autoencoder with attention mechanism."""
        logger.info(f"Building enhanced autoencoder with input shape {self.input_shape}")
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # Encoder with bidirectional LSTM and attention
        encoder_lstm1 = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ),
            name='encoder_lstm1'
        )(inputs)
        
        encoder_lstm2 = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units // 2,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ),
            name='encoder_lstm2'
        )(encoder_lstm1)
        
        # Attention mechanism
        attention = self._add_attention_layer(encoder_lstm2)
        
        # Encoder final layer
        encoded = layers.LSTM(
            self.latent_dim,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='encoder_final'
        )(attention)
        
        # Bottleneck with batch normalization
        bottleneck = layers.BatchNormalization(name='bottleneck_bn')(encoded)
        bottleneck = layers.Dropout(self.dropout_rate, name='bottleneck_dropout')(bottleneck)
        
        # Decoder
        decoded = layers.RepeatVector(self.input_shape[0], name='repeat_vector')(bottleneck)
        
        decoder_lstm1 = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units // 2,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ),
            name='decoder_lstm1'
        )(decoded)
        
        decoder_lstm2 = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ),
            name='decoder_lstm2'
        )(decoder_lstm1)
        
        # Output layer with residual connection
        output = layers.TimeDistributed(
            layers.Dense(
                self.input_shape[1],
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ),
            name='output_dense'
        )(decoder_lstm2)
        
        # Build model
        self.model = models.Model(inputs, output, name='enhanced_autoencoder')
        
        # Compile with adaptive learning rate
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=self._composite_loss,
            metrics=['mse', 'mae']
        )
        
        logger.info(f"Model built successfully with {self.model.count_params()} parameters")
        return self.model
    
    def _add_attention_layer(self, lstm_output):
        """Add attention mechanism to LSTM output."""
        attention_weights = layers.Dense(1, activation='tanh', name='attention_weights')(lstm_output)
        attention_weights = layers.Softmax(axis=1, name='attention_softmax')(attention_weights)
        attended_output = layers.multiply([lstm_output, attention_weights], name='attention_multiply')
        return attended_output
    
    def _composite_loss(self, y_true, y_pred):
        """Composite loss combining MSE and MAE."""
        mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        mae_loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
        return 0.7 * mse_loss + 0.3 * mae_loss
    
    def train_with_callbacks(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10
    ):
        """Train model with enhanced callbacks."""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_enhanced_autoencoder.h5',
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        validation_data = (X_val, X_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, X_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def detect_anomalies(
        self,
        X: np.ndarray,
        threshold_percentile: float = 95.0
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Enhanced anomaly detection with dynamic thresholding."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Get reconstructions
        reconstructions = self.model.predict(X, verbose=0)
        
        # Calculate reconstruction errors
        mse_errors = np.mean(np.square(X - reconstructions), axis=(1, 2))
        mae_errors = np.mean(np.abs(X - reconstructions), axis=(1, 2))
        
        # Composite error score
        errors = 0.7 * mse_errors + 0.3 * mae_errors
        
        # Dynamic threshold
        threshold = np.percentile(errors, threshold_percentile)
        
        # Binary anomaly predictions
        anomalies = errors > threshold
        
        logger.info(f"Detected {np.sum(anomalies)} anomalies out of {len(X)} samples")
        logger.info(f"Threshold: {threshold:.6f}, Max error: {np.max(errors):.6f}")
        
        return anomalies, errors, threshold


class RealTimeAnomalyDetector:
    """Real-time anomaly detection with streaming capabilities."""
    
    def __init__(self, model_path: str, threshold: float = None):
        self.model = tf.keras.models.load_model(model_path)
        self.threshold = threshold
        self.error_buffer = []
        self.buffer_size = 1000
        
    def process_sample(self, sample: np.ndarray) -> Tuple[bool, float]:
        """Process single sample for real-time detection."""
        if sample.ndim == 2:
            sample = np.expand_dims(sample, axis=0)
        
        reconstruction = self.model.predict(sample, verbose=0)
        error = np.mean(np.square(sample - reconstruction))
        
        # Update buffer
        self.error_buffer.append(error)
        if len(self.error_buffer) > self.buffer_size:
            self.error_buffer.pop(0)
        
        # Adaptive threshold
        if self.threshold is None and len(self.error_buffer) > 50:
            self.threshold = np.percentile(self.error_buffer, 95)
        
        is_anomaly = error > self.threshold if self.threshold else False
        
        return is_anomaly, error


def create_sample_data(
    num_samples: int = 1000,
    sequence_length: int = 50,
    num_features: int = 5,
    anomaly_rate: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic time series data with anomalies."""
    logger.info(f"Generating {num_samples} samples with {anomaly_rate*100}% anomalies")
    
    # Normal data
    normal_data = []
    for _ in range(num_samples):
        # Sinusoidal patterns with noise
        t = np.linspace(0, 4*np.pi, sequence_length)
        sample = np.zeros((sequence_length, num_features))
        
        for i in range(num_features):
            freq = 0.5 + i * 0.3
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = 1 + i * 0.2
            
            signal = amplitude * np.sin(freq * t + phase)
            noise = np.random.normal(0, 0.1, sequence_length)
            sample[:, i] = signal + noise
        
        normal_data.append(sample)
    
    data = np.array(normal_data)
    
    # Add anomalies
    num_anomalies = int(num_samples * anomaly_rate)
    anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
    
    labels = np.zeros(num_samples, dtype=bool)
    labels[anomaly_indices] = True
    
    # Introduce anomalies
    for idx in anomaly_indices:
        # Random spike anomalies
        anomaly_start = np.random.randint(0, sequence_length - 10)
        anomaly_length = np.random.randint(5, 15)
        anomaly_magnitude = np.random.uniform(3, 5)
        
        data[idx, anomaly_start:anomaly_start+anomaly_length, :] *= anomaly_magnitude
    
    logger.info(f"Created dataset with shape {data.shape}")
    return data, labels


if __name__ == "__main__":
    # Generation 1 implementation demonstration
    logger.info("=== GENERATION 1: ENHANCED AUTOENCODER DEMO ===")
    
    # Create sample data
    X, y_true = create_sample_data(num_samples=1000, sequence_length=30, num_features=3)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_test = y_true[split_idx:]
    
    # Build and train enhanced autoencoder
    autoencoder = EnhancedAutoencoder(
        input_shape=(30, 3),
        latent_dim=16,
        lstm_units=32,
        dropout_rate=0.1
    )
    
    logger.info("Training enhanced autoencoder...")
    autoencoder.train_with_callbacks(
        X_train,
        X_val=X_test,
        epochs=20,
        batch_size=32,
        patience=5
    )
    
    # Detect anomalies
    logger.info("Detecting anomalies...")
    anomalies, errors, threshold = autoencoder.detect_anomalies(X_test)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, roc_auc_score
    
    auc_score = roc_auc_score(y_test, errors)
    logger.info(f"AUC Score: {auc_score:.4f}")
    logger.info("\nClassification Report:")
    print(classification_report(y_test, anomalies))
    
    logger.info("=== GENERATION 1 IMPLEMENTATION COMPLETE ===")