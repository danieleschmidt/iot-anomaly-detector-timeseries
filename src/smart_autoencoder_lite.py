"""Smart lightweight autoencoder for Generation 1: MAKE IT WORK."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None

from .logging_config import get_logger
from .config import get_config


class SmartAutoencoderLite:
    """Lightweight LSTM Autoencoder for IoT Anomaly Detection - Generation 1."""
    
    def __init__(
        self,
        window_size: int = 30,
        encoding_dim: int = 16,
        learning_rate: float = 0.001,
        scaler_type: str = "standard",
        random_seed: int = 42
    ):
        """Initialize the lightweight autoencoder.
        
        Args:
            window_size: Size of the sliding window for sequences
            encoding_dim: Dimension of the encoding layer
            learning_rate: Learning rate for training
            scaler_type: Type of scaler ('standard' or 'minmax')
            random_seed: Random seed for reproducibility
        """
        self.window_size = window_size
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.scaler_type = scaler_type
        self.random_seed = random_seed
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        self.logger = get_logger(__name__)
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(random_seed)
        
        # Initialize scaler
        if scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
    
    def _create_model(self, input_shape: Tuple[int, int]) -> None:
        """Create the LSTM autoencoder model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required but not available")
        
        # Encoder
        encoder_inputs = keras.Input(shape=input_shape, name='encoder_input')
        encoder_lstm = layers.LSTM(
            self.encoding_dim, 
            return_state=True,
            name='encoder_lstm'
        )
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = layers.RepeatVector(
            input_shape[0], 
            name='decoder_repeat'
        )(encoder_outputs)
        decoder_lstm = layers.LSTM(
            self.encoding_dim, 
            return_sequences=True,
            name='decoder_lstm'
        )
        decoder_outputs = decoder_lstm(
            decoder_inputs,
            initial_state=encoder_states
        )
        
        # Output layer
        decoder_dense = layers.TimeDistributed(
            layers.Dense(input_shape[1], activation='linear'),
            name='decoder_output'
        )
        outputs = decoder_dense(decoder_outputs)
        
        # Create model
        self.model = keras.Model(encoder_inputs, outputs, name='smart_autoencoder_lite')
        
        # Compile with optimizer - using string names for compatibility
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        self.logger.info(f"Created LSTM autoencoder model with encoding_dim={self.encoding_dim}")
    
    def _prepare_sequences(self, data: np.ndarray) -> np.ndarray:
        """Prepare sliding window sequences from time series data."""
        sequences = []
        
        for i in range(len(data) - self.window_size + 1):
            sequence = data[i:(i + self.window_size)]
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def fit(
        self, 
        data: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """Train the autoencoder on normal data.
        
        Args:
            data: Training data (normal patterns only)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            verbose: Verbosity level for training
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Training autoencoder on {len(data)} samples")
        
        # Store feature names
        self.feature_names = data.columns.tolist()
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.values)
        
        # Create sequences
        sequences = self._prepare_sequences(scaled_data)
        self.logger.info(f"Created {len(sequences)} sequences of length {self.window_size}")
        
        # Create model if not exists
        if self.model is None:
            input_shape = (self.window_size, len(self.feature_names))
            self._create_model(input_shape)
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=verbose
        )
        
        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=verbose
        )
        
        callbacks = [early_stopping, reduce_lr]
        
        # Train the model
        history = self.model.fit(
            sequences, sequences,  # Autoencoder: input = output
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        self.logger.info("Training completed successfully")
        
        return history.history
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions and reconstruction errors.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Tuple of (reconstructed_sequences, reconstruction_errors)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Validate feature consistency
        if self.feature_names and data.columns.tolist() != self.feature_names:
            self.logger.warning("Feature names don't match training data")
        
        # Scale the data
        scaled_data = self.scaler.transform(data.values)
        
        # Create sequences
        sequences = self._prepare_sequences(scaled_data)
        
        if len(sequences) == 0:
            raise ValueError(f"Not enough data to create sequences. Need at least {self.window_size} samples")
        
        # Predict reconstructions
        reconstructions = self.model.predict(sequences, verbose=0)
        
        # Calculate reconstruction errors (MSE for each sequence)
        reconstruction_errors = np.mean((sequences - reconstructions) ** 2, axis=(1, 2))
        
        self.logger.info(f"Generated predictions for {len(sequences)} sequences")
        
        return reconstructions, reconstruction_errors
    
    def detect_anomalies(
        self, 
        data: pd.DataFrame,
        threshold: Optional[float] = None,
        contamination_rate: float = 0.1
    ) -> Tuple[np.ndarray, float]:
        """Detect anomalies based on reconstruction error.
        
        Args:
            data: Input data for anomaly detection
            threshold: Fixed threshold for anomaly detection
            contamination_rate: Expected contamination rate for threshold calculation
            
        Returns:
            Tuple of (anomaly_flags, threshold_used)
        """
        _, reconstruction_errors = self.predict(data)
        
        if threshold is None:
            # Calculate threshold based on contamination rate
            threshold = np.percentile(reconstruction_errors, (1 - contamination_rate) * 100)
            self.logger.info(f"Calculated threshold: {threshold:.6f}")
        
        # Detect anomalies
        anomalies = reconstruction_errors > threshold
        
        anomaly_count = np.sum(anomalies)
        self.logger.info(f"Detected {anomaly_count} anomalies out of {len(anomalies)} sequences")
        
        return anomalies, threshold
    
    def save(self, model_path: str, scaler_path: str) -> None:
        """Save the trained model and scaler.
        
        Args:
            model_path: Path to save the model
            scaler_path: Path to save the scaler
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directories if they don't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        
        # Save scaler and metadata
        scaler_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'window_size': self.window_size,
            'encoding_dim': self.encoding_dim,
            'scaler_type': self.scaler_type
        }
        joblib.dump(scaler_data, scaler_path)
        
        self.logger.info(f"Saved model to {model_path} and scaler to {scaler_path}")
    
    @classmethod
    def load(cls, model_path: str, scaler_path: str) -> 'SmartAutoencoderLite':
        """Load a trained model and scaler.
        
        Args:
            model_path: Path to the saved model
            scaler_path: Path to the saved scaler
            
        Returns:
            Loaded SmartAutoencoderLite instance
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required but not available")
        
        # Load scaler and metadata
        scaler_data = joblib.load(scaler_path)
        
        # Create instance with loaded parameters
        instance = cls(
            window_size=scaler_data['window_size'],
            encoding_dim=scaler_data['encoding_dim'],
            scaler_type=scaler_data['scaler_type']
        )
        
        # Load model
        instance.model = keras.models.load_model(model_path)
        instance.scaler = scaler_data['scaler']
        instance.feature_names = scaler_data['feature_names']
        instance.is_trained = True
        
        instance.logger.info(f"Loaded model from {model_path} and scaler from {scaler_path}")
        
        return instance
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not created yet"
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            self.model.summary()
            summary = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        
        return summary