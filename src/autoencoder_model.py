from tensorflow.keras import layers, models, losses, optimizers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Tuple, Optional, List
import logging


def build_autoencoder(
    input_shape: Tuple[int, int], 
    latent_dim: int = 16, 
    lstm_units: int = 32,
    dropout_rate: float = 0.1,
    learning_rate: float = 0.001,
    use_regularization: bool = True
) -> Model:
    """Build an enhanced LSTM autoencoder with regularization and improved architecture."""
    logging.info(f"Building autoencoder with input_shape={input_shape}, latent_dim={latent_dim}")
    
    inputs = layers.Input(shape=input_shape)
    
    # Encoder with improved architecture
    x = layers.LSTM(
        lstm_units, 
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        kernel_regularizer=regularizers.l2(0.01) if use_regularization else None
    )(inputs)
    x = layers.BatchNormalization()(x)
    
    encoded = layers.LSTM(
        latent_dim,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        kernel_regularizer=regularizers.l2(0.01) if use_regularization else None
    )(x)
    encoded = layers.BatchNormalization()(encoded)
    
    # Decoder with improved architecture
    x = layers.RepeatVector(input_shape[0])(encoded)
    x = layers.LSTM(
        latent_dim, 
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        kernel_regularizer=regularizers.l2(0.01) if use_regularization else None
    )(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.LSTM(
        lstm_units, 
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        kernel_regularizer=regularizers.l2(0.01) if use_regularization else None
    )(x)
    x = layers.BatchNormalization()(x)
    
    decoded = layers.TimeDistributed(
        layers.Dense(input_shape[1], activation='linear')
    )(x)
    
    model = models.Model(inputs, decoded)
    
    # Enhanced optimizer configuration
    optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer, 
        loss=losses.MeanSquaredError(),
        metrics=['mae']
    )
    
    logging.info(f"Autoencoder built successfully with {model.count_params()} parameters")
    return model


def get_training_callbacks(patience: int = 10, min_delta: float = 0.001) -> List:
    """Get standard training callbacks for improved training stability."""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-6,
            verbose=1
        )
    ]
