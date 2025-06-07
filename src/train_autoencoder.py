import os
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Assuming src is in PYTHONPATH or script is run from project root
from data_preprocessor import TimeSeriesPreprocessor
from autoencoder_model import build_lstm_autoencoder

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
DATA_FILE_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'sensor_data.csv')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
MODEL_FILE_PATH = os.path.join(SAVED_MODELS_DIR, 'lstm_autoencoder.keras')
PREPROCESSOR_FILE_PATH = os.path.join(SAVED_MODELS_DIR, 'preprocessor.pkl')

WINDOW_SIZE = 5 # Adjusted to be less than the number of normal data points
FEATURE_COLS = ['sensor1', 'sensor2', 'sensor3']
LSTM_ENCODER_UNITS = [64, 32]
LSTM_DECODER_UNITS = [32, 64]
DENSE_ACTIVATION = 'sigmoid' # Sigmoid for output scaled to [0,1] by MinMaxScaler

EPOCHS = 10 # For dummy data, a small number is fine
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.1 # Use 10% of training data for validation

def train_model():
    """
    Loads data, preprocesses it, builds, trains, and saves an LSTM autoencoder model
    and the data preprocessor.
    """
    print("Starting training process...")

    # --- Ensure directories exist ---
    if not os.path.exists(SAVED_MODELS_DIR):
        os.makedirs(SAVED_MODELS_DIR)
        print(f"Created directory: {SAVED_MODELS_DIR}")

    # --- Load Data ---
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        print(f"Successfully loaded data from {DATA_FILE_PATH}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE_PATH}")
        return

    # --- Filter Normal Data for Training ---
    if 'anomaly_label' in df.columns:
        df_normal = df[df['anomaly_label'] == 'normal'].copy()
        if df_normal.empty:
            print("Error: No 'normal' data available for training after filtering.")
            return
        print(f"Filtered normal data for training. Shape: {df_normal.shape}")
    else:
        print("Warning: 'anomaly_label' column not found. Using all data for training.")
        df_normal = df.copy()

    if len(df_normal) < WINDOW_SIZE:
        print(f"Error: Not enough normal data points ({len(df_normal)}) to create even one sequence with window size {WINDOW_SIZE}.")
        return

    # --- Initialize and Use Preprocessor ---
    preprocessor = TimeSeriesPreprocessor(window_size=WINDOW_SIZE, feature_cols=FEATURE_COLS)

    try:
        df_scaled = preprocessor.fit_transform_scale(df_normal)
        print("Data scaled successfully.")
    except ValueError as e:
        print(f"Error during data scaling: {e}")
        return

    X_train = preprocessor.create_sequences(df_scaled)
    if X_train.shape[0] == 0:
        print(f"Error: No sequences created from the data. X_train is empty. Check data length and window size.")
        print(f"Scaled data shape: {df_scaled.shape}, Window size: {WINDOW_SIZE}")
        return
    print(f"Sequences created successfully. X_train shape: {X_train.shape}")

    # --- Build Model ---
    input_shape = X_train.shape[1:] # (WINDOW_SIZE, n_features)

    autoencoder = build_lstm_autoencoder(
        input_shape=input_shape,
        lstm_units_encoder=LSTM_ENCODER_UNITS,
        lstm_units_decoder=LSTM_DECODER_UNITS,
        dense_activation=DENSE_ACTIVATION
    )
    print("LSTM Autoencoder model built successfully.")
    autoencoder.summary()

    # --- Train Model ---
    print("Starting model training...")
    history = autoencoder.fit(
        X_train,
        X_train, # Autoencoder learns to reconstruct its input
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        shuffle=True, # Good practice to shuffle training data
        verbose=1 # To log training progress
    )
    print("Model training completed.")
    print("Training MAE history:", history.history['loss'])
    if 'val_loss' in history.history:
        print("Validation MAE history:", history.history['val_loss'])


    # --- Save Artifacts ---
    # Save Keras model
    try:
        autoencoder.save(MODEL_FILE_PATH)
        print(f"Trained Keras model saved to {MODEL_FILE_PATH}")
    except Exception as e:
        print(f"Error saving Keras model: {e}")

    # Save TimeSeriesPreprocessor instance
    try:
        with open(PREPROCESSOR_FILE_PATH, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"TimeSeriesPreprocessor saved to {PREPROCESSOR_FILE_PATH}")
    except Exception as e:
        print(f"Error saving TimeSeriesPreprocessor: {e}")

if __name__ == '__main__':
    # Set random seeds for reproducibility (optional, but good for consistent results)
    tf.random.set_seed(42)
    np.random.seed(42)

    train_model()
