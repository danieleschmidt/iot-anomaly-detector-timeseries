"""
Basic IoT Anomaly Detection Pipeline
Generation 1: Simple, functional implementation
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import joblib

from .data_preprocessor import DataPreprocessor
from .autoencoder_model import build_autoencoder, get_training_callbacks
from .anomaly_detector import AnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicAnomalyPipeline:
    """Simple, end-to-end anomaly detection pipeline for IoT time series data."""
    
    def __init__(
        self,
        window_size: int = 30,
        latent_dim: int = 16,
        lstm_units: int = 32,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ):
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        self.preprocessor = None
        self.model = None
        self.detector = None
        
        logger.info("BasicAnomalyPipeline initialized")
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load time series data from CSV file."""
        logger.info(f"Loading data from {data_path}")
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess and create windowed sequences from time series data."""
        logger.info("Preparing data for training")
        
        # Initialize preprocessor if not exists
        if self.preprocessor is None:
            self.preprocessor = DataPreprocessor()
        
        # Scale the data
        scaled_data = self.preprocessor.fit_transform(df)
        
        # Create windowed sequences
        sequences = []
        for i in range(len(scaled_data) - self.window_size + 1):
            sequences.append(scaled_data[i:i + self.window_size])
        
        sequences = np.array(sequences)
        logger.info(f"Created {len(sequences)} sequences of shape {sequences.shape}")
        
        # For autoencoder, input and output are the same
        return sequences, sequences
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build the autoencoder model."""
        logger.info(f"Building model with input shape {input_shape}")
        
        self.model = build_autoencoder(
            input_shape=input_shape,
            latent_dim=self.latent_dim,
            lstm_units=self.lstm_units
        )
        
        logger.info("Model architecture:")
        self.model.summary(print_fn=logger.info)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the autoencoder model."""
        logger.info("Starting model training")
        
        if self.model is None:
            self.build_model(input_shape=X.shape[1:])
        
        # Get training callbacks
        callbacks = get_training_callbacks()
        
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log training results
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        logger.info(f"Training completed - Final loss: {final_loss:.4f}, Val loss: {final_val_loss:.4f}")
        
        return history.history
    
    def detect_anomalies(self, df: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Detect anomalies in the provided data."""
        logger.info("Detecting anomalies")
        
        if self.model is None:
            raise ValueError("Model must be trained before detecting anomalies")
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted before detecting anomalies")
        
        # Initialize detector if not exists
        if self.detector is None:
            self.detector = AnomalyDetector(model=self.model, scaler=self.preprocessor.scaler)
        
        # Prepare data for detection
        scaled_data = self.preprocessor.transform(df)
        
        # Create windowed sequences
        sequences = []
        for i in range(len(scaled_data) - self.window_size + 1):
            sequences.append(scaled_data[i:i + self.window_size])
        
        if not sequences:
            logger.warning("No sequences created for anomaly detection")
            return np.array([])
        
        sequences = np.array(sequences)
        
        # Calculate reconstruction errors
        reconstructions = self.model.predict(sequences, batch_size=self.batch_size, verbose=0)
        errors = np.mean((sequences - reconstructions) ** 2, axis=(1, 2))
        
        # Set threshold if not provided
        if threshold is None:
            threshold = np.percentile(errors, 95)
            logger.info(f"Using automatic threshold: {threshold:.4f}")
        else:
            logger.info(f"Using provided threshold: {threshold:.4f}")
        
        # Detect anomalies
        anomalies = (errors > threshold).astype(int)
        anomaly_count = np.sum(anomalies)
        
        logger.info(f"Detected {anomaly_count} anomalies out of {len(anomalies)} sequences")
        
        return anomalies
    
    def save_models(self, model_path: str, scaler_path: str) -> None:
        """Save trained model and preprocessor."""
        logger.info(f"Saving model to {model_path} and scaler to {scaler_path}")
        
        # Create directories if they don't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.model is not None:
            self.model.save(model_path)
            logger.info("Model saved successfully")
        else:
            logger.warning("No model to save")
        
        # Save preprocessor
        if self.preprocessor is not None:
            self.preprocessor.save(scaler_path)
            logger.info("Preprocessor saved successfully")
        else:
            logger.warning("No preprocessor to save")
    
    def load_models(self, model_path: str, scaler_path: str) -> None:
        """Load trained model and preprocessor."""
        logger.info(f"Loading model from {model_path} and scaler from {scaler_path}")
        
        # Load preprocessor
        self.preprocessor = DataPreprocessor.load(scaler_path)
        
        # Load model
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)
        
        logger.info("Models loaded successfully")
    
    def run_complete_pipeline(
        self, 
        data_path: str, 
        model_output_path: str = "saved_models/basic_autoencoder.h5",
        scaler_output_path: str = "saved_models/basic_scaler.pkl",
        threshold: float = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run the complete anomaly detection pipeline from data to results."""
        logger.info("Starting complete pipeline execution")
        
        # Load data
        df = self.load_data(data_path)
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Train model
        training_history = self.train(X, y)
        
        # Save trained models
        self.save_models(model_output_path, scaler_output_path)
        
        # Detect anomalies on the same data
        anomalies = self.detect_anomalies(df, threshold=threshold)
        
        results = {
            'training_history': training_history,
            'anomaly_count': np.sum(anomalies),
            'total_sequences': len(anomalies),
            'anomaly_percentage': (np.sum(anomalies) / len(anomalies) * 100) if len(anomalies) > 0 else 0
        }
        
        logger.info("Pipeline execution completed successfully")
        logger.info(f"Results: {results['anomaly_count']} anomalies detected ({results['anomaly_percentage']:.2f}%)")
        
        return anomalies, results


def main():
    """Example usage of the basic pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Basic IoT Anomaly Detection Pipeline")
    parser.add_argument("--data-path", required=True, help="Path to CSV data file")
    parser.add_argument("--model-path", default="saved_models/basic_autoencoder.h5", help="Model output path")
    parser.add_argument("--scaler-path", default="saved_models/basic_scaler.pkl", help="Scaler output path")
    parser.add_argument("--window-size", type=int, default=30, help="Sequence window size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--threshold", type=float, help="Anomaly detection threshold")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = BasicAnomalyPipeline(
        window_size=args.window_size,
        epochs=args.epochs
    )
    
    # Run complete pipeline
    anomalies, results = pipeline.run_complete_pipeline(
        data_path=args.data_path,
        model_output_path=args.model_path,
        scaler_output_path=args.scaler_path,
        threshold=args.threshold
    )
    
    print(f"\nPipeline Results:")
    print(f"Total sequences: {results['total_sequences']}")
    print(f"Anomalies detected: {results['anomaly_count']}")
    print(f"Anomaly percentage: {results['anomaly_percentage']:.2f}%")


if __name__ == "__main__":
    main()