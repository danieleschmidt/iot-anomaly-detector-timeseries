"""Generation 1 Pipeline: MAKE IT WORK - Simple and Reliable Anomaly Detection."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import joblib
import time
import warnings

from .smart_autoencoder_lite import SmartAutoencoderLite
from .data_preprocessor import DataPreprocessor
from .logging_config import get_logger, setup_logging
from .config import get_config


class Generation1Pipeline:
    """Simple, reliable pipeline for IoT anomaly detection - Generation 1."""
    
    def __init__(
        self,
        window_size: int = 30,
        encoding_dim: int = 16,
        contamination_rate: float = 0.1,
        enable_logging: bool = True
    ):
        """Initialize Generation 1 pipeline.
        
        Args:
            window_size: Size of sliding window for sequences
            encoding_dim: Encoding dimension for autoencoder
            contamination_rate: Expected contamination rate
            enable_logging: Whether to enable logging
        """
        self.window_size = window_size
        self.encoding_dim = encoding_dim
        self.contamination_rate = contamination_rate
        
        if enable_logging:
            setup_logging(level="INFO")
        
        self.logger = get_logger(__name__)
        self.autoencoder = None
        self.preprocessor = DataPreprocessor()
        self.is_trained = False
        
        self.logger.info("Initialized Generation 1 Pipeline")
    
    def prepare_data(
        self, 
        data_path: str,
        target_column: Optional[str] = None,
        validate_data: bool = True
    ) -> pd.DataFrame:
        """Prepare data for training/inference.
        
        Args:
            data_path: Path to the CSV data file
            target_column: Optional target column to exclude
            validate_data: Whether to perform data validation
            
        Returns:
            Prepared DataFrame
        """
        self.logger.info(f"Loading data from {data_path}")
        
        # Load data
        data = pd.read_csv(data_path)
        self.logger.info(f"Loaded {len(data)} samples with {len(data.columns)} features")
        
        # Remove target column if specified
        if target_column and target_column in data.columns:
            data = data.drop(columns=[target_column])
            self.logger.info(f"Removed target column: {target_column}")
        
        # Basic data validation
        if validate_data:
            self._validate_data(data)
        
        # Handle missing values (simple forward fill)
        if data.isnull().any().any():
            data = data.fillna(method='ffill').fillna(method='bfill')
            self.logger.warning("Handled missing values with forward/backward fill")
        
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Basic data validation."""
        # Check for empty data
        if len(data) == 0:
            raise ValueError("Data is empty")
        
        # Check for minimum samples
        if len(data) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} samples, got {len(data)}")
        
        # Check for non-numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) != len(data.columns):
            non_numeric = set(data.columns) - set(numeric_columns)
            self.logger.warning(f"Non-numeric columns detected: {non_numeric}")
            # Keep only numeric columns
            data = data[numeric_columns]
        
        self.logger.info("Data validation passed")
    
    def train(
        self,
        train_data_path: str,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        save_model: bool = True,
        model_dir: str = "saved_models"
    ) -> Dict[str, Any]:
        """Train the anomaly detection pipeline.
        
        Args:
            train_data_path: Path to training data (normal patterns)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation split ratio
            save_model: Whether to save the trained model
            model_dir: Directory to save models
            
        Returns:
            Training metrics and results
        """
        start_time = time.time()
        self.logger.info("Starting Generation 1 training pipeline")
        
        # Prepare training data
        train_data = self.prepare_data(train_data_path)
        
        # Initialize autoencoder
        self.autoencoder = SmartAutoencoderLite(
            window_size=self.window_size,
            encoding_dim=self.encoding_dim
        )
        
        # Train the model
        self.logger.info("Training autoencoder...")
        history = self.autoencoder.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        self.is_trained = True
        training_time = time.time() - start_time
        
        # Save model if requested
        if save_model:
            model_path = Path(model_dir)
            model_path.mkdir(parents=True, exist_ok=True)
            
            self.autoencoder.save(
                str(model_path / "generation1_autoencoder.h5"),
                str(model_path / "generation1_scaler.pkl")
            )
            
            # Save pipeline metadata
            metadata = {
                'window_size': self.window_size,
                'encoding_dim': self.encoding_dim,
                'contamination_rate': self.contamination_rate,
                'training_samples': len(train_data),
                'feature_count': len(train_data.columns),
                'training_time_seconds': training_time,
                'final_loss': history['loss'][-1] if 'loss' in history else None,
                'final_val_loss': history['val_loss'][-1] if 'val_loss' in history else None
            }
            
            joblib.dump(metadata, model_path / "generation1_metadata.pkl")
            self.logger.info(f"Saved model and metadata to {model_path}")
        
        # Prepare results
        results = {
            'training_time_seconds': training_time,
            'training_samples': len(train_data),
            'feature_count': len(train_data.columns),
            'epochs_completed': len(history['loss']) if 'loss' in history else epochs,
            'final_training_loss': history['loss'][-1] if 'loss' in history else None,
            'final_validation_loss': history['val_loss'][-1] if 'val_loss' in history else None,
            'history': history
        }
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Final training loss: {results['final_training_loss']:.6f}")
        self.logger.info(f"Final validation loss: {results['final_validation_loss']:.6f}")
        
        return results
    
    def detect_anomalies(
        self,
        data_path: str,
        threshold: Optional[float] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect anomalies in new data.
        
        Args:
            data_path: Path to data for anomaly detection
            threshold: Optional fixed threshold
            output_path: Optional path to save results
            
        Returns:
            Detection results and statistics
        """
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before anomaly detection")
        
        start_time = time.time()
        self.logger.info("Starting anomaly detection")
        
        # Prepare data
        data = self.prepare_data(data_path, validate_data=False)
        
        # Detect anomalies
        anomalies, threshold_used = self.autoencoder.detect_anomalies(
            data,
            threshold=threshold,
            contamination_rate=self.contamination_rate
        )
        
        # Get reconstruction errors for analysis
        _, reconstruction_errors = self.autoencoder.predict(data)
        
        detection_time = time.time() - start_time
        
        # Calculate statistics
        anomaly_count = np.sum(anomalies)
        anomaly_rate = anomaly_count / len(anomalies)
        
        results = {
            'total_sequences': len(anomalies),
            'anomaly_count': int(anomaly_count),
            'anomaly_rate': float(anomaly_rate),
            'threshold_used': float(threshold_used),
            'detection_time_seconds': detection_time,
            'mean_reconstruction_error': float(np.mean(reconstruction_errors)),
            'std_reconstruction_error': float(np.std(reconstruction_errors)),
            'max_reconstruction_error': float(np.max(reconstruction_errors)),
            'min_reconstruction_error': float(np.min(reconstruction_errors)),
            'anomaly_flags': anomalies.tolist(),
            'reconstruction_errors': reconstruction_errors.tolist()
        }
        
        # Save results if requested
        if output_path:
            output_df = pd.DataFrame({
                'sequence_index': range(len(anomalies)),
                'is_anomaly': anomalies.astype(int),
                'reconstruction_error': reconstruction_errors,
                'threshold': threshold_used
            })
            
            output_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved detection results to {output_path}")
        
        self.logger.info(f"Anomaly detection completed in {detection_time:.2f} seconds")
        self.logger.info(f"Detected {anomaly_count} anomalies ({anomaly_rate:.2%} rate)")
        self.logger.info(f"Threshold used: {threshold_used:.6f}")
        
        return results
    
    def evaluate_performance(
        self,
        test_data_path: str,
        labels_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate pipeline performance on test data.
        
        Args:
            test_data_path: Path to test data
            labels_path: Optional path to ground truth labels
            
        Returns:
            Performance metrics
        """
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before evaluation")
        
        self.logger.info("Evaluating pipeline performance")
        
        # Detect anomalies
        results = self.detect_anomalies(test_data_path)
        
        # If labels are provided, calculate classification metrics
        if labels_path:
            try:
                labels_df = pd.read_csv(labels_path)
                # Assume labels are in the first column or named 'label'
                if 'label' in labels_df.columns:
                    true_labels = labels_df['label'].values
                else:
                    true_labels = labels_df.iloc[:, 0].values
                
                # Align labels with predictions (handle sequence alignment)
                predicted_labels = np.array(results['anomaly_flags'])
                
                # Truncate to minimum length
                min_len = min(len(true_labels), len(predicted_labels))
                true_labels = true_labels[:min_len]
                predicted_labels = predicted_labels[:min_len]
                
                # Calculate classification metrics
                from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                
                precision = precision_score(true_labels, predicted_labels, zero_division=0)
                recall = recall_score(true_labels, predicted_labels, zero_division=0)
                f1 = f1_score(true_labels, predicted_labels, zero_division=0)
                accuracy = accuracy_score(true_labels, predicted_labels)
                
                results.update({
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'accuracy': float(accuracy),
                    'true_positive_rate': float(recall),
                    'false_positive_rate': float(1 - precision) if precision > 0 else 0.0
                })
                
                self.logger.info(f"Classification metrics: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Could not calculate classification metrics: {e}")
        
        return results
    
    @classmethod
    def load_trained_pipeline(
        cls,
        model_dir: str = "saved_models"
    ) -> 'Generation1Pipeline':
        """Load a previously trained pipeline.
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            Loaded pipeline instance
        """
        model_path = Path(model_dir)
        
        # Load metadata
        metadata = joblib.load(model_path / "generation1_metadata.pkl")
        
        # Create pipeline instance
        pipeline = cls(
            window_size=metadata['window_size'],
            encoding_dim=metadata['encoding_dim'],
            contamination_rate=metadata['contamination_rate']
        )
        
        # Load autoencoder
        pipeline.autoencoder = SmartAutoencoderLite.load(
            str(model_path / "generation1_autoencoder.h5"),
            str(model_path / "generation1_scaler.pkl")
        )
        
        pipeline.is_trained = True
        pipeline.logger.info(f"Loaded trained pipeline from {model_dir}")
        
        return pipeline
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'is_trained': self.is_trained,
            'window_size': self.window_size,
            'encoding_dim': self.encoding_dim,
            'contamination_rate': self.contamination_rate,
            'model_available': self.autoencoder is not None,
            'model_trained': self.autoencoder.is_trained if self.autoencoder else False
        }