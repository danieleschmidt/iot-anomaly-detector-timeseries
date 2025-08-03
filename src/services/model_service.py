"""
Model Service

Business logic for model training, versioning, and lifecycle management.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from ..train_autoencoder import train_model
from ..model_manager import ModelManager
from ..model_metadata import ModelMetadata
from ..data_preprocessor import DataPreprocessor
from ..flexible_autoencoder import FlexibleAutoencoder
from ..training_callbacks import get_callbacks

logger = logging.getLogger(__name__)


class ModelService:
    """
    Service for managing machine learning models.
    
    Handles training, versioning, deployment, and lifecycle management
    of autoencoder models for anomaly detection.
    """
    
    def __init__(
        self,
        model_dir: str = "saved_models",
        enable_versioning: bool = True,
        auto_backup: bool = True
    ):
        """
        Initialize the model service.
        
        Args:
            model_dir: Directory for storing models
            enable_versioning: Whether to enable model versioning
            auto_backup: Whether to automatically backup models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.enable_versioning = enable_versioning
        self.auto_backup = auto_backup
        self.model_manager = ModelManager(base_dir=model_dir)
        self.preprocessor = DataPreprocessor()
        self._training_history: Dict[str, Any] = {}
        
    def train_model(
        self,
        training_data: pd.DataFrame,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        validation_split: float = 0.2,
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train a new anomaly detection model.
        
        Args:
            training_data: Training data DataFrame
            model_config: Model architecture configuration
            training_config: Training hyperparameters
            validation_split: Validation data split ratio
            experiment_name: Name for this training experiment
            
        Returns:
            Training results including model version and metrics
        """
        start_time = datetime.now()
        
        # Set default configurations
        model_config = model_config or self._get_default_model_config()
        training_config = training_config or self._get_default_training_config()
        
        # Generate experiment name if not provided
        if not experiment_name:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting model training experiment: {experiment_name}")
        
        # Preprocess data
        logger.info("Preprocessing training data")
        scaled_data = self.preprocessor.fit_transform(training_data)
        
        # Create train/validation split
        split_idx = int(len(scaled_data) * (1 - validation_split))
        train_data = scaled_data[:split_idx]
        val_data = scaled_data[split_idx:]
        
        # Create windowed sequences
        window_size = model_config.get('window_size', 30)
        step_size = model_config.get('step_size', 1)
        
        X_train = self._create_sequences(train_data, window_size, step_size)
        X_val = self._create_sequences(val_data, window_size, step_size)
        
        logger.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
        
        # Build model
        model = self._build_model(
            input_shape=(window_size, train_data.shape[1]),
            **model_config
        )
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=training_config.get('learning_rate', 0.001)
            ),
            loss='mse',
            metrics=['mae']
        )
        
        # Set up callbacks
        callbacks = self._setup_callbacks(experiment_name, training_config)
        
        # Train model
        logger.info("Training model...")
        history = model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=training_config.get('epochs', 100),
            batch_size=training_config.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate final metrics
        train_loss = model.evaluate(X_train, X_train, verbose=0)
        val_loss = model.evaluate(X_val, X_val, verbose=0)
        
        # Generate model version
        model_version = self._generate_model_version(experiment_name)
        
        # Save model and metadata
        model_path = self.model_dir / f"model_{model_version}.h5"
        scaler_path = self.model_dir / f"scaler_{model_version}.pkl"
        
        model.save(model_path)
        self.preprocessor.save_scaler(scaler_path)
        
        # Create and save metadata
        metadata = {
            'version': model_version,
            'experiment_name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'model_config': model_config,
            'training_config': training_config,
            'input_shape': list(X_train.shape[1:]),
            'parameters': int(model.count_params()),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'final_train_loss': float(train_loss[0]) if isinstance(train_loss, list) else float(train_loss),
            'final_val_loss': float(val_loss[0]) if isinstance(val_loss, list) else float(val_loss),
            'training_time': (datetime.now() - start_time).total_seconds(),
            'data_stats': {
                'mean': train_data.mean().tolist() if hasattr(train_data, 'mean') else 0,
                'std': train_data.std().tolist() if hasattr(train_data, 'std') else 1
            }
        }
        
        # Save metadata
        metadata_path = self.model_dir / f"metadata_{model_version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Store training history
        self._training_history[model_version] = {
            'history': history.history,
            'metadata': metadata
        }
        
        # Register model
        if self.enable_versioning:
            self.model_manager.register_model(
                model_path=model_path,
                metadata=metadata,
                scaler_path=scaler_path
            )
        
        logger.info(f"Model training complete. Version: {model_version}")
        
        return {
            'model_version': model_version,
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'metadata': metadata,
            'training_history': history.history
        }
    
    def retrain_model(
        self,
        base_version: str,
        new_data: pd.DataFrame,
        fine_tune_epochs: int = 50
    ) -> Dict[str, Any]:
        """
        Retrain an existing model with new data.
        
        Args:
            base_version: Version of model to retrain
            new_data: New training data
            fine_tune_epochs: Number of epochs for fine-tuning
            
        Returns:
            Results from retraining
        """
        logger.info(f"Retraining model version: {base_version}")
        
        # Load existing model
        model_path, scaler_path = self.model_manager.get_model_paths(base_version)
        model = keras.models.load_model(model_path)
        
        # Load existing metadata
        metadata = self.model_manager.get_model_metadata(base_version)
        
        # Use existing preprocessor
        if scaler_path:
            self.preprocessor.load_scaler(scaler_path)
        
        # Preprocess new data
        scaled_data = self.preprocessor.transform(new_data)
        
        # Create sequences
        window_size = metadata.get('model_config', {}).get('window_size', 30)
        step_size = metadata.get('model_config', {}).get('step_size', 1)
        X_new = self._create_sequences(scaled_data, window_size, step_size)
        
        # Fine-tune model
        logger.info(f"Fine-tuning with {len(X_new)} new samples")
        history = model.fit(
            X_new, X_new,
            epochs=fine_tune_epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Save retrained model
        new_version = f"{base_version}_retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_model_path = self.model_dir / f"model_{new_version}.h5"
        model.save(new_model_path)
        
        # Update metadata
        new_metadata = metadata.copy()
        new_metadata.update({
            'version': new_version,
            'base_version': base_version,
            'retrained_at': datetime.now().isoformat(),
            'retraining_samples': len(X_new),
            'fine_tune_epochs': fine_tune_epochs
        })
        
        # Save updated metadata
        metadata_path = self.model_dir / f"metadata_{new_version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(new_metadata, f, indent=2)
        
        logger.info(f"Retraining complete. New version: {new_version}")
        
        return {
            'new_version': new_version,
            'base_version': base_version,
            'model_path': str(new_model_path),
            'retraining_history': history.history,
            'metadata': new_metadata
        }
    
    def compare_models(
        self,
        version1: str,
        version2: str,
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compare performance of two model versions.
        
        Args:
            version1: First model version
            version2: Second model version
            test_data: Test data for comparison
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing models: {version1} vs {version2}")
        
        results = {}
        
        for version in [version1, version2]:
            # Load model
            model_path, scaler_path = self.model_manager.get_model_paths(version)
            model = keras.models.load_model(model_path)
            
            # Load metadata
            metadata = self.model_manager.get_model_metadata(version)
            
            # Preprocess test data
            if scaler_path:
                self.preprocessor.load_scaler(scaler_path)
            scaled_data = self.preprocessor.transform(test_data)
            
            # Create sequences
            window_size = metadata.get('model_config', {}).get('window_size', 30)
            step_size = metadata.get('model_config', {}).get('step_size', 1)
            X_test = self._create_sequences(scaled_data, window_size, step_size)
            
            # Evaluate model
            loss = model.evaluate(X_test, X_test, verbose=0)
            predictions = model.predict(X_test)
            reconstruction_errors = np.mean(np.square(X_test - predictions), axis=(1, 2))
            
            results[version] = {
                'loss': float(loss[0]) if isinstance(loss, list) else float(loss),
                'mean_reconstruction_error': float(np.mean(reconstruction_errors)),
                'std_reconstruction_error': float(np.std(reconstruction_errors)),
                'min_error': float(np.min(reconstruction_errors)),
                'max_error': float(np.max(reconstruction_errors)),
                'percentile_95': float(np.percentile(reconstruction_errors, 95)),
                'model_size_mb': model_path.stat().st_size / (1024 * 1024),
                'parameters': metadata.get('parameters', 0)
            }
        
        # Calculate relative differences
        comparison = {
            'loss_improvement': (
                (results[version1]['loss'] - results[version2]['loss']) / 
                results[version1]['loss'] * 100
            ),
            'error_improvement': (
                (results[version1]['mean_reconstruction_error'] - 
                 results[version2]['mean_reconstruction_error']) / 
                results[version1]['mean_reconstruction_error'] * 100
            ),
            'size_difference_mb': (
                results[version2]['model_size_mb'] - 
                results[version1]['model_size_mb']
            )
        }
        
        return {
            'version1': version1,
            'version2': version2,
            'results': results,
            'comparison': comparison,
            'recommendation': self._get_model_recommendation(results, comparison)
        }
    
    def deploy_model(
        self,
        version: str,
        deployment_target: str = 'production'
    ) -> Dict[str, Any]:
        """
        Deploy a model version to a target environment.
        
        Args:
            version: Model version to deploy
            deployment_target: Target environment (production, staging, etc.)
            
        Returns:
            Deployment status and information
        """
        logger.info(f"Deploying model {version} to {deployment_target}")
        
        # Verify model exists
        model_path, scaler_path = self.model_manager.get_model_paths(version)
        if not model_path.exists():
            raise ValueError(f"Model version {version} not found")
        
        # Load and validate model
        try:
            model = keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully: {model.count_params()} parameters")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Create deployment package
        deployment_info = {
            'version': version,
            'deployment_target': deployment_target,
            'deployed_at': datetime.now().isoformat(),
            'model_path': str(model_path),
            'scaler_path': str(scaler_path) if scaler_path else None,
            'status': 'deployed',
            'health_check': 'passed'
        }
        
        # Save deployment info
        deployment_path = self.model_dir / f"deployment_{deployment_target}.json"
        with open(deployment_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        # Update model manager
        self.model_manager.set_active_model(version, deployment_target)
        
        logger.info(f"Model {version} successfully deployed to {deployment_target}")
        
        return deployment_info
    
    def rollback_model(
        self,
        deployment_target: str = 'production'
    ) -> Dict[str, Any]:
        """
        Rollback to previous model version.
        
        Args:
            deployment_target: Target environment to rollback
            
        Returns:
            Rollback status
        """
        logger.info(f"Rolling back model in {deployment_target}")
        
        # Get deployment history
        deployment_path = self.model_dir / f"deployment_{deployment_target}.json"
        if not deployment_path.exists():
            raise ValueError(f"No deployment found for {deployment_target}")
        
        with open(deployment_path, 'r') as f:
            current_deployment = json.load(f)
        
        # Get previous version (simplified - in production would maintain history)
        previous_version = self.model_manager.get_previous_version(
            current_deployment['version']
        )
        
        if not previous_version:
            raise ValueError("No previous version available for rollback")
        
        # Deploy previous version
        return self.deploy_model(previous_version, deployment_target)
    
    def get_model_metrics(
        self,
        version: str,
        test_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive metrics for a model version.
        
        Args:
            version: Model version
            test_data: Optional test data for evaluation
            
        Returns:
            Model metrics and statistics
        """
        # Load metadata
        metadata = self.model_manager.get_model_metadata(version)
        
        metrics = {
            'version': version,
            'created_at': metadata.get('created_at'),
            'training_samples': metadata.get('training_samples'),
            'validation_samples': metadata.get('validation_samples'),
            'parameters': metadata.get('parameters'),
            'final_train_loss': metadata.get('final_train_loss'),
            'final_val_loss': metadata.get('final_val_loss'),
            'training_time': metadata.get('training_time')
        }
        
        # Add test metrics if data provided
        if test_data is not None:
            model_path, scaler_path = self.model_manager.get_model_paths(version)
            model = keras.models.load_model(model_path)
            
            if scaler_path:
                self.preprocessor.load_scaler(scaler_path)
            scaled_data = self.preprocessor.transform(test_data)
            
            window_size = metadata.get('model_config', {}).get('window_size', 30)
            X_test = self._create_sequences(scaled_data, window_size, 1)
            
            test_loss = model.evaluate(X_test, X_test, verbose=0)
            metrics['test_loss'] = float(test_loss[0]) if isinstance(test_loss, list) else float(test_loss)
        
        return metrics
    
    def cleanup_old_models(
        self,
        keep_latest: int = 5,
        keep_deployed: bool = True
    ) -> Dict[str, Any]:
        """
        Clean up old model versions to save space.
        
        Args:
            keep_latest: Number of latest versions to keep
            keep_deployed: Whether to keep deployed models
            
        Returns:
            Cleanup summary
        """
        logger.info(f"Cleaning up old models (keeping {keep_latest} latest)")
        
        # Get all model versions
        all_versions = self.model_manager.list_models()
        
        # Sort by creation date
        versions_with_dates = []
        for version in all_versions:
            metadata = self.model_manager.get_model_metadata(version)
            versions_with_dates.append((
                version,
                datetime.fromisoformat(metadata.get('created_at', '2000-01-01'))
            ))
        
        versions_with_dates.sort(key=lambda x: x[1], reverse=True)
        
        # Determine which to keep
        to_keep = set([v[0] for v in versions_with_dates[:keep_latest]])
        
        # Add deployed models if requested
        if keep_deployed:
            for target in ['production', 'staging']:
                deployment_path = self.model_dir / f"deployment_{target}.json"
                if deployment_path.exists():
                    with open(deployment_path, 'r') as f:
                        deployment = json.load(f)
                        to_keep.add(deployment['version'])
        
        # Delete old models
        deleted = []
        for version, _ in versions_with_dates:
            if version not in to_keep:
                model_path = self.model_dir / f"model_{version}.h5"
                scaler_path = self.model_dir / f"scaler_{version}.pkl"
                metadata_path = self.model_dir / f"metadata_{version}.json"
                
                for path in [model_path, scaler_path, metadata_path]:
                    if path.exists():
                        path.unlink()
                
                deleted.append(version)
                logger.info(f"Deleted model version: {version}")
        
        return {
            'total_models': len(all_versions),
            'kept_models': len(to_keep),
            'deleted_models': len(deleted),
            'deleted_versions': deleted,
            'space_freed_mb': sum([
                (self.model_dir / f"model_{v}.h5").stat().st_size / (1024 * 1024)
                for v in deleted
                if (self.model_dir / f"model_{v}.h5").exists()
            ])
        }
    
    def _build_model(
        self,
        input_shape: Tuple[int, int],
        latent_dim: int = 32,
        encoder_units: List[int] = None,
        decoder_units: List[int] = None,
        dropout_rate: float = 0.2,
        **kwargs
    ) -> keras.Model:
        """Build autoencoder model."""
        encoder_units = encoder_units or [128, 64]
        decoder_units = decoder_units or [64, 128]
        
        # Encoder
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        for units in encoder_units:
            x = keras.layers.LSTM(
                units,
                return_sequences=True,
                dropout=dropout_rate
            )(x)
        
        # Latent representation
        x = keras.layers.LSTM(latent_dim, return_sequences=True)(x)
        
        # Decoder
        for units in decoder_units:
            x = keras.layers.LSTM(
                units,
                return_sequences=True,
                dropout=dropout_rate
            )(x)
        
        # Output layer
        outputs = keras.layers.TimeDistributed(
            keras.layers.Dense(input_shape[1])
        )(x)
        
        return keras.Model(inputs, outputs)
    
    def _create_sequences(
        self,
        data: np.ndarray,
        window_size: int,
        step_size: int
    ) -> np.ndarray:
        """Create sequences for training."""
        sequences = []
        for i in range(0, len(data) - window_size + 1, step_size):
            sequences.append(data[i:i + window_size])
        return np.array(sequences)
    
    def _setup_callbacks(
        self,
        experiment_name: str,
        training_config: Dict[str, Any]
    ) -> List[keras.callbacks.Callback]:
        """Set up training callbacks."""
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = self.model_dir / f"checkpoint_{experiment_name}.h5"
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ))
        
        # Early stopping
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=training_config.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=1
        ))
        
        # Reduce learning rate
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ))
        
        return callbacks
    
    def _generate_model_version(self, experiment_name: str) -> str:
        """Generate unique model version."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hash_input = f"{experiment_name}_{timestamp}"
        version_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"v{timestamp}_{version_hash}"
    
    def _get_default_model_config(self) -> Dict[str, Any]:
        """Get default model configuration."""
        return {
            'window_size': 30,
            'step_size': 1,
            'latent_dim': 32,
            'encoder_units': [128, 64],
            'decoder_units': [64, 128],
            'dropout_rate': 0.2
        }
    
    def _get_default_training_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'early_stopping_patience': 10
        }
    
    def _get_model_recommendation(
        self,
        results: Dict[str, Any],
        comparison: Dict[str, Any]
    ) -> str:
        """Generate model recommendation based on comparison."""
        if comparison['loss_improvement'] > 5 and comparison['error_improvement'] > 5:
            return "version2 shows significant improvement"
        elif comparison['loss_improvement'] < -5:
            return "version1 performs better"
        else:
            return "Both models show similar performance"