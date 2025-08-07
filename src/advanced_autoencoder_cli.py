"""Advanced Autoencoder CLI for Next-Generation IoT Anomaly Detection.

Unified command-line interface for training and using advanced autoencoder architectures
including Transformer-based, Variational, and Quantum-Classical Hybrid models.
Part of Generation 3+ research implementation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from .transformer_autoencoder import (
    create_transformer_autoencoder,
    get_transformer_presets,
    TransformerAutoencoderBuilder
)
from .variational_autoencoder import (
    create_variational_autoencoder,
    get_vae_presets,
    VAEAutoencoderBuilder
)
from .quantum_hybrid_autoencoder import (
    create_quantum_hybrid_autoencoder,
    get_quantum_hybrid_presets,
    QuantumHybridAutoencoderBuilder
)
from .flexible_autoencoder import (
    create_autoencoder_from_config,
    get_predefined_architectures
)
from .data_preprocessor import DataPreprocessor
from .anomaly_detector import AnomalyDetector
from .logging_config import setup_logging, get_logger


class AdvancedAutoencoderManager:
    """Manager class for advanced autoencoder operations."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.supported_architectures = {
            'transformer': 'Transformer-based Autoencoder',
            'variational': 'Variational Autoencoder (VAE)',
            'quantum_hybrid': 'Quantum-Classical Hybrid Autoencoder',
            'classical': 'Classical Flexible Autoencoder'
        }
        
    def get_available_presets(self, architecture_type: str) -> Dict[str, Dict[str, Any]]:
        """Get available presets for specified architecture type."""
        if architecture_type == 'transformer':
            return get_transformer_presets()
        elif architecture_type == 'variational':
            return get_vae_presets()
        elif architecture_type == 'quantum_hybrid':
            return get_quantum_hybrid_presets()
        elif architecture_type == 'classical':
            return get_predefined_architectures()
        else:
            raise ValueError(f"Unsupported architecture type: {architecture_type}")
    
    def create_model(self, architecture_type: str, input_shape: Tuple[int, int],
                    preset: str = None, custom_config: Dict[str, Any] = None):
        """Create model based on architecture type and configuration."""
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required for model creation")
        
        self.logger.info(f"Creating {architecture_type} model with input shape {input_shape}")
        
        if architecture_type == 'transformer':
            if preset:
                return create_transformer_autoencoder(input_shape, preset, **(custom_config or {}))
            else:
                builder = TransformerAutoencoderBuilder(input_shape)
                if custom_config:
                    self._configure_transformer_builder(builder, custom_config)
                return builder.build()
                
        elif architecture_type == 'variational':
            if preset:
                return create_variational_autoencoder(input_shape, preset, **(custom_config or {}))
            else:
                builder = VAEAutoencoderBuilder(input_shape)
                if custom_config:
                    self._configure_vae_builder(builder, custom_config)
                return builder.build()
                
        elif architecture_type == 'quantum_hybrid':
            if preset:
                return create_quantum_hybrid_autoencoder(input_shape, preset, **(custom_config or {}))
            else:
                builder = QuantumHybridAutoencoderBuilder(input_shape)
                if custom_config:
                    self._configure_quantum_builder(builder, custom_config)
                return builder.build()
                
        elif architecture_type == 'classical':
            if preset:
                presets = get_predefined_architectures()
                if preset in presets:
                    config = presets[preset].copy()
                    config.update(custom_config or {})
                    return create_autoencoder_from_config(config)
            else:
                raise ValueError("Classical architectures require preset specification")
                
        else:
            raise ValueError(f"Unsupported architecture type: {architecture_type}")
    
    def _configure_transformer_builder(self, builder: TransformerAutoencoderBuilder, 
                                     config: Dict[str, Any]):
        """Configure transformer builder with custom settings."""
        if 'd_model' in config or 'latent_dim' in config:
            builder.set_model_dimensions(
                config.get('d_model', 256),
                config.get('latent_dim', 64)
            )
        
        if 'num_heads' in config or 'dff' in config:
            builder.set_attention_config(
                config.get('num_heads', 8),
                config.get('dff', 512)
            )
        
        if 'num_encoder_layers' in config or 'num_decoder_layers' in config:
            builder.set_architecture_depth(
                config.get('num_encoder_layers', 4),
                config.get('num_decoder_layers', 4)
            )
        
        if 'dropout_rate' in config or 'use_positional_encoding' in config:
            builder.set_regularization(
                config.get('dropout_rate', 0.1),
                config.get('use_positional_encoding', True)
            )
        
        if 'optimizer' in config or 'loss' in config or 'metrics' in config:
            builder.set_compilation(
                config.get('optimizer', 'adam'),
                config.get('loss', 'mse'),
                config.get('metrics')
            )
    
    def _configure_vae_builder(self, builder: VAEAutoencoderBuilder, config: Dict[str, Any]):
        """Configure VAE builder with custom settings."""
        if 'latent_dim' in config or 'beta' in config:
            builder.set_latent_config(
                config.get('latent_dim', 32),
                config.get('beta', 1.0)
            )
        
        if 'architecture_type' in config or 'hidden_units' in config:
            builder.set_architecture(
                config.get('architecture_type', 'lstm'),
                config.get('hidden_units')
            )
        
        if 'dropout_rate' in config or 'use_batch_norm' in config:
            builder.set_regularization(
                config.get('dropout_rate', 0.1),
                config.get('use_batch_norm', True)
            )
        
        if 'optimizer' in config or 'learning_rate' in config:
            builder.set_optimizer(
                config.get('optimizer', 'adam'),
                config.get('learning_rate', 1e-3)
            )
    
    def _configure_quantum_builder(self, builder: QuantumHybridAutoencoderBuilder, 
                                  config: Dict[str, Any]):
        """Configure quantum hybrid builder with custom settings."""
        quantum_config = config.get('quantum_config', {})
        if quantum_config:
            builder.set_quantum_config(**quantum_config)
        
        if 'classical_latent_dim' in config or 'quantum_latent_dim' in config:
            builder.set_latent_dimensions(
                config.get('classical_latent_dim', 64),
                config.get('quantum_latent_dim', 32)
            )
        
        if 'use_quantum_attention' in config or 'fusion_method' in config:
            builder.set_hybrid_options(
                config.get('use_quantum_attention', True),
                config.get('fusion_method', 'concatenate')
            )
        
        if 'optimizer' in config or 'learning_rate' in config or 'loss' in config:
            builder.set_training_config(
                config.get('optimizer', 'adam'),
                config.get('learning_rate', 1e-4),
                config.get('loss', 'mse')
            )
    
    def train_model(self, model, train_data: np.ndarray, val_data: np.ndarray = None,
                   epochs: int = 50, batch_size: int = 32, 
                   callbacks: List = None, verbose: int = 1):
        """Train the advanced autoencoder model."""
        self.logger.info(f"Training model for {epochs} epochs with batch size {batch_size}")
        
        # Prepare validation data
        validation_data = val_data if val_data is not None else None
        
        # Configure callbacks
        default_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        if callbacks:
            default_callbacks.extend(callbacks)
        
        # Train the model
        history = model.fit(
            train_data,
            train_data,  # Autoencoder target is input
            validation_data=(validation_data, validation_data) if validation_data is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=default_callbacks,
            verbose=verbose
        )
        
        self.logger.info("Training completed successfully")
        return history
    
    def evaluate_model(self, model, test_data: np.ndarray, 
                      architecture_type: str) -> Dict[str, Any]:
        """Evaluate model performance and extract insights."""
        self.logger.info(f"Evaluating {architecture_type} model")
        
        # Basic reconstruction metrics
        reconstructions = model.predict(test_data)
        mse_error = np.mean(np.square(test_data - reconstructions))
        mae_error = np.mean(np.abs(test_data - reconstructions))
        
        results = {
            'mse_error': float(mse_error),
            'mae_error': float(mae_error),
            'num_parameters': model.count_params()
        }
        
        # Architecture-specific evaluations
        if architecture_type == 'variational':
            if hasattr(model, 'get_reconstruction_error'):
                vae_metrics = model.get_reconstruction_error(test_data)
                results['uncertainty_metrics'] = {
                    'mean_uncertainty': float(tf.reduce_mean(vae_metrics['uncertainty'])),
                    'uncertainty_std': float(tf.reduce_std(vae_metrics['uncertainty']))
                }
        
        elif architecture_type == 'quantum_hybrid':
            if hasattr(model, 'compute_quantum_advantage_metric'):
                quantum_metrics = model.compute_quantum_advantage_metric(test_data)
                results['quantum_metrics'] = {
                    'quantum_advantage': float(quantum_metrics['quantum_advantage']),
                    'classical_variance': float(quantum_metrics['classical_variance']),
                    'quantum_variance': float(quantum_metrics['quantum_variance'])
                }
        
        elif architecture_type == 'transformer':
            if hasattr(model, 'get_latent_representation'):
                latent_repr = model.get_latent_representation(test_data)
                results['latent_analysis'] = {
                    'latent_dimension': latent_repr.shape[-1],
                    'latent_variance': float(tf.reduce_mean(tf.math.reduce_variance(latent_repr, axis=0)))
                }
        
        self.logger.info(f"Evaluation completed. MSE: {mse_error:.6f}, MAE: {mae_error:.6f}")
        return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Advanced Autoencoder CLI for IoT Anomaly Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available architectures and presets
  python -m src.advanced_autoencoder_cli list-presets --architecture transformer
  
  # Train transformer autoencoder with standard preset
  python -m src.advanced_autoencoder_cli train \\
    --architecture transformer \\
    --preset standard_transformer \\
    --data-path data/processed/train_data.npy \\
    --model-path models/transformer_model.h5 \\
    --epochs 100
  
  # Train VAE with custom configuration
  python -m src.advanced_autoencoder_cli train \\
    --architecture variational \\
    --config-file configs/custom_vae.json \\
    --data-path data/processed/train_data.npy \\
    --val-data-path data/processed/val_data.npy \\
    --model-path models/vae_model.h5
  
  # Train quantum-hybrid with minimal setup
  python -m src.advanced_autoencoder_cli train \\
    --architecture quantum_hybrid \\
    --preset lightweight_quantum \\
    --data-path data/processed/train_data.npy \\
    --model-path models/quantum_model.h5 \\
    --window-size 30 --features 5
  
  # Evaluate model performance
  python -m src.advanced_autoencoder_cli evaluate \\
    --architecture transformer \\
    --model-path models/transformer_model.h5 \\
    --test-data-path data/processed/test_data.npy \\
    --output-path results/evaluation.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List presets command
    list_parser = subparsers.add_parser('list-presets', help='List available presets')
    list_parser.add_argument('--architecture', choices=['transformer', 'variational', 'quantum_hybrid', 'classical'],
                           required=True, help='Architecture type')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train advanced autoencoder')
    train_parser.add_argument('--architecture', choices=['transformer', 'variational', 'quantum_hybrid', 'classical'],
                            required=True, help='Architecture type')
    train_parser.add_argument('--preset', help='Preset configuration name')
    train_parser.add_argument('--config-file', help='Custom configuration JSON file')
    train_parser.add_argument('--data-path', required=True, help='Training data file path')
    train_parser.add_argument('--val-data-path', help='Validation data file path')
    train_parser.add_argument('--model-path', required=True, help='Output model path')
    train_parser.add_argument('--window-size', type=int, default=30, help='Sequence window size')
    train_parser.add_argument('--features', type=int, default=3, help='Number of features')
    train_parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--verbose', type=int, default=1, help='Training verbosity')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--architecture', choices=['transformer', 'variational', 'quantum_hybrid', 'classical'],
                           required=True, help='Architecture type')
    eval_parser.add_argument('--model-path', required=True, help='Trained model path')
    eval_parser.add_argument('--test-data-path', required=True, help='Test data file path')
    eval_parser.add_argument('--output-path', help='Evaluation results output path')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple architectures')
    compare_parser.add_argument('--model-configs', required=True, nargs='+', 
                              help='List of model configuration files')
    compare_parser.add_argument('--test-data-path', required=True, help='Test data file path')
    compare_parser.add_argument('--output-path', help='Comparison results output path')
    
    # Logging arguments
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger = get_logger(__name__)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        manager = AdvancedAutoencoderManager()
        
        if args.command == 'list-presets':
            presets = manager.get_available_presets(args.architecture)
            print(f"\nAvailable presets for {args.architecture} architecture:")
            print("=" * 60)
            for preset_name, config in presets.items():
                print(f"\n{preset_name}:")
                print(f"  Name: {config.get('name', 'N/A')}")
                print(f"  Description: {config.get('description', 'N/A')}")
                if 'input_shape' in config:
                    print(f"  Default Input Shape: {config['input_shape']}")
        
        elif args.command == 'train':
            # Load or prepare training data
            if args.data_path.endswith('.npy'):
                train_data = np.load(args.data_path)
            else:
                logger.error(f"Unsupported data format: {args.data_path}")
                return
            
            # Prepare input shape
            if len(train_data.shape) == 2:
                # Add sequence dimension if missing
                input_shape = (args.window_size, args.features)
                logger.warning(f"Reshaping data to {input_shape}")
            else:
                input_shape = train_data.shape[1:]  # Exclude batch dimension
            
            # Load custom configuration if provided
            custom_config = {}
            if args.config_file:
                with open(args.config_file, 'r') as f:
                    custom_config = json.load(f)
            
            # Create model
            model = manager.create_model(
                args.architecture, 
                input_shape, 
                args.preset, 
                custom_config
            )
            
            if model is None:
                logger.error("Failed to create model")
                return
            
            # Load validation data if provided
            val_data = None
            if args.val_data_path:
                val_data = np.load(args.val_data_path)
            
            # Train model
            history = manager.train_model(
                model, train_data, val_data,
                args.epochs, args.batch_size, verbose=args.verbose
            )
            
            # Save model
            model_path = Path(args.model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.model_path)
            logger.info(f"Model saved to {args.model_path}")
            
            # Save training history
            history_path = model_path.with_suffix('.json')
            with open(history_path, 'w') as f:
                history_dict = {k: [float(v) for v in values] 
                              for k, values in history.history.items()}
                json.dump(history_dict, f, indent=2)
            logger.info(f"Training history saved to {history_path}")
        
        elif args.command == 'evaluate':
            # Load test data
            test_data = np.load(args.test_data_path)
            
            # Load model
            model = tf.keras.models.load_model(args.model_path, compile=False)
            
            # Evaluate model
            results = manager.evaluate_model(model, test_data, args.architecture)
            
            # Output results
            if args.output_path:
                output_path = Path(args.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Evaluation results saved to {args.output_path}")
            else:
                print("\nEvaluation Results:")
                print("=" * 40)
                for key, value in results.items():
                    if isinstance(value, dict):
                        print(f"{key}:")
                        for sub_key, sub_value in value.items():
                            print(f"  {sub_key}: {sub_value}")
                    else:
                        print(f"{key}: {value}")
        
        logger.info("Command completed successfully")
        
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()