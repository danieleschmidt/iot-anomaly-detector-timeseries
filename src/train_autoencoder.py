from pathlib import Path
import argparse
import logging

from .data_preprocessor import DataPreprocessor
from .autoencoder_model import build_autoencoder
from .flexible_autoencoder import (
    FlexibleAutoencoderBuilder,
    create_autoencoder_from_config,
    load_architecture_config,
    get_predefined_architectures
)
from .config import get_config
from .model_metadata import save_model_with_metadata
from .training_callbacks import create_training_callbacks


def main(
    csv_path: str = "data/raw/sensor_data.csv",
    epochs: int | None = None,
    window_size: int | None = None,
    step: int = 1,
    latent_dim: int | None = None,
    lstm_units: int | None = None,
    scaler: str | None = None,
    model_path: str = "saved_models/autoencoder.h5",
    scaler_path: str | None = None,
    config_file: str | None = None,
    enable_progress: bool = True,
    enable_early_stopping: bool = True,
    architecture_config: str | None = None,
    architecture_name: str | None = None,
    use_flexible_architecture: bool = False,
) -> str:
    """Train the LSTM autoencoder and write it to ``model_path``.

    Parameters
    ----------
    csv_path : str
        Sensor data CSV file used for training.
    epochs : int, optional
        Number of training epochs. Uses config default if None.
    window_size : int, optional
        Length of each sliding window of time steps. Uses config default if None.
    step : int, optional
        Step size for the sliding window.
    latent_dim : int, optional
        Size of the latent representation. Uses config default if None.
    lstm_units : int, optional
        Units for the LSTM layers. Uses config default if None.
    scaler : str or None, optional
        Scaling method to use. ``"standard"`` selects ``StandardScaler``; any
        other value uses ``MinMaxScaler``.
    model_path : str, optional
        Destination path for saving the trained model. Parent directories are
        created automatically.
    scaler_path : str or None, optional
        If given, the fitted scaler is written to this file for reuse.
    config_file : str or None, optional
        Path to configuration file. If None, uses global config.
    enable_progress : bool, optional
        Whether to enable detailed progress indication during training.
    enable_early_stopping : bool, optional
        Whether to enable early stopping based on loss improvement.
    architecture_config : str, optional
        Path to JSON file containing flexible architecture configuration.
    architecture_name : str, optional
        Name of predefined architecture to use.
    use_flexible_architecture : bool, optional
        Whether to use flexible architecture system instead of legacy autoencoder.
    """
    # Load configuration
    config = get_config()
    if config_file:
        from .config import reload_config
        config = reload_config(config_file)
    
    # Use configuration values as defaults for None parameters
    epochs = epochs if epochs is not None else config.EPOCHS
    window_size = window_size if window_size is not None else config.WINDOW_SIZE
    latent_dim = latent_dim if latent_dim is not None else config.LATENT_DIM
    lstm_units = lstm_units if lstm_units is not None else config.LSTM_UNITS
    
    logging.info(f"Training with config: epochs={epochs}, window_size={window_size}, "
                f"latent_dim={latent_dim}, lstm_units={lstm_units}")
    
    if scaler == "standard":
        from sklearn.preprocessing import StandardScaler
        dp = DataPreprocessor(StandardScaler())
    else:
        dp = DataPreprocessor()

    windows = dp.load_and_preprocess(csv_path, window_size=window_size, step=step)
    
    # Determine which architecture system to use
    if use_flexible_architecture or architecture_config or architecture_name:
        model = _build_flexible_model(
            input_shape=(windows.shape[1], windows.shape[2]),
            architecture_config=architecture_config,
            architecture_name=architecture_name,
            latent_dim=latent_dim,
            lstm_units=lstm_units
        )
        model_type = "flexible"
    else:
        # Use legacy autoencoder
        model = build_autoencoder(
            (windows.shape[1], windows.shape[2]),
            latent_dim=latent_dim,
            lstm_units=lstm_units,
        )
        model_type = "legacy"
    
    batch_size = config.BATCH_SIZE
    
    # Create training callbacks for progress indication
    callbacks = []
    if enable_progress:
        callbacks = create_training_callbacks(
            epochs=epochs,
            enable_early_stopping=enable_early_stopping,
            early_stopping_patience=max(5, epochs // 10),  # Adaptive patience
            progress_log_frequency=max(1, epochs // 20) if epochs > 20 else 1,
            metrics_log_frequency=max(1, epochs // 10) if epochs > 10 else 5
        )
        logging.info(f"Training with {len(callbacks)} callbacks enabled")
    
    # Train the model with callbacks
    verbose_level = 1 if not enable_progress else 0  # Reduce TensorFlow verbosity when using custom progress
    history = model.fit(
        windows, windows, 
        epochs=epochs, 
        batch_size=batch_size, 
        verbose=verbose_level,
        callbacks=callbacks
    )
    
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_file)
    logging.info(f"Model saved to {model_file}")
    
    # Save model metadata
    training_params = {
        "epochs": epochs,
        "window_size": window_size,
        "step": step,
        "latent_dim": latent_dim,
        "lstm_units": lstm_units,
        "batch_size": batch_size,
        "scaler_type": "standard" if scaler == "standard" else "minmax",
        "model_type": model_type,
        "architecture_config": architecture_config,
        "architecture_name": architecture_name,
        "use_flexible_architecture": use_flexible_architecture
    }
    
    dataset_info = {
        "csv_path": csv_path,
        "num_windows": len(windows),
        "window_shape": windows.shape,
        "training_data_file": str(Path(csv_path).absolute())
    }
    
    # Extract final training loss from history
    final_loss = float(history.history['loss'][-1]) if history.history['loss'] else None
    performance_metrics = {"final_training_loss": final_loss} if final_loss else {}
    
    model_path_str, metadata_path = save_model_with_metadata(
        model_path=str(model_file),
        training_params=training_params,
        performance_metrics=performance_metrics,
        dataset_info=dataset_info,
        metadata_directory=str(model_file.parent)
    )
    
    logging.info(f"Model metadata saved to {metadata_path}")
    
    if scaler_path:
        dp.save(scaler_path)
        logging.info(f"Scaler saved to {scaler_path}")
    
    return str(model_file)


def _build_flexible_model(
    input_shape: tuple[int, int],
    architecture_config: str | None = None,
    architecture_name: str | None = None,
    latent_dim: int | None = None,
    lstm_units: int | None = None
):
    """Build model using flexible architecture system.
    
    Parameters
    ----------
    input_shape : tuple[int, int]
        Shape of input data (time_steps, features)
    architecture_config : str, optional
        Path to architecture configuration file
    architecture_name : str, optional
        Name of predefined architecture
    latent_dim : int, optional
        Override latent dimension
    lstm_units : int, optional
        Override LSTM units (for simple architectures)
        
    Returns
    -------
    Model
        Built Keras model
    """
    logger = logging.getLogger(__name__)
    
    if architecture_config:
        # Load from configuration file
        config = load_architecture_config(architecture_config)
        logger.info(f"Loaded architecture config from {architecture_config}")
        
        # Override input shape to match data
        config['input_shape'] = list(input_shape)
        
        # Apply parameter overrides
        if latent_dim is not None:
            config['latent_config']['dim'] = latent_dim
            
        return create_autoencoder_from_config(config)
        
    elif architecture_name:
        # Use predefined architecture
        predefined = get_predefined_architectures()
        if architecture_name not in predefined:
            available = list(predefined.keys())
            raise ValueError(f"Unknown architecture '{architecture_name}'. "
                           f"Available: {available}")
        
        config = predefined[architecture_name].copy()
        logger.info(f"Using predefined architecture: {architecture_name}")
        
        # Override input shape to match data
        config['input_shape'] = list(input_shape)
        
        # Apply parameter overrides
        if latent_dim is not None:
            config['latent_config']['dim'] = latent_dim
            
        return create_autoencoder_from_config(config)
        
    else:
        # Create simple flexible architecture equivalent to legacy
        builder = FlexibleAutoencoderBuilder(input_shape)
        builder.add_encoder_layer('lstm', units=lstm_units or 64, return_sequences=True)
        builder.add_encoder_layer('lstm', units=(lstm_units or 64) // 2, return_sequences=False)
        builder.set_latent_config(dim=latent_dim or 16)
        builder.set_compilation(optimizer='adam', loss='mse')
        
        logger.info("Using simple flexible architecture (legacy equivalent)")
        return builder.build()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM autoencoder")
    parser.add_argument("--csv-path", default="data/raw/sensor_data.csv")
    parser.add_argument("--epochs", type=int, help="Number of training epochs (overrides config)")
    parser.add_argument("--window-size", type=int, help="Window size (overrides config)")
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size for sliding windows",
    )
    parser.add_argument("--latent-dim", type=int, help="Latent dimension (overrides config)")
    parser.add_argument("--lstm-units", type=int, help="LSTM units (overrides config)")
    parser.add_argument(
        "--scaler",
        choices=["minmax", "standard"],
        default="minmax",
        help="Scaling method for preprocessing",
    )
    parser.add_argument(
        "--model-path",
        default="saved_models/autoencoder.h5",
        help="Where to store the trained autoencoder",
    )
    parser.add_argument(
        "--scaler-path",
        default=None,
        help="Optional path to save the fitted scaler",
    )
    parser.add_argument(
        "--config-file",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable detailed progress indication during training"
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Disable early stopping during training"
    )
    parser.add_argument(
        "--architecture-config",
        help="Path to JSON file with flexible architecture configuration"
    )
    parser.add_argument(
        "--architecture-name",
        help="Name of predefined architecture (simple_lstm, deep_lstm, hybrid_conv_lstm, gru_based, lightweight)"
    )
    parser.add_argument(
        "--use-flexible-architecture",
        action="store_true",
        help="Use flexible architecture system instead of legacy autoencoder"
    )
    parser.add_argument(
        "--list-architectures",
        action="store_true",
        help="List available predefined architectures and exit"
    )
    args = parser.parse_args()
    
    if args.list_architectures:
        print("Available predefined architectures:")
        for name, config in get_predefined_architectures().items():
            print(f"  {name}: {config.get('description', 'No description')}")
        import sys
        sys.exit(0)
    
    main(
        csv_path=args.csv_path,
        epochs=args.epochs,
        window_size=args.window_size,
        step=args.step,
        latent_dim=args.latent_dim,
        lstm_units=args.lstm_units,
        scaler=args.scaler,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        config_file=args.config_file,
        enable_progress=not args.disable_progress,
        enable_early_stopping=not args.disable_early_stopping,
        architecture_config=args.architecture_config,
        architecture_name=args.architecture_name,
        use_flexible_architecture=args.use_flexible_architecture,
    )
