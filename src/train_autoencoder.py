from pathlib import Path
import argparse

from .data_preprocessor import DataPreprocessor
from .autoencoder_model import build_autoencoder


def main(
    csv_path: str = "data/raw/sensor_data.csv",
    epochs: int = 3,
    window_size: int = 30,
    latent_dim: int = 16,
    lstm_units: int = 32,
    scaler: str | None = None,
    model_path: str = "saved_models/autoencoder.h5",
):
    """Train the LSTM autoencoder and write it to ``model_path``.

    Parameters
    ----------
    csv_path : str
        Sensor data CSV file used for training.
    epochs : int, optional
        Number of training epochs.
    window_size : int, optional
        Length of each sliding window of time steps.
    latent_dim : int, optional
        Size of the latent representation.
    lstm_units : int, optional
        Units for the LSTM layers.
    scaler : str or None, optional
        Scaling method to use. ``"standard"`` selects ``StandardScaler``; any
        other value uses ``MinMaxScaler``.
    model_path : str, optional
        Destination path for saving the trained model. Parent directories are
        created automatically.
    """
    if scaler == "standard":
        from sklearn.preprocessing import StandardScaler

        dp = DataPreprocessor(StandardScaler())
    else:
        dp = DataPreprocessor()

    windows = dp.load_and_preprocess(csv_path, window_size=window_size)
    model = build_autoencoder(
        (windows.shape[1], windows.shape[2]),
        latent_dim=latent_dim,
        lstm_units=lstm_units,
    )
    model.fit(windows, windows, epochs=epochs, batch_size=32, verbose=0)
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM autoencoder")
    parser.add_argument("--csv-path", default="data/raw/sensor_data.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--lstm-units", type=int, default=32)
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
    args = parser.parse_args()
    main(
        csv_path=args.csv_path,
        epochs=args.epochs,
        window_size=args.window_size,
        latent_dim=args.latent_dim,
        lstm_units=args.lstm_units,
        scaler=args.scaler,
        model_path=args.model_path,
    )
