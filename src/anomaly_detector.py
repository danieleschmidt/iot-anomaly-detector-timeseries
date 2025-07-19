"""Anomaly detection utilities using a trained autoencoder."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from .data_preprocessor import DataPreprocessor
from .logging_config import get_logger


class AnomalyDetector:
    """Compute anomaly scores and predictions using an autoencoder."""

    def __init__(
        self,
        model_path: str = "saved_models/autoencoder.h5",
        scaler_path: str | None = None,
    ) -> None:
        """Load a trained model from ``model_path`` and optional scaler."""
        self.logger = get_logger(__name__)
        
        try:
            self.logger.info(f"Loading autoencoder model from {model_path}")
            self.model = load_model(model_path)
            self.logger.info(f"Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            raise ValueError(f"Unable to load model from {model_path}: {e}") from e
        
        if scaler_path and Path(scaler_path).exists():
            self.logger.info(f"Loading scaler from {scaler_path}")
            self.preprocessor = DataPreprocessor.load(scaler_path)
        else:
            if scaler_path:
                self.logger.warning(f"Scaler path {scaler_path} not found, using default MinMaxScaler")
            else:
                self.logger.info("No scaler path provided, using default MinMaxScaler")
            self.preprocessor = DataPreprocessor()

    def score(self, sequences: np.ndarray) -> np.ndarray:
        """Compute reconstruction error scores for sequences."""
        self.logger.debug(f"Computing reconstruction scores for {len(sequences)} sequences")
        reconstructed = self.model.predict(sequences, verbose=0)
        scores = np.mean(np.square(sequences - reconstructed), axis=(1, 2))
        self.logger.debug(f"Score statistics: mean={scores.mean():.4f}, std={scores.std():.4f}")
        return scores

    def predict(
        self,
        csv_path: str,
        window_size: int = 30,
        step: int = 1,
        threshold: float | None = None,
        quantile: float | None = None,
    ) -> np.ndarray:
        """Return a boolean array indicating anomalous windows.

        Parameters
        ----------
        csv_path : str
            Source CSV containing the sensor data.
        window_size : int, optional
            Length of each sliding window.
        step : int, optional
            Step size for the sliding window.
        threshold : float or None, optional
            Manual threshold for anomaly detection. If ``None`` and ``quantile``
            is also ``None``, a threshold of ``mean + 3 * std`` is used.
        quantile : float or None, optional
            Quantile of the reconstruction error distribution used to derive the
            threshold. Must be between 0 and 1 (exclusive) and is mutually
            exclusive with ``threshold``.
        """
        self.logger.info(f"Predicting anomalies for {csv_path}")
        windows = self.preprocessor.load_and_preprocess(csv_path, window_size, step)
        scores = self.score(windows)
        
        if threshold is not None and quantile is not None:
            raise ValueError("Provide either threshold or quantile, not both")

        if quantile is not None:
            if not 0 < quantile < 1:
                raise ValueError("quantile must be between 0 and 1 (exclusive)")

        if threshold is None:
            if quantile is not None:
                threshold = float(np.quantile(scores, quantile))
                self.logger.info(f"Using quantile-based threshold: {threshold:.4f} (quantile={quantile})")
            else:
                threshold = scores.mean() + 3 * scores.std()
                self.logger.info(f"Using statistical threshold: {threshold:.4f} (mean + 3*std)")
        else:
            self.logger.info(f"Using manual threshold: {threshold:.4f}")
        
        predictions = scores > threshold
        anomaly_count = predictions.sum()
        self.logger.info(f"Detected {anomaly_count} anomalous windows out of {len(predictions)} total")
        
        return predictions


def main(
    csv_path: str = "data/raw/sensor_data.csv",
    model_path: str = "saved_models/autoencoder.h5",
    scaler_path: str | None = None,
    window_size: int = 30,
    step: int = 1,
    threshold: float | None = None,
    quantile: float | None = None,
    output_path: str = "predictions.csv",
) -> None:
    """Run anomaly detection on ``csv_path`` and write flags to ``output_path``.

    Parameters
    ----------
    csv_path : str, optional
        Path to the CSV file containing the sensor data.
    model_path : str, optional
        Path to the trained autoencoder model.
    window_size : int, optional
        Length of each sliding window.
    scaler_path : str or None, optional
        Path to a saved scaler fitted during training.
    step : int, optional
        Step size for the sliding window.
    threshold : float or None, optional
        Manual threshold for anomaly detection.
    quantile : float or None, optional
        Quantile-based threshold to use instead of ``threshold``. The value must
        be between 0 and 1 (exclusive) and cannot be combined with
        ``threshold``.
    output_path : str, optional
        Where to write the anomaly flags as a CSV file.
    """
    logger = get_logger(__name__)
    logger.info(f"Starting anomaly detection: input={csv_path}, output={output_path}")
    
    detector = AnomalyDetector(model_path, scaler_path)
    if quantile is not None and not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1 (exclusive)")

    preds = detector.predict(csv_path, window_size, step, threshold, quantile)
    pd.Series(preds.astype(int)).to_csv(output_path, index=False, header=False)
    logger.info(f"Anomaly predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect anomalies in a dataset")
    parser.add_argument("--csv-path", default="data/raw/sensor_data.csv")
    parser.add_argument("--model-path", default="saved_models/autoencoder.h5")
    parser.add_argument(
        "--scaler-path",
        default=None,
        help="Path to a saved scaler fitted during training",
    )
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size for sliding windows",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--threshold",
        type=float,
        help="Manual threshold for anomaly detection",
    )
    group.add_argument(
        "--quantile",
        type=float,
        help=(
            "Quantile of reconstruction error used for the threshold. "
            "Must be between 0 and 1 (exclusive)."
        ),
    )
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        window_size=args.window_size,
        step=args.step,
        threshold=args.threshold,
        quantile=args.quantile,
        output_path=args.output,
    )
