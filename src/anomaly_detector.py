"""Anomaly detection utilities using a trained autoencoder."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from .data_preprocessor import DataPreprocessor


class AnomalyDetector:
    """Compute anomaly scores and predictions using an autoencoder."""

    def __init__(
        self,
        model_path: str = "saved_models/autoencoder.h5",
        scaler_path: str | None = None,
    ) -> None:
        """Load a trained model from ``model_path`` and optional scaler."""
        self.model = load_model(model_path)
        if scaler_path and Path(scaler_path).exists():
            self.preprocessor = DataPreprocessor.load(scaler_path)
        else:
            self.preprocessor = DataPreprocessor()

    def score(self, sequences: np.ndarray) -> np.ndarray:
        reconstructed = self.model.predict(sequences, verbose=0)
        return np.mean(np.square(sequences - reconstructed), axis=(1, 2))

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
            else:
                threshold = scores.mean() + 3 * scores.std()
        return scores > threshold


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
    detector = AnomalyDetector(model_path, scaler_path)
    if quantile is not None and not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1 (exclusive)")

    preds = detector.predict(csv_path, window_size, step, threshold, quantile)
    pd.Series(preds.astype(int)).to_csv(output_path, index=False, header=False)


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
