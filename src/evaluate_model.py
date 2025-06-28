from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .anomaly_detector import AnomalyDetector
from . import train_autoencoder


def evaluate(
    csv_path: str = "data/raw/sensor_data.csv",
    window_size: int = 30,
    step: int = 1,
    threshold_factor: float = 3.0,
    quantile: float | None = None,
    labels_path: str | None = None,
    output_path: str | None = None,
    model_path: str = "saved_models/autoencoder.h5",
    scaler_path: str | None = None,
    train_epochs: int = 1,
) -> dict[str, float]:
    """Evaluate model reconstruction error statistics.

    Parameters
    ----------
    csv_path : str
        Path to the CSV containing sensor data.
    window_size : int
        Length of each sliding window.
    step : int, optional
        Step size for the sliding window.
    threshold_factor : float, optional
        Factor for the standard deviation when computing the anomaly threshold.
        Ignored if ``quantile`` is provided.
    quantile : float or None, optional
        Derive the anomaly threshold from this quantile of the reconstruction
        error distribution. Must be between 0 and 1 (exclusive).
    output_path : str or None, optional
        If given, write a JSON report with the evaluation statistics.
    labels_path : str or None, optional
        CSV file containing ground truth anomaly flags for each time step.
    model_path : str, optional
        Path to a trained autoencoder. If it does not exist it will be trained
        on ``csv_path`` with ``train_epochs`` epochs.
    scaler_path : str or None, optional
        Location of a fitted scaler used during training.
    train_epochs : int, optional
        Number of epochs used for fallback training when the model is missing.
    """

    model_file = Path(model_path)
    if not model_file.exists():
        train_autoencoder.main(
            csv_path=csv_path,
            epochs=train_epochs,
            window_size=window_size,
            step=step,
            model_path=model_path,
            scaler_path=scaler_path,
        )

    detector = AnomalyDetector(model_path, scaler_path)
    windows = detector.preprocessor.load_and_preprocess(
        csv_path, window_size, step
    )
    scores = detector.score(windows)
    mse_mean = float(scores.mean())
    mse_std = float(scores.std())
    if quantile is not None:
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1 (exclusive)")
        threshold = float(np.quantile(scores, quantile))
    else:
        threshold = mse_mean + threshold_factor * mse_std
    percent_anomaly = float((scores > threshold).mean() * 100)

    stats = {
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "threshold": threshold,
        "percent_anomaly": percent_anomaly,
    }

    if labels_path:
        import pandas as pd
        from sklearn.metrics import precision_recall_fscore_support

        true_labels = pd.read_csv(labels_path, header=None)[0].to_numpy()
        window_labels = []
        for start in range(0, len(true_labels) - window_size + 1, step):
            window_labels.append(int(true_labels[start : start + window_size].any()))
        window_labels = np.array(window_labels, dtype=bool)
        preds = scores > threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            window_labels, preds, average="binary", zero_division=0
        )
        stats.update(
            {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    if output_path:
        Path(output_path).write_text(json.dumps(stats, indent=2))

    print(f"Average MSE: {mse_mean:.4f}")
    print(f"Std MSE: {mse_std:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Percent anomalies at threshold: {percent_anomaly:.2f}%")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate autoencoder")
    parser.add_argument("--csv-path", default="data/raw/sensor_data.csv")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size for sliding windows",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--threshold-factor",
        type=float,
        default=3.0,
        help="Factor for std when deriving threshold",
    )
    group.add_argument(
        "--quantile",
        type=float,
        help=(
            "Quantile of reconstruction error used for the threshold. "
            "Must be between 0 and 1 (exclusive)."
        ),
    )
    parser.add_argument("--output", help="Write JSON report to this path")
    parser.add_argument(
        "--labels-path",
        help="CSV file with ground truth anomaly flags",
    )
    parser.add_argument(
        "--model-path",
        default="saved_models/autoencoder.h5",
        help="Autoencoder model location",
    )
    parser.add_argument(
        "--scaler-path",
        default=None,
        help="Path to the scaler used during training",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=1,
        help="Epochs for fallback training if the model is missing",
    )
    args = parser.parse_args()

    evaluate(
        csv_path=args.csv_path,
        window_size=args.window_size,
        step=args.step,
        threshold_factor=args.threshold_factor,
        quantile=args.quantile,
        output_path=args.output,
        labels_path=args.labels_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        train_epochs=args.train_epochs,
    )
