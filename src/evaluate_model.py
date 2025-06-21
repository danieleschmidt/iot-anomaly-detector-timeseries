from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .anomaly_detector import AnomalyDetector


def evaluate(
    csv_path: str = "data/raw/sensor_data.csv",
    window_size: int = 30,
    threshold_factor: float = 3.0,
    output_path: str | None = None,
) -> dict[str, float]:
    """Evaluate model reconstruction error statistics.

    Parameters
    ----------
    csv_path : str
        Path to the CSV containing sensor data.
    window_size : int
        Length of each sliding window.
    threshold_factor : float, optional
        Factor for the standard deviation when computing the anomaly threshold.
    output_path : str or None, optional
        If given, write a JSON report with the evaluation statistics.
    """

    detector = AnomalyDetector()
    windows = detector.preprocessor.load_and_preprocess(csv_path, window_size)
    scores = detector.score(windows)
    mse_mean = float(scores.mean())
    mse_std = float(scores.std())
    threshold = mse_mean + threshold_factor * mse_std
    percent_anomaly = float((scores > threshold).mean() * 100)

    stats = {
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "threshold": threshold,
        "percent_anomaly": percent_anomaly,
    }

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
    parser.add_argument("--threshold-factor", type=float, default=3.0)
    parser.add_argument("--output", help="Write JSON report to this path")
    args = parser.parse_args()

    evaluate(
        csv_path=args.csv_path,
        window_size=args.window_size,
        threshold_factor=args.threshold_factor,
        output_path=args.output,
    )
