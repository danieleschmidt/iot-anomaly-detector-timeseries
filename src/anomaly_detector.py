"""Anomaly detection utilities using a trained autoencoder."""

import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from .data_preprocessor import DataPreprocessor


class AnomalyDetector:
    """Compute anomaly scores and predictions using an autoencoder."""

    def __init__(self, model_path: str = "saved_models/autoencoder.h5"):
        """Load a trained model from ``model_path``."""
        self.model = load_model(model_path)
        self.preprocessor = DataPreprocessor()

    def score(self, sequences: np.ndarray) -> np.ndarray:
        reconstructed = self.model.predict(sequences, verbose=0)
        return np.mean(np.square(sequences - reconstructed), axis=(1, 2))

    def predict(
        self,
        csv_path: str,
        window_size: int = 30,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Return a boolean array indicating anomalous windows."""
        windows = self.preprocessor.load_and_preprocess(csv_path, window_size)
        scores = self.score(windows)
        if threshold is None:
            threshold = scores.mean() + 3 * scores.std()
        return scores > threshold


def main(
    csv_path: str = "data/raw/sensor_data.csv",
    model_path: str = "saved_models/autoencoder.h5",
    window_size: int = 30,
    threshold: float | None = None,
    output_path: str = "predictions.csv",
) -> None:
    """Run anomaly detection on ``csv_path`` and write flags to ``output_path``."""
    detector = AnomalyDetector(model_path)
    preds = detector.predict(csv_path, window_size, threshold)
    pd.Series(preds.astype(int)).to_csv(output_path, index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect anomalies in a dataset")
    parser.add_argument("--csv-path", default="data/raw/sensor_data.csv")
    parser.add_argument("--model-path", default="saved_models/autoencoder.h5")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        model_path=args.model_path,
        window_size=args.window_size,
        threshold=args.threshold,
        output_path=args.output,
    )
