"""Anomaly detection utilities using a trained autoencoder."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from .data_preprocessor import DataPreprocessor
from .logging_config import get_logger
from .caching_strategy import cache_prediction, get_cache_stats


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

    @cache_prediction
    def score(self, sequences: np.ndarray) -> np.ndarray:
        """Compute reconstruction error scores for sequences."""
        self.logger.debug(f"Computing reconstruction scores for {len(sequences)} sequences")
        reconstructed = self.model.predict(sequences, verbose=0)
        scores = np.mean(np.square(sequences - reconstructed), axis=(1, 2))
        self.logger.debug(f"Score statistics: mean={scores.mean():.4f}, std={scores.std():.4f}")
        return scores

    def score_batched(self, sequences: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Compute reconstruction error scores for sequences using batched processing.
        
        This method processes sequences in batches to optimize memory usage and provide
        progress feedback for large datasets. It produces identical results to the 
        score() method but with better scalability.
        
        Parameters
        ----------
        sequences : np.ndarray
            Input sequences of shape (n_sequences, window_size, n_features)
        batch_size : int, optional
            Number of sequences to process in each batch (default: 256)
            
        Returns
        -------
        np.ndarray
            Reconstruction error scores for each sequence
            
        Examples
        --------
        >>> detector = AnomalyDetector("model.h5")
        >>> sequences = np.random.randn(10000, 30, 3)  # Large dataset
        >>> scores = detector.score_batched(sequences, batch_size=512)
        """
        if len(sequences) == 0:
            return np.array([])
        
        n_sequences = len(sequences)
        scores = np.zeros(n_sequences)
        
        self.logger.info(f"Processing {n_sequences} sequences in batches of {batch_size}")
        
        for i in range(0, n_sequences, batch_size):
            batch_end = min(i + batch_size, n_sequences)
            batch = sequences[i:batch_end]
            
            # Compute reconstruction for this batch
            batch_reconstructed = self.model.predict(batch, verbose=0)
            batch_scores = np.mean(np.square(batch - batch_reconstructed), axis=(1, 2))
            
            scores[i:batch_end] = batch_scores
            
            # Log progress every 10 batches or at completion
            if i % (batch_size * 10) == 0 or batch_end == n_sequences:
                progress_pct = (batch_end / n_sequences) * 100
                self.logger.info(f"Processed {batch_end:,}/{n_sequences:,} sequences ({progress_pct:.1f}%)")
        
        self.logger.debug(f"Batch processing complete. Score statistics: mean={scores.mean():.4f}, std={scores.std():.4f}")
        return scores

    def predict(
        self,
        csv_path: str,
        window_size: int = 30,
        step: int = 1,
        threshold: float | None = None,
        quantile: float | None = None,
        batch_size: int | None = None,
        use_batched: bool = False,
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
        batch_size : int or None, optional
            Batch size for processing (only used when use_batched=True).
            If None, defaults to 256.
        use_batched : bool, optional
            Whether to use batched processing for improved memory efficiency
            and progress tracking. Recommended for large datasets.
        """
        self.logger.info(f"Predicting anomalies for {csv_path}")
        windows = self.preprocessor.load_and_preprocess(csv_path, window_size, step)
        
        # Automatically use batched processing for large datasets
        n_windows = len(windows)
        if use_batched or n_windows > 1000:
            if n_windows > 1000 and not use_batched:
                self.logger.info(f"Large dataset detected ({n_windows:,} windows), automatically using batched processing")
            
            if batch_size is None:
                batch_size = 256
            scores = self.score_batched(windows, batch_size=batch_size)
        else:
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
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics for prediction operations."""
        stats = get_cache_stats()
        return {
            "prediction_cache": stats.get("prediction", {}),
            "all_cache_stats": stats
        }
    
    def clear_cache(self) -> None:
        """Clear prediction cache."""
        from .caching_strategy import clear_all_caches
        clear_all_caches()
        self.logger.info("Prediction cache cleared")


def main(
    csv_path: str = "data/raw/sensor_data.csv",
    model_path: str = "saved_models/autoencoder.h5",
    scaler_path: str | None = None,
    window_size: int = 30,
    step: int = 1,
    threshold: float | None = None,
    quantile: float | None = None,
    output_path: str = "predictions.csv",
    batch_size: int | None = None,
    use_batched: bool = False,
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
    batch_size : int or None, optional
        Batch size for processing (only used when use_batched=True).
        If None, defaults to 256.
    use_batched : bool, optional
        Force batched processing even for smaller datasets. Automatically
        enabled for datasets with >1000 windows.
    """
    logger = get_logger(__name__)
    logger.info(f"Starting anomaly detection: input={csv_path}, output={output_path}")
    
    detector = AnomalyDetector(model_path, scaler_path)
    if quantile is not None and not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1 (exclusive)")

    preds = detector.predict(csv_path, window_size, step, threshold, quantile, 
                           batch_size, use_batched)
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
    parser.add_argument(
        "--batch-size", 
        type=int, 
        help="Batch size for processing (default: 256 when batched processing is used)"
    )
    parser.add_argument(
        "--use-batched", 
        action="store_true",
        help="Force batched processing (automatically enabled for >1000 windows)"
    )
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
        batch_size=args.batch_size,
        use_batched=args.use_batched,
    )
