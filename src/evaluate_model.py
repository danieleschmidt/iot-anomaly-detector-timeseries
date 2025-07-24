from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .anomaly_detector import AnomalyDetector
from . import train_autoencoder
from .logging_config import get_logger
from .model_metadata import ModelMetadata


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
    logger = get_logger(__name__)
    
    model_file = Path(model_path)
    if not model_file.exists():
        logger.info(f"Model not found at {model_path}, training new model with {train_epochs} epochs")
        train_autoencoder.main(
            csv_path=csv_path,
            epochs=train_epochs,
            window_size=window_size,
            step=step,
            model_path=model_path,
            scaler_path=scaler_path,
        )

    logger.info(f"Loading model from {model_path}")
    detector = AnomalyDetector(model_path, scaler_path)
    
    logger.info(f"Processing data from {csv_path} with window_size={window_size}")
    windows = detector.preprocessor.load_and_preprocess(
        csv_path, window_size, step
    )
    
    logger.info(f"Computing reconstruction scores for {len(windows)} windows")
    scores = detector.score(windows)
    mse_mean = float(scores.mean())
    mse_std = float(scores.std())
    
    if quantile is not None:
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1 (exclusive)")
        threshold = float(np.quantile(scores, quantile))
        logger.info(f"Using quantile-based threshold: {threshold:.4f} (quantile={quantile})")
    else:
        threshold = mse_mean + threshold_factor * mse_std
        logger.info(f"Using statistical threshold: {threshold:.4f} (mean + {threshold_factor}*std)")
    
    percent_anomaly = float((scores > threshold).mean() * 100)

    stats = {
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "threshold": threshold,
        "percent_anomaly": percent_anomaly,
    }

    if labels_path:
        import pandas as pd
        from sklearn.metrics import (
            precision_recall_fscore_support,
            roc_auc_score,
            confusion_matrix
        )

        logger.info(f"Loading ground truth labels from {labels_path}")
        true_labels = pd.read_csv(labels_path, header=None)[0].to_numpy()
        window_labels = []
        for start in range(0, len(true_labels) - window_size + 1, step):
            window_labels.append(int(true_labels[start : start + window_size].any()))
        window_labels = np.array(window_labels, dtype=bool)
        preds = scores > threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            window_labels, preds, average="binary", zero_division=0
        )
        
        # Calculate ROC AUC using reconstruction scores as probabilities
        try:
            roc_auc = roc_auc_score(window_labels, scores)
        except ValueError:
            # Handle case where all labels are the same class
            roc_auc = 0.0
            logger.warning("ROC AUC could not be calculated - all labels are the same class")
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(window_labels, preds).ravel()
        
        # Calculate additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        logger.info(f"Classification metrics: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}, roc_auc={roc_auc:.3f}")
        logger.info(f"Confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        logger.info(f"Additional metrics: accuracy={accuracy:.3f}, specificity={specificity:.3f}")
        
        stats.update(
            {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": float(roc_auc),
                "accuracy": float(accuracy),
                "specificity": float(specificity),
                "confusion_matrix": {
                    "true_positives": int(tp),
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn)
                }
            }
        )

    if output_path:
        Path(output_path).write_text(json.dumps(stats, indent=2))
        logger.info(f"Evaluation results saved to {output_path}")
    
    # Update model metadata with evaluation results if labels were provided
    if labels_path:
        try:
            metadata_manager = ModelMetadata(Path(model_path).parent)
            model_versions = metadata_manager.list_model_versions()
            
            # Find the most recent version that matches our model file
            model_name = Path(model_path).name
            matching_version = None
            for version_info in model_versions:
                if version_info["model_file"] == model_name:
                    matching_version = version_info["version"]
                    break
            
            if matching_version:
                # Load existing metadata and update with evaluation results
                metadata_path = Path(model_path).parent / f"metadata_{matching_version}.json"
                if metadata_path.exists():
                    metadata = metadata_manager.load_metadata(str(metadata_path))
                    
                    # Update performance metrics with evaluation results
                    evaluation_metrics = {
                        f"eval_{k}": v for k, v in stats.items() 
                        if k in ["precision", "recall", "f1", "roc_auc", "accuracy", "specificity"]
                    }
                    metadata["performance_metrics"].update(evaluation_metrics)
                    
                    # Save updated metadata
                    metadata_manager.save_metadata(metadata, str(metadata_path))
                    logger.info("Updated model metadata with evaluation results")
        except Exception as e:
            logger.warning(f"Could not update model metadata: {e}")

    # Log evaluation summary
    logger.info("Evaluation Summary", extra={
        "average_mse": round(mse_mean, 4),
        "std_mse": round(mse_std, 4),
        "threshold": round(threshold, 4),
        "percent_anomalies": round(percent_anomaly, 2)
    })

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
