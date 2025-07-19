import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging

from .config import get_config


def simulate_sensor_data(
    num_samples: int = 1000,
    num_features: int = 3,
    seed: int | None = None,
    return_labels: bool = False,
    anomaly_start: int | None = None,
    anomaly_length: int | None = None,
    anomaly_magnitude: float | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Simulate multivariate sensor data.

    Parameters
    ----------
    num_samples : int, optional
        Number of time steps to generate.
    num_features : int, optional
        How many sensor streams to simulate.
    seed : int or None, optional
        Random seed for reproducibility.
    return_labels : bool, optional
        If ``True``, also return a Series marking anomalous indices.
    anomaly_start : int, optional
        Index of the first anomalous time step. Uses config default if None.
    anomaly_length : int, optional
        How many time steps the anomaly spans. Uses config default if None.
    anomaly_magnitude : float, optional
        Amount added to ``sensor_1`` during the anomaly window. Uses config default if None.
    """
    
    # Load configuration defaults
    config = get_config()
    anomaly_start = anomaly_start if anomaly_start is not None else config.ANOMALY_START
    anomaly_length = anomaly_length if anomaly_length is not None else config.ANOMALY_LENGTH
    anomaly_magnitude = anomaly_magnitude if anomaly_magnitude is not None else config.ANOMALY_MAGNITUDE

    if seed is not None:
        np.random.seed(seed)

    t = np.arange(num_samples)
    data = {
        f"sensor_{i+1}": np.sin(0.02 * t + i)
        + np.random.normal(scale=0.1, size=num_samples)
        for i in range(num_features)
    }
    labels = np.zeros(num_samples, dtype=int)
    if num_samples > anomaly_start:
        start = anomaly_start
        end = min(anomaly_start + anomaly_length, num_samples)
        data["sensor_1"][start:end] += anomaly_magnitude
        labels[start:end] = 1

    df = pd.DataFrame(data)
    if return_labels:
        return df, pd.Series(labels)
    return df


def main(
    num_samples: int = 1000,
    num_features: int = 3,
    seed: int | None = None,
    output_path: str = "data/raw/sensor_data.csv",
    labels_path: str | None = None,
    anomaly_start: int | None = None,
    anomaly_length: int | None = None,
    anomaly_magnitude: float | None = None,
    config_file: str | None = None,
) -> None:
    """Generate synthetic sensor data and write it to ``output_path``."""
    
    # Load configuration
    if config_file:
        from .config import reload_config
        reload_config(config_file)

    df, labels = simulate_sensor_data(
        num_samples,
        num_features,
        seed=seed,
        return_labels=True,
        anomaly_start=anomaly_start,
        anomaly_length=anomaly_length,
        anomaly_magnitude=anomaly_magnitude,
    )
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    logging.info(f"Generated {num_samples} samples with {num_features} features, saved to {out_file}")
    
    if labels_path:
        Path(labels_path).write_text("\n".join(map(str, labels.tolist())))
        logging.info(f"Anomaly labels saved to {labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic sensor data")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-features", type=int, default=3)
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--output-path",
        default="data/raw/sensor_data.csv",
        help="Where to write the generated CSV",
    )
    parser.add_argument(
        "--labels-path",
        help="Optional path to write anomaly labels",
    )
    parser.add_argument(
        "--anomaly-start",
        type=int,
        help="Index of the first anomalous time step (overrides config)",
    )
    parser.add_argument(
        "--anomaly-length",
        type=int,
        help="Length of the injected anomaly (overrides config)",
    )
    parser.add_argument(
        "--anomaly-magnitude",
        type=float,
        help="Magnitude added to sensor_1 during the anomaly (overrides config)",
    )
    parser.add_argument(
        "--config-file",
        help="Path to configuration file",
    )
    args = parser.parse_args()
    main(
        num_samples=args.num_samples,
        num_features=args.num_features,
        seed=args.seed,
        output_path=args.output_path,
        labels_path=args.labels_path,
        anomaly_start=args.anomaly_start,
        anomaly_length=args.anomaly_length,
        anomaly_magnitude=args.anomaly_magnitude,
        config_file=args.config_file,
    )
