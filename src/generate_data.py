import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def simulate_sensor_data(
    num_samples: int = 1000,
    num_features: int = 3,
    seed: int | None = None,
    return_labels: bool = False,
    anomaly_start: int = 200,
    anomaly_length: int = 20,
    anomaly_magnitude: float = 3.0,
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
        Index of the first anomalous time step. If ``num_samples`` is less
        than ``anomaly_start`` no anomalies are inserted.
    anomaly_length : int, optional
        How many time steps the anomaly spans.
    anomaly_magnitude : float, optional
        Amount added to ``sensor_1`` during the anomaly window.
    """

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
    anomaly_start: int = 200,
    anomaly_length: int = 20,
    anomaly_magnitude: float = 3.0,
) -> None:
    """Generate synthetic sensor data and write it to ``output_path``."""

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
    if labels_path:
        Path(labels_path).write_text("\n".join(map(str, labels.tolist())))


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
        default=200,
        help="Index of the first anomalous time step",
    )
    parser.add_argument(
        "--anomaly-length",
        type=int,
        default=20,
        help="Length of the injected anomaly",
    )
    parser.add_argument(
        "--anomaly-magnitude",
        type=float,
        default=3.0,
        help="Magnitude added to sensor_1 during the anomaly",
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
    )
