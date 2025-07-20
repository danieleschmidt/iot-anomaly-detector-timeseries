import argparse
import matplotlib.pyplot as plt
import pandas as pd


def plot_sequences(csv_path: str, anomalies: pd.Series = None, output: str = 'plot.png') -> None:
    df = pd.read_csv(csv_path)
    df.plot(subplots=True, figsize=(10, 8))
    if anomalies is not None:
        for idx, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                plt.axvspan(idx, idx + 1, color='red', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main(
    csv_path: str = "data/raw/sensor_data.csv",
    anomalies_path: str | None = None,
    output: str = "plot.png",
) -> None:
    """Plot sequences from ``csv_path`` highlighting anomalies if provided."""

    anomalies = None
    if anomalies_path:
        anomalies = pd.read_csv(anomalies_path, header=None)[0]
    plot_sequences(csv_path, anomalies, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sensor sequences")
    parser.add_argument("--csv-path", default="data/raw/sensor_data.csv")
    parser.add_argument("--anomalies", help="CSV file containing anomaly flags")
    parser.add_argument("--output", default="plot.png", help="Output image path")
    args = parser.parse_args()
    main(csv_path=args.csv_path, anomalies_path=args.anomalies, output=args.output)
