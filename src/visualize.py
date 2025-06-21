import matplotlib.pyplot as plt
import pandas as pd


def plot_sequences(csv_path: str, anomalies: pd.Series = None, output='plot.png'):
    df = pd.read_csv(csv_path)
    df.plot(subplots=True, figsize=(10, 8))
    if anomalies is not None:
        for idx, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                plt.axvspan(idx, idx + 1, color='red', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
