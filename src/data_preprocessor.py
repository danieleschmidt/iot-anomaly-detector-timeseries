import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessor:
    def __init__(self, scaler=None):
        """Preprocess raw sensor data.

        Parameters
        ----------
        scaler : sklearn-like scaler or None, optional
            Scaler used for normalization. Defaults to ``MinMaxScaler``.
        """
        self.scaler = scaler or MinMaxScaler()

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        scaled = self.scaler.fit_transform(df)
        return scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df)

    def create_windows(self, data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
        windows = []
        for start in range(0, len(data) - window_size + 1, step):
            windows.append(data[start:start + window_size])
        return np.stack(windows)

    def load_and_preprocess(self, csv_path: str, window_size: int, step: int = 1) -> np.ndarray:
        df = pd.read_csv(csv_path)
        scaled = self.fit_transform(df)
        return self.create_windows(scaled, window_size, step)
