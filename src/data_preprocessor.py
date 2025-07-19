import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler
import logging


class DataPreprocessor:
    def __init__(self, scaler=None):
        """Preprocess raw sensor data.

        Parameters
        ----------
        scaler : sklearn-like scaler or None, optional
            Scaler used for normalization. Defaults to ``MinMaxScaler``.
        """
        self.scaler = scaler or MinMaxScaler()

    def save(self, path: str) -> None:
        """Persist the underlying scaler to ``path``."""
        try:
            joblib.dump(self.scaler, Path(path))
            logging.info(f"Scaler saved successfully to {path}")
        except Exception as e:
            logging.error(f"Failed to save scaler to {path}: {e}")
            raise ValueError(f"Unable to save scaler to {path}: {e}") from e

    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """Load a scaler from ``path`` and return a new ``DataPreprocessor``."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")
        
        try:
            scaler = joblib.load(path_obj)
            logging.info(f"Scaler loaded successfully from {path}")
            return cls(scaler)
        except Exception as e:
            logging.error(f"Failed to load scaler from {path}: {e}")
            raise ValueError(f"Unable to load scaler from {path}: {e}") from e

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit scaler to data and transform it."""
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if df.isnull().any().any():
            raise ValueError("DataFrame contains missing values")
        
        try:
            scaled = self.scaler.fit_transform(df)
            logging.info(f"Data fitted and transformed: shape {scaled.shape}")
            return scaled
        except Exception as e:
            logging.error(f"Failed to fit and transform data: {e}")
            raise ValueError(f"Unable to fit and transform data: {e}") from e

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted scaler."""
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if df.isnull().any().any():
            raise ValueError("DataFrame contains missing values")
        
        try:
            transformed = self.scaler.transform(df)
            logging.info(f"Data transformed: shape {transformed.shape}")
            return transformed
        except Exception as e:
            logging.error(f"Failed to transform data: {e}")
            raise ValueError(f"Unable to transform data: {e}") from e

    def create_windows(self, data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
        """Create sliding windows from data."""
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        
        if step <= 0:
            raise ValueError("step must be positive")
        
        if window_size > len(data):
            raise ValueError("window_size cannot be larger than data length")
        
        try:
            windows = []
            for start in range(0, len(data) - window_size + 1, step):
                windows.append(data[start:start + window_size])
            
            if not windows:
                raise ValueError("No windows could be created with given parameters")
            
            result = np.stack(windows)
            logging.info(f"Created {len(windows)} windows with shape {result.shape}")
            return result
        except Exception as e:
            logging.error(f"Failed to create windows: {e}")
            raise ValueError(f"Unable to create windows: {e}") from e

    def load_and_preprocess(self, csv_path: str, window_size: int, step: int = 1) -> np.ndarray:
        """Load CSV data, preprocess it, and create windows."""
        csv_path_obj = Path(csv_path)
        
        if not csv_path_obj.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path_obj)
            logging.info(f"Loaded CSV file: {csv_path}, shape: {df.shape}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {csv_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Unable to parse CSV file {csv_path}: {e}")
        except Exception as e:
            raise ValueError(f"Unable to read CSV file {csv_path}: {e}")
        
        if df.empty:
            raise ValueError(f"CSV file is empty: {csv_path}")
        
        scaled = self.fit_transform(df)
        return self.create_windows(scaled, window_size, step)
