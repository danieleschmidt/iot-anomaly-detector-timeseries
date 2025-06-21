import numpy as np
from tensorflow.keras.models import load_model

from .data_preprocessor import DataPreprocessor


class AnomalyDetector:
    def __init__(self, model_path='saved_models/autoencoder.h5'):
        self.model = load_model(model_path)
        self.preprocessor = DataPreprocessor()

    def score(self, sequences: np.ndarray) -> np.ndarray:
        reconstructed = self.model.predict(sequences, verbose=0)
        return np.mean(np.square(sequences - reconstructed), axis=(1, 2))

    def predict(self, csv_path: str, window_size: int = 30, threshold: float = None):
        windows = self.preprocessor.load_and_preprocess(csv_path, window_size)
        scores = self.score(windows)
        if threshold is None:
            threshold = scores.mean() + 3 * scores.std()
        return scores > threshold
