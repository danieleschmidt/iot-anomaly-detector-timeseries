import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

import numpy as np
import pandas as pd
from src.data_preprocessor import DataPreprocessor


def test_create_windows():
    dp = DataPreprocessor()
    data = np.arange(10).reshape(-1, 1)
    windows = dp.create_windows(data, window_size=5, step=2)
    assert windows.shape == (3, 5, 1)
    assert (windows[0].flatten() == np.arange(5)).all()


def test_load_and_preprocess(tmp_path):
    df = pd.DataFrame({
        'a': np.arange(10),
        'b': np.arange(0, 20, 2)
    })
    csv = tmp_path / 'data.csv'
    df.to_csv(csv, index=False)
    dp = DataPreprocessor()
    windows = dp.load_and_preprocess(str(csv), window_size=3, step=1)
    # values should be between 0 and 1
    assert windows.min() >= 0
    assert windows.max() <= 1
    assert windows.shape[1] == 3


def test_standard_scaler(tmp_path):
    df = pd.DataFrame({"a": np.arange(10), "b": np.arange(0, 20, 2)})
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    from sklearn.preprocessing import StandardScaler

    dp = DataPreprocessor(StandardScaler())
    windows = dp.load_and_preprocess(str(csv), window_size=5, step=5)
    # StandardScaler -> mean close to 0, std close to 1
    flat = windows.reshape(-1, windows.shape[-1])
    assert np.allclose(flat.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(flat.std(axis=0), 1, atol=1e-6)


def test_save_and_load_scaler(tmp_path):
    df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10, 20)})
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    dp = DataPreprocessor()
    dp.load_and_preprocess(str(csv), window_size=2, step=1)
    scaler_file = tmp_path / "scaler.pkl"
    dp.save(scaler_file)
    assert scaler_file.is_file()
    dp2 = DataPreprocessor.load(scaler_file)
    transformed = dp2.transform(df)
    assert transformed.shape == (10, 2)
