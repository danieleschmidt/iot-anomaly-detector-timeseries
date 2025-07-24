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


def test_load_nonexistent_csv():
    """Test error handling for non-existent CSV file."""
    dp = DataPreprocessor()
    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        dp.load_and_preprocess("nonexistent.csv", window_size=3)


def test_load_invalid_csv(tmp_path):
    """Test error handling for invalid CSV content."""
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("not,valid,csv\ncontent")
    dp = DataPreprocessor()
    with pytest.raises(ValueError, match="Data validation failed"):
        dp.load_and_preprocess(str(invalid_csv), window_size=3)


def test_load_malformed_csv(tmp_path):
    """Test error handling for malformed CSV content."""
    malformed_csv = tmp_path / "malformed.csv"
    malformed_csv.write_text("\"unclosed quote field\na,b,c\n1,2,3")
    dp = DataPreprocessor()
    with pytest.raises(ValueError, match="Unable to parse CSV file"):
        dp.load_and_preprocess(str(malformed_csv), window_size=3)


def test_load_empty_csv(tmp_path):
    """Test error handling for empty CSV file."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")
    dp = DataPreprocessor()
    with pytest.raises(ValueError, match="CSV file is empty"):
        dp.load_and_preprocess(str(empty_csv), window_size=3)


def test_invalid_window_size():
    """Test validation for window_size parameter."""
    dp = DataPreprocessor()
    data = np.arange(10).reshape(-1, 1)
    
    # Test negative window size
    with pytest.raises(ValueError, match="window_size must be positive"):
        dp.create_windows(data, window_size=-1)
    
    # Test zero window size
    with pytest.raises(ValueError, match="window_size must be positive"):
        dp.create_windows(data, window_size=0)
    
    # Test window size larger than data
    with pytest.raises(ValueError, match="window_size cannot be larger than data length"):
        dp.create_windows(data, window_size=15)


def test_invalid_step_size():
    """Test validation for step parameter."""
    dp = DataPreprocessor()
    data = np.arange(10).reshape(-1, 1)
    
    # Test negative step
    with pytest.raises(ValueError, match="step must be positive"):
        dp.create_windows(data, window_size=3, step=-1)
    
    # Test zero step
    with pytest.raises(ValueError, match="step must be positive"):
        dp.create_windows(data, window_size=3, step=0)


def test_load_corrupted_scaler(tmp_path):
    """Test error handling for corrupted scaler file."""
    corrupted_file = tmp_path / "corrupted.pkl"
    corrupted_file.write_text("not a valid pickle file")
    
    with pytest.raises(ValueError, match="Unable to load scaler"):
        DataPreprocessor.load(str(corrupted_file))


def test_load_nonexistent_scaler():
    """Test error handling for non-existent scaler file."""
    with pytest.raises(ValueError, match="File does not exist"):
        DataPreprocessor.load("nonexistent.pkl")


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    dp = DataPreprocessor()
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError, match="DataFrame is empty"):
        dp.fit_transform(empty_df)


def test_dataframe_with_missing_values():
    """Test handling of DataFrame with NaN values."""
    dp = DataPreprocessor()
    df_with_nan = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, 6, 7, 8]})
    
    with pytest.raises(ValueError, match="DataFrame contains missing values"):
        dp.fit_transform(df_with_nan)
