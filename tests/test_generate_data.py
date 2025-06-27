import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

from src.generate_data import main, simulate_sensor_data
import pandas as pd
import numpy as np


def test_simulate_sensor_data_shape():
    df = simulate_sensor_data(num_samples=50, num_features=4)
    assert df.shape == (50, 4)


def test_cli_generates_file(tmp_path):
    out = tmp_path / "data.csv"
    labels = tmp_path / "labels.csv"
    main(
        num_samples=20,
        num_features=2,
        output_path=str(out),
        labels_path=str(labels),
        anomaly_start=5,
        anomaly_length=3,
        anomaly_magnitude=2.0,
    )
    df = pd.read_csv(out)
    assert df.shape == (20, 2)
    assert labels.is_file()
    lbls = pd.read_csv(labels, header=None)
    assert lbls.shape == (20, 1)


def test_seed_reproducible():
    df1 = simulate_sensor_data(num_samples=30, num_features=2, seed=123)
    df2 = simulate_sensor_data(num_samples=30, num_features=2, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_return_labels():
    df, labels = simulate_sensor_data(num_samples=40, num_features=2, return_labels=True)
    assert len(labels) == 40
    assert isinstance(labels.iloc[0], (int, np.integer))


def test_custom_anomaly_params():
    df, labels = simulate_sensor_data(
        num_samples=30,
        num_features=2,
        return_labels=True,
        anomaly_start=10,
        anomaly_length=5,
        anomaly_magnitude=5.0,
    )
    assert labels.sum() == 5
    assert (labels[10:15] == 1).all()

