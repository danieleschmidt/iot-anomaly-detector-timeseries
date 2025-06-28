import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("tensorflow")

from src import train_autoencoder
from src.anomaly_detector import AnomalyDetector, main
import pandas as pd
import subprocess
import sys


def test_predict_flags_anomalies(tmp_path):
    model = tmp_path / "model.h5"
    train_autoencoder.main(
        csv_path='data/raw/sensor_data.csv',
        epochs=1,
        window_size=10,
        step=2,
        latent_dim=8,
        lstm_units=16,
        model_path=str(model),
        scaler_path=str(tmp_path / "scaler.pkl"),
    )
    detector = AnomalyDetector(str(model), str(tmp_path / "scaler.pkl"))
    preds = detector.predict('data/raw/sensor_data.csv', window_size=10, step=2, threshold=0.5)
    assert preds.dtype == bool
    assert len(preds) > 0


def test_predict_with_quantile(tmp_path):
    model = tmp_path / "model.h5"
    train_autoencoder.main(
        csv_path='data/raw/sensor_data.csv',
        epochs=1,
        window_size=10,
        step=2,
        latent_dim=8,
        lstm_units=16,
        model_path=str(model),
        scaler_path=str(tmp_path / "scaler.pkl"),
    )
    detector = AnomalyDetector(str(model), str(tmp_path / "scaler.pkl"))
    preds = detector.predict(
        'data/raw/sensor_data.csv', window_size=10, step=2, quantile=0.9
    )
    assert preds.dtype == bool
    assert len(preds) > 0


@pytest.mark.parametrize("bad_quantile", [1.5, 1.0, 0.0, -0.2])
def test_quantile_validation(tmp_path, bad_quantile):
    model = tmp_path / "model.h5"
    train_autoencoder.main(
        csv_path='data/raw/sensor_data.csv',
        epochs=1,
        window_size=10,
        step=2,
        latent_dim=8,
        lstm_units=16,
        model_path=str(model),
        scaler_path=str(tmp_path / "scaler.pkl"),
    )
    detector = AnomalyDetector(str(model), str(tmp_path / "scaler.pkl"))
    with pytest.raises(ValueError):
        detector.predict('data/raw/sensor_data.csv', step=2, quantile=bad_quantile)


def test_threshold_and_quantile_exclusive(tmp_path):
    model = tmp_path / "model.h5"
    train_autoencoder.main(
        csv_path='data/raw/sensor_data.csv',
        epochs=1,
        window_size=10,
        step=2,
        latent_dim=8,
        lstm_units=16,
        model_path=str(model),
        scaler_path=str(tmp_path / "scaler.pkl"),
    )
    detector = AnomalyDetector(str(model), str(tmp_path / "scaler.pkl"))
    with pytest.raises(ValueError):
        detector.predict(
            'data/raw/sensor_data.csv', step=2, threshold=0.5, quantile=0.9
        )


def test_cli_writes_predictions(tmp_path):
    model = tmp_path / "model.h5"
    output = tmp_path / "preds.csv"
    train_autoencoder.main(
        csv_path='data/raw/sensor_data.csv',
        epochs=1,
        window_size=10,
        step=2,
        latent_dim=8,
        lstm_units=16,
        model_path=str(model),
        scaler_path=str(tmp_path / "scaler.pkl"),
    )
    main(
        csv_path='data/raw/sensor_data.csv',
        model_path=str(model),
        scaler_path=str(tmp_path / "scaler.pkl"),
        window_size=10,
        step=2,
        quantile=0.9,
        output_path=str(output),
    )
    flags = pd.read_csv(output, header=None)[0]
    assert len(flags) > 0


def test_cli_threshold_and_quantile_exclusive(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.anomaly_detector",
            "--csv-path",
            "data/raw/sensor_data.csv",
            "--model-path",
            str(tmp_path / "model.h5"),
            "--scaler-path",
            str(tmp_path / "scaler.pkl"),
            "--window-size",
            "10",
            "--step",
            "2",
            "--threshold",
            "0.5",
            "--quantile",
            "0.9",
            "--output",
            str(tmp_path / "out.csv"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_invalid_quantile(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.anomaly_detector",
            "--csv-path",
            "data/raw/sensor_data.csv",
            "--model-path",
            str(tmp_path / "model.h5"),
            "--scaler-path",
            str(tmp_path / "scaler.pkl"),
            "--window-size",
            "10",
            "--step",
            "2",
            "--quantile",
            "1.2",
            "--output",
            str(tmp_path / "out.csv"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
