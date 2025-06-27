import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("tensorflow")

from src import train_autoencoder
from src.anomaly_detector import AnomalyDetector, main
import pandas as pd


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
        threshold=0.5,
        output_path=str(output),
    )
    flags = pd.read_csv(output, header=None)[0]
    assert len(flags) > 0
