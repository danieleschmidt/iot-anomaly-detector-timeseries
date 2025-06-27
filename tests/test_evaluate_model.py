import json
import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("tensorflow")

from src.evaluate_model import evaluate

def test_evaluate_returns_stats(tmp_path):
    csv = 'data/raw/sensor_data.csv'
    out = tmp_path / 'stats.json'
    model = tmp_path / "model.h5"
    stats = evaluate(
        csv_path=csv,
        window_size=30,
        step=2,
        threshold_factor=2.0,
        output_path=str(out),
        model_path=str(model),
        scaler_path=str(tmp_path / "scaler.pkl"),
        labels_path=None,
    )
    saved = json.loads(out.read_text())
    assert set(stats) == {"mse_mean", "mse_std", "threshold", "percent_anomaly"}
    assert saved == stats


def test_evaluate_trains_if_model_missing(tmp_path):
    csv = 'data/raw/sensor_data.csv'
    model = tmp_path / 'autoencoder.h5'
    assert not model.exists()
    evaluate(csv_path=csv, model_path=str(model), scaler_path=str(tmp_path / "scaler.pkl"), train_epochs=1, step=2)
    assert model.is_file()


def test_evaluate_with_labels(tmp_path):
    from src.generate_data import simulate_sensor_data

    df, labels = simulate_sensor_data(num_samples=300, return_labels=True)
    csv = tmp_path / "data.csv"
    lab = tmp_path / "labels.csv"
    df.to_csv(csv, index=False)
    labels.to_csv(lab, index=False, header=False)
    model = tmp_path / "auto.h5"
    stats = evaluate(
        csv_path=str(csv),
        window_size=30,
        step=1,
        threshold_factor=3.0,
        model_path=str(model),
        scaler_path=str(tmp_path / "scaler.pkl"),
        labels_path=str(lab),
        train_epochs=1,
    )
    assert {"precision", "recall", "f1"}.issubset(stats)
