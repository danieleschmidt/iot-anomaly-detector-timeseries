import json
from src.evaluate_model import evaluate

def test_evaluate_returns_stats(tmp_path):
    csv = 'data/raw/sensor_data.csv'
    out = tmp_path / 'stats.json'
    model = tmp_path / "model.h5"
    stats = evaluate(
        csv_path=csv,
        window_size=30,
        threshold_factor=2.0,
        output_path=str(out),
        model_path=str(model),
    )
    saved = json.loads(out.read_text())
    assert set(stats) == {"mse_mean", "mse_std", "threshold", "percent_anomaly"}
    assert saved == stats


def test_evaluate_trains_if_model_missing(tmp_path):
    csv = 'data/raw/sensor_data.csv'
    model = tmp_path / 'autoencoder.h5'
    assert not model.exists()
    evaluate(csv_path=csv, model_path=str(model), train_epochs=1)
    assert model.is_file()
