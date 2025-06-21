import json
from src.evaluate_model import evaluate

def test_evaluate_returns_stats(tmp_path):
    csv = 'data/raw/sensor_data.csv'
    out = tmp_path / 'stats.json'
    stats = evaluate(csv_path=csv, window_size=30, threshold_factor=2.0, output_path=str(out))
    saved = json.loads(out.read_text())
    assert set(stats) == {"mse_mean", "mse_std", "threshold", "percent_anomaly"}
    assert saved == stats
