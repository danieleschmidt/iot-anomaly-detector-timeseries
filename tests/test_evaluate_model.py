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


def test_evaluate_with_quantile(tmp_path):
    csv = 'data/raw/sensor_data.csv'
    model = tmp_path / 'model.h5'
    stats = evaluate(
        csv_path=csv,
        window_size=30,
        step=2,
        quantile=0.9,
        model_path=str(model),
        scaler_path=str(tmp_path / 'scaler.pkl'),
    )
    assert "threshold" in stats


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
    assert {"precision", "recall", "f1", "roc_auc", "accuracy", "specificity", "confusion_matrix"}.issubset(stats)
    # Verify confusion matrix structure
    cm = stats["confusion_matrix"]
    assert set(cm.keys()) == {"true_positives", "true_negatives", "false_positives", "false_negatives"}
    assert all(isinstance(v, int) for v in cm.values())


def test_enhanced_metrics_edge_cases(tmp_path):
    """Test enhanced metrics with edge cases like all-same-class labels."""
    import pandas as pd
    import numpy as np
    
    # Create data with all normal labels (no anomalies)
    df = pd.DataFrame(np.random.randn(100, 3), columns=['feature1', 'feature2', 'feature3'])
    labels = pd.Series([0] * 100)  # All normal
    
    csv = tmp_path / "edge_data.csv"
    lab = tmp_path / "edge_labels.csv"
    df.to_csv(csv, index=False)
    labels.to_csv(lab, index=False, header=False)
    
    model = tmp_path / "edge_auto.h5"
    stats = evaluate(
        csv_path=str(csv),
        window_size=10,
        step=1,
        threshold_factor=1.0,
        model_path=str(model),
        scaler_path=str(tmp_path / "edge_scaler.pkl"),
        labels_path=str(lab),
        train_epochs=1,
    )
    
    # With all-normal labels, ROC AUC should be 0.0 (our fallback)
    assert "roc_auc" in stats
    assert stats["roc_auc"] >= 0.0
    assert "accuracy" in stats
    assert "specificity" in stats
    assert "confusion_matrix" in stats


@pytest.mark.parametrize("bad_q", [1.1, 0.0, -0.3])
def test_evaluate_invalid_quantile(tmp_path, bad_q):
    csv = 'data/raw/sensor_data.csv'
    model = tmp_path / 'bad.h5'
    with pytest.raises(ValueError):
        evaluate(
            csv_path=csv,
            step=2,
            quantile=bad_q,
            model_path=str(model),
            scaler_path=str(tmp_path / 'scaler.pkl'),
        )
