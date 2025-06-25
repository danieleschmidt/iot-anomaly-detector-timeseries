import pytest
from src.generate_data import main as gen_main
from src.train_autoencoder import main as train_main
from src.anomaly_detector import AnomalyDetector
import tensorflow as tf


@pytest.mark.integration
def test_end_to_end_pipeline(tmp_path):
    csv = tmp_path / "data.csv"
    labels = tmp_path / "labels.csv"
    gen_main(
        num_samples=100,
        num_features=3,
        output_path=str(csv),
        labels_path=str(labels),
        anomaly_start=50,
        anomaly_length=10,
    )
    model = tmp_path / "model.h5"
    scaler = tmp_path / "scaler.pkl"
    train_main(
        csv_path=str(csv),
        epochs=1,
        window_size=20,
        step=2,
        latent_dim=8,
        lstm_units=16,
        model_path=str(model),
        scaler_path=str(scaler),
    )
    detector = AnomalyDetector(str(model), str(scaler))
    preds = detector.predict(str(csv), window_size=20, step=2)
    assert preds.dtype == bool
    assert len(preds) > 0


@pytest.mark.integration
def test_detection_accuracy(tmp_path):
    tf.keras.utils.set_random_seed(1)
    csv = tmp_path / "data.csv"
    labels = tmp_path / "labels.csv"
    gen_main(
        num_samples=300,
        num_features=3,
        seed=2,
        output_path=str(csv),
        labels_path=str(labels),
        anomaly_start=150,
        anomaly_length=30,
    )
    model = tmp_path / "model.h5"
    scaler = tmp_path / "scaler.pkl"
    train_main(
        csv_path=str(csv),
        epochs=5,
        window_size=30,
        step=1,
        latent_dim=8,
        lstm_units=16,
        model_path=str(model),
        scaler_path=str(scaler),
    )
    from src.evaluate_model import evaluate

    stats = evaluate(
        csv_path=str(csv),
        window_size=30,
        step=1,
        threshold_factor=1.0,
        labels_path=str(labels),
        model_path=str(model),
        scaler_path=str(scaler),
    )
    assert stats["f1"] >= 0.7
