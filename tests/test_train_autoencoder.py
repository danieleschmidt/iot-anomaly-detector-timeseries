import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("tensorflow")

from src.train_autoencoder import main


def test_model_is_saved(tmp_path):
    model_path = tmp_path / 'model_dir' / 'model.h5'
    main(
        csv_path='data/raw/sensor_data.csv',
        epochs=1,
        window_size=10,
        step=2,
        latent_dim=8,
        lstm_units=16,
        model_path=str(model_path),
        scaler_path=str(tmp_path / "scaler.pkl"),
        enable_progress=False  # Disable for testing speed
    )
    assert model_path.is_file()
    assert (tmp_path / "scaler.pkl").is_file()


def test_main_with_progress_callbacks(tmp_path):
    """Test training with progress callbacks enabled."""
    model_path = tmp_path / "progress_model.h5"
    result = main(
        csv_path="data/raw/sensor_data.csv",
        epochs=2,
        model_path=str(model_path),
        scaler_path=str(tmp_path / "progress_scaler.pkl"),
        enable_progress=True,
        enable_early_stopping=True
    )
    assert model_path.exists()
    assert result == str(model_path)
    
    # Check that metadata was created
    metadata_files = list(model_path.parent.glob("metadata_*.json"))
    assert len(metadata_files) > 0


def test_main_without_progress_callbacks(tmp_path):
    """Test training with progress callbacks disabled."""
    model_path = tmp_path / "no_progress_model.h5"
    result = main(
        csv_path="data/raw/sensor_data.csv",
        epochs=2,
        model_path=str(model_path),
        scaler_path=str(tmp_path / "no_progress_scaler.pkl"),
        enable_progress=False,
        enable_early_stopping=False
    )
    assert model_path.exists()
    assert result == str(model_path)
