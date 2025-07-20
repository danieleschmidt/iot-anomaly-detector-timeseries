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


def test_main_with_flexible_architecture(tmp_path):
    """Test training with flexible architecture system."""
    model_path = tmp_path / "flexible_model.h5"
    result = main(
        csv_path="data/raw/sensor_data.csv",
        epochs=1,
        model_path=str(model_path),
        scaler_path=str(tmp_path / "flexible_scaler.pkl"),
        enable_progress=False,
        use_flexible_architecture=True
    )
    assert model_path.exists()
    assert result == str(model_path)


def test_main_with_predefined_architecture(tmp_path):
    """Test training with predefined architecture."""
    model_path = tmp_path / "predefined_model.h5"
    result = main(
        csv_path="data/raw/sensor_data.csv",
        epochs=1,
        model_path=str(model_path),
        scaler_path=str(tmp_path / "predefined_scaler.pkl"),
        enable_progress=False,
        architecture_name="simple_lstm"
    )
    assert model_path.exists()
    assert result == str(model_path)


def test_main_with_architecture_config_file(tmp_path):
    """Test training with architecture configuration file."""
    import json
    
    # Create a simple architecture config
    config = {
        "name": "test_architecture",
        "description": "Test architecture for validation",
        "input_shape": [30, 3],
        "encoder_layers": [
            {"type": "lstm", "units": 32, "return_sequences": False}
        ],
        "latent_config": {"dim": 8, "activation": "linear"},
        "compilation": {"optimizer": "adam", "loss": "mse"}
    }
    
    config_path = tmp_path / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    model_path = tmp_path / "config_model.h5"
    result = main(
        csv_path="data/raw/sensor_data.csv",
        epochs=1,
        model_path=str(model_path),
        scaler_path=str(tmp_path / "config_scaler.pkl"),
        enable_progress=False,
        architecture_config=str(config_path)
    )
    assert model_path.exists()
    assert result == str(model_path)
