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
    )
    assert model_path.is_file()
    assert (tmp_path / "scaler.pkl").is_file()
