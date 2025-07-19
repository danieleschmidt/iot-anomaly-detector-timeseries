import pytest
import os
import tempfile
from pathlib import Path
from src.config import Config


def test_default_config():
    """Test that default configuration values are loaded correctly."""
    config = Config()
    
    # Test default values exist
    assert hasattr(config, 'WINDOW_SIZE')
    assert hasattr(config, 'LSTM_UNITS')
    assert hasattr(config, 'LATENT_DIM')
    assert hasattr(config, 'BATCH_SIZE')
    assert hasattr(config, 'EPOCHS')
    assert hasattr(config, 'THRESHOLD_FACTOR')
    assert hasattr(config, 'ANOMALY_START')
    assert hasattr(config, 'ANOMALY_LENGTH')
    assert hasattr(config, 'ANOMALY_MAGNITUDE')
    
    # Test default values are reasonable
    assert config.WINDOW_SIZE == 30
    assert config.LSTM_UNITS == 32
    assert config.LATENT_DIM == 16
    assert config.BATCH_SIZE == 32
    assert config.EPOCHS == 100
    assert config.THRESHOLD_FACTOR == 3.0
    assert config.ANOMALY_START == 200
    assert config.ANOMALY_LENGTH == 20
    assert config.ANOMALY_MAGNITUDE == 3.0


def test_config_from_file(tmp_path):
    """Test loading configuration from a file."""
    config_file = tmp_path / "test_config.yaml"
    config_content = """
window_size: 50
lstm_units: 64
latent_dim: 32
batch_size: 16
epochs: 200
threshold_factor: 2.5
anomaly_start: 100
anomaly_length: 30
anomaly_magnitude: 4.0
"""
    config_file.write_text(config_content)
    
    config = Config(str(config_file))
    
    assert config.WINDOW_SIZE == 50
    assert config.LSTM_UNITS == 64
    assert config.LATENT_DIM == 32
    assert config.BATCH_SIZE == 16
    assert config.EPOCHS == 200
    assert config.THRESHOLD_FACTOR == 2.5
    assert config.ANOMALY_START == 100
    assert config.ANOMALY_LENGTH == 30
    assert config.ANOMALY_MAGNITUDE == 4.0


def test_config_from_env_vars(monkeypatch):
    """Test loading configuration from environment variables."""
    monkeypatch.setenv("IOT_WINDOW_SIZE", "25")
    monkeypatch.setenv("IOT_LSTM_UNITS", "48")
    monkeypatch.setenv("IOT_BATCH_SIZE", "8")
    monkeypatch.setenv("IOT_THRESHOLD_FACTOR", "2.0")
    
    config = Config()
    
    assert config.WINDOW_SIZE == 25
    assert config.LSTM_UNITS == 48
    assert config.BATCH_SIZE == 8
    assert config.THRESHOLD_FACTOR == 2.0
    # Non-env vars should use defaults
    assert config.LATENT_DIM == 16
    assert config.EPOCHS == 100


def test_config_validation():
    """Test configuration validation rules."""
    with pytest.raises(ValueError, match="window_size must be positive"):
        Config(config_dict={"window_size": 0})
    
    with pytest.raises(ValueError, match="window_size must be positive"):
        Config(config_dict={"window_size": -5})
    
    with pytest.raises(ValueError, match="lstm_units must be positive"):
        Config(config_dict={"lstm_units": 0})
    
    with pytest.raises(ValueError, match="batch_size must be positive"):
        Config(config_dict={"batch_size": -1})
    
    with pytest.raises(ValueError, match="threshold_factor must be positive"):
        Config(config_dict={"threshold_factor": 0})
    
    with pytest.raises(ValueError, match="epochs must be positive"):
        Config(config_dict={"epochs": -10})


def test_config_precedence(tmp_path, monkeypatch):
    """Test configuration precedence: env vars > file > defaults."""
    # Create config file
    config_file = tmp_path / "config.yaml"
    config_content = """
window_size: 40
lstm_units: 64
batch_size: 16
"""
    config_file.write_text(config_content)
    
    # Set env var (should override file)
    monkeypatch.setenv("IOT_WINDOW_SIZE", "60")
    monkeypatch.setenv("IOT_EPOCHS", "150")
    
    config = Config(str(config_file))
    
    # Env var should override file
    assert config.WINDOW_SIZE == 60  # From env
    # File should override defaults
    assert config.LSTM_UNITS == 64   # From file
    assert config.BATCH_SIZE == 16   # From file
    # Env var should override defaults
    assert config.EPOCHS == 150      # From env
    # Default should be used if not in file or env
    assert config.LATENT_DIM == 16   # Default


def test_config_invalid_file():
    """Test handling of invalid configuration file."""
    with pytest.raises(FileNotFoundError):
        Config("nonexistent_config.yaml")


def test_config_malformed_file(tmp_path):
    """Test handling of malformed configuration file."""
    config_file = tmp_path / "bad_config.yaml"
    config_file.write_text("invalid: yaml: content: [")
    
    with pytest.raises(ValueError, match="Invalid YAML configuration"):
        Config(str(config_file))


def test_config_to_dict():
    """Test converting configuration to dictionary."""
    config = Config()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert config_dict['window_size'] == 30
    assert config_dict['lstm_units'] == 32
    assert config_dict['batch_size'] == 32
    assert len(config_dict) == 9  # All config parameters


def test_config_save(tmp_path):
    """Test saving configuration to file."""
    config = Config()
    save_path = tmp_path / "saved_config.yaml"
    
    config.save(str(save_path))
    
    assert save_path.exists()
    
    # Load saved config and verify
    loaded_config = Config(str(save_path))
    assert loaded_config.WINDOW_SIZE == config.WINDOW_SIZE
    assert loaded_config.LSTM_UNITS == config.LSTM_UNITS