import pytest
import pandas as pd
import numpy as np

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from src.visualize import (
    main, 
    plot_sequences,
    plot_sequences_enhanced,
    plot_reconstruction_error,
    plot_training_history,
    create_dashboard
)


def test_plot_sequences_basic(tmp_path):
    """Test basic plot_sequences function (legacy)."""
    csv = 'data/raw/sensor_data.csv'
    out = tmp_path / 'plot.png'
    plot_sequences(csv, output=str(out))
    assert out.is_file()


def test_cli_creates_image(tmp_path):
    out = tmp_path / 'out.png'
    main(csv_path='data/raw/sensor_data.csv', output=str(out))
    assert out.is_file()


def test_plot_sequences_enhanced_custom_settings(tmp_path):
    """Test enhanced plotting with custom settings."""
    csv = 'data/raw/sensor_data.csv'
    out = tmp_path / 'enhanced_plot.png'
    
    plot_sequences_enhanced(
        csv_path=csv,
        output=str(out),
        figure_size=(12, 10),
        anomaly_color='orange',
        anomaly_alpha=0.5,
        line_style='--',
        line_width=2.0,
        show_grid=True,
        title="Custom Enhanced Plot"
    )
    assert out.is_file()


def test_plot_sequences_enhanced_with_anomalies(tmp_path):
    """Test enhanced plotting with anomaly highlighting."""
    # Create test data
    df = pd.DataFrame({
        'feature1': np.sin(np.linspace(0, 4*np.pi, 100)),
        'feature2': np.cos(np.linspace(0, 4*np.pi, 100)),
        'feature3': np.random.randn(100) * 0.1
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    # Create anomaly flags
    anomalies = pd.Series([0] * 80 + [1] * 10 + [0] * 10)
    anomaly_path = tmp_path / "anomalies.csv"
    anomalies.to_csv(anomaly_path, index=False, header=False)
    
    out = tmp_path / 'anomaly_plot.png'
    plot_sequences_enhanced(
        csv_path=str(csv_path),
        anomalies_path=str(anomaly_path),
        output=str(out),
        highlight_style='fill',
        anomaly_color='red',
        anomaly_alpha=0.3
    )
    assert out.is_file()


def test_plot_reconstruction_error(tmp_path):
    """Test reconstruction error visualization."""
    # Create mock reconstruction error data
    scores = np.random.exponential(0.1, 200)  # Typical reconstruction error distribution
    scores[150:170] = np.random.exponential(0.5, 20)  # Simulate anomalies
    
    error_df = pd.DataFrame({'reconstruction_error': scores})
    error_path = tmp_path / "errors.csv"
    error_df.to_csv(error_path, index=False)
    
    out = tmp_path / 'error_plot.png'
    plot_reconstruction_error(
        error_csv=str(error_path),
        output=str(out),
        threshold=0.3,
        show_distribution=True,
        show_threshold_line=True
    )
    assert out.is_file()


def test_plot_training_history(tmp_path):
    """Test training history visualization."""
    # Create mock training history
    epochs = 20
    history = {
        'loss': [0.5 * np.exp(-i/10) + 0.05 + np.random.normal(0, 0.01) for i in range(epochs)],
        'val_loss': [0.5 * np.exp(-i/8) + 0.08 + np.random.normal(0, 0.02) for i in range(epochs)]
    }
    
    out = tmp_path / 'training_plot.png'
    plot_training_history(
        history=history,
        output=str(out),
        show_validation=True,
        highlight_best_epoch=True,
        smooth_curves=True
    )
    assert out.is_file()


def test_create_dashboard(tmp_path):
    """Test comprehensive dashboard creation."""
    # Create test sensor data
    df = pd.DataFrame({
        'feature1': np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100),
        'feature2': np.cos(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100),
        'feature3': np.random.randn(100) * 0.3
    })
    csv_path = tmp_path / "dashboard_data.csv"
    df.to_csv(csv_path, index=False)
    
    # Create reconstruction errors
    errors = np.random.exponential(0.1, 100)
    errors[70:80] = np.random.exponential(0.4, 10)  # Anomalies
    error_df = pd.DataFrame({'reconstruction_error': errors})
    error_path = tmp_path / "dashboard_errors.csv"
    error_df.to_csv(error_path, index=False)
    
    # Create training history
    history = {
        'loss': [0.3 * np.exp(-i/8) + 0.05 for i in range(15)],
        'val_loss': [0.3 * np.exp(-i/6) + 0.08 for i in range(15)]
    }
    
    out = tmp_path / 'dashboard.png'
    create_dashboard(
        sensor_data_csv=str(csv_path),
        error_csv=str(error_path),
        training_history=history,
        output=str(out),
        threshold=0.25,
        title="IoT Anomaly Detection Dashboard"
    )
    assert out.is_file()


def test_plot_sequences_enhanced_different_highlight_styles(tmp_path):
    """Test different anomaly highlighting styles."""
    csv = 'data/raw/sensor_data.csv'
    
    # Test different highlight styles
    for style in ['fill', 'line', 'marker', 'background']:
        out = tmp_path / f'highlight_{style}.png'
        plot_sequences_enhanced(
            csv_path=csv,
            output=str(out),
            highlight_style=style,
            anomaly_color='red'
        )
        assert out.is_file()


def test_plot_sequences_enhanced_error_handling(tmp_path):
    """Test error handling in enhanced plotting."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        plot_sequences_enhanced(
            csv_path="non_existent.csv",
            output=str(tmp_path / "error.png")
        )
    
    # Test with invalid highlight style
    with pytest.raises(ValueError):
        plot_sequences_enhanced(
            csv_path='data/raw/sensor_data.csv',
            output=str(tmp_path / "invalid.png"),
            highlight_style='invalid_style'
        )


def test_plot_sequences_enhanced_feature_selection(tmp_path):
    """Test plotting specific features only."""
    csv = 'data/raw/sensor_data.csv'
    out = tmp_path / 'feature_selection.png'
    
    plot_sequences_enhanced(
        csv_path=csv,
        output=str(out),
        selected_features=['feature1', 'feature2'],  # Only plot first two features
        figure_size=(10, 6)
    )
    assert out.is_file()
