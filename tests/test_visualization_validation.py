"""Validation tests for enhanced visualization functionality."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Mock matplotlib to avoid import errors in test environment
with patch.dict('sys.modules', {
    'matplotlib': MagicMock(),
    'matplotlib.pyplot': MagicMock(),
    'scipy': MagicMock(),
    'scipy.ndimage': MagicMock()
}):
    from src.visualize import (
        plot_sequences_enhanced,
        plot_reconstruction_error,
        plot_training_history,
        create_dashboard,
        _add_anomaly_highlights
    )


class TestVisualizationValidation:
    """Test the enhanced visualization logic without actual plotting."""
    
    def test_plot_sequences_enhanced_parameter_validation(self, tmp_path):
        """Test parameter validation in enhanced plotting."""
        # Test invalid highlight style
        with pytest.raises(ValueError, match="Invalid highlight_style"):
            plot_sequences_enhanced(
                csv_path="data/raw/sensor_data.csv",
                output=str(tmp_path / "test.png"),
                highlight_style="invalid_style"
            )
    
    def test_plot_sequences_enhanced_file_validation(self, tmp_path):
        """Test file validation in enhanced plotting."""
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            plot_sequences_enhanced(
                csv_path="non_existent_file.csv",
                output=str(tmp_path / "test.png")
            )
    
    def test_selected_features_validation(self, tmp_path):
        """Test feature selection validation."""
        # Create test data
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'feature3': [7, 8, 9]
        })
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Test with non-existent features
        with pytest.raises(ValueError, match="None of the selected features"):
            plot_sequences_enhanced(
                csv_path=str(csv_path),
                output=str(tmp_path / "test.png"),
                selected_features=['non_existent_feature']
            )
    
    @patch('src.visualize.plt')
    @patch('src.visualize.pd.read_csv')
    def test_plot_sequences_enhanced_basic_flow(self, mock_read_csv, mock_plt, tmp_path):
        """Test the basic flow of enhanced plotting."""
        # Mock data
        mock_df = pd.DataFrame({
            'feature1': np.sin(np.linspace(0, 4*np.pi, 100)),
            'feature2': np.cos(np.linspace(0, 4*np.pi, 100))
        })
        mock_read_csv.return_value = mock_df
        
        # Mock matplotlib components
        mock_fig, mock_axes = MagicMock(), [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        # Test successful execution
        plot_sequences_enhanced(
            csv_path="mock_data.csv",
            output=str(tmp_path / "test.png"),
            figure_size=(12, 8),
            anomaly_color='orange',
            show_grid=True,
            title="Test Plot"
        )
        
        # Verify matplotlib functions were called
        mock_plt.subplots.assert_called_once()
        mock_plt.tight_layout.assert_called_once()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()
    
    @patch('src.visualize.plt')
    @patch('src.visualize.pd.read_csv')
    def test_plot_reconstruction_error_flow(self, mock_read_csv, mock_plt, tmp_path):
        """Test reconstruction error plotting flow."""
        # Mock error data
        mock_df = pd.DataFrame({'error': np.random.exponential(0.1, 100)})
        mock_read_csv.return_value = mock_df
        
        # Test with distribution
        plot_reconstruction_error(
            error_csv="mock_errors.csv",
            output=str(tmp_path / "error_plot.png"),
            threshold=0.3,
            show_distribution=True
        )
        
        # Verify plotting calls
        assert mock_plt.subplots.called
        assert mock_plt.savefig.called
        assert mock_plt.close.called
    
    @patch('src.visualize.plt')
    def test_plot_training_history_flow(self, mock_plt, tmp_path):
        """Test training history plotting flow."""
        history = {
            'loss': [0.5, 0.3, 0.2, 0.15, 0.1],
            'val_loss': [0.6, 0.35, 0.25, 0.18, 0.12]
        }
        
        plot_training_history(
            history=history,
            output=str(tmp_path / "history_plot.png"),
            show_validation=True,
            highlight_best_epoch=True
        )
        
        # Verify plotting calls
        assert mock_plt.subplots.called
        assert mock_plt.savefig.called
        assert mock_plt.close.called
    
    @patch('src.visualize.plt')
    @patch('src.visualize.pd.read_csv')
    def test_create_dashboard_flow(self, mock_read_csv, mock_plt, tmp_path):
        """Test dashboard creation flow."""
        # Mock data
        sensor_df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        error_df = pd.DataFrame({'error': np.random.exponential(0.1, 100)})
        
        def read_csv_side_effect(path, **kwargs):
            if 'sensor' in str(path):
                return sensor_df
            elif 'error' in str(path):
                return error_df
            return sensor_df
        
        mock_read_csv.side_effect = read_csv_side_effect
        
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_gs = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_gridspec.return_value = mock_gs
        mock_fig.add_subplot.return_value = MagicMock()
        
        history = {
            'loss': [0.5, 0.3, 0.2, 0.15, 0.1],
            'val_loss': [0.6, 0.35, 0.25, 0.18, 0.12]
        }
        
        create_dashboard(
            sensor_data_csv="sensor_data.csv",
            error_csv="error_data.csv",
            training_history=history,
            output=str(tmp_path / "dashboard.png"),
            threshold=0.3,
            title="Test Dashboard"
        )
        
        # Verify dashboard creation calls
        assert mock_plt.figure.called
        assert mock_plt.savefig.called
        assert mock_plt.close.called
    
    def test_add_anomaly_highlights_styles(self):
        """Test different anomaly highlighting styles."""
        # Mock axis
        mock_ax = MagicMock()
        anomalies = pd.Series([0, 1, 0, 1, 0])
        
        # Test fill style
        _add_anomaly_highlights(mock_ax, anomalies, 'fill', 'red', 0.3)
        assert mock_ax.axvspan.called
        
        # Reset and test line style
        mock_ax.reset_mock()
        _add_anomaly_highlights(mock_ax, anomalies, 'line', 'red', 0.3)
        assert mock_ax.axvline.called
        
        # Reset and test marker style
        mock_ax.reset_mock()
        mock_ax.get_ylim.return_value = (0, 1)
        _add_anomaly_highlights(mock_ax, anomalies, 'marker', 'red', 0.3)
        assert mock_ax.scatter.called
        
        # Reset and test background style
        mock_ax.reset_mock()
        _add_anomaly_highlights(mock_ax, anomalies, 'background', 'red', 0.3)
        assert mock_ax.axvspan.called


def test_visualization_imports():
    """Test that visualization module can be imported without errors."""
    # This test verifies that the module structure is correct
    import src.visualize
    
    # Check that key functions exist
    assert hasattr(src.visualize, 'plot_sequences_enhanced')
    assert hasattr(src.visualize, 'plot_reconstruction_error')
    assert hasattr(src.visualize, 'plot_training_history')
    assert hasattr(src.visualize, 'create_dashboard')
    assert hasattr(src.visualize, '_add_anomaly_highlights')


def test_parameter_types():
    """Test that function signatures have correct parameter types."""
    import inspect
    from src.visualize import plot_sequences_enhanced
    
    sig = inspect.signature(plot_sequences_enhanced)
    
    # Check that key parameters exist
    assert 'csv_path' in sig.parameters
    assert 'output' in sig.parameters
    assert 'figure_size' in sig.parameters
    assert 'anomaly_color' in sig.parameters
    assert 'highlight_style' in sig.parameters