import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .logging_config import get_logger


def plot_sequences(csv_path: str, anomalies: pd.Series = None, output: str = 'plot.png') -> None:
    """Legacy function for basic sequence plotting."""
    df = pd.read_csv(csv_path)
    df.plot(subplots=True, figsize=(10, 8))
    if anomalies is not None:
        for idx, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                plt.axvspan(idx, idx + 1, color='red', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_sequences_enhanced(
    csv_path: str,
    anomalies_path: Optional[str] = None,
    output: str = 'enhanced_plot.png',
    figure_size: tuple[int, int] = (12, 8),
    anomaly_color: str = 'red',
    anomaly_alpha: float = 0.3,
    line_style: str = '-',
    line_width: float = 1.0,
    show_grid: bool = True,
    title: Optional[str] = None,
    highlight_style: str = 'fill',
    selected_features: Optional[List[str]] = None,
    color_palette: Optional[List[str]] = None,
    dpi: int = 300
) -> None:
    """Enhanced sequence plotting with customizable options.
    
    Parameters
    ----------
    csv_path : str
        Path to sensor data CSV file
    anomalies_path : str, optional
        Path to anomaly flags CSV file
    output : str
        Output image path
    figure_size : tuple[int, int]
        Figure size (width, height)
    anomaly_color : str
        Color for anomaly highlighting
    anomaly_alpha : float
        Transparency for anomaly highlighting
    line_style : str
        Line style for plots ('-', '--', '-.', ':')
    line_width : float
        Line width
    show_grid : bool
        Whether to show grid
    title : str, optional
        Plot title
    highlight_style : str
        Anomaly highlighting style ('fill', 'line', 'marker', 'background')
    selected_features : List[str], optional
        Specific features to plot
    color_palette : List[str], optional
        Custom color palette
    dpi : int
        Image DPI
    """
    logger = get_logger(__name__)
    
    # Validate inputs
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    valid_highlight_styles = ['fill', 'line', 'marker', 'background']
    if highlight_style not in valid_highlight_styles:
        raise ValueError(f"Invalid highlight_style. Must be one of: {valid_highlight_styles}")
    
    # Load data
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded data with shape {df.shape} from {csv_path}")
    
    # Filter features if specified
    if selected_features:
        available_features = [col for col in selected_features if col in df.columns]
        if not available_features:
            raise ValueError(f"None of the selected features {selected_features} found in data")
        df = df[available_features]
        logger.info(f"Plotting selected features: {available_features}")
    
    # Load anomalies if provided
    anomalies = None
    if anomalies_path and Path(anomalies_path).exists():
        anomalies = pd.read_csv(anomalies_path, header=None)[0]
        logger.info(f"Loaded {anomalies.sum()} anomalies from {anomalies_path}")
    
    # Set up the plot
    fig, axes = plt.subplots(len(df.columns), 1, figsize=figure_size, sharex=True, dpi=dpi)
    if len(df.columns) == 1:
        axes = [axes]
    
    # Set color palette
    colors = color_palette or plt.cm.tab10.colors[:len(df.columns)]
    
    # Plot each feature
    for i, (column, ax) in enumerate(zip(df.columns, axes)):
        color = colors[i % len(colors)]
        
        # Plot the time series
        ax.plot(df.index, df[column], 
               color=color, 
               linestyle=line_style, 
               linewidth=line_width,
               label=column)
        
        # Add anomaly highlighting
        if anomalies is not None:
            _add_anomaly_highlights(ax, anomalies, highlight_style, anomaly_color, anomaly_alpha)
        
        # Customize appearance
        ax.set_ylabel(column)
        ax.legend(loc='upper right')
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        # Set background style for background highlighting
        if highlight_style == 'background' and anomalies is not None:
            ax.set_facecolor('#f8f8f8')
    
    # Set title and labels
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    axes[-1].set_xlabel('Time Step')
    
    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Enhanced plot saved to {output}")


def _add_anomaly_highlights(ax, anomalies: pd.Series, style: str, color: str, alpha: float) -> None:
    """Add anomaly highlighting to a plot axis."""
    anomaly_indices = anomalies[anomalies == 1].index
    
    if style == 'fill':
        for idx in anomaly_indices:
            ax.axvspan(idx, idx + 1, color=color, alpha=alpha)
    
    elif style == 'line':
        for idx in anomaly_indices:
            ax.axvline(x=idx, color=color, alpha=alpha, linestyle='--', linewidth=2)
    
    elif style == 'marker':
        y_min, y_max = ax.get_ylim()
        for idx in anomaly_indices:
            ax.scatter(idx, y_max * 0.95, color=color, s=50, marker='v', alpha=alpha)
    
    elif style == 'background':
        for idx in anomaly_indices:
            ax.axvspan(idx, idx + 1, color=color, alpha=alpha*0.5, zorder=-1)


def plot_reconstruction_error(
    error_csv: str,
    output: str = 'reconstruction_error.png',
    threshold: Optional[float] = None,
    show_distribution: bool = True,
    show_threshold_line: bool = True,
    figure_size: tuple[int, int] = (12, 6),
    dpi: int = 300
) -> None:
    """Plot reconstruction error with distribution and threshold visualization.
    
    Parameters
    ----------
    error_csv : str
        Path to CSV file with reconstruction errors
    output : str
        Output image path
    threshold : float, optional
        Anomaly threshold to highlight
    show_distribution : bool
        Whether to show error distribution histogram
    show_threshold_line : bool
        Whether to show threshold line
    figure_size : tuple[int, int]
        Figure size
    dpi : int
        Image DPI
    """
    logger = get_logger(__name__)
    
    # Load reconstruction errors
    df = pd.read_csv(error_csv)
    errors = df.iloc[:, 0]  # Assume first column contains errors
    
    logger.info(f"Plotting reconstruction errors: mean={errors.mean():.4f}, std={errors.std():.4f}")
    
    if show_distribution:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figure_size, dpi=dpi, height_ratios=[2, 1])
    else:
        fig, ax1 = plt.subplots(figsize=figure_size, dpi=dpi)
    
    # Plot time series of reconstruction errors
    ax1.plot(errors.index, errors, color='blue', alpha=0.7, linewidth=1)
    ax1.set_ylabel('Reconstruction Error')
    ax1.set_title('Reconstruction Error Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Add threshold line if provided
    if threshold and show_threshold_line:
        ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {threshold:.3f}')
        # Highlight anomalies
        anomalies = errors > threshold
        if anomalies.any():
            ax1.fill_between(errors.index, 0, errors, 
                           where=anomalies, color='red', alpha=0.3, 
                           label=f'Anomalies: {anomalies.sum()}')
        ax1.legend()
    
    # Plot distribution if requested
    if show_distribution:
        ax2.hist(errors, bins=50, alpha=0.7, color='blue', density=True)
        ax2.set_xlabel('Reconstruction Error')
        ax2.set_ylabel('Density')
        ax2.set_title('Error Distribution')
        
        if threshold and show_threshold_line:
            ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
        
        ax2.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel('Time Step')
    
    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Reconstruction error plot saved to {output}")


def plot_training_history(
    history: Dict[str, List[float]],
    output: str = 'training_history.png',
    show_validation: bool = True,
    highlight_best_epoch: bool = True,
    smooth_curves: bool = False,
    figure_size: tuple[int, int] = (10, 6),
    dpi: int = 300
) -> None:
    """Plot training history with loss curves.
    
    Parameters
    ----------
    history : Dict[str, List[float]]
        Training history dictionary
    output : str
        Output image path
    show_validation : bool
        Whether to show validation loss
    highlight_best_epoch : bool
        Whether to highlight best epoch
    smooth_curves : bool
        Whether to smooth the curves
    figure_size : tuple[int, int]
        Figure size
    dpi : int
        Image DPI
    """
    logger = get_logger(__name__)
    
    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Plot training loss
    train_loss = history['loss']
    if smooth_curves and HAS_SCIPY:
        train_loss = ndimage.gaussian_filter1d(train_loss, sigma=1.0)
    elif smooth_curves:
        logger.warning("scipy not available, skipping curve smoothing")
    
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    
    # Plot validation loss if available
    if show_validation and 'val_loss' in history:
        val_loss = history['val_loss']
        if smooth_curves and HAS_SCIPY:
            val_loss = ndimage.gaussian_filter1d(val_loss, sigma=1.0)
        
        ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
        
        # Highlight best epoch
        if highlight_best_epoch:
            best_epoch = np.argmin(val_loss) + 1
            best_loss = min(val_loss)
            ax.scatter(best_epoch, best_loss, color='red', s=100, marker='*', 
                      zorder=5, label=f'Best Epoch: {best_epoch}')
    elif highlight_best_epoch:
        best_epoch = np.argmin(train_loss) + 1
        best_loss = min(train_loss)
        ax.scatter(best_epoch, best_loss, color='blue', s=100, marker='*', 
                  zorder=5, label=f'Best Epoch: {best_epoch}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training history plot saved to {output}")


def create_dashboard(
    sensor_data_csv: str,
    error_csv: str,
    training_history: Dict[str, List[float]],
    output: str = 'dashboard.png',
    threshold: Optional[float] = None,
    title: str = 'IoT Anomaly Detection Dashboard',
    figure_size: tuple[int, int] = (16, 12),
    dpi: int = 300
) -> None:
    """Create a comprehensive dashboard with multiple visualizations.
    
    Parameters
    ----------
    sensor_data_csv : str
        Path to sensor data CSV
    error_csv : str
        Path to reconstruction error CSV
    training_history : Dict[str, List[float]]
        Training history dictionary
    output : str
        Output image path
    threshold : float, optional
        Anomaly threshold
    title : str
        Dashboard title
    figure_size : tuple[int, int]
        Figure size
    dpi : int
        Image DPI
    """
    logger = get_logger(__name__)
    
    # Load data
    sensor_df = pd.read_csv(sensor_data_csv)
    error_df = pd.read_csv(error_csv)
    errors = error_df.iloc[:, 0]
    
    # Create dashboard layout
    fig = plt.figure(figsize=figure_size, dpi=dpi)
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    # 1. Sensor data with anomalies (top, spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot first 3 features
    features_to_plot = sensor_df.columns[:min(3, len(sensor_df.columns))]
    colors = ['blue', 'green', 'orange']
    
    for i, feature in enumerate(features_to_plot):
        ax1.plot(sensor_df.index, sensor_df[feature], 
                color=colors[i], alpha=0.7, linewidth=1, label=feature)
    
    # Add anomaly highlighting if threshold provided
    if threshold:
        anomalies = errors > threshold
        for idx in anomalies[anomalies].index:
            ax1.axvspan(idx, idx + 1, color='red', alpha=0.2)
    
    ax1.set_title('Sensor Data with Anomaly Highlighting')
    ax1.set_ylabel('Sensor Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reconstruction error (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(errors.index, errors, color='blue', alpha=0.7, linewidth=1)
    if threshold:
        ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2)
        anomalies = errors > threshold
        ax2.fill_between(errors.index, 0, errors, where=anomalies, 
                        color='red', alpha=0.3)
    ax2.set_title('Reconstruction Error')
    ax2.set_ylabel('Error')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error distribution (bottom middle-left)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(errors, bins=30, alpha=0.7, color='blue', density=True)
    if threshold:
        ax3.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
    ax3.set_title('Error Distribution')
    ax3.set_xlabel('Reconstruction Error')
    ax3.set_ylabel('Density')
    ax3.grid(True, alpha=0.3)
    
    # 4. Training history (bottom right)
    ax4 = fig.add_subplot(gs[2, :])
    epochs = range(1, len(training_history['loss']) + 1)
    ax4.plot(epochs, training_history['loss'], 'b-', linewidth=2, label='Training Loss')
    if 'val_loss' in training_history:
        ax4.plot(epochs, training_history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax4.set_title('Training History')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add main title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Dashboard saved to {output}")


def main(
    csv_path: str = "data/raw/sensor_data.csv",
    anomalies_path: str | None = None,
    output: str = "plot.png",
    enhanced: bool = False,
    **kwargs
) -> None:
    """Plot sequences from ``csv_path`` highlighting anomalies if provided.
    
    Parameters
    ----------
    csv_path : str
        Path to sensor data CSV file
    anomalies_path : str, optional
        Path to anomaly flags CSV file
    output : str
        Output image path
    enhanced : bool
        Whether to use enhanced plotting
    **kwargs
        Additional arguments for enhanced plotting
    """
    if enhanced:
        plot_sequences_enhanced(
            csv_path=csv_path,
            anomalies_path=anomalies_path,
            output=output,
            **kwargs
        )
    else:
        # Legacy plotting
        anomalies = None
        if anomalies_path:
            anomalies = pd.read_csv(anomalies_path, header=None)[0]
        plot_sequences(csv_path, anomalies, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot sensor sequences with enhanced visualization options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic plotting
  python -m src.visualize --csv-path data/raw/sensor_data.csv --output basic.png
  
  # Enhanced plotting with custom options
  python -m src.visualize --csv-path data/raw/sensor_data.csv --enhanced \
      --figure-size 14 10 --anomaly-color orange --show-grid --title "Custom Plot"
  
  # Create reconstruction error plot
  python -m src.visualize --mode error --error-csv errors.csv --threshold 0.3
  
  # Create comprehensive dashboard
  python -m src.visualize --mode dashboard --csv-path data.csv --error-csv errors.csv
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        choices=["sequence", "error", "dashboard"], 
        default="sequence",
        help="Visualization mode"
    )
    
    # Common arguments
    parser.add_argument("--csv-path", default="data/raw/sensor_data.csv", help="Sensor data CSV path")
    parser.add_argument("--output", default="plot.png", help="Output image path")
    parser.add_argument("--anomalies", help="CSV file containing anomaly flags")
    
    # Enhanced plotting options
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced plotting")
    parser.add_argument("--figure-size", nargs=2, type=int, default=[12, 8], help="Figure size (width height)")
    parser.add_argument("--anomaly-color", default="red", help="Anomaly highlight color")
    parser.add_argument("--anomaly-alpha", type=float, default=0.3, help="Anomaly highlight transparency")
    parser.add_argument("--line-style", default="-", help="Line style")
    parser.add_argument("--line-width", type=float, default=1.0, help="Line width")
    parser.add_argument("--show-grid", action="store_true", help="Show grid")
    parser.add_argument("--title", help="Plot title")
    parser.add_argument("--highlight-style", choices=["fill", "line", "marker", "background"], 
                       default="fill", help="Anomaly highlighting style")
    parser.add_argument("--selected-features", nargs="+", help="Specific features to plot")
    parser.add_argument("--dpi", type=int, default=300, help="Image DPI")
    
    # Error plot specific options
    parser.add_argument("--error-csv", help="Reconstruction error CSV path")
    parser.add_argument("--threshold", type=float, help="Anomaly threshold")
    parser.add_argument("--show-distribution", action="store_true", help="Show error distribution")
    
    # Dashboard specific options
    parser.add_argument("--dashboard-title", default="IoT Anomaly Detection Dashboard", 
                       help="Dashboard title")
    
    args = parser.parse_args()
    
    if args.mode == "sequence":
        if args.enhanced:
            plot_sequences_enhanced(
                csv_path=args.csv_path,
                anomalies_path=args.anomalies,
                output=args.output,
                figure_size=tuple(args.figure_size),
                anomaly_color=args.anomaly_color,
                anomaly_alpha=args.anomaly_alpha,
                line_style=args.line_style,
                line_width=args.line_width,
                show_grid=args.show_grid,
                title=args.title,
                highlight_style=args.highlight_style,
                selected_features=args.selected_features,
                dpi=args.dpi
            )
        else:
            main(csv_path=args.csv_path, anomalies_path=args.anomalies, output=args.output)
    
    elif args.mode == "error":
        if not args.error_csv:
            parser.error("--error-csv is required for error mode")
        plot_reconstruction_error(
            error_csv=args.error_csv,
            output=args.output,
            threshold=args.threshold,
            show_distribution=args.show_distribution,
            figure_size=tuple(args.figure_size),
            dpi=args.dpi
        )
    
    elif args.mode == "dashboard":
        if not args.error_csv:
            parser.error("--error-csv is required for dashboard mode")
        
        # Create mock training history if not provided
        training_history = {
            'loss': [0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.09, 0.08],
            'val_loss': [0.6, 0.35, 0.25, 0.18, 0.15, 0.13, 0.12, 0.11]
        }
        
        create_dashboard(
            sensor_data_csv=args.csv_path,
            error_csv=args.error_csv,
            training_history=training_history,
            output=args.output,
            threshold=args.threshold,
            title=args.dashboard_title,
            figure_size=tuple(args.figure_size),
            dpi=args.dpi
        )
