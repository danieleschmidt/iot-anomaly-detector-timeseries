"""Custom training callbacks for progress indication and monitoring."""

import time
from typing import Dict, Any, Optional
from tensorflow.keras.callbacks import Callback

from .logging_config import get_logger


class ProgressCallback(Callback):
    """Custom callback to provide detailed progress indication during training."""
    
    def __init__(self, 
                 total_epochs: int,
                 log_frequency: int = 1,
                 time_estimate: bool = True):
        """Initialize progress callback.
        
        Parameters
        ----------
        total_epochs : int
            Total number of training epochs
        log_frequency : int, optional
            Log progress every N epochs (default: 1)
        time_estimate : bool, optional
            Whether to estimate remaining time (default: True)
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.log_frequency = log_frequency
        self.time_estimate = time_estimate
        self.logger = get_logger(__name__)
        
        # Timing attributes
        self.start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None
        self.epoch_durations: list[float] = []
        
        # Training metrics
        self.best_loss: float = float('inf')
        self.loss_improvement_count: int = 0
        
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called when training begins."""
        self.start_time = time.time()
        self.logger.info(f"Starting training for {self.total_epochs} epochs", extra={
            "total_epochs": self.total_epochs,
            "estimated_duration": "calculating..."
        })
        
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each epoch."""
        self.epoch_start_time = time.time()
        
        if epoch % self.log_frequency == 0:
            progress_percent = (epoch / self.total_epochs) * 100
            self.logger.info(f"Starting epoch {epoch + 1}/{self.total_epochs} ({progress_percent:.1f}%)")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each epoch."""
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            self.epoch_durations.append(epoch_duration)
        
        current_loss = logs.get('loss', float('inf')) if logs else float('inf')
        
        # Track loss improvements
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.loss_improvement_count += 1
            improvement_status = "improved ✓"
        else:
            improvement_status = "no improvement"
        
        # Calculate time estimates
        time_info = {}
        if self.time_estimate and self.epoch_durations:
            avg_epoch_time = sum(self.epoch_durations) / len(self.epoch_durations)
            remaining_epochs = self.total_epochs - (epoch + 1)
            estimated_remaining = avg_epoch_time * remaining_epochs
            
            time_info = {
                "epoch_duration": round(epoch_duration, 2),
                "avg_epoch_duration": round(avg_epoch_time, 2),
                "estimated_remaining_minutes": round(estimated_remaining / 60, 1)
            }
        
        # Log progress
        if (epoch + 1) % self.log_frequency == 0 or (epoch + 1) == self.total_epochs:
            progress_percent = ((epoch + 1) / self.total_epochs) * 100
            
            log_extra = {
                "epoch": epoch + 1,
                "total_epochs": self.total_epochs,
                "progress_percent": round(progress_percent, 1),
                "current_loss": round(current_loss, 6),
                "best_loss": round(self.best_loss, 6),
                "improvement_status": improvement_status,
                **time_info
            }
            
            self.logger.info(
                f"Epoch {epoch + 1}/{self.total_epochs} - Loss: {current_loss:.6f} - {improvement_status}",
                extra=log_extra
            )
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called when training completes."""
        if self.start_time is not None:
            total_duration = time.time() - self.start_time
            avg_epoch_time = sum(self.epoch_durations) / len(self.epoch_durations) if self.epoch_durations else 0
            
            self.logger.info("Training completed", extra={
                "total_duration_minutes": round(total_duration / 60, 2),
                "average_epoch_duration": round(avg_epoch_time, 2),
                "total_epochs": self.total_epochs,
                "final_loss": round(self.best_loss, 6),
                "loss_improvements": self.loss_improvement_count
            })


class EarlyStoppingWithLogging(Callback):
    """Enhanced early stopping with detailed logging."""
    
    def __init__(self, 
                 monitor: str = 'loss',
                 patience: int = 10,
                 min_delta: float = 0.0001,
                 restore_best_weights: bool = True):
        """Initialize early stopping callback.
        
        Parameters
        ----------
        monitor : str
            Metric to monitor (default: 'loss')
        patience : int
            Number of epochs with no improvement after which training stops
        min_delta : float
            Minimum change to qualify as an improvement
        restore_best_weights : bool
            Whether to restore best weights when stopping
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.logger = get_logger(__name__)
        
        self.best_value: Optional[float] = None
        self.best_weights: Optional[Any] = None
        self.wait: int = 0
        self.stopped_epoch: int = 0
        
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Initialize early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float('inf')  # Assuming we want to minimize the metric
        
        self.logger.info(f"Early stopping configured: monitor={self.monitor}, patience={self.patience}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check for early stopping condition."""
        if logs is None:
            return
            
        current_value = logs.get(self.monitor)
        if current_value is None:
            self.logger.warning(f"Early stopping metric '{self.monitor}' not found in logs")
            return
        
        # Check if we have improvement
        if self.best_value is None or (current_value < self.best_value - self.min_delta):
            self.best_value = current_value
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            
            self.logger.debug(f"New best {self.monitor}: {current_value:.6f}")
        else:
            self.wait += 1
            self.logger.debug(f"No improvement for {self.wait}/{self.patience} epochs")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs", extra={
                    "stopped_epoch": epoch + 1,
                    "best_value": self.best_value,
                    "patience": self.patience,
                    "monitor": self.monitor
                })
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Restore best weights if early stopping occurred."""
        if self.stopped_epoch > 0:
            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)
                self.logger.info("Restored best weights from early stopping")
            
            self.logger.info(f"Training stopped early at epoch {self.stopped_epoch + 1}")
        

class MetricsLoggingCallback(Callback):
    """Callback to log detailed training metrics."""
    
    def __init__(self, log_frequency: int = 5):
        """Initialize metrics logging callback.
        
        Parameters
        ----------
        log_frequency : int
            Log detailed metrics every N epochs
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.logger = get_logger(__name__)
        self.metrics_history: Dict[str, list[float]] = {}
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log detailed metrics."""
        if logs is None:
            return
        
        # Store metrics history
        for metric, value in logs.items():
            if metric not in self.metrics_history:
                self.metrics_history[metric] = []
            self.metrics_history[metric].append(value)
        
        # Log detailed metrics periodically
        if (epoch + 1) % self.log_frequency == 0:
            log_extra = {f"metric_{k}": round(v, 6) for k, v in logs.items()}
            log_extra["epoch"] = epoch + 1
            
            self.logger.info(f"Detailed metrics at epoch {epoch + 1}", extra=log_extra)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log final metrics summary."""
        if not self.metrics_history:
            return
        
        summary = {}
        for metric, values in self.metrics_history.items():
            if values:
                summary[f"{metric}_min"] = round(min(values), 6)
                summary[f"{metric}_max"] = round(max(values), 6)
                summary[f"{metric}_final"] = round(values[-1], 6)
                summary[f"{metric}_mean"] = round(sum(values) / len(values), 6)
        
        self.logger.info("Training metrics summary", extra=summary)


def create_training_callbacks(
    epochs: int,
    enable_early_stopping: bool = True,
    early_stopping_patience: int = 10,
    progress_log_frequency: int = 1,
    metrics_log_frequency: int = 5
) -> list[Callback]:
    """Create a standard set of training callbacks.
    
    Parameters
    ----------
    epochs : int
        Total number of training epochs
    enable_early_stopping : bool
        Whether to enable early stopping
    early_stopping_patience : int
        Patience for early stopping
    progress_log_frequency : int
        Frequency for progress logging
    metrics_log_frequency : int
        Frequency for detailed metrics logging
        
    Returns
    -------
    list[Callback]
        List of configured callbacks
    """
    callbacks = [
        ProgressCallback(
            total_epochs=epochs,
            log_frequency=progress_log_frequency
        ),
        MetricsLoggingCallback(
            log_frequency=metrics_log_frequency
        )
    ]
    
    if enable_early_stopping and epochs > early_stopping_patience:
        callbacks.append(
            EarlyStoppingWithLogging(
                patience=early_stopping_patience,
                min_delta=0.0001
            )
        )
    
    return callbacks