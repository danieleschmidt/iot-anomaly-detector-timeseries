import pytest
import unittest.mock
import tempfile
from pathlib import Path

pytest.importorskip("tensorflow")

from src.training_callbacks import (
    ProgressCallback,
    EarlyStoppingWithLogging,
    MetricsLoggingCallback,
    create_training_callbacks
)


class MockModel:
    """Mock Keras model for testing callbacks."""
    
    def __init__(self):
        self.stop_training = False
        self.weights = [1, 2, 3]  # Mock weights
    
    def get_weights(self):
        return self.weights.copy()
    
    def set_weights(self, weights):
        self.weights = weights


def test_progress_callback_basic_functionality():
    """Test basic functionality of ProgressCallback."""
    callback = ProgressCallback(total_epochs=5, log_frequency=1)
    callback.model = MockModel()
    
    # Test training begin
    callback.on_train_begin()
    assert callback.start_time is not None
    
    # Test epoch begin/end cycle
    callback.on_epoch_begin(0)
    assert callback.epoch_start_time is not None
    
    callback.on_epoch_end(0, logs={'loss': 0.5})
    assert len(callback.epoch_durations) == 1
    assert callback.best_loss == 0.5
    assert callback.loss_improvement_count == 1
    
    # Test improvement tracking
    callback.on_epoch_end(1, logs={'loss': 0.3})
    assert callback.best_loss == 0.3
    assert callback.loss_improvement_count == 2
    
    # Test no improvement
    callback.on_epoch_end(2, logs={'loss': 0.4})
    assert callback.best_loss == 0.3  # Should remain the same
    assert callback.loss_improvement_count == 2  # Should not increase
    
    # Test training end
    callback.on_train_end()


def test_progress_callback_time_estimation():
    """Test time estimation functionality."""
    callback = ProgressCallback(total_epochs=10, time_estimate=True)
    callback.model = MockModel()
    
    callback.on_train_begin()
    
    # Simulate some epochs with known durations
    import time
    for epoch in range(3):
        callback.on_epoch_begin(epoch)
        time.sleep(0.01)  # Small delay to simulate training time
        callback.on_epoch_end(epoch, logs={'loss': 0.5 - epoch * 0.1})
    
    assert len(callback.epoch_durations) == 3
    assert all(duration > 0 for duration in callback.epoch_durations)


def test_early_stopping_callback():
    """Test early stopping functionality."""
    callback = EarlyStoppingWithLogging(monitor='loss', patience=2, min_delta=0.01)
    callback.model = MockModel()
    
    callback.on_train_begin()
    assert callback.wait == 0
    assert callback.best_value == float('inf')
    
    # First epoch - improvement
    callback.on_epoch_end(0, logs={'loss': 0.5})
    assert callback.wait == 0
    assert callback.best_value == 0.5
    assert not callback.model.stop_training
    
    # Second epoch - small improvement (less than min_delta)
    callback.on_epoch_end(1, logs={'loss': 0.495})
    assert callback.wait == 1  # Should increment wait since improvement < min_delta
    
    # Third epoch - no improvement
    callback.on_epoch_end(2, logs={'loss': 0.51})
    assert callback.wait == 2
    assert callback.model.stop_training  # Should trigger early stopping
    
    callback.on_train_end()


def test_early_stopping_weight_restoration():
    """Test weight restoration in early stopping."""
    callback = EarlyStoppingWithLogging(
        monitor='loss', 
        patience=1, 
        restore_best_weights=True
    )
    callback.model = MockModel()
    
    callback.on_train_begin()
    
    # Best epoch
    callback.model.weights = [1, 2, 3]
    callback.on_epoch_end(0, logs={'loss': 0.3})
    best_weights = callback.model.get_weights()
    
    # Worse epoch that triggers early stopping
    callback.model.weights = [4, 5, 6]  # Simulate weight change
    callback.on_epoch_end(1, logs={'loss': 0.5})
    
    # Weights should be restored
    callback.on_train_end()
    assert callback.model.weights == best_weights


def test_metrics_logging_callback():
    """Test metrics logging functionality."""
    callback = MetricsLoggingCallback(log_frequency=2)
    callback.model = MockModel()
    
    # Test metrics collection
    callback.on_epoch_end(0, logs={'loss': 0.5, 'accuracy': 0.8})
    callback.on_epoch_end(1, logs={'loss': 0.4, 'accuracy': 0.85})
    callback.on_epoch_end(2, logs={'loss': 0.3, 'accuracy': 0.9})
    
    assert 'loss' in callback.metrics_history
    assert 'accuracy' in callback.metrics_history
    assert len(callback.metrics_history['loss']) == 3
    assert callback.metrics_history['loss'] == [0.5, 0.4, 0.3]
    
    # Test training end summary
    callback.on_train_end()


def test_create_training_callbacks():
    """Test the callback factory function."""
    # Test basic callback creation
    callbacks = create_training_callbacks(epochs=10)
    assert len(callbacks) >= 2  # Should have at least ProgressCallback and MetricsLoggingCallback
    
    callback_types = [type(cb).__name__ for cb in callbacks]
    assert 'ProgressCallback' in callback_types
    assert 'MetricsLoggingCallback' in callback_types
    assert 'EarlyStoppingWithLogging' in callback_types  # Should be enabled for 10 epochs
    
    # Test without early stopping
    callbacks_no_early = create_training_callbacks(epochs=5, enable_early_stopping=False)
    callback_types_no_early = [type(cb).__name__ for cb in callbacks_no_early]
    assert 'EarlyStoppingWithLogging' not in callback_types_no_early
    
    # Test with very few epochs (early stopping should be disabled)
    callbacks_few_epochs = create_training_callbacks(epochs=3)
    callback_types_few = [type(cb).__name__ for cb in callbacks_few_epochs]
    assert 'EarlyStoppingWithLogging' not in callback_types_few


def test_progress_callback_handles_none_logs():
    """Test that callbacks handle None logs gracefully."""
    callback = ProgressCallback(total_epochs=5)
    callback.model = MockModel()
    
    callback.on_train_begin()
    callback.on_epoch_begin(0)
    
    # Should not crash with None logs
    callback.on_epoch_end(0, logs=None)
    callback.on_train_end()


def test_early_stopping_handles_missing_metric():
    """Test early stopping when monitored metric is missing."""
    callback = EarlyStoppingWithLogging(monitor='validation_loss', patience=2)
    callback.model = MockModel()
    
    callback.on_train_begin()
    
    # Epoch with missing metric - should not crash
    callback.on_epoch_end(0, logs={'loss': 0.5})  # validation_loss not present
    assert callback.wait == 0  # Should not change wait counter
    assert not callback.model.stop_training


def test_progress_callback_log_frequency():
    """Test that progress callback respects log frequency."""
    callback = ProgressCallback(total_epochs=10, log_frequency=3)
    callback.model = MockModel()
    
    callback.on_train_begin()
    
    # Only epochs 0, 3, 6, 9 should trigger detailed logging
    # This is tested by ensuring the callback doesn't crash and handles the frequency correctly
    for epoch in range(10):
        callback.on_epoch_begin(epoch)
        callback.on_epoch_end(epoch, logs={'loss': 0.5})
    
    callback.on_train_end()
    assert len(callback.epoch_durations) == 10


def test_callbacks_with_empty_logs():
    """Test callbacks handle empty logs dictionary."""
    progress_cb = ProgressCallback(total_epochs=2)
    metrics_cb = MetricsLoggingCallback()
    early_cb = EarlyStoppingWithLogging()
    
    for cb in [progress_cb, metrics_cb, early_cb]:
        cb.model = MockModel()
        cb.on_train_begin()
        cb.on_epoch_end(0, logs={})  # Empty logs
        cb.on_train_end()


@pytest.mark.parametrize("epochs,expected_min_callbacks", [
    (1, 2),    # ProgressCallback + MetricsLoggingCallback (no early stopping)
    (5, 2),    # Same, early stopping disabled for few epochs
    (15, 3),   # All three callbacks enabled
    (100, 3),  # All three callbacks enabled
])
def test_callback_count_by_epochs(epochs, expected_min_callbacks):
    """Test that appropriate number of callbacks are created based on epoch count."""
    callbacks = create_training_callbacks(epochs=epochs)
    assert len(callbacks) >= expected_min_callbacks