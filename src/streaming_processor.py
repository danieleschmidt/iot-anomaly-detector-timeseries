"""Real-time streaming data processing for IoT anomaly detection."""

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Callable, Optional, Any, Union
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from .data_preprocessor import DataPreprocessor
from .logging_config import get_logger, performance_monitor


@dataclass
class StreamingConfig:
    """Configuration for streaming data processing."""
    
    window_size: int = 50
    step_size: int = 1
    batch_size: int = 32
    anomaly_threshold: float = 0.5
    buffer_size: int = 1000
    processing_interval: float = 1.0  # seconds
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.window_size <= 0:
            raise ValueError("Window size must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.anomaly_threshold < 0:
            raise ValueError("Anomaly threshold must be non-negative")
        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
        if self.processing_interval <= 0:
            raise ValueError("Processing interval must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'StreamingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


class StreamingProcessor:
    """Real-time streaming processor for IoT anomaly detection."""
    
    def __init__(
        self,
        model_path: str,
        config: StreamingConfig,
        scaler_path: Optional[str] = None
    ):
        """Initialize streaming processor.
        
        Parameters
        ----------
        model_path : str
            Path to trained autoencoder model
        config : StreamingConfig
            Streaming processing configuration
        scaler_path : str, optional
            Path to trained scaler
        """
        self.logger = get_logger(__name__)
        self.config = config
        
        # Load model and preprocessor
        self.logger.info(f"Loading model from {model_path}")
        self.model = load_model(model_path)
        self.preprocessor = self._load_preprocessor(scaler_path)
        
        # Initialize streaming components
        self.buffer = deque(maxlen=config.buffer_size)
        self.results_history: List[Dict[str, Any]] = []
        self.anomaly_callbacks: List[Callable] = []
        
        # Threading and state management
        self.is_running = False
        self.streaming_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Performance metrics
        self.total_processed = 0
        self.anomalies_detected = 0
        self.start_time = time.time()
        
        self.logger.info("StreamingProcessor initialized successfully")
    
    def _load_preprocessor(self, scaler_path: Optional[str]) -> DataPreprocessor:
        """Load or create data preprocessor."""
        if scaler_path and Path(scaler_path).exists():
            self.logger.info(f"Loading preprocessor from {scaler_path}")
            return DataPreprocessor.load(scaler_path)
        else:
            if scaler_path:
                self.logger.warning(f"Scaler path {scaler_path} not found, using default")
            return DataPreprocessor()
    
    @performance_monitor
    def ingest_data(self, data_point: Dict[str, Any]) -> None:
        """Ingest a single data point into the streaming buffer.
        
        Parameters
        ----------
        data_point : Dict[str, Any]
            Single data point with sensor values and timestamp
        """
        self.buffer.append(data_point)
        self.logger.debug(f"Ingested data point, buffer size: {len(self.buffer)}")
    
    @performance_monitor
    def ingest_batch(self, batch_data: List[Dict[str, Any]]) -> None:
        """Ingest a batch of data points.
        
        Parameters
        ----------
        batch_data : List[Dict[str, Any]]
            Batch of data points to ingest
        """
        for data_point in batch_data:
            self.buffer.append(data_point)
        
        self.logger.debug(f"Ingested batch of {len(batch_data)} points, buffer size: {len(self.buffer)}")
    
    def _create_windows_from_buffer(self) -> Optional[np.ndarray]:
        """Create sliding windows from current buffer data.
        
        Returns
        -------
        np.ndarray or None
            Windowed sequences or None if insufficient data
        """
        if len(self.buffer) < self.config.window_size:
            return None
        
        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.buffer))
        
        # Remove timestamp column if present
        feature_columns = [col for col in df.columns if col != 'timestamp']
        if not feature_columns:
            self.logger.warning("No feature columns found in buffer data")
            return None
        
        # Use numerical columns only
        numerical_data = df[feature_columns].select_dtypes(include=[np.number])
        
        if numerical_data.empty:
            self.logger.warning("No numerical data found for window creation")
            return None
        
        try:
            windows = self.preprocessor.create_sliding_windows(
                numerical_data.values,
                self.config.window_size,
                self.config.step_size
            )
            return windows
        except Exception as e:
            self.logger.error(f"Error creating windows from buffer: {e}")
            return None
    
    @performance_monitor
    def detect_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in current buffer data.
        
        Returns
        -------
        Dict[str, Any]
            Detection results with scores, anomalies, and metadata
        """
        windows = self._create_windows_from_buffer()
        
        if windows is None:
            return {
                'timestamp': pd.Timestamp.now().isoformat(),
                'scores': [],
                'anomalies': [],
                'message': 'Insufficient data for anomaly detection'
            }
        
        # Compute reconstruction scores
        reconstructed = self.model.predict(windows, verbose=0)
        scores = np.mean(np.square(windows - reconstructed), axis=(1, 2))
        
        # Determine anomalies
        anomalies = scores > self.config.anomaly_threshold
        
        # Update metrics
        self.total_processed += len(scores)
        self.anomalies_detected += np.sum(anomalies)
        
        result = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'scores': scores.tolist(),
            'anomalies': anomalies.tolist(),
            'num_windows': len(windows),
            'anomaly_count': int(np.sum(anomalies))
        }
        
        # Store in history
        self.results_history.append(result)
        
        # Trigger callbacks if anomalies detected
        if np.any(anomalies):
            self.logger.warning(f"Detected {np.sum(anomalies)} anomalies")
            for callback in self.anomaly_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"Error in anomaly callback: {e}")
        
        self.logger.info(
            f"Processed {len(scores)} windows, found {np.sum(anomalies)} anomalies"
        )
        
        return result
    
    def add_anomaly_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback function for anomaly detection events.
        
        Parameters
        ----------
        callback : Callable
            Function to call when anomalies are detected
        """
        self.anomaly_callbacks.append(callback)
        self.logger.info("Added anomaly detection callback")
    
    def start_streaming(self) -> None:
        """Start streaming processing in background thread."""
        if self.is_running:
            self.logger.warning("Streaming already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        self.streaming_thread = threading.Thread(target=self._streaming_loop)
        self.streaming_thread.start()
        
        self.logger.info("Started streaming processing")
    
    def stop_streaming(self) -> None:
        """Stop streaming processing."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.streaming_thread:
            self.streaming_thread.join(timeout=5.0)
            self.streaming_thread = None
        
        self.logger.info("Stopped streaming processing")
    
    def _streaming_loop(self) -> None:
        """Main streaming processing loop."""
        self.logger.info("Starting streaming loop")
        
        while not self.stop_event.is_set():
            try:
                if len(self.buffer) >= self.config.window_size:
                    self.detect_anomalies()
                
                # Wait for next processing interval
                self.stop_event.wait(self.config.processing_interval)
                
            except Exception as e:
                self.logger.error(f"Error in streaming loop: {e}")
                time.sleep(1.0)  # Brief pause on error
        
        self.logger.info("Streaming loop ended")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns
        -------
        Dict[str, Any]
            Performance metrics
        """
        elapsed_time = time.time() - self.start_time
        
        return {
            'total_processed': self.total_processed,
            'anomalies_detected': self.anomalies_detected,
            'processing_rate': self.total_processed / max(elapsed_time, 1e-6),
            'anomaly_rate': self.anomalies_detected / max(self.total_processed, 1),
            'buffer_utilization': len(self.buffer) / self.config.buffer_size,
            'buffer_size': len(self.buffer),
            'elapsed_time': elapsed_time,
            'is_running': self.is_running
        }
    
    def export_results(self, output_path: str, format: str = 'json') -> None:
        """Export streaming results to file.
        
        Parameters
        ----------
        output_path : str
            Path to output file
        format : str, default 'json'
            Export format ('json' or 'csv')
        """
        if not self.results_history:
            self.logger.warning("No results to export")
            return
        
        output_path = Path(output_path)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.results_history, f, indent=2)
        
        elif format.lower() == 'csv':
            # Flatten results for CSV export
            flattened_results = []
            for result in self.results_history:
                base_data = {
                    'timestamp': result['timestamp'],
                    'num_windows': result.get('num_windows', 0),
                    'anomaly_count': result.get('anomaly_count', 0)
                }
                
                # Add individual scores and anomalies
                scores = result.get('scores', [])
                anomalies = result.get('anomalies', [])
                
                for i, (score, anomaly) in enumerate(zip(scores, anomalies)):
                    row_data = base_data.copy()
                    row_data.update({
                        'window_index': i,
                        'score': score,
                        'anomaly': anomaly
                    })
                    flattened_results.append(row_data)
            
            df = pd.DataFrame(flattened_results)
            df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported {len(self.results_history)} results to {output_path}")
    
    def clear_history(self) -> None:
        """Clear results history to free memory."""
        self.results_history.clear()
        self.logger.info("Cleared results history")
    
    def get_buffer_data(self) -> List[Dict[str, Any]]:
        """Get current buffer data as list.
        
        Returns
        -------
        List[Dict[str, Any]]
            Current buffer contents
        """
        return list(self.buffer)