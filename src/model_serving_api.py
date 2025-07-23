"""
Model Serving REST API for IoT Anomaly Detection.
FastAPI implementation with health checks, versioning, and comprehensive endpoints.
"""
import time
import logging
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import asyncio

try:
    from fastapi import FastAPI, HTTPException, Request, status
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Mock classes for when FastAPI is not available
    class BaseModel:
        pass
    class Field:
        def __init__(self, *args, **kwargs):
            pass
    HTTPException = Exception
    status = type('status', (), {'HTTP_500_INTERNAL_SERVER_ERROR': 500})()

try:
    from .anomaly_detector import AnomalyDetector
    from .logging_config import get_logger
    from .security_utils import sanitize_error_message
except ImportError:
    # Handle imports when running as standalone module
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from anomaly_detector import AnomalyDetector
    from logging_config import get_logger
    from security_utils import sanitize_error_message

logger = get_logger(__name__)


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    data: List[List[float]] = Field(..., description="Time series data points")
    window_size: int = Field(30, ge=1, le=100, description="Window size for preprocessing")
    threshold_factor: float = Field(3.0, ge=0.1, le=10.0, description="Anomaly threshold factor")
    
    @validator('data')
    def validate_data(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Data cannot be empty")
        for row in v:
            if not isinstance(row, list) or len(row) == 0:
                raise ValueError("Each data point must be a non-empty list")
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    batch: List[PredictionRequest] = Field(..., description="Batch of prediction requests")
    
    @validator('batch')
    def validate_batch(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Batch cannot be empty")
        if len(v) > 100:  # Limit batch size for performance
            raise ValueError("Batch size cannot exceed 100")
        return v


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    anomaly_scores: List[float]
    is_anomaly: List[bool]
    threshold: float
    processing_time_ms: float
    window_count: int


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[PredictionResponse]
    batch_size: int
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    version: str
    model_loaded: bool
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str
    version: str
    architecture: str
    input_shape: List[int]
    trained_at: Optional[str]
    loaded_at: str


class MetricsResponse(BaseModel):
    """Response model for model metrics."""
    total_predictions: int
    total_batch_predictions: int
    average_processing_time_ms: float
    anomalies_detected: int
    uptime_seconds: float
    last_prediction_at: Optional[str]
    memory_usage_mb: float


class ModelServer:
    """Model server for handling anomaly detection requests."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the model server.
        
        Args:
            model_path: Optional path to pre-load a model
        """
        self.model: Optional[AnomalyDetector] = None
        self.model_path: Optional[str] = None
        self.start_time = time.time()
        self.metrics = {
            'total_predictions': 0,
            'total_batch_predictions': 0,
            'total_processing_time': 0.0,
            'anomalies_detected': 0,
            'last_prediction_at': None
        }
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load model from path.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model = AnomalyDetector.load(model_path)
            self.model_path = model_path
            logger.info(f"Model loaded successfully from {sanitize_error_message(model_path)}")
            return True
        except Exception as e:
            sanitized_error = sanitize_error_message(str(e))
            logger.error(f"Failed to load model: {sanitized_error}")
            self.model = None
            self.model_path = None
            return False
    
    def predict(self, data: List[List[float]], window_size: int = 30, 
                threshold_factor: float = 3.0) -> Dict[str, Any]:
        """Make a prediction on the input data.
        
        Args:
            data: Time series data points
            window_size: Window size for preprocessing
            threshold_factor: Anomaly threshold factor
            
        Returns:
            Prediction results with scores and anomaly flags
            
        Raises:
            RuntimeError: If no model is loaded
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Load a model first.")
        
        start_time = time.time()
        
        try:
            # Convert data to the format expected by AnomalyDetector
            import numpy as np
            import pandas as pd
            
            # Create DataFrame from input data
            df = pd.DataFrame(data)
            
            # Use the anomaly detector to get predictions
            is_anomaly, scores = self.model.predict(df, window_size=window_size)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Update metrics
            self._update_metrics(processing_time, is_anomaly)
            
            return {
                'anomaly_scores': scores.tolist() if hasattr(scores, 'tolist') else list(scores),
                'is_anomaly': is_anomaly.tolist() if hasattr(is_anomaly, 'tolist') else list(is_anomaly),
                'threshold': threshold_factor,  # For now, return the input threshold
                'processing_time_ms': processing_time,
                'window_count': len(scores) if scores is not None else 0
            }
            
        except Exception as e:
            sanitized_error = sanitize_error_message(str(e))
            logger.error(f"Prediction failed: {sanitized_error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {sanitized_error}"
            )
    
    def predict_batch(self, batch_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make predictions on a batch of requests.
        
        Args:
            batch_requests: List of prediction requests
            
        Returns:
            Batch prediction results
        """
        start_time = time.time()
        predictions = []
        
        for request in batch_requests:
            try:
                pred_result = self.predict(
                    request['data'],
                    request.get('window_size', 30),
                    request.get('threshold_factor', 3.0)
                )
                predictions.append(pred_result)
            except Exception as e:
                # Continue with other predictions if one fails
                sanitized_error = sanitize_error_message(str(e))
                logger.warning(f"Batch prediction item failed: {sanitized_error}")
                predictions.append({
                    'error': sanitized_error,
                    'anomaly_scores': [],
                    'is_anomaly': [],
                    'threshold': 3.0,
                    'processing_time_ms': 0.0,
                    'window_count': 0
                })
        
        total_processing_time = (time.time() - start_time) * 1000
        self.metrics['total_batch_predictions'] += 1
        
        return {
            'predictions': predictions,
            'batch_size': len(batch_requests),
            'total_processing_time_ms': total_processing_time
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        if self.model is None:
            return {
                'model_name': 'No model loaded',
                'version': 'N/A',
                'architecture': 'N/A',
                'input_shape': [],
                'trained_at': None,
                'loaded_at': 'N/A'
            }
        
        model_name = Path(self.model_path).stem if self.model_path else 'unknown'
        
        # Try to get metadata from model if available
        metadata = getattr(self.model, 'model_metadata', {})
        
        return {
            'model_name': model_name,
            'version': metadata.get('version', '1.0.0'),
            'architecture': metadata.get('architecture', 'Autoencoder'),
            'input_shape': metadata.get('input_shape', [30, 1]),
            'trained_at': metadata.get('trained_at'),
            'loaded_at': datetime.now().isoformat()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics.
        
        Returns:
            Metrics dictionary
        """
        import psutil
        
        # Calculate memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        avg_processing_time = (
            self.metrics['total_processing_time'] / self.metrics['total_predictions']
            if self.metrics['total_predictions'] > 0 else 0.0
        )
        
        return {
            'total_predictions': self.metrics['total_predictions'],
            'total_batch_predictions': self.metrics['total_batch_predictions'],
            'average_processing_time_ms': avg_processing_time,
            'anomalies_detected': self.metrics['anomalies_detected'],
            'uptime_seconds': time.time() - self.start_time,
            'last_prediction_at': self.metrics['last_prediction_at'],
            'memory_usage_mb': memory_mb
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health status dictionary
        """
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'model_loaded': self.model is not None,
            'uptime_seconds': time.time() - self.start_time
        }
    
    def _update_metrics(self, processing_time: float, is_anomaly: List[bool]) -> None:
        """Update internal metrics.
        
        Args:
            processing_time: Processing time in milliseconds
            is_anomaly: List of anomaly flags
        """
        self.metrics['total_predictions'] += 1
        self.metrics['total_processing_time'] += processing_time
        self.metrics['anomalies_detected'] += sum(is_anomaly) if is_anomaly else 0
        self.metrics['last_prediction_at'] = datetime.now().isoformat()


# Global model server instance
model_server = ModelServer()

# FastAPI app initialization
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="IoT Anomaly Detection API",
        description="REST API for real-time IoT anomaly detection using autoencoders",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors."""
        sanitized_error = sanitize_error_message(str(exc))
        logger.error(f"Unhandled exception: {sanitized_error}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {sanitized_error}"}
        )
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return model_server.health_check()
    
    @app.get("/model/info", response_model=ModelInfoResponse)
    async def get_model_info():
        """Get information about the loaded model."""
        return model_server.get_model_info()
    
    @app.get("/model/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """Get server and model metrics."""
        return model_server.get_metrics()
    
    @app.post("/model/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Make a prediction on time series data."""
        try:
            result = model_server.predict(
                request.data,
                request.window_size,
                request.threshold_factor
            )
            return result
        except RuntimeError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e)
            )
        except Exception as e:
            sanitized_error = sanitize_error_message(str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {sanitized_error}"
            )
    
    @app.post("/model/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(request: BatchPredictionRequest):
        """Make predictions on a batch of time series data."""
        try:
            batch_data = [req.dict() for req in request.batch]
            result = model_server.predict_batch(batch_data)
            return result
        except Exception as e:
            sanitized_error = sanitize_error_message(str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch prediction failed: {sanitized_error}"
            )
    
    @app.post("/model/load")
    async def load_model(model_path: str):
        """Load a model from the specified path."""
        success = model_server.load_model(model_path)
        if success:
            return {"message": "Model loaded successfully", "model_path": sanitize_error_message(model_path)}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to load model"
            )
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "IoT Anomaly Detection API",
            "version": "1.0.0",
            "description": "REST API for real-time IoT anomaly detection",
            "endpoints": {
                "health": "/health",
                "docs": "/docs",
                "model_info": "/model/info",
                "predict": "/model/predict",
                "batch_predict": "/model/predict/batch",
                "metrics": "/model/metrics"
            }
        }

else:
    # Fallback when FastAPI is not available
    app = None
    logger.warning("FastAPI not available. API server cannot be started.")


def start_server(host: str = "0.0.0.0", port: int = 8000, model_path: Optional[str] = None):
    """Start the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        model_path: Optional path to pre-load a model
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is not available. Please install with: pip install fastapi uvicorn")
    
    if model_path:
        success = model_server.load_model(model_path)
        if not success:
            logger.warning(f"Failed to load initial model from {model_path}")
    
    try:
        import uvicorn
        logger.info(f"Starting API server on {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")
    except ImportError:
        raise RuntimeError("uvicorn is not available. Please install with: pip install uvicorn")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start the IoT Anomaly Detection API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", help="Path to model file to pre-load")
    
    args = parser.parse_args()
    
    try:
        start_server(args.host, args.port, args.model)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        exit(1)