"""
Test cases for Model Serving REST API.
Tests FastAPI endpoints, health checks, and versioning.
"""
import pytest
import os
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock FastAPI and related dependencies since they may not be installed
try:
    from fastapi.testclient import TestClient
    from model_serving_api import app, ModelServer
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    pytest.skip("FastAPI not available", allow_module_level=True)


class TestModelServingAPI:
    """Test the Model Serving REST API."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")
        return TestClient(app)
    
    @pytest.fixture
    def mock_model_server(self):
        """Create a mock model server."""
        with patch('model_serving_api.ModelServer') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance
    
    def test_health_check_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["status"] == "healthy"
    
    def test_model_info_endpoint(self, client, mock_model_server):
        """Test the model info endpoint."""
        # Mock model info
        mock_model_server.get_model_info.return_value = {
            "model_name": "autoencoder_v1",
            "version": "1.0.0",
            "architecture": "LSTM",
            "input_shape": [30, 1],
            "trained_at": "2024-01-01T00:00:00Z"
        }
        
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_name" in data
        assert "version" in data
        assert "architecture" in data
    
    def test_predict_endpoint_valid_input(self, client, mock_model_server):
        """Test prediction endpoint with valid input."""
        # Mock prediction
        mock_model_server.predict.return_value = {
            "anomaly_scores": [0.1, 0.2, 0.8, 0.1],
            "is_anomaly": [False, False, True, False],
            "threshold": 0.5,
            "processing_time_ms": 45.2
        }
        
        # Valid input data
        input_data = {
            "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "window_size": 30
        }
        
        response = client.post("/model/predict", json=input_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "anomaly_scores" in data
        assert "is_anomaly" in data
        assert "threshold" in data
        assert "processing_time_ms" in data
    
    def test_predict_endpoint_invalid_input(self, client):
        """Test prediction endpoint with invalid input."""
        # Invalid input - missing required fields
        invalid_data = {"invalid": "data"}
        
        response = client.post("/model/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_malformed_data(self, client):
        """Test prediction endpoint with malformed data."""
        # Malformed data
        malformed_data = {
            "data": "not_a_list",
            "window_size": "not_a_number"
        }
        
        response = client.post("/model/predict", json=malformed_data)
        assert response.status_code == 422
    
    def test_batch_predict_endpoint(self, client, mock_model_server):
        """Test batch prediction endpoint."""
        # Mock batch prediction
        mock_model_server.predict_batch.return_value = {
            "predictions": [
                {"anomaly_scores": [0.1, 0.2], "is_anomaly": [False, False]},
                {"anomaly_scores": [0.8, 0.9], "is_anomaly": [True, True]}
            ],
            "batch_size": 2,
            "total_processing_time_ms": 120.5
        }
        
        # Batch input data
        batch_data = {
            "batch": [
                {"data": [[1.0, 2.0], [3.0, 4.0]], "window_size": 30},
                {"data": [[5.0, 6.0], [7.0, 8.0]], "window_size": 30}
            ]
        }
        
        response = client.post("/model/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "batch_size" in data
        assert "total_processing_time_ms" in data
    
    def test_model_metrics_endpoint(self, client, mock_model_server):
        """Test model metrics endpoint."""
        # Mock metrics
        mock_model_server.get_metrics.return_value = {
            "total_predictions": 1000,
            "average_processing_time_ms": 42.3,
            "anomalies_detected": 45,
            "uptime_seconds": 86400,
            "last_prediction_at": "2024-01-01T12:00:00Z"
        }
        
        response = client.get("/model/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_predictions" in data
        assert "average_processing_time_ms" in data
        assert "anomalies_detected" in data


class TestModelServer:
    """Test the ModelServer class."""
    
    def test_model_server_initialization(self):
        """Test ModelServer initialization."""
        with patch('model_serving_api.AnomalyDetector'):
            server = ModelServer()
            assert server is not None
    
    def test_load_model_success(self):
        """Test successful model loading."""
        with patch('model_serving_api.AnomalyDetector') as mock_detector:
            server = ModelServer()
            
            # Mock successful loading
            mock_detector.load.return_value = MagicMock()
            
            result = server.load_model("/path/to/model")
            assert result is True
    
    def test_load_model_failure(self):
        """Test model loading failure."""
        with patch('model_serving_api.AnomalyDetector') as mock_detector:
            server = ModelServer()
            
            # Mock loading failure
            mock_detector.load.side_effect = Exception("Model not found")
            
            result = server.load_model("/invalid/path")
            assert result is False
    
    def test_predict_with_loaded_model(self):
        """Test prediction with loaded model."""
        with patch('model_serving_api.AnomalyDetector'):
            server = ModelServer()
            
            # Mock model methods
            mock_model = MagicMock()
            mock_model.predict.return_value = ([False, True, False], [0.2, 0.7, 0.1])
            server.model = mock_model
            
            # Test data
            test_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            
            result = server.predict(test_data, window_size=30)
            
            assert "anomaly_scores" in result
            assert "is_anomaly" in result
            assert "processing_time_ms" in result
    
    def test_predict_without_model(self):
        """Test prediction without loaded model."""
        server = ModelServer()
        
        test_data = [[1.0, 2.0, 3.0]]
        
        with pytest.raises(RuntimeError, match="No model loaded"):
            server.predict(test_data, window_size=30)
    
    def test_get_model_info(self):
        """Test getting model information."""
        with patch('model_serving_api.AnomalyDetector'):
            server = ModelServer()
            
            # Mock model with metadata
            mock_model = MagicMock()
            mock_model.model_metadata = {
                "version": "1.0.0",
                "architecture": "LSTM",
                "input_shape": [30, 1]
            }
            server.model = mock_model
            server.model_path = "/path/to/model.h5"
            
            info = server.get_model_info()
            
            assert "model_name" in info
            assert "version" in info
            assert "architecture" in info
    
    def test_health_check(self):
        """Test health check functionality."""
        server = ModelServer()
        
        health = server.health_check()
        
        assert "status" in health
        assert "timestamp" in health
        assert "model_loaded" in health


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        if not FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not available")
        return TestClient(app)
    
    def test_predict_with_server_error(self, client):
        """Test prediction endpoint when server error occurs."""
        with patch('model_serving_api.model_server') as mock_server:
            mock_server.predict.side_effect = Exception("Internal server error")
            
            input_data = {
                "data": [[1.0, 2.0, 3.0]],
                "window_size": 30
            }
            
            response = client.post("/model/predict", json=input_data)
            assert response.status_code == 500
    
    def test_invalid_endpoint(self, client):
        """Test accessing invalid endpoint."""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404
    
    def test_unsupported_method(self, client):
        """Test unsupported HTTP method."""
        response = client.put("/model/predict")
        assert response.status_code == 405  # Method not allowed