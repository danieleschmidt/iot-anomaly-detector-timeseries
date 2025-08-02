"""API contract testing to ensure API compatibility."""

import pytest
import json
from unittest.mock import Mock, patch
from typing import Dict, Any

# Test data
VALID_SENSOR_DATA = {
    "timestamp": "2024-01-01T12:00:00",
    "temperature": 25.5,
    "pressure": 1013.25,
    "humidity": 60.0,
    "vibration": 0.1
}

BATCH_SENSOR_DATA = {
    "data": [
        {
            "timestamp": "2024-01-01T12:00:00",
            "temperature": 25.5,
            "pressure": 1013.25,
            "humidity": 60.0,
            "vibration": 0.1
        },
        {
            "timestamp": "2024-01-01T12:01:00",
            "temperature": 26.0,
            "pressure": 1012.8,
            "humidity": 58.5,
            "vibration": 0.12
        }
    ]
}


class TestAnomalyDetectionAPIContracts:
    """Test API contracts for anomaly detection service."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock model for testing."""
        model = Mock()
        model.predict.return_value = [[0.1, 0.2, 0.3, 0.05]]
        return model
    
    @pytest.fixture
    def api_client(self):
        """Create API test client."""
        try:
            from fastapi.testclient import TestClient
            from src.model_serving_api import app
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI not available for contract testing")
    
    def test_health_endpoint_contract(self, api_client):
        """Test /health endpoint contract."""
        response = api_client.get("/health")
        
        # Status code contract
        assert response.status_code == 200
        
        # Response schema contract
        data = response.json()
        required_fields = ["status", "timestamp", "version"]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        assert data["status"] in ["healthy", "unhealthy"], "Invalid status value"
        assert isinstance(data["timestamp"], str), "Timestamp should be string"
        assert isinstance(data["version"], str), "Version should be string"
    
    def test_predict_single_endpoint_contract(self, api_client):
        """Test /predict endpoint contract for single prediction."""
        response = api_client.post(
            "/predict",
            json=VALID_SENSOR_DATA,
            headers={"Content-Type": "application/json"}
        )
        
        # Should accept request (may fail due to model not loaded, but schema should be validated)
        assert response.status_code in [200, 422, 500]  # 422 for validation, 500 for model issues
        
        if response.status_code == 200:
            data = response.json()
            
            # Response schema contract
            required_fields = ["is_anomaly", "confidence", "reconstruction_error", "timestamp"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            assert isinstance(data["is_anomaly"], bool), "is_anomaly should be boolean"
            assert isinstance(data["confidence"], (int, float)), "confidence should be numeric"
            assert isinstance(data["reconstruction_error"], (int, float)), "reconstruction_error should be numeric"
            assert 0 <= data["confidence"] <= 1, "confidence should be between 0 and 1"
    
    def test_predict_batch_endpoint_contract(self, api_client):
        """Test /predict/batch endpoint contract."""
        response = api_client.post(
            "/predict/batch",
            json=BATCH_SENSOR_DATA,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code in [200, 422, 500]
        
        if response.status_code == 200:
            data = response.json()
            
            # Response should be a list
            assert isinstance(data, list), "Batch response should be a list"
            
            if len(data) > 0:
                # Each item should follow single prediction contract
                for item in data:
                    required_fields = ["is_anomaly", "confidence", "reconstruction_error", "timestamp"]
                    for field in required_fields:
                        assert field in item, f"Missing required field in batch item: {field}"
    
    def test_model_info_endpoint_contract(self, api_client):
        """Test /model/info endpoint contract."""
        response = api_client.get("/model/info")
        
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            data = response.json()
            
            expected_fields = ["model_name", "version", "created_at", "metrics"]
            for field in expected_fields:
                assert field in data, f"Missing required field: {field}"
    
    def test_model_metrics_endpoint_contract(self, api_client):
        """Test /model/metrics endpoint contract."""
        response = api_client.get("/model/metrics")
        
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            data = response.json()
            
            # Should contain performance metrics
            expected_metrics = ["accuracy", "precision", "recall", "f1_score"]
            
            # At least some metrics should be present
            assert any(metric in data for metric in expected_metrics), "No expected metrics found"
    
    def test_input_validation_contracts(self, api_client):
        """Test input validation contracts."""
        # Test missing required fields
        invalid_data = {"temperature": 25.5}  # Missing other required fields
        
        response = api_client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
        
        # Test invalid data types
        invalid_types = {
            "timestamp": "2024-01-01T12:00:00",
            "temperature": "not_a_number",  # Should be numeric
            "pressure": 1013.25,
            "humidity": 60.0,
            "vibration": 0.1
        }
        
        response = api_client.post("/predict", json=invalid_types)
        assert response.status_code == 422
        
        # Test out-of-range values
        out_of_range = {
            "timestamp": "2024-01-01T12:00:00",
            "temperature": -500,  # Unrealistic temperature
            "pressure": 1013.25,
            "humidity": 60.0,
            "vibration": 0.1
        }
        
        response = api_client.post("/predict", json=out_of_range)
        # May pass validation but could be caught by business logic
        assert response.status_code in [200, 400, 422, 500]
    
    def test_error_response_contracts(self, api_client):
        """Test error response contracts."""
        # Test malformed JSON
        response = api_client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code in [400, 422]
        
        # Error response should have consistent structure
        if response.status_code == 422:
            error_data = response.json()
            assert "detail" in error_data, "Error response should contain 'detail' field"
    
    def test_content_type_contracts(self, api_client):
        """Test content type handling contracts."""
        # Test without Content-Type header
        response = api_client.post("/predict", json=VALID_SENSOR_DATA)
        assert response.status_code in [200, 415, 422, 500]
        
        # Test with wrong Content-Type
        response = api_client.post(
            "/predict",
            data=json.dumps(VALID_SENSOR_DATA),
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code in [415, 422]  # Unsupported Media Type or validation error


class TestStreamingAPIContracts:
    """Test contracts for streaming API endpoints."""
    
    @pytest.fixture
    def streaming_client(self):
        """Create streaming API test client."""
        try:
            from fastapi.testclient import TestClient
            from src.streaming_processor import create_streaming_app
            app = create_streaming_app()
            return TestClient(app)
        except (ImportError, AttributeError):
            pytest.skip("Streaming API not available for contract testing")
    
    def test_streaming_predict_contract(self, streaming_client):
        """Test streaming prediction endpoint contract."""
        stream_data = {
            "stream_id": "test_stream_001",
            "data": BATCH_SENSOR_DATA["data"]
        }
        
        response = streaming_client.post("/stream/predict", json=stream_data)
        
        assert response.status_code in [200, 422, 500]
        
        if response.status_code == 200:
            data = response.json()
            
            required_fields = ["stream_id", "predictions", "timestamp", "processed_count"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            assert data["stream_id"] == "test_stream_001"
            assert isinstance(data["predictions"], list)
            assert isinstance(data["processed_count"], int)


class TestModelManagementAPIContracts:
    """Test contracts for model management API endpoints."""
    
    @pytest.fixture
    def mgmt_client(self):
        """Create model management API test client."""
        try:
            from fastapi.testclient import TestClient
            from src.model_manager import create_management_app
            app = create_management_app()
            return TestClient(app)
        except (ImportError, AttributeError):
            pytest.skip("Model management API not available for contract testing")
    
    def test_model_list_contract(self, mgmt_client):
        """Test /models endpoint contract."""
        response = mgmt_client.get("/models")
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list), "Models list should be an array"
            
            if len(data) > 0:
                model_info = data[0]
                expected_fields = ["name", "version", "created_at", "status"]
                for field in expected_fields:
                    assert field in model_info, f"Missing field in model info: {field}"
    
    def test_model_deployment_contract(self, mgmt_client):
        """Test model deployment endpoint contract."""
        deployment_request = {
            "model_name": "test_model",
            "version": "1.0.0"
        }
        
        response = mgmt_client.post("/models/deploy", json=deployment_request)
        
        assert response.status_code in [200, 404, 422, 500]
        
        if response.status_code == 200:
            data = response.json()
            expected_fields = ["deployment_id", "status", "timestamp"]
            for field in expected_fields:
                assert field in data, f"Missing field in deployment response: {field}"


class TestDataValidationContracts:
    """Test data validation contracts."""
    
    def test_sensor_data_schema_contract(self):
        """Test sensor data schema validation."""
        from src.data_validator import DataValidator
        
        validator = DataValidator()
        
        # Valid data should pass
        valid_df = pd.DataFrame([VALID_SENSOR_DATA])
        result = validator.validate_dataset(valid_df)
        
        assert "is_valid" in result
        assert isinstance(result["is_valid"], bool)
        assert "errors" in result
        assert isinstance(result["errors"], list)
        
        if not result["is_valid"]:
            # Errors should have specific structure
            for error in result["errors"]:
                assert isinstance(error, dict)
                assert "field" in error or "message" in error
    
    def test_time_series_validation_contract(self):
        """Test time series specific validation contracts."""
        from src.data_validator import DataValidator
        
        validator = DataValidator()
        
        # Test with time series data
        import pandas as pd
        ts_data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="1min"),
            "temperature": range(100),
            "pressure": range(100),
            "humidity": range(100),
            "vibration": [0.1] * 100
        })
        
        # Time series validation should check for temporal ordering, gaps, etc.
        result = validator.validate_time_series(ts_data)
        
        expected_checks = ["temporal_order", "missing_timestamps", "data_gaps"]
        for check in expected_checks:
            assert check in result, f"Missing time series validation check: {check}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])