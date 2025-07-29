# API Reference

This document provides comprehensive API documentation for the IoT Anomaly Detector service.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Base URL](#base-url)
- [Common Headers](#common-headers)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [API Endpoints](#api-endpoints)
- [Data Models](#data-models)
- [Examples](#examples)
- [SDK Usage](#sdk-usage)

## Overview

The IoT Anomaly Detector API provides RESTful endpoints for:

- **Data Ingestion**: Submit sensor data for processing
- **Anomaly Detection**: Detect anomalies in real-time or batch mode
- **Model Management**: Train, update, and manage ML models
- **Health Monitoring**: Check system health and status
- **Configuration**: Manage detection thresholds and parameters

### API Version

Current API Version: `v1`

All endpoints are prefixed with `/api/v1/` unless otherwise specified.

## Authentication

### API Key Authentication

Most endpoints require API key authentication via the `X-API-Key` header:

```http
GET /api/v1/detect
X-API-Key: your-api-key-here
Content-Type: application/json
```

### JWT Token Authentication

For user-specific operations, JWT tokens are required:

```http
POST /api/v1/models/train
Authorization: Bearer your-jwt-token-here
Content-Type: application/json
```

### Obtaining API Keys

API keys can be obtained by:

1. Registering through the web interface
2. Using the CLI: `iot-detector auth create-key`
3. Contacting system administrators

## Base URL

**Production**: `https://api.iot-anomaly-detector.com`  
**Staging**: `https://staging-api.iot-anomaly-detector.com`  
**Local Development**: `http://localhost:8000`

## Common Headers

### Required Headers

```http
Content-Type: application/json
X-API-Key: your-api-key
Accept: application/json
```

### Optional Headers

```http
X-Request-ID: unique-request-identifier
X-Client-Version: client-version
User-Agent: your-application/1.0.0
```

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 422 | Validation Error |
| 429 | Rate Limited |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data format",
    "details": {
      "field": "sensor_data",
      "reason": "Missing required field 'timestamp'"
    },
    "request_id": "req_123456789",
    "timestamp": "2024-01-29T10:30:00Z"
  }
}
```

### Common Error Codes

- `INVALID_API_KEY`: API key is missing or invalid
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `VALIDATION_ERROR`: Request data validation failed
- `MODEL_NOT_FOUND`: Requested model doesn't exist
- `PROCESSING_ERROR`: Error during data processing
- `INSUFFICIENT_PERMISSIONS`: User lacks required permissions

## Rate Limiting

### Default Limits

- **Free Tier**: 100 requests/hour
- **Basic Tier**: 1000 requests/hour  
- **Premium Tier**: 10000 requests/hour
- **Enterprise**: Custom limits

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1640995200
X-RateLimit-Window: 3600
```

## API Endpoints

### Health & Status

#### GET /health

Check basic service health.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-29T10:30:00Z",
  "version": "0.0.3"
}
```

#### GET /health/detailed

Get comprehensive system status.

**Response**:
```json
{
  "status": "healthy",
  "components": {
    "database": "healthy",
    "cache": "healthy", 
    "ml_models": "healthy",
    "external_services": "degraded"
  },
  "metrics": {
    "uptime": 86400,
    "requests_per_minute": 150,
    "error_rate": 0.001
  }
}
```

### Data Ingestion

#### POST /api/v1/data/ingest

Submit sensor data for processing.

**Request**:
```json
{
  "sensor_id": "sensor-001",
  "timestamp": "2024-01-29T10:30:00Z",
  "data": {
    "temperature": 25.5,
    "humidity": 60.2,
    "pressure": 1013.25
  },
  "metadata": {
    "location": "Building A, Floor 2",
    "device_type": "environmental_sensor"
  }
}
```

**Response**:
```json
{
  "id": "ingestion-12345",
  "status": "accepted",
  "timestamp": "2024-01-29T10:30:01Z",
  "processing_time_ms": 45
}
```

#### POST /api/v1/data/batch

Submit multiple sensor readings in batch.

**Request**:
```json
{
  "batch_id": "batch-001",
  "readings": [
    {
      "sensor_id": "sensor-001",
      "timestamp": "2024-01-29T10:30:00Z",
      "data": {
        "temperature": 25.5,
        "humidity": 60.2
      }
    },
    {
      "sensor_id": "sensor-002", 
      "timestamp": "2024-01-29T10:30:05Z",
      "data": {
        "temperature": 24.8,
        "humidity": 58.1
      }
    }
  ]
}
```

### Anomaly Detection

#### POST /api/v1/detect

Detect anomalies in sensor data.

**Request**:
```json
{
  "sensor_data": [
    [25.5, 60.2, 1013.25],
    [26.1, 62.3, 1012.80],
    [88.9, 95.5, 950.00]
  ],
  "model_id": "autoencoder-v1",
  "threshold": 0.95,
  "return_scores": true
}
```

**Response**:
```json
{
  "anomalies": [
    {
      "index": 2,
      "score": 0.97,
      "is_anomaly": true,
      "confidence": 0.89
    }
  ],
  "summary": {
    "total_points": 3,
    "anomalies_detected": 1,
    "processing_time_ms": 123
  },
  "model_info": {
    "model_id": "autoencoder-v1",
    "version": "1.2.0",
    "threshold": 0.95
  }
}
```

#### GET /api/v1/detect/stream

Server-sent events endpoint for real-time anomaly detection.

**Usage**:
```javascript
const eventSource = new EventSource('/api/v1/detect/stream?model_id=autoencoder-v1');

eventSource.onmessage = function(event) {
  const anomaly = JSON.parse(event.data);
  console.log('Anomaly detected:', anomaly);
};
```

**Event Data**:
```json
{
  "event": "anomaly_detected",
  "data": {
    "sensor_id": "sensor-001",
    "timestamp": "2024-01-29T10:30:00Z",
    "score": 0.97,
    "severity": "high"
  }
}
```

### Model Management

#### GET /api/v1/models

List available models.

**Response**:
```json
{
  "models": [
    {
      "id": "autoencoder-v1",
      "name": "LSTM Autoencoder v1",
      "status": "active",
      "accuracy": 0.94,
      "created_at": "2024-01-15T08:00:00Z",
      "last_trained": "2024-01-29T06:00:00Z"
    }
  ],
  "total": 1
}
```

#### POST /api/v1/models/train

Train a new model or retrain existing model.

**Request**:
```json
{
  "model_type": "autoencoder",
  "parameters": {
    "window_size": 30,
    "latent_dim": 16,
    "epochs": 100,
    "batch_size": 32
  },
  "training_data": {
    "source": "uploaded_dataset",
    "validation_split": 0.2
  },
  "name": "autoencoder-v2"
}
```

**Response**:
```json
{
  "job_id": "train-job-12345",
  "status": "started",
  "estimated_completion": "2024-01-29T12:00:00Z",
  "progress_url": "/api/v1/jobs/train-job-12345"
}
```

#### GET /api/v1/models/{model_id}

Get detailed model information.

**Response**:
```json
{
  "id": "autoencoder-v1",
  "name": "LSTM Autoencoder v1",
  "status": "active",
  "metrics": {
    "accuracy": 0.94,
    "precision": 0.92,
    "recall": 0.89,
    "f1_score": 0.90
  },
  "parameters": {
    "window_size": 30,
    "latent_dim": 16,
    "epochs": 100
  },
  "training_info": {
    "trained_at": "2024-01-29T06:00:00Z",
    "training_samples": 50000,
    "validation_accuracy": 0.93
  }
}
```

### Configuration

#### GET /api/v1/config

Get current system configuration.

**Response**:
```json
{
  "detection": {
    "default_threshold": 0.95,
    "auto_retrain": true,
    "batch_size": 1000
  },
  "data_retention": {
    "raw_data_days": 30,
    "processed_data_days": 90,
    "model_versions": 5
  },
  "notifications": {
    "email_enabled": true,
    "webhook_enabled": false,
    "severity_threshold": "medium"
  }
}
```

#### PUT /api/v1/config

Update system configuration.

**Request**:
```json
{
  "detection": {
    "default_threshold": 0.90
  },
  "notifications": {
    "severity_threshold": "high"
  }
}
```

### Data Export

#### GET /api/v1/export/anomalies

Export detected anomalies.

**Query Parameters**:
- `start_date`: Start date (ISO 8601)
- `end_date`: End date (ISO 8601)
- `format`: Export format (json, csv, xlsx)
- `sensor_ids`: Comma-separated sensor IDs

**Example**:
```
GET /api/v1/export/anomalies?start_date=2024-01-01&end_date=2024-01-29&format=csv&sensor_ids=sensor-001,sensor-002
```

**Response** (CSV format):
```csv
timestamp,sensor_id,score,is_anomaly,severity
2024-01-29T10:30:00Z,sensor-001,0.97,true,high
2024-01-29T10:35:00Z,sensor-002,0.89,false,low
```

## Data Models

### SensorReading

```json
{
  "sensor_id": "string",
  "timestamp": "string (ISO 8601)",
  "data": {
    "field1": "number",
    "field2": "number"
  },
  "metadata": {
    "location": "string",
    "device_type": "string"
  }
}
```

### AnomalyResult

```json
{
  "id": "string",
  "sensor_id": "string", 
  "timestamp": "string (ISO 8601)",
  "score": "number (0-1)",
  "is_anomaly": "boolean",
  "confidence": "number (0-1)",
  "severity": "string (low|medium|high|critical)",
  "model_id": "string"
}
```

### ModelInfo

```json
{
  "id": "string",
  "name": "string",
  "type": "string",
  "status": "string (training|active|inactive|error)",
  "version": "string",
  "accuracy": "number (0-1)",
  "created_at": "string (ISO 8601)",
  "parameters": "object"
}
```

## Examples

### Python SDK

```python
from iot_anomaly_detector import AnomalyDetectorClient

# Initialize client
client = AnomalyDetectorClient(
    base_url="https://api.iot-anomaly-detector.com",
    api_key="your-api-key"
)

# Submit sensor data
result = client.ingest_data(
    sensor_id="sensor-001",
    data={"temperature": 25.5, "humidity": 60.2}
)

# Detect anomalies
anomalies = client.detect_anomalies([
    [25.5, 60.2, 1013.25],
    [88.9, 95.5, 950.00]
])

print(f"Detected {len(anomalies)} anomalies")
```

### JavaScript/Node.js

```javascript
const { AnomalyDetectorClient } = require('@iot-anomaly-detector/client');

const client = new AnomalyDetectorClient({
  baseUrl: 'https://api.iot-anomaly-detector.com',
  apiKey: 'your-api-key'
});

// Detect anomalies
const result = await client.detectAnomalies({
  sensor_data: [[25.5, 60.2, 1013.25], [88.9, 95.5, 950.00]],
  model_id: 'autoencoder-v1',
  threshold: 0.95
});

console.log('Anomalies:', result.anomalies);
```

### cURL Examples

#### Submit Data

```bash
curl -X POST https://api.iot-anomaly-detector.com/api/v1/data/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "sensor_id": "sensor-001",
    "timestamp": "2024-01-29T10:30:00Z",
    "data": {
      "temperature": 25.5,
      "humidity": 60.2
    }
  }'
```

#### Detect Anomalies

```bash
curl -X POST https://api.iot-anomaly-detector.com/api/v1/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "sensor_data": [[25.5, 60.2], [88.9, 95.5]],
    "model_id": "autoencoder-v1",
    "threshold": 0.95
  }'
```

## SDK Usage

### Installation

```bash
# Python
pip install iot-anomaly-detector-client

# Node.js
npm install @iot-anomaly-detector/client

# Go
go get github.com/terragonlabs/iot-anomaly-detector-go
```

### Configuration

```python
# Python configuration
import os
from iot_anomaly_detector import AnomalyDetectorClient

client = AnomalyDetectorClient(
    base_url=os.getenv('IOT_API_URL', 'http://localhost:8000'),
    api_key=os.getenv('IOT_API_KEY'),
    timeout=30,
    retry_attempts=3
)
```

```javascript
// Node.js configuration
const client = new AnomalyDetectorClient({
  baseUrl: process.env.IOT_API_URL || 'http://localhost:8000',
  apiKey: process.env.IOT_API_KEY,
  timeout: 30000,
  retryAttempts: 3
});
```

### Error Handling

```python
from iot_anomaly_detector.exceptions import (
    APIKeyError, 
    ValidationError, 
    RateLimitError
)

try:
    result = client.detect_anomalies(data)
except APIKeyError:
    print("Invalid API key")
except ValidationError as e:
    print(f"Validation error: {e.details}")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
```

For additional information and examples, see the [SDK documentation](https://docs.iot-anomaly-detector.com/sdk) or check the [GitHub repository](https://github.com/terragonlabs/iot-anomaly-detector).