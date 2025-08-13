#!/bin/bash
set -e

echo "Building Docker images..."

# Build API service
docker build -t iot-anomaly-detector/api:latest -f deployment/dockerfiles/Dockerfile.web_api .

# Build inference engine
docker build -t iot-anomaly-detector/inference:latest -f deployment/dockerfiles/Dockerfile.inference_engine .

# Build monitoring service
docker build -t iot-anomaly-detector/monitoring:latest -f deployment/dockerfiles/Dockerfile.monitoring .

echo "Build completed successfully!"
