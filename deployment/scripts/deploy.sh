#!/bin/bash
set -e

ENVIRONMENT=${1:-development}
echo "Deploying to $ENVIRONMENT environment..."

# Load environment-specific configuration
if [ -f "deployment/configs/$ENVIRONMENT.env" ]; then
    source "deployment/configs/$ENVIRONMENT.env"
fi

# Deploy with docker-compose
if [ "$ENVIRONMENT" = "development" ] || [ "$ENVIRONMENT" = "staging" ]; then
    echo "Using Docker Compose for deployment..."
    docker-compose -f deployment/docker-compose.yml up -d
    
    echo "Waiting for services to be ready..."
    sleep 30
    
    # Health checks
    echo "Performing health checks..."
    curl -f http://localhost:8000/health || echo "API health check failed"
    curl -f http://localhost:8002/health || echo "Monitoring health check failed"
    
elif [ "$ENVIRONMENT" = "production" ]; then
    echo "Using Kubernetes for production deployment..."
    
    # Apply Kubernetes manifests
    kubectl apply -f deployment/k8s/
    
    # Wait for rollout
    kubectl rollout status deployment/iot-anomaly-detector-api -n iot-anomaly-detector
    kubectl rollout status deployment/iot-anomaly-detector-inference -n iot-anomaly-detector
    kubectl rollout status deployment/iot-anomaly-detector-monitoring -n iot-anomaly-detector
    
    echo "Production deployment completed!"
fi
