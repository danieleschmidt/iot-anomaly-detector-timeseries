#!/bin/bash

ENVIRONMENT=${1:-development}
echo "Cleaning up $ENVIRONMENT environment..."

if [ "$ENVIRONMENT" = "development" ] || [ "$ENVIRONMENT" = "staging" ]; then
    echo "Stopping Docker Compose services..."
    docker-compose -f deployment/docker-compose.yml down
    
    # Optional: Remove volumes (uncomment if needed)
    # docker-compose -f deployment/docker-compose.yml down -v
    
    # Optional: Remove images (uncomment if needed)
    # docker rmi $(docker images "iot-anomaly-detector/*" -q) || true
    
elif [ "$ENVIRONMENT" = "production" ]; then
    echo "Cleaning up Kubernetes resources..."
    kubectl delete namespace iot-anomaly-detector --ignore-not-found=true
fi

echo "Cleanup completed!"
