#!/bin/bash

echo "System Status:"
echo "=============="

# Docker containers status
if command -v docker &> /dev/null; then
    echo "Docker Containers:"
    docker ps --filter "name=iot-anomaly-detector" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
fi

echo ""

# Kubernetes pods status
if command -v kubectl &> /dev/null; then
    echo "Kubernetes Pods:"
    kubectl get pods -n iot-anomaly-detector
fi

echo ""

# Health checks
echo "Health Checks:"
echo "API: $(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "Connection failed")"
echo "Monitoring: $(curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/health || echo "Connection failed")"

echo ""

# Resource usage
echo "Resource Usage:"
if command -v docker &> /dev/null; then
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" $(docker ps --filter "name=iot-anomaly-detector" -q)
fi
