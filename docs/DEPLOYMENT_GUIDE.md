# Deployment Guide

This guide provides comprehensive instructions for deploying the IoT Anomaly Detector in various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Deployment Options](#deployment-options)
- [Production Deployment](#production-deployment)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS 10.15+, or Windows 10+
- **CPU**: 4+ cores recommended
- **Memory**: 8GB+ RAM (16GB+ for production)
- **Storage**: 50GB+ available disk space
- **Network**: Internet connectivity for dependencies and monitoring

### Required Software

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.8+ (for local development)
- Git 2.25+

## Environment Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Application Configuration
APP_ENV=production
APP_DEBUG=false
APP_PORT=8000
APP_HOST=0.0.0.0

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/iot_anomaly_db
REDIS_URL=redis://localhost:6379/0

# ML Model Configuration
MODEL_PATH=/app/models/autoencoder.h5
SCALER_PATH=/app/models/scaler.pkl
PREDICTION_THRESHOLD=0.95

# Monitoring Configuration
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
LOG_LEVEL=INFO

# Security Configuration
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
ENCRYPT_KEY=your-encryption-key-here

# External Services
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=notifications@example.com
SMTP_PASSWORD=your-smtp-password
```

### Docker Environment

For containerized deployments, use `docker-compose.override.yml`:

```yaml
version: '3.8'
services:
  app:
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/iot_anomaly_db
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
```

## Deployment Options

### 1. Docker Compose (Recommended)

#### Quick Start

```bash
# Clone repository
git clone https://github.com/terragonlabs/iot-anomaly-detector.git
cd iot-anomaly-detector

# Configure environment
cp .env.example .env
# Edit .env with your configuration

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

#### Services Included

- **Application**: Main IoT anomaly detection service
- **PostgreSQL**: Primary database
- **Redis**: Caching and session storage  
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards
- **Nginx**: Reverse proxy and load balancer

#### Scaling Services

```bash
# Scale application instances
docker-compose up -d --scale app=3

# Scale with specific resource limits
docker-compose -f docker-compose.yml -f docker-compose.production.yml up -d
```

### 2. Kubernetes Deployment

#### Namespace Creation

```bash
kubectl create namespace iot-anomaly-detector
```

#### Configuration

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: iot-anomaly-detector
data:
  APP_ENV: "production"
  LOG_LEVEL: "INFO"
  PROMETHEUS_ENABLED: "true"
```

#### Deployment

```bash
# Apply all Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n iot-anomaly-detector

# View logs
kubectl logs -f deployment/iot-anomaly-detector -n iot-anomaly-detector
```

### 3. Cloud Deployment

#### AWS Deployment

Using AWS ECS with Fargate:

```bash
# Install AWS CLI and configure credentials
aws configure

# Create ECS cluster
aws ecs create-cluster --cluster-name iot-anomaly-detector

# Register task definition
aws ecs register-task-definition --cli-input-json file://aws/task-definition.json

# Create service
aws ecs create-service --cluster iot-anomaly-detector --service-name app --task-definition iot-anomaly-detector:1 --desired-count 2
```

#### Azure Deployment

Using Azure Container Instances:

```bash
# Login to Azure
az login

# Create resource group
az group create --name iot-anomaly-detector --location eastus

# Deploy container group
az container create --resource-group iot-anomaly-detector --file azure/container-group.yaml
```

#### Google Cloud Deployment

Using Google Cloud Run:

```bash
# Authenticate with gcloud
gcloud auth login

# Build and deploy
gcloud run deploy iot-anomaly-detector --source . --platform managed --region us-central1
```

## Production Deployment

### High Availability Setup

#### Load Balancer Configuration

```nginx
# nginx/nginx.conf
upstream app {
    server app1:8000 max_fails=3 fail_timeout=30s;
    server app2:8000 max_fails=3 fail_timeout=30s;
    server app3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Health check
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
    }
}
```

#### Database Replication

```yaml
# docker-compose.production.yml
services:
  postgres-primary:
    image: postgres:15
    environment:
      POSTGRES_REPLICATION_MODE: master
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: replicator_password
    
  postgres-replica:
    image: postgres:15
    environment:
      POSTGRES_REPLICATION_MODE: slave
      POSTGRES_MASTER_HOST: postgres-primary
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: replicator_password
```

### Resource Optimization

#### Memory Management

```yaml
# docker-compose.production.yml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    environment:
      - PYTHON_MEMORY_LIMIT=1500MB
      - TF_MEMORY_GROWTH=true
```

#### CPU Optimization

```python
# config/production.py
import multiprocessing

# Gunicorn configuration
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True
```

## Monitoring and Observability

### Metrics Collection

The application exposes metrics at `/metrics` endpoint:

```bash
# Check metrics endpoint
curl http://localhost:8000/metrics
```

### Grafana Dashboards

Import the provided dashboard:

1. Open Grafana (http://localhost:3000)
2. Go to "+" → Import
3. Upload `config/grafana-dashboard.json`
4. Configure data source (Prometheus: http://prometheus:9090)

### Log Aggregation

#### ELK Stack Integration

```yaml
# docker-compose.logging.yml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.10.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    
  logstash:
    image: docker.elastic.co/logstash/logstash:8.10.0
    volumes:
      - ./config/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    
  kibana:
    image: docker.elastic.co/kibana/kibana:8.10.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
```

### Alerting Setup

Configure Prometheus alerting rules:

```bash
# Validate alert rules
promtool check rules config/monitoring/alert_rules.yml

# Test alerts
curl -XPOST http://localhost:9093/api/v1/alerts
```

## Security Considerations

### SSL/TLS Configuration

```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/private.key -out ssl/certificate.crt

# Update nginx configuration
server {
    listen 443 ssl;
    ssl_certificate /etc/ssl/certificate.crt;
    ssl_certificate_key /etc/ssl/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
}
```

### Secrets Management

#### Using Docker Secrets

```bash
# Create secrets
echo "my-secret-password" | docker secret create db_password -

# Use in compose
services:
  app:
    secrets:
      - db_password
    environment:
      - DATABASE_PASSWORD_FILE=/run/secrets/db_password
```

#### Using HashiCorp Vault

```bash
# Store secrets in Vault
vault kv put secret/iot-anomaly-detector \
  database_password="secure-password" \
  jwt_secret="secure-jwt-secret"

# Retrieve in application
vault kv get -field=database_password secret/iot-anomaly-detector
```

### Network Security

```yaml
# docker-compose.security.yml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true

services:
  app:
    networks:
      - frontend
      - backend
  
  postgres:
    networks:
      - backend
```

## Troubleshooting

### Common Issues

#### Application Won't Start

```bash
# Check logs
docker-compose logs app

# Check resource usage
docker stats

# Verify configuration
docker-compose config
```

#### Database Connection Issues

```bash
# Test database connectivity
docker-compose exec app python -c "
import psycopg2
conn = psycopg2.connect('postgresql://user:pass@postgres:5432/db')
print('Database connection successful')
"

# Check database logs
docker-compose logs postgres
```

#### Performance Issues

```bash
# Check system resources
docker-compose exec app htop

# Monitor application metrics
curl http://localhost:8000/health

# Check database performance
docker-compose exec postgres psql -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC LIMIT 10;
"
```

### Health Checks

The application provides several health check endpoints:

- `/health` - Basic health check
- `/health/ready` - Readiness probe
- `/health/live` - Liveness probe
- `/health/detailed` - Comprehensive health status

### Log Analysis

```bash
# Filter error logs
docker-compose logs app | grep ERROR

# Follow real-time logs
docker-compose logs -f --tail=100 app

# Search specific patterns
docker-compose logs app | grep -i "anomaly detected"
```

### Performance Monitoring

```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null http://localhost:8000/predict

# Monitor resource usage
docker-compose exec app python -m memory_profiler src/anomaly_detector.py

# Database query performance
docker-compose exec postgres pg_stat_statements
```

## Backup and Recovery

### Database Backup

```bash
# Create backup
docker-compose exec postgres pg_dump -U user iot_anomaly_db > backup.sql

# Restore backup  
docker-compose exec postgres psql -U user iot_anomaly_db < backup.sql
```

### Model Backup

```bash
# Backup trained models
tar -czf models-backup-$(date +%Y%m%d).tar.gz saved_models/

# Restore models
tar -xzf models-backup-20240129.tar.gz
```

For additional support, please refer to the [troubleshooting section](../TROUBLESHOOTING.md) or create an issue on GitHub.