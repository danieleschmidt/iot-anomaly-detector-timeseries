# Deployment Guide

This guide covers deploying the IoT Anomaly Detection system in various environments.

## Table of Contents

- [Overview](#overview)
- [Quick Deployment](#quick-deployment)
- [Production Deployment](#production-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring Setup](#monitoring-setup)
- [Security Configuration](#security-configuration)
- [Troubleshooting](#troubleshooting)

## Overview

The system supports multiple deployment strategies:

- **Local Development**: Docker Compose for development
- **Production Single Node**: Docker with production optimizations
- **Cloud Native**: Kubernetes deployment
- **Serverless**: AWS Lambda/Azure Functions (planned)

### Architecture Components

```mermaid
graph TB
    LB[Load Balancer] --> API[API Gateway]
    API --> APP[Application Pods]
    APP --> DB[(PostgreSQL)]
    APP --> CACHE[(Redis)]
    APP --> MODELS[Model Storage]
    
    MON[Prometheus] --> APP
    MON --> GRAFANA[Grafana]
    
    LOG[Centralized Logging] --> APP
```

## Quick Deployment

### Local Development

```bash
# Clone repository
git clone https://github.com/terragonlabs/iot-anomaly-detector.git
cd iot-anomaly-detector

# Start all services
make docker-run

# Access services
# API: http://localhost:8000
# Grafana: http://localhost:3000
# Jupyter: http://localhost:8888
```

### Production Single Node

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# With external monitoring
docker-compose -f docker-compose.prod.yml --profile monitoring up -d
```

## Production Deployment

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM
- 50GB+ disk space
- SSL certificates (for HTTPS)

### Environment Configuration

Create production environment file:

```bash
# Copy and customize
cp .env.example .env.prod
```

Key production settings:

```bash
# .env.prod
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Security
SECRET_KEY=your-production-secret-key
JWT_SECRET=your-jwt-secret
ENCRYPTION_KEY=your-32-byte-encryption-key

# Database
DATABASE_URL=postgresql://user:password@postgres:5432/anomaly_db
REDIS_URL=redis://redis:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true

# SSL/TLS
FORCE_HTTPS=true
TLS_VERSION_MIN=1.3
```

### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      target: production
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env.prod
    volumes:
      - models-volume:/app/saved_models
      - logs-volume:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: anomaly_db
      POSTGRES_USER: user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:
  models-volume:
  logs-volume:
```

### SSL/TLS Configuration

#### Using Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

#### Nginx Configuration

```nginx
# nginx/nginx.conf
upstream app {
    server app:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.3;
    ssl_ciphers ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS;

    location / {
        proxy_pass http://app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        proxy_pass http://app/health;
        access_log off;
    }
}
```

## Cloud Deployment

### Kubernetes Deployment

#### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3.0+ (optional)

#### Namespace Setup

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: anomaly-detection
```

#### ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: anomaly-detection
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
```

#### Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: anomaly-detection
type: Opaque
data:
  SECRET_KEY: <base64-encoded-secret>
  DATABASE_URL: <base64-encoded-db-url>
  REDIS_URL: <base64-encoded-redis-url>
```

#### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-detector
  namespace: anomaly-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: anomaly-detector
  template:
    metadata:
      labels:
        app: anomaly-detector
    spec:
      containers:
      - name: app
        image: ghcr.io/terragonlabs/iot-anomaly-detector:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: app-config
        - secretRef:
            name: app-secrets
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

#### Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: anomaly-detector-service
  namespace: anomaly-detection
spec:
  selector:
    app: anomaly-detector
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: anomaly-detector-ingress
  namespace: anomaly-detection
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.your-domain.com
    secretName: anomaly-detector-tls
  rules:
  - host: api.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: anomaly-detector-service
            port:
              number: 80
```

#### Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f k8s/

# Check deployment
kubectl get pods -n anomaly-detection
kubectl get svc -n anomaly-detection
kubectl get ingress -n anomaly-detection

# View logs
kubectl logs -f deployment/anomaly-detector -n anomaly-detection
```

### AWS Deployment

#### ECS with Fargate

```json
{
  "family": "anomaly-detector",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/anomaly-detector-task-role",
  "containerDefinitions": [
    {
      "name": "anomaly-detector",
      "image": "ghcr.io/terragonlabs/iot-anomaly-detector:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:db-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/anomaly-detector",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### EKS Deployment

```bash
# Create EKS cluster
eksctl create cluster --name anomaly-detector --region us-west-2

# Deploy application
kubectl apply -f k8s/

# Setup load balancer
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.0.0/deploy/static/provider/aws/deploy.yaml
```

### Azure Deployment

#### Container Instances

```bash
# Create resource group
az group create --name anomaly-detector --location eastus

# Deploy container
az container create \
  --resource-group anomaly-detector \
  --name anomaly-detector-app \
  --image ghcr.io/terragonlabs/iot-anomaly-detector:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables ENVIRONMENT=production \
  --secure-environment-variables DATABASE_URL=$DATABASE_URL
```

### Google Cloud Deployment

#### Cloud Run

```bash
# Deploy to Cloud Run
gcloud run deploy anomaly-detector \
  --image ghcr.io/terragonlabs/iot-anomaly-detector:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ENVIRONMENT=production \
  --memory 2Gi \
  --cpu 2
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'anomaly-detector'
    static_configs:
      - targets: ['app:9090']
```

### Grafana Dashboard

Import dashboard from `monitoring/grafana/dashboards/anomaly-detection-overview.json`

### Log Aggregation

#### ELK Stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.0.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.0.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.0.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

## Security Configuration

### Network Security

```bash
# Firewall rules (UFW example)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw deny 8000/tcp   # Block direct API access
sudo ufw enable
```

### Container Security

```bash
# Run security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image ghcr.io/terragonlabs/iot-anomaly-detector:latest
```

### Secrets Management

#### Kubernetes Secrets

```bash
# Create secret from file
kubectl create secret generic app-secrets \
  --from-file=.env.prod \
  --namespace=anomaly-detection
```

#### AWS Secrets Manager

```bash
# Store secret
aws secretsmanager create-secret \
  --name "anomaly-detector/database-url" \
  --secret-string "postgresql://user:pass@host:5432/db"
```

## Health Checks and Monitoring

### Health Check Endpoints

- `/health` - Basic health check
- `/ready` - Readiness probe
- `/metrics` - Prometheus metrics

### Monitoring Alerts

```yaml
# prometheus/alerts.yml
groups:
- name: anomaly-detector
  rules:
  - alert: ServiceDown
    expr: up{job="anomaly-detector"} == 0
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Anomaly Detection service is down"

  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
```

## Backup and Disaster Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# PostgreSQL backup
docker exec postgres pg_dump -U user anomaly_db > \
  $BACKUP_DIR/anomaly_db_$DATE.sql

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/anomaly_db_$DATE.sql \
  s3://your-backup-bucket/database/
```

### Model Backup

```bash
# Backup trained models
docker run --rm -v models-volume:/models \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/models_$DATE.tar.gz /models
```

### Restore Procedures

```bash
# Restore database
docker exec -i postgres psql -U user anomaly_db < backup.sql

# Restore models
docker run --rm -v models-volume:/models \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/models_backup.tar.gz -C /
```

## Performance Tuning

### Application Tuning

```python
# gunicorn.conf.py
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 120
keepalive = 5
```

### Database Tuning

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
SELECT pg_reload_conf();
```

### Cache Configuration

```python
# Redis configuration
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://redis:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 20,
                'retry_on_timeout': True,
            }
        }
    }
}
```

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker logs container_name

# Check resource usage
docker stats

# Inspect container
docker inspect container_name
```

#### Database Connection Issues

```bash
# Test database connectivity
docker exec app pg_isready -h postgres -p 5432

# Check database logs
docker logs postgres
```

#### Performance Issues

```bash
# Monitor resource usage
htop
iotop
netstat -tulpn

# Application metrics
curl http://localhost:8000/metrics
```

#### SSL/TLS Issues

```bash
# Test SSL configuration
openssl s_client -connect your-domain.com:443

# Check certificate
openssl x509 -in cert.pem -text -noout
```

### Log Analysis

```bash
# Application logs
docker logs -f app

# Structured log analysis
docker logs app | jq '.level == "ERROR"'

# Performance analysis
docker logs app | grep "slow_query"
```

### Emergency Procedures

#### Service Recovery

```bash
# Quick restart
docker-compose restart app

# Full system restart
docker-compose down
docker-compose up -d

# Rollback deployment
docker-compose down
docker pull ghcr.io/terragonlabs/iot-anomaly-detector:previous-tag
docker-compose up -d
```

#### Data Recovery

```bash
# Restore from backup
docker-compose down
docker volume rm project_postgres-data
docker-compose up -d postgres
# Wait for startup, then restore
```

For additional support, check the troubleshooting section in the main documentation or create an issue in the GitHub repository.