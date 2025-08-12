# IoT Anomaly Detection System - Production Deployment Guide

## Overview

This guide covers the complete deployment of the IoT Anomaly Detection System to a production Kubernetes environment with high availability, scalability, and monitoring.

## System Architecture

```mermaid
graph TB
    A[Load Balancer] --> B[API Gateway]
    B --> C[Anomaly Detection API Pods]
    C --> D[Redis Cache Cluster]
    C --> E[Model Storage]
    C --> F[Monitoring Stack]
    
    subgraph "Kubernetes Cluster"
        C
        D
        E
        F
    end
    
    G[IoT Sensors] --> A
    H[Data Pipeline] --> A
```

## Prerequisites

### Infrastructure Requirements

- **Kubernetes Cluster**: v1.24+
- **Node Requirements**: 
  - Minimum 3 nodes
  - 8 CPU cores per node
  - 32GB RAM per node
  - 500GB SSD storage per node
- **Network**: CNI plugin (Calico, Flannel, or Cilium)
- **Load Balancer**: NGINX Ingress Controller or cloud provider LB
- **Certificate Manager**: cert-manager for TLS

### Software Dependencies

- `kubectl` CLI tool
- `helm` v3.8+
- Docker registry access
- Monitoring stack (Prometheus, Grafana)

## Pre-Deployment Setup

### 1. Build and Push Docker Images

```bash
# Build the production image
docker build -f deploy/Dockerfile --target production -t terragonlabs/iot-anomaly-detector:v1.0.0 .

# Build development image
docker build -f deploy/Dockerfile --target development -t terragonlabs/iot-anomaly-detector:v1.0.0-dev .

# Push to registry
docker push terragonlabs/iot-anomaly-detector:v1.0.0
docker push terragonlabs/iot-anomaly-detector:v1.0.0-dev
```

### 2. Prepare Models

```bash
# Train models (if not already done)
python -m src.train_autoencoder --epochs 50 --model-path models/autoencoder.h5

# Prepare model archive
tar -czf models.tar.gz models/

# Upload to persistent storage (example with cloud storage)
gsutil cp models.tar.gz gs://your-bucket/models/
```

### 3. Configure Secrets

```bash
# Create namespace
kubectl create namespace iot-anomaly-detection

# Create Redis credentials secret
kubectl create secret generic redis-credentials \
  --from-literal=url="redis://user:password@redis.example.com:6379/0" \
  -n iot-anomaly-detection

# Create TLS certificates (if using cert-manager)
kubectl apply -f deploy/certificates.yaml
```

## Deployment Steps

### 1. Deploy Infrastructure Components

#### Redis Cache Cluster

```bash
# Add Redis Helm repository
helm repo add bitnami https://charts.bitnami.com/bitnami

# Install Redis cluster
helm install redis-cluster bitnami/redis-cluster \
  --namespace iot-anomaly-detection \
  --set auth.enabled=true \
  --set auth.password="secure-password" \
  --set cluster.nodes=6 \
  --set cluster.replicas=1 \
  --set persistence.enabled=true \
  --set persistence.size=50Gi
```

#### Monitoring Stack

```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

# Install Prometheus and Grafana
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi
```

### 2. Deploy the Application

```bash
# Apply the main deployment
kubectl apply -f deploy/production_deployment.yaml

# Verify deployment
kubectl get pods -n iot-anomaly-detection
kubectl get services -n iot-anomaly-detection
kubectl get ingress -n iot-anomaly-detection
```

### 3. Configure Model Storage

```bash
# Create persistent volume for models
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-storage-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: fast-ssd
  nfs:
    server: nfs.example.com
    path: /exports/models
EOF

# Load models into the persistent volume
kubectl run model-loader --image=alpine --rm -it --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"models","persistentVolumeClaim":{"claimName":"model-storage-pvc"}}],"containers":[{"name":"model-loader","image":"alpine","command":["sh"],"volumeMounts":[{"name":"models","mountPath":"/models"}]}]}}' \
  -n iot-anomaly-detection

# Inside the container, download and extract models
wget https://your-bucket/models.tar.gz
tar -xzf models.tar.gz -C /models/
exit
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment | `production` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `REDIS_URL` | Redis connection URL | - | Yes |
| `MODEL_PATH` | Path to model files | `/app/models` | No |
| `MAX_WORKERS` | Number of worker processes | `8` | No |
| `ENABLE_QUANTUM` | Enable quantum algorithms | `true` | No |
| `API_RATE_LIMIT` | Requests per minute limit | `1000` | No |

### ConfigMap Configuration

The application uses a ConfigMap for configuration:

```yaml
# config.yaml in ConfigMap
api:
  host: 0.0.0.0
  port: 8080
  workers: 8
  timeout: 300

detection:
  window_size: 30
  batch_size: 32
  enable_quantum: true
  enable_caching: true
  cache_ttl: 3600

performance:
  max_queue_size: 1000
  auto_scaling: true
  metrics_enabled: true

security:
  enable_authentication: true
  rate_limiting: true
  max_requests_per_minute: 1000
```

## Security Configuration

### Network Policies

Network policies restrict communication between pods:

```bash
# Applied automatically with deployment
kubectl get networkpolicy -n iot-anomaly-detection
```

### Pod Security Standards

Pods run with security constraints:

- Non-root user (UID 1000)
- Read-only root filesystem (where possible)
- No privileged escalation
- Dropped capabilities

### RBAC

Role-Based Access Control limits pod permissions:

- ServiceAccount: `anomaly-detection-sa`
- Permissions: Read access to ConfigMaps and Secrets

## Scaling and Performance

### Horizontal Pod Autoscaling

The HPA automatically scales pods based on:

- CPU utilization (70% target)
- Memory utilization (80% target)  
- Custom metrics (requests per second)

```bash
# Check HPA status
kubectl get hpa -n iot-anomaly-detection

# View HPA details
kubectl describe hpa anomaly-detection-hpa -n iot-anomaly-detection
```

### Vertical Pod Autoscaling (Optional)

```bash
# Install VPA if not available
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/download/vertical-pod-autoscaler-0.13.0/vpa-release-0.13.0.yaml

# Apply VPA for the deployment
kubectl apply -f - <<EOF
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: anomaly-detection-vpa
  namespace: iot-anomaly-detection
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: anomaly-detection-api
  updatePolicy:
    updateMode: "Auto"
EOF
```

## Monitoring and Observability

### Metrics Collection

The application exposes Prometheus metrics on port 9090:

- Request rates and latencies
- Model prediction accuracy
- Cache hit rates
- Resource utilization
- Error rates

### Grafana Dashboards

Import the pre-built dashboard:

```bash
# Get Grafana admin password
kubectl get secret monitoring-grafana -n monitoring -o jsonpath="{.data.admin-password}" | base64 --decode

# Access Grafana UI
kubectl port-forward svc/monitoring-grafana 3000:80 -n monitoring

# Import dashboard from deploy/monitoring/grafana-dashboard.json
```

### Log Aggregation

Logs are structured and can be aggregated with:

- ELK Stack (Elasticsearch, Logstash, Kibana)
- Loki + Grafana
- Cloud provider logging solutions

### Alerting Rules

Key alerts are configured in Prometheus:

```yaml
# Example alerting rules
groups:
- name: anomaly-detection
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"

  - alert: HighMemoryUsage
    expr: container_memory_usage_bytes{pod=~"anomaly-detection-api.*"} / container_spec_memory_limit_bytes > 0.9
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage in anomaly detection pods"
```

## Health Checks and Probes

### Liveness Probe

Checks if the container is running and healthy:

- Endpoint: `/health`
- Initial delay: 30 seconds
- Period: 10 seconds

### Readiness Probe

Checks if the container is ready to receive traffic:

- Endpoint: `/ready`
- Initial delay: 5 seconds
- Period: 5 seconds

### Custom Health Check Script

The container includes a comprehensive health check:

```bash
# Run manual health check
kubectl exec -it deployment/anomaly-detection-api -n iot-anomaly-detection -- python healthcheck.py
```

## Disaster Recovery

### Backup Strategy

1. **Model Backups**: Automated backups of trained models
2. **Configuration Backups**: ConfigMaps and Secrets
3. **Persistent Volume Backups**: Using Velero or cloud-native tools

### Recovery Procedures

```bash
# Restore from backup (example with Velero)
velero restore create --from-backup anomaly-detection-backup-20231201

# Verify restoration
kubectl get pods -n iot-anomaly-detection
```

## Troubleshooting

### Common Issues

#### 1. Pod Startup Issues

```bash
# Check pod logs
kubectl logs -f deployment/anomaly-detection-api -n iot-anomaly-detection

# Check events
kubectl get events -n iot-anomaly-detection --sort-by=.metadata.creationTimestamp
```

#### 2. Model Loading Problems

```bash
# Check model directory contents
kubectl exec deployment/anomaly-detection-api -n iot-anomaly-detection -- ls -la /app/models/

# Test model loading
kubectl exec deployment/anomaly-detection-api -n iot-anomaly-detection -- python -c "
from src.resilient_anomaly_pipeline import ResilientAnomalyPipeline
pipeline = ResilientAnomalyPipeline()
print('Model loading test successful')
"
```

#### 3. Performance Issues

```bash
# Check resource utilization
kubectl top pods -n iot-anomaly-detection

# Check HPA status
kubectl describe hpa anomaly-detection-hpa -n iot-anomaly-detection

# Check application metrics
kubectl port-forward svc/anomaly-detection-api 9090:9090 -n iot-anomaly-detection
curl http://localhost:9090/metrics
```

### Performance Tuning

1. **Memory Optimization**:
   - Adjust JVM heap size if using Java components
   - Tune worker processes based on available RAM
   - Configure cache sizes appropriately

2. **CPU Optimization**:
   - Set appropriate CPU requests and limits
   - Use CPU affinity for consistent performance
   - Consider CPU governor settings on nodes

3. **Storage Optimization**:
   - Use fast SSDs for model storage
   - Configure appropriate I/O limits
   - Monitor disk usage and performance

## Maintenance

### Rolling Updates

```bash
# Update the deployment with new image
kubectl set image deployment/anomaly-detection-api \
  anomaly-detection-api=terragonlabs/iot-anomaly-detector:v1.1.0 \
  -n iot-anomaly-detection

# Monitor rollout status
kubectl rollout status deployment/anomaly-detection-api -n iot-anomaly-detection

# Rollback if needed
kubectl rollout undo deployment/anomaly-detection-api -n iot-anomaly-detection
```

### Model Updates

```bash
# Update models in persistent volume
kubectl run model-updater --image=alpine --rm -it --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"models","persistentVolumeClaim":{"claimName":"model-storage-pvc"}}],"containers":[{"name":"model-updater","image":"alpine","command":["sh"],"volumeMounts":[{"name":"models","mountPath":"/models"}]}]}}' \
  -n iot-anomaly-detection

# Restart pods to reload models
kubectl rollout restart deployment/anomaly-detection-api -n iot-anomaly-detection
```

### Log Rotation

Logs are automatically managed by Kubernetes, but for persistent logging:

```bash
# Configure log retention in container runtime
# Example for containerd: edit /etc/containerd/config.toml
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
  max_log_size = 100
  max_log_files = 5
```

## API Usage

### Health Check Endpoints

```bash
# Health check
curl https://api.anomaly-detection.terragonlabs.com/health

# Readiness check
curl https://api.anomaly-detection.terragonlabs.com/ready

# Metrics
curl https://api.anomaly-detection.terragonlabs.com/metrics
```

### Anomaly Detection API

```bash
# Submit data for analysis
curl -X POST https://api.anomaly-detection.terragonlabs.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[1.2, 3.4, 5.6], [2.3, 4.5, 6.7]],
    "window_size": 30,
    "method": "ensemble"
  }'

# Get prediction status
curl https://api.anomaly-detection.terragonlabs.com/predict/status/request-id-123

# Batch prediction
curl -X POST https://api.anomaly-detection.terragonlabs.com/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      [[1.2, 3.4, 5.6], [2.3, 4.5, 6.7]],
      [[3.4, 5.6, 7.8], [4.5, 6.7, 8.9]]
    ]
  }'
```

## Cost Optimization

1. **Resource Right-Sizing**:
   - Use VPA recommendations
   - Monitor actual vs. requested resources
   - Adjust based on usage patterns

2. **Auto-Scaling**:
   - Configure appropriate scaling thresholds
   - Use Cluster Autoscaler for node scaling
   - Consider spot instances for non-critical workloads

3. **Storage Optimization**:
   - Use appropriate storage classes
   - Implement data lifecycle policies
   - Compress model files where possible

## Compliance and Governance

### Security Compliance

- Regular security scans with tools like Falco
- Image vulnerability scanning
- Network policy enforcement
- Audit logging enabled

### Data Governance

- Data retention policies
- Privacy controls (GDPR, CCPA compliance)
- Data encryption at rest and in transit
- Access logging and monitoring

## Support and Documentation

- **API Documentation**: Available at `/docs` endpoint
- **Monitoring Dashboards**: Grafana dashboards for operational visibility
- **Runbooks**: Detailed procedures for common operational tasks
- **Support Channels**: Issue tracking and escalation procedures

This deployment guide ensures a production-ready, scalable, and maintainable IoT anomaly detection system with comprehensive monitoring, security, and operational excellence.