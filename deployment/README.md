# Iot-Anomaly-Detector Deployment Guide

This directory contains all the necessary files for deploying the IoT Anomaly Detection system across different environments.

## Directory Structure

```
deployment/
├── README.md                 # This file
├── deployment-config.json    # Deployment configuration
├── docker-compose.yml        # Docker Compose configuration
├── configs/                  # Environment-specific configurations
│   ├── development.env
│   ├── staging.env
│   └── production.env
├── dockerfiles/             # Service-specific Dockerfiles
│   ├── Dockerfile.web_api
│   ├── Dockerfile.inference_engine
│   └── Dockerfile.monitoring
├── k8s/                     # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── deployment-*.yaml
│   ├── service-*.yaml
│   ├── ingress-*.yaml
│   └── hpa.yaml
└── scripts/                 # Deployment scripts
    ├── build.sh
    ├── deploy.sh
    ├── monitor.sh
    └── cleanup.sh
```

## Quick Start

### Development Environment

```bash
# Build images
./scripts/build.sh

# Deploy with Docker Compose
./scripts/deploy.sh development

# Monitor services
./scripts/monitor.sh

# Cleanup
./scripts/cleanup.sh development
```

### Staging Environment

```bash
# Set environment variables
export SECRET_KEY="your-secret-key"
export JWT_SECRET="your-jwt-secret"

# Deploy
./scripts/deploy.sh staging
```

### Production Environment

```bash
# Set production environment variables
export DATABASE_URL="postgresql://user:pass@prod-db:5432/db"
export REDIS_URL="redis://prod-cache:6379/0"
export SECRET_KEY="production-secret-key"
export JWT_SECRET="production-jwt-secret"

# Deploy to Kubernetes
./scripts/deploy.sh production
```

## Services

### API Service (Port 8000)
- Web API for anomaly detection
- RESTful endpoints
- Authentication and authorization
- Rate limiting and caching

### Inference Engine (Port 8001)
- Real-time anomaly detection
- Batch processing capabilities
- Model serving and management
- Auto-scaling based on load

### Monitoring Service (Port 8002)
- System health monitoring
- Performance metrics collection
- Alerting and notifications
- Resource usage tracking

## Environment Variables

See the `configs/` directory for environment-specific configuration files.

### Required Environment Variables for Production

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: Application secret key
- `JWT_SECRET`: JWT signing secret

## Health Checks

All services expose health check endpoints:

- API: `http://localhost:8000/health`
- Inference: `http://localhost:8001/health`
- Monitoring: `http://localhost:8002/health`

## Scaling

### Docker Compose
Modify the `docker-compose.yml` file to adjust service replicas.

### Kubernetes
Use the HorizontalPodAutoscaler (HPA) for automatic scaling based on CPU/memory usage.

## Security

- All services run as non-root users
- Container images are built with security best practices
- Network policies restrict inter-service communication
- Secrets are managed through environment variables

## Monitoring and Logging

- Application logs are centralized and structured
- Metrics are collected and exposed via Prometheus
- Health checks ensure service availability
- Alerting is configured for critical issues

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8000-8002 are available
2. **Database connection**: Verify DATABASE_URL is correct
3. **Memory issues**: Increase container memory limits
4. **Permission errors**: Check file ownership and permissions

### Debug Commands

```bash
# Check container logs
docker-compose logs -f [service-name]

# Check Kubernetes pods
kubectl get pods -n iot-anomaly-detector
kubectl logs -f deployment/[deployment-name] -n iot-anomaly-detector

# Test service endpoints
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

## Performance Tuning

### Resource Limits
Adjust CPU and memory limits in:
- `docker-compose.yml` for Docker Compose
- Kubernetes deployment manifests for K8s

### Model Optimization
- Use model quantization for faster inference
- Implement model caching strategies
- Configure batch processing parameters

## Backup and Recovery

### Database Backup
```bash
# PostgreSQL backup
pg_dump $DATABASE_URL > backup.sql

# Restore
psql $DATABASE_URL < backup.sql
```

### Model Backup
Models are stored in persistent volumes and should be backed up regularly.

## Support

For deployment issues or questions:
1. Check the logs for error messages
2. Verify environment configuration
3. Ensure all dependencies are installed
4. Contact the development team with specific error details
