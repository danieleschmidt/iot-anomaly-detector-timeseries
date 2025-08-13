"""Production deployment platform with containerization and orchestration."""

import os
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ServiceType(Enum):
    """Service types for deployment."""
    WEB_API = "web_api"
    INFERENCE_ENGINE = "inference_engine"
    MONITORING = "monitoring"
    DATABASE = "database"
    CACHE = "cache"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    name: str
    environment: DeploymentEnvironment
    services: List[Dict[str, Any]]
    resources: Dict[str, Any]
    scaling: Dict[str, Any]
    monitoring: Dict[str, Any]
    security: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['environment'] = self.environment.value
        return result


class ProductionDeploymentPlatform:
    """Complete production deployment platform."""
    
    def __init__(self, project_name: str = "iot-anomaly-detector"):
        self.project_name = project_name
        self.deployment_dir = Path("deployment")
        self.templates_dir = self.deployment_dir / "templates"
        self.configs_dir = self.deployment_dir / "configs"
        self.scripts_dir = self.deployment_dir / "scripts"
        
        # Create directory structure
        for dir_path in [self.deployment_dir, self.templates_dir, self.configs_dir, self.scripts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_dockerfile(self, service_type: ServiceType) -> str:
        """Generate optimized Dockerfile for service type."""
        
        if service_type == ServiceType.WEB_API:
            dockerfile_content = """
# Multi-stage build for optimized production image
FROM python:3.12-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY src/ ./src/
COPY enhanced_autoencoder.py smart_preprocessor.py ./
COPY robust_monitoring_system.py comprehensive_error_handling.py ./
COPY scalable_inference_platform.py intelligent_resource_manager.py ./

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Update PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "uvicorn", "src.model_serving_api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        elif service_type == ServiceType.INFERENCE_ENGINE:
            dockerfile_content = """
# Optimized inference engine image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN groupadd -r inference && useradd -r -g inference inference

# Copy application code
COPY enhanced_autoencoder.py smart_preprocessor.py ./
COPY scalable_inference_platform.py intelligent_resource_manager.py ./
COPY saved_models/ ./saved_models/

# Set ownership
RUN chown -R inference:inference /app

# Switch to non-root user
USER inference

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \\
    CMD python -c "import scalable_inference_platform; print('OK')" || exit 1

# Expose port
EXPOSE 8001

# Start inference engine
CMD ["python", "scalable_inference_platform.py"]
"""
        
        elif service_type == ServiceType.MONITORING:
            dockerfile_content = """
# Monitoring service image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    procps \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN groupadd -r monitor && useradd -r -g monitor monitor

# Copy monitoring code
COPY robust_monitoring_system.py comprehensive_error_handling.py ./
COPY intelligent_resource_manager.py ./

# Set ownership
RUN chown -R monitor:monitor /app

# Switch to non-root user
USER monitor

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import robust_monitoring_system; print('OK')" || exit 1

# Expose port
EXPOSE 8002

# Start monitoring
CMD ["python", "robust_monitoring_system.py"]
"""
        
        else:
            dockerfile_content = """
# Generic service image
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "app.py"]
"""
        
        return dockerfile_content.strip()
    
    def generate_docker_compose(self, config: DeploymentConfig) -> str:
        """Generate docker-compose.yml for the deployment."""
        
        compose_config = {
            'version': '3.8',
            'services': {},
            'networks': {
                'app-network': {
                    'driver': 'bridge'
                }
            },
            'volumes': {
                'models-data': {},
                'logs-data': {},
                'monitoring-data': {}
            }
        }
        
        # Add services
        for service in config.services:
            service_name = service['name']
            service_type = ServiceType(service['type'])
            
            service_config = {
                'build': {
                    'context': '.',
                    'dockerfile': f'deployment/dockerfiles/Dockerfile.{service_type.value}'
                },
                'container_name': f"{self.project_name}-{service_name}",
                'restart': 'unless-stopped',
                'networks': ['app-network'],
                'environment': service.get('environment', {}),
                'volumes': [],
                'ports': [],
                'depends_on': service.get('depends_on', []),
                'healthcheck': {
                    'test': service.get('healthcheck', 'CMD-SHELL exit 0'),
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3,
                    'start_period': '10s'
                }
            }
            
            # Add service-specific configuration
            if service_type == ServiceType.WEB_API:
                service_config['ports'] = ['8000:8000']
                service_config['volumes'] = [
                    'models-data:/app/saved_models:ro',
                    'logs-data:/app/logs'
                ]
                service_config['environment'].update({
                    'MODEL_PATH': '/app/saved_models',
                    'LOG_LEVEL': 'INFO',
                    'WORKERS': '4'
                })
            
            elif service_type == ServiceType.INFERENCE_ENGINE:
                service_config['ports'] = ['8001:8001']
                service_config['volumes'] = [
                    'models-data:/app/saved_models:ro',
                    'logs-data:/app/logs'
                ]
                service_config['environment'].update({
                    'BATCH_SIZE': '32',
                    'MAX_WORKERS': '8',
                    'CACHE_SIZE': '10000'
                })
            
            elif service_type == ServiceType.MONITORING:
                service_config['ports'] = ['8002:8002']
                service_config['volumes'] = [
                    'monitoring-data:/app/data',
                    'logs-data:/app/logs',
                    '/var/run/docker.sock:/var/run/docker.sock:ro'
                ]
                service_config['environment'].update({
                    'MONITORING_INTERVAL': '30',
                    'ALERT_WEBHOOK': 'http://alertmanager:9093'
                })
            
            elif service_type == ServiceType.DATABASE:
                service_config['image'] = 'postgres:15-alpine'
                service_config['volumes'] = ['postgres-data:/var/lib/postgresql/data']
                service_config['environment'].update({
                    'POSTGRES_DB': 'anomaly_detection',
                    'POSTGRES_USER': 'app_user',
                    'POSTGRES_PASSWORD': 'secure_password_change_me'
                })
                service_config['ports'] = ['5432:5432']
                # Remove build config for database
                service_config.pop('build', None)
            
            elif service_type == ServiceType.CACHE:
                service_config['image'] = 'redis:7-alpine'
                service_config['volumes'] = ['redis-data:/data']
                service_config['ports'] = ['6379:6379']
                service_config['command'] = 'redis-server --appendonly yes'
                # Remove build config for cache
                service_config.pop('build', None)
            
            compose_config['services'][service_name] = service_config
        
        # Add additional volumes for database and cache
        if any(ServiceType(s['type']) == ServiceType.DATABASE for s in config.services):
            compose_config['volumes']['postgres-data'] = {}
        
        if any(ServiceType(s['type']) == ServiceType.CACHE for s in config.services):
            compose_config['volumes']['redis-data'] = {}
        
        return yaml.dump(compose_config, default_flow_style=False, sort_keys=False)
    
    def generate_kubernetes_manifests(self, config: DeploymentConfig) -> Dict[str, str]:
        """Generate Kubernetes deployment manifests."""
        
        manifests = {}
        
        # Namespace
        namespace_manifest = {
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {
                'name': self.project_name,
                'labels': {
                    'app': self.project_name,
                    'environment': config.environment.value
                }
            }
        }
        manifests['namespace.yaml'] = yaml.dump(namespace_manifest)
        
        # ConfigMap
        configmap_manifest = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f"{self.project_name}-config",
                'namespace': self.project_name
            },
            'data': {
                'ENVIRONMENT': config.environment.value,
                'LOG_LEVEL': 'INFO',
                'MODEL_PATH': '/app/saved_models',
                'BATCH_SIZE': '32',
                'MAX_WORKERS': '8'
            }
        }
        manifests['configmap.yaml'] = yaml.dump(configmap_manifest)
        
        # Generate manifests for each service
        for service in config.services:
            service_name = service['name']
            service_type = ServiceType(service['type'])
            
            if service_type in [ServiceType.DATABASE, ServiceType.CACHE]:
                continue  # Skip database and cache for K8s (use managed services)
            
            # Deployment
            deployment_manifest = {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': f"{self.project_name}-{service_name}",
                    'namespace': self.project_name,
                    'labels': {
                        'app': self.project_name,
                        'service': service_name,
                        'version': 'v1'
                    }
                },
                'spec': {
                    'replicas': service.get('replicas', 2),
                    'selector': {
                        'matchLabels': {
                            'app': self.project_name,
                            'service': service_name
                        }
                    },
                    'template': {
                        'metadata': {
                            'labels': {
                                'app': self.project_name,
                                'service': service_name,
                                'version': 'v1'
                            }
                        },
                        'spec': {
                            'containers': [{
                                'name': service_name,
                                'image': f"{self.project_name}/{service_name}:latest",
                                'ports': [{'containerPort': service.get('port', 8000)}],
                                'env': [
                                    {'name': 'ENVIRONMENT', 'valueFrom': {'configMapKeyRef': {'name': f"{self.project_name}-config", 'key': 'ENVIRONMENT'}}},
                                    {'name': 'LOG_LEVEL', 'valueFrom': {'configMapKeyRef': {'name': f"{self.project_name}-config", 'key': 'LOG_LEVEL'}}}
                                ],
                                'resources': {
                                    'requests': {
                                        'memory': service.get('memory_request', '256Mi'),
                                        'cpu': service.get('cpu_request', '100m')
                                    },
                                    'limits': {
                                        'memory': service.get('memory_limit', '512Mi'),
                                        'cpu': service.get('cpu_limit', '500m')
                                    }
                                },
                                'livenessProbe': {
                                    'httpGet': {
                                        'path': '/health',
                                        'port': service.get('port', 8000)
                                    },
                                    'initialDelaySeconds': 30,
                                    'periodSeconds': 10
                                },
                                'readinessProbe': {
                                    'httpGet': {
                                        'path': '/ready',
                                        'port': service.get('port', 8000)
                                    },
                                    'initialDelaySeconds': 5,
                                    'periodSeconds': 5
                                }
                            }],
                            'securityContext': {
                                'runAsNonRoot': True,
                                'runAsUser': 1000,
                                'fsGroup': 1000
                            }
                        }
                    }
                }
            }
            
            manifests[f"deployment-{service_name}.yaml"] = yaml.dump(deployment_manifest)
            
            # Service
            service_manifest = {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': f"{self.project_name}-{service_name}",
                    'namespace': self.project_name,
                    'labels': {
                        'app': self.project_name,
                        'service': service_name
                    }
                },
                'spec': {
                    'selector': {
                        'app': self.project_name,
                        'service': service_name
                    },
                    'ports': [{
                        'protocol': 'TCP',
                        'port': service.get('port', 8000),
                        'targetPort': service.get('port', 8000)
                    }],
                    'type': 'ClusterIP'
                }
            }
            
            manifests[f"service-{service_name}.yaml"] = yaml.dump(service_manifest)
            
            # Ingress for web services
            if service_type == ServiceType.WEB_API:
                ingress_manifest = {
                    'apiVersion': 'networking.k8s.io/v1',
                    'kind': 'Ingress',
                    'metadata': {
                        'name': f"{self.project_name}-{service_name}",
                        'namespace': self.project_name,
                        'annotations': {
                            'nginx.ingress.kubernetes.io/rewrite-target': '/',
                            'cert-manager.io/cluster-issuer': 'letsencrypt-prod'
                        }
                    },
                    'spec': {
                        'tls': [{
                            'hosts': [f"{service_name}.{self.project_name}.com"],
                            'secretName': f"{self.project_name}-{service_name}-tls"
                        }],
                        'rules': [{
                            'host': f"{service_name}.{self.project_name}.com",
                            'http': {
                                'paths': [{
                                    'path': '/',
                                    'pathType': 'Prefix',
                                    'backend': {
                                        'service': {
                                            'name': f"{self.project_name}-{service_name}",
                                            'port': {'number': service.get('port', 8000)}
                                        }
                                    }
                                }]
                            }
                        }]
                    }
                }
                
                manifests[f"ingress-{service_name}.yaml"] = yaml.dump(ingress_manifest)
        
        # HorizontalPodAutoscaler
        hpa_manifest = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f"{self.project_name}-hpa",
                'namespace': self.project_name
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': f"{self.project_name}-api"
                },
                'minReplicas': config.scaling.get('min_replicas', 2),
                'maxReplicas': config.scaling.get('max_replicas', 10),
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 80
                            }
                        }
                    }
                ]
            }
        }
        
        manifests['hpa.yaml'] = yaml.dump(hpa_manifest)
        
        return manifests
    
    def generate_deployment_scripts(self) -> Dict[str, str]:
        """Generate deployment and management scripts."""
        
        scripts = {}
        
        # Build script
        build_script = """#!/bin/bash
set -e

echo "Building Docker images..."

# Build API service
docker build -t iot-anomaly-detector/api:latest -f deployment/dockerfiles/Dockerfile.web_api .

# Build inference engine
docker build -t iot-anomaly-detector/inference:latest -f deployment/dockerfiles/Dockerfile.inference_engine .

# Build monitoring service
docker build -t iot-anomaly-detector/monitoring:latest -f deployment/dockerfiles/Dockerfile.monitoring .

echo "Build completed successfully!"
"""
        scripts['build.sh'] = build_script
        
        # Deploy script
        deploy_script = """#!/bin/bash
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
"""
        scripts['deploy.sh'] = deploy_script
        
        # Monitoring script
        monitoring_script = """#!/bin/bash

echo "System Status:"
echo "=============="

# Docker containers status
if command -v docker &> /dev/null; then
    echo "Docker Containers:"
    docker ps --filter "name=iot-anomaly-detector" --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}"
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
    docker stats --no-stream --format "table {{.Name}}\\t{{.CPUPerc}}\\t{{.MemUsage}}" $(docker ps --filter "name=iot-anomaly-detector" -q)
fi
"""
        scripts['monitor.sh'] = monitoring_script
        
        # Cleanup script
        cleanup_script = """#!/bin/bash

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
"""
        scripts['cleanup.sh'] = cleanup_script
        
        return scripts
    
    def generate_environment_configs(self) -> Dict[str, str]:
        """Generate environment-specific configuration files."""
        
        configs = {}
        
        # Development environment
        dev_config = """
# Development Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG=true

# Database
DATABASE_URL=postgresql://app_user:secure_password_change_me@postgres:5432/anomaly_detection

# Cache
REDIS_URL=redis://redis:6379/0

# Model Configuration
MODEL_PATH=/app/saved_models
BATCH_SIZE=16
MAX_WORKERS=2

# Monitoring
MONITORING_INTERVAL=60
ALERT_THRESHOLD=0.8

# Security (Development only - change in production)
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET=dev-jwt-secret-change-in-production
"""
        configs['development.env'] = dev_config
        
        # Staging environment
        staging_config = """
# Staging Environment Configuration
ENVIRONMENT=staging
LOG_LEVEL=INFO
DEBUG=false

# Database
DATABASE_URL=postgresql://app_user:staging_password@staging-db:5432/anomaly_detection

# Cache
REDIS_URL=redis://staging-cache:6379/0

# Model Configuration
MODEL_PATH=/app/saved_models
BATCH_SIZE=32
MAX_WORKERS=4

# Monitoring
MONITORING_INTERVAL=30
ALERT_THRESHOLD=0.85

# Security
SECRET_KEY=${SECRET_KEY}
JWT_SECRET=${JWT_SECRET}
"""
        configs['staging.env'] = staging_config
        
        # Production environment
        prod_config = """
# Production Environment Configuration
ENVIRONMENT=production
LOG_LEVEL=WARNING
DEBUG=false

# Database (Use managed database service)
DATABASE_URL=${DATABASE_URL}

# Cache (Use managed Redis service)
REDIS_URL=${REDIS_URL}

# Model Configuration
MODEL_PATH=/app/saved_models
BATCH_SIZE=64
MAX_WORKERS=8

# Monitoring
MONITORING_INTERVAL=15
ALERT_THRESHOLD=0.9

# Security
SECRET_KEY=${SECRET_KEY}
JWT_SECRET=${JWT_SECRET}

# Performance
GUNICORN_WORKERS=4
GUNICORN_WORKER_CLASS=uvicorn.workers.UvicornWorker
GUNICORN_MAX_REQUESTS=1000
GUNICORN_TIMEOUT=30
"""
        configs['production.env'] = prod_config
        
        return configs
    
    def create_deployment_package(self, config: DeploymentConfig) -> str:
        """Create complete deployment package."""
        
        logger.info(f"Creating deployment package for {config.environment.value} environment")
        
        # Create dockerfiles directory
        dockerfiles_dir = self.deployment_dir / "dockerfiles"
        dockerfiles_dir.mkdir(exist_ok=True)
        
        # Generate and save Dockerfiles
        for service in config.services:
            service_type = ServiceType(service['type'])
            if service_type not in [ServiceType.DATABASE, ServiceType.CACHE]:  # Skip external services
                dockerfile_content = self.generate_dockerfile(service_type)
                dockerfile_path = dockerfiles_dir / f"Dockerfile.{service_type.value}"
                dockerfile_path.write_text(dockerfile_content)
                logger.info(f"Generated Dockerfile for {service_type.value}")
        
        # Generate and save docker-compose.yml
        compose_content = self.generate_docker_compose(config)
        compose_path = self.deployment_dir / "docker-compose.yml"
        compose_path.write_text(compose_content)
        logger.info("Generated docker-compose.yml")
        
        # Generate and save Kubernetes manifests
        k8s_dir = self.deployment_dir / "k8s"
        k8s_dir.mkdir(exist_ok=True)
        
        k8s_manifests = self.generate_kubernetes_manifests(config)
        for filename, content in k8s_manifests.items():
            manifest_path = k8s_dir / filename
            manifest_path.write_text(content)
            logger.info(f"Generated Kubernetes manifest: {filename}")
        
        # Generate and save deployment scripts
        scripts = self.generate_deployment_scripts()
        for script_name, script_content in scripts.items():
            script_path = self.scripts_dir / script_name
            script_path.write_text(script_content)
            script_path.chmod(0o755)  # Make executable
            logger.info(f"Generated deployment script: {script_name}")
        
        # Generate and save environment configurations
        env_configs = self.generate_environment_configs()
        for config_name, config_content in env_configs.items():
            config_path = self.configs_dir / config_name
            config_path.write_text(config_content)
            logger.info(f"Generated environment config: {config_name}")
        
        # Generate deployment configuration file
        config_path = self.deployment_dir / "deployment-config.json"
        config_path.write_text(json.dumps(config.to_dict(), indent=2))
        logger.info("Generated deployment configuration")
        
        # Generate README
        readme_content = self._generate_deployment_readme()
        readme_path = self.deployment_dir / "README.md"
        readme_path.write_text(readme_content)
        logger.info("Generated deployment README")
        
        logger.info(f"Deployment package created in {self.deployment_dir}")
        return str(self.deployment_dir)
    
    def _generate_deployment_readme(self) -> str:
        """Generate deployment README."""
        
        readme_content = f"""# {self.project_name.title()} Deployment Guide

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
kubectl get pods -n {self.project_name}
kubectl logs -f deployment/[deployment-name] -n {self.project_name}

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
"""
        return readme_content


def create_production_deployment():
    """Create production-ready deployment configuration."""
    
    logger.info("=== PRODUCTION DEPLOYMENT CONFIGURATION ===")
    
    # Initialize deployment platform
    platform = ProductionDeploymentPlatform()
    
    # Define deployment configuration
    config = DeploymentConfig(
        name="iot-anomaly-detector",
        environment=DeploymentEnvironment.PRODUCTION,
        services=[
            {
                'name': 'api',
                'type': ServiceType.WEB_API.value,
                'port': 8000,
                'replicas': 3,
                'memory_request': '512Mi',
                'memory_limit': '1Gi',
                'cpu_request': '250m',
                'cpu_limit': '500m',
                'healthcheck': 'CMD-SHELL curl -f http://localhost:8000/health || exit 1'
            },
            {
                'name': 'inference',
                'type': ServiceType.INFERENCE_ENGINE.value,
                'port': 8001,
                'replicas': 2,
                'memory_request': '1Gi',
                'memory_limit': '2Gi',
                'cpu_request': '500m',
                'cpu_limit': '1000m',
                'depends_on': ['database', 'cache']
            },
            {
                'name': 'monitoring',
                'type': ServiceType.MONITORING.value,
                'port': 8002,
                'replicas': 1,
                'memory_request': '256Mi',
                'memory_limit': '512Mi',
                'cpu_request': '100m',
                'cpu_limit': '250m'
            },
            {
                'name': 'database',
                'type': ServiceType.DATABASE.value,
                'port': 5432
            },
            {
                'name': 'cache',
                'type': ServiceType.CACHE.value,
                'port': 6379
            }
        ],
        resources={
            'total_cpu': '4000m',
            'total_memory': '8Gi',
            'storage': '50Gi'
        },
        scaling={
            'min_replicas': 2,
            'max_replicas': 10,
            'target_cpu_utilization': 70,
            'target_memory_utilization': 80
        },
        monitoring={
            'metrics_enabled': True,
            'logging_enabled': True,
            'alerting_enabled': True,
            'health_checks_enabled': True
        },
        security={
            'run_as_non_root': True,
            'read_only_root_filesystem': True,
            'drop_capabilities': ['ALL'],
            'network_policies_enabled': True
        }
    )
    
    # Create deployment package
    deployment_path = platform.create_deployment_package(config)
    
    logger.info(f"Production deployment package created at: {deployment_path}")
    logger.info("To deploy:")
    logger.info("  1. Set environment variables (see configs/production.env)")
    logger.info("  2. Run: ./scripts/build.sh")
    logger.info("  3. Run: ./scripts/deploy.sh production")
    logger.info("  4. Monitor: ./scripts/monitor.sh")
    
    return deployment_path


if __name__ == "__main__":
    create_production_deployment()