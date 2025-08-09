"""Production deployment utilities and infrastructure management."""

import os
import json
import yaml
import logging
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pathlib import Path
import tempfile
import shutil

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False


class DeploymentTarget(Enum):
    """Deployment target environments."""
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    CLOUD_AWS = "cloud_aws"
    CLOUD_GCP = "cloud_gcp"
    CLOUD_AZURE = "cloud_azure"
    EDGE = "edge"


class ServiceType(Enum):
    """Types of services to deploy."""
    API = "api"
    WORKER = "worker"
    SCHEDULER = "scheduler"
    MONITORING = "monitoring"
    EDGE_NODE = "edge_node"
    DATABASE = "database"
    CACHE = "cache"


@dataclass
class DeploymentConfig:
    """Configuration for production deployment."""
    target: DeploymentTarget
    environment: str  # dev, staging, prod
    service_name: str
    service_type: ServiceType
    version: str = "latest"
    
    # Resource requirements
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    
    # Scaling configuration
    min_replicas: int = 1
    max_replicas: int = 5
    target_cpu_utilization: int = 70
    
    # Networking
    port: int = 8000
    health_check_path: str = "/health"
    readiness_probe_path: str = "/ready"
    
    # Environment variables
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # Configuration files
    config_files: Dict[str, str] = field(default_factory=dict)
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)


@dataclass
class DeploymentResult:
    """Result of deployment operation."""
    success: bool
    deployment_id: str
    service_url: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class DockerDeploymentManager:
    """Docker deployment management."""
    
    def __init__(self):
        self.client = None
        self.logger = logging.getLogger(__name__)
        
        if DOCKER_AVAILABLE:
            try:
                self.client = docker.from_env()
                self.client.ping()
            except Exception as e:
                self.logger.error(f"Docker connection failed: {e}")
    
    def build_image(self, dockerfile_path: str, image_name: str, tag: str = "latest") -> bool:
        """Build Docker image."""
        if not self.client:
            self.logger.error("Docker client not available")
            return False
        
        try:
            self.logger.info(f"Building Docker image: {image_name}:{tag}")
            
            # Build image
            image, build_logs = self.client.images.build(
                path=str(Path(dockerfile_path).parent),
                dockerfile=Path(dockerfile_path).name,
                tag=f"{image_name}:{tag}",
                rm=True,
                forcerm=True
            )
            
            # Log build process
            for log in build_logs:
                if 'stream' in log:
                    self.logger.debug(log['stream'].strip())
            
            self.logger.info(f"Successfully built image: {image.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Docker build failed: {e}")
            return False
    
    def deploy_container(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy container using Docker."""
        if not self.client:
            return DeploymentResult(
                success=False,
                deployment_id="",
                error_message="Docker client not available"
            )
        
        try:
            container_name = f"{config.service_name}-{config.environment}"
            image_name = f"{config.service_name}:{config.version}"
            
            # Stop existing container if running
            try:
                existing_container = self.client.containers.get(container_name)
                existing_container.stop()
                existing_container.remove()
                self.logger.info(f"Stopped existing container: {container_name}")
            except docker.errors.NotFound:
                pass
            
            # Prepare environment variables
            env_vars = {
                "SERVICE_NAME": config.service_name,
                "SERVICE_TYPE": config.service_type.value,
                "ENVIRONMENT": config.environment,
                **config.environment_variables
            }
            
            # Create container
            container = self.client.containers.run(
                image_name,
                name=container_name,
                ports={f"{config.port}/tcp": config.port},
                environment=env_vars,
                detach=True,
                restart_policy={"Name": "unless-stopped"},
                healthcheck={
                    "test": ["CMD", "curl", "-f", f"http://localhost:{config.port}{config.health_check_path}"],
                    "interval": 30000000000,  # 30 seconds in nanoseconds
                    "timeout": 10000000000,   # 10 seconds
                    "retries": 3
                }
            )
            
            # Wait for container to be healthy
            self.logger.info(f"Waiting for container {container_name} to be healthy...")
            for _ in range(30):  # Wait up to 5 minutes
                container.reload()
                if container.status == "running":
                    # Check if health check passes
                    try:
                        health = container.attrs.get("State", {}).get("Health", {})
                        if health.get("Status") == "healthy":
                            break
                    except:
                        pass
                
                time.sleep(10)
            else:
                self.logger.warning(f"Container {container_name} may not be fully healthy")
            
            service_url = f"http://localhost:{config.port}"
            
            return DeploymentResult(
                success=True,
                deployment_id=container.id,
                service_url=service_url,
                logs=[f"Container deployed: {container_name}"]
            )
            
        except Exception as e:
            self.logger.error(f"Container deployment failed: {e}")
            return DeploymentResult(
                success=False,
                deployment_id="",
                error_message=str(e)
            )
    
    def get_container_logs(self, container_id: str, tail: int = 100) -> List[str]:
        """Get container logs."""
        if not self.client:
            return []
        
        try:
            container = self.client.containers.get(container_id)
            logs = container.logs(tail=tail, timestamps=True).decode('utf-8')
            return logs.split('\n')
        except Exception as e:
            self.logger.error(f"Failed to get container logs: {e}")
            return []


class KubernetesDeploymentManager:
    """Kubernetes deployment management."""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.v1 = None
        self.apps_v1 = None
        self.logger = logging.getLogger(__name__)
        
        if KUBERNETES_AVAILABLE:
            try:
                if kubeconfig_path:
                    config.load_kube_config(config_file=kubeconfig_path)
                else:
                    try:
                        config.load_incluster_config()
                    except:
                        config.load_kube_config()
                
                self.v1 = client.CoreV1Api()
                self.apps_v1 = client.AppsV1Api()
                
            except Exception as e:
                self.logger.error(f"Kubernetes connection failed: {e}")
    
    def create_deployment_manifest(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{config.service_name}-{config.environment}",
                "labels": {
                    "app": config.service_name,
                    "environment": config.environment,
                    "service-type": config.service_type.value
                }
            },
            "spec": {
                "replicas": config.min_replicas,
                "selector": {
                    "matchLabels": {
                        "app": config.service_name,
                        "environment": config.environment
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.service_name,
                            "environment": config.environment,
                            "service-type": config.service_type.value
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": config.service_name,
                            "image": f"{config.service_name}:{config.version}",
                            "ports": [{
                                "containerPort": config.port,
                                "protocol": "TCP"
                            }],
                            "env": [
                                {"name": k, "value": v}
                                for k, v in config.environment_variables.items()
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": config.cpu_request,
                                    "memory": config.memory_request
                                },
                                "limits": {
                                    "cpu": config.cpu_limit,
                                    "memory": config.memory_limit
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": config.health_check_path,
                                    "port": config.port
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 30
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": config.readiness_probe_path,
                                    "port": config.port
                                },
                                "initialDelaySeconds": 10,
                                "periodSeconds": 10
                            }
                        }]
                    }
                }
            }
        }
    
    def create_service_manifest(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create Kubernetes service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{config.service_name}-{config.environment}-service",
                "labels": {
                    "app": config.service_name,
                    "environment": config.environment
                }
            },
            "spec": {
                "type": "ClusterIP",
                "ports": [{
                    "port": 80,
                    "targetPort": config.port,
                    "protocol": "TCP"
                }],
                "selector": {
                    "app": config.service_name,
                    "environment": config.environment
                }
            }
        }
    
    def deploy_to_kubernetes(self, config: DeploymentConfig, namespace: str = "default") -> DeploymentResult:
        """Deploy to Kubernetes cluster."""
        if not self.apps_v1:
            return DeploymentResult(
                success=False,
                deployment_id="",
                error_message="Kubernetes client not available"
            )
        
        try:
            deployment_name = f"{config.service_name}-{config.environment}"
            
            # Create deployment manifest
            deployment_manifest = self.create_deployment_manifest(config)
            service_manifest = self.create_service_manifest(config)
            
            # Deploy or update deployment
            try:
                # Try to update existing deployment
                self.apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=deployment_manifest
                )
                self.logger.info(f"Updated existing deployment: {deployment_name}")
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    # Create new deployment
                    self.apps_v1.create_namespaced_deployment(
                        namespace=namespace,
                        body=deployment_manifest
                    )
                    self.logger.info(f"Created new deployment: {deployment_name}")
                else:
                    raise
            
            # Deploy or update service
            service_name = f"{config.service_name}-{config.environment}-service"
            try:
                self.v1.patch_namespaced_service(
                    name=service_name,
                    namespace=namespace,
                    body=service_manifest
                )
                self.logger.info(f"Updated existing service: {service_name}")
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    self.v1.create_namespaced_service(
                        namespace=namespace,
                        body=service_manifest
                    )
                    self.logger.info(f"Created new service: {service_name}")
                else:
                    raise
            
            # Wait for deployment to be ready
            self.logger.info("Waiting for deployment to be ready...")
            for _ in range(60):  # Wait up to 10 minutes
                try:
                    deployment = self.apps_v1.read_namespaced_deployment(
                        name=deployment_name,
                        namespace=namespace
                    )
                    
                    if (deployment.status.ready_replicas and
                        deployment.status.ready_replicas >= config.min_replicas):
                        break
                except:
                    pass
                
                time.sleep(10)
            else:
                self.logger.warning(f"Deployment {deployment_name} may not be fully ready")
            
            service_url = f"http://{service_name}.{namespace}.svc.cluster.local"
            
            return DeploymentResult(
                success=True,
                deployment_id=deployment_name,
                service_url=service_url,
                logs=[f"Deployed to Kubernetes: {deployment_name}"]
            )
            
        except Exception as e:
            self.logger.error(f"Kubernetes deployment failed: {e}")
            return DeploymentResult(
                success=False,
                deployment_id="",
                error_message=str(e)
            )


class ProductionDeploymentManager:
    """Main production deployment manager."""
    
    def __init__(self):
        self.docker_manager = DockerDeploymentManager()
        self.k8s_manager = KubernetesDeploymentManager()
        self.logger = logging.getLogger(__name__)
        
        # Deployment history
        self.deployment_history: List[DeploymentResult] = []
    
    def generate_dockerfile(
        self, 
        service_type: ServiceType,
        python_version: str = "3.11",
        requirements_file: str = "requirements.txt"
    ) -> str:
        """Generate optimized Dockerfile for service."""
        
        base_dockerfile = f"""# Multi-stage build for production optimization
FROM python:{python_version}-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY {requirements_file} .
RUN pip install --no-cache-dir --user -r {requirements_file}

# Production stage
FROM python:{python_version}-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /home/app/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY config/ ./config/ 
COPY scripts/ ./scripts/

# Set permissions
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Add local packages to PATH
ENV PATH=/home/app/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000
"""
        
        # Add service-specific commands
        if service_type == ServiceType.API:
            base_dockerfile += """
# Start API server
CMD ["python", "-m", "src.model_serving_api", "--host", "0.0.0.0", "--port", "8000"]
"""
        elif service_type == ServiceType.WORKER:
            base_dockerfile += """
# Start worker process
CMD ["python", "-m", "src.distributed_worker", "--role", "worker"]
"""
        elif service_type == ServiceType.SCHEDULER:
            base_dockerfile += """
# Start scheduler
CMD ["python", "-m", "src.task_scheduler"]
"""
        elif service_type == ServiceType.MONITORING:
            base_dockerfile += """
# Start monitoring service
CMD ["python", "-m", "src.comprehensive_monitoring", "--export-metrics", "/app/metrics"]
"""
        else:
            base_dockerfile += """
# Default command
CMD ["python", "-m", "src.main"]
"""
        
        return base_dockerfile
    
    def generate_docker_compose(self, configs: List[DeploymentConfig]) -> str:
        """Generate docker-compose.yml for multi-service deployment."""
        compose_config = {
            "version": "3.8",
            "services": {},
            "networks": {
                "anomaly_detection_network": {
                    "driver": "bridge"
                }
            },
            "volumes": {
                "redis_data": {},
                "postgres_data": {},
                "model_storage": {}
            }
        }
        
        # Add services
        for config in configs:
            service_config = {
                "build": {
                    "context": ".",
                    "dockerfile": f"Dockerfile.{config.service_type.value}"
                },
                "image": f"{config.service_name}:{config.version}",
                "container_name": f"{config.service_name}-{config.environment}",
                "ports": [f"{config.port}:{config.port}"],
                "environment": {
                    "SERVICE_NAME": config.service_name,
                    "SERVICE_TYPE": config.service_type.value,
                    "ENVIRONMENT": config.environment,
                    **config.environment_variables
                },
                "networks": ["anomaly_detection_network"],
                "restart": "unless-stopped",
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", f"http://localhost:{config.port}{config.health_check_path}"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3,
                    "start_period": "60s"
                }
            }
            
            # Add dependencies
            if config.dependencies:
                service_config["depends_on"] = config.dependencies
            
            # Add volumes for specific services
            if config.service_type == ServiceType.DATABASE:
                service_config["volumes"] = ["postgres_data:/var/lib/postgresql/data"]
            elif config.service_type == ServiceType.CACHE:
                service_config["volumes"] = ["redis_data:/data"]
            elif config.service_type in [ServiceType.API, ServiceType.WORKER]:
                service_config["volumes"] = ["model_storage:/app/models"]
            
            compose_config["services"][config.service_name] = service_config
        
        # Add infrastructure services
        if not any(c.service_type == ServiceType.DATABASE for c in configs):
            compose_config["services"]["postgres"] = {
                "image": "postgres:15-alpine",
                "environment": {
                    "POSTGRES_DB": "anomaly_detection",
                    "POSTGRES_USER": "anomaly_user",
                    "POSTGRES_PASSWORD": "anomaly_password"
                },
                "volumes": ["postgres_data:/var/lib/postgresql/data"],
                "networks": ["anomaly_detection_network"],
                "ports": ["5432:5432"]
            }
        
        if not any(c.service_type == ServiceType.CACHE for c in configs):
            compose_config["services"]["redis"] = {
                "image": "redis:7-alpine",
                "volumes": ["redis_data:/data"],
                "networks": ["anomaly_detection_network"],
                "ports": ["6379:6379"]
            }
        
        return yaml.dump(compose_config, default_flow_style=False, sort_keys=False)
    
    def deploy_service(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy service based on target."""
        self.logger.info(f"Deploying {config.service_name} to {config.target.value}")
        
        try:
            if config.target == DeploymentTarget.DOCKER:
                result = self.docker_manager.deploy_container(config)
            elif config.target == DeploymentTarget.KUBERNETES:
                result = self.k8s_manager.deploy_to_kubernetes(config)
            elif config.target == DeploymentTarget.LOCAL:
                result = self._deploy_local(config)
            else:
                result = DeploymentResult(
                    success=False,
                    deployment_id="",
                    error_message=f"Unsupported deployment target: {config.target.value}"
                )
            
            self.deployment_history.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            result = DeploymentResult(
                success=False,
                deployment_id="",
                error_message=str(e)
            )
            self.deployment_history.append(result)
            return result
    
    def _deploy_local(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy service locally."""
        try:
            # Create local service script
            service_script = self._generate_service_script(config)
            
            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(service_script)
                script_path = f.name
            
            # Start service in background
            env = os.environ.copy()
            env.update(config.environment_variables)
            
            process = subprocess.Popen(
                ["python", script_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment to check if service starts successfully
            time.sleep(2)
            
            if process.poll() is None:  # Process is still running
                service_url = f"http://localhost:{config.port}"
                return DeploymentResult(
                    success=True,
                    deployment_id=str(process.pid),
                    service_url=service_url,
                    logs=[f"Local service started with PID: {process.pid}"]
                )
            else:
                stdout, stderr = process.communicate()
                error_msg = stderr.decode('utf-8') if stderr else "Service failed to start"
                return DeploymentResult(
                    success=False,
                    deployment_id="",
                    error_message=error_msg
                )
                
        except Exception as e:
            return DeploymentResult(
                success=False,
                deployment_id="",
                error_message=str(e)
            )
    
    def _generate_service_script(self, config: DeploymentConfig) -> str:
        """Generate service startup script."""
        script = f"""#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables
os.environ["SERVICE_NAME"] = "{config.service_name}"
os.environ["SERVICE_TYPE"] = "{config.service_type.value}"
os.environ["ENVIRONMENT"] = "{config.environment}"
os.environ["PORT"] = "{config.port}"

# Additional environment variables
"""
        
        for key, value in config.environment_variables.items():
            script += f'os.environ["{key}"] = "{value}"\n'
        
        # Add service-specific startup code
        if config.service_type == ServiceType.API:
            script += """
from src.model_serving_api import create_app, run_server
app = create_app()
run_server(app, host="0.0.0.0", port=int(os.environ["PORT"]))
"""
        elif config.service_type == ServiceType.WORKER:
            script += """
from src.distributed_anomaly_detection import create_worker
worker = create_worker()
worker.start_processing()

# Keep running
import time
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    worker.shutdown()
"""
        elif config.service_type == ServiceType.MONITORING:
            script += """
from src.comprehensive_monitoring import create_monitoring_system
monitoring = create_monitoring_system()

# Keep running and export metrics periodically
import time
try:
    while True:
        time.sleep(300)  # Export every 5 minutes
        monitoring.export_metrics("metrics.json")
except KeyboardInterrupt:
    monitoring.shutdown()
"""
        else:
            script += """
print(f"Service {os.environ['SERVICE_NAME']} started")
import time
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("Service shutting down")
"""
        
        return script
    
    def create_deployment_package(
        self,
        configs: List[DeploymentConfig],
        output_dir: str = "deployment_package"
    ) -> str:
        """Create complete deployment package."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate Dockerfiles
        for config in configs:
            dockerfile_content = self.generate_dockerfile(config.service_type)
            dockerfile_path = output_path / f"Dockerfile.{config.service_type.value}"
            
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
        
        # Generate docker-compose.yml
        compose_content = self.generate_docker_compose(configs)
        with open(output_path / "docker-compose.yml", 'w') as f:
            f.write(compose_content)
        
        # Generate Kubernetes manifests
        k8s_dir = output_path / "kubernetes"
        k8s_dir.mkdir(exist_ok=True)
        
        for config in configs:
            deployment_manifest = self.k8s_manager.create_deployment_manifest(config)
            service_manifest = self.k8s_manager.create_service_manifest(config)
            
            with open(k8s_dir / f"{config.service_name}-deployment.yaml", 'w') as f:
                yaml.dump(deployment_manifest, f, default_flow_style=False)
            
            with open(k8s_dir / f"{config.service_name}-service.yaml", 'w') as f:
                yaml.dump(service_manifest, f, default_flow_style=False)
        
        # Generate deployment scripts
        scripts_dir = output_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Docker deployment script
        docker_script = """#!/bin/bash
set -e

echo "Building and deploying with Docker Compose..."
docker-compose build
docker-compose up -d

echo "Waiting for services to be healthy..."
sleep 30

echo "Checking service health..."
docker-compose ps

echo "Deployment complete!"
"""
        
        with open(scripts_dir / "deploy_docker.sh", 'w') as f:
            f.write(docker_script)
        
        # Kubernetes deployment script
        k8s_script = """#!/bin/bash
set -e

echo "Deploying to Kubernetes..."

# Apply all manifests
for manifest in kubernetes/*.yaml; do
    echo "Applying $manifest"
    kubectl apply -f "$manifest"
done

echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment --all

echo "Getting service status..."
kubectl get services
kubectl get pods

echo "Deployment complete!"
"""
        
        with open(scripts_dir / "deploy_k8s.sh", 'w') as f:
            f.write(k8s_script)
        
        # Make scripts executable
        for script in scripts_dir.glob("*.sh"):
            script.chmod(0o755)
        
        # Generate configuration files
        config_dir = output_path / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Create environment-specific configs
        for env in ["dev", "staging", "prod"]:
            env_config = {
                "environment": env,
                "log_level": "INFO" if env == "prod" else "DEBUG",
                "metrics_enabled": True,
                "security_enabled": env == "prod",
                "monitoring": {
                    "enabled": True,
                    "export_interval": 300,
                    "retention_days": 30 if env == "prod" else 7
                }
            }
            
            with open(config_dir / f"{env}.json", 'w') as f:
                json.dump(env_config, f, indent=2)
        
        # Generate README
        readme_content = """# Deployment Package

This package contains all necessary files for deploying the IoT Anomaly Detection system.

## Structure

- `Dockerfile.*`: Service-specific Docker images
- `docker-compose.yml`: Multi-service Docker Compose configuration
- `kubernetes/`: Kubernetes manifests for all services
- `scripts/`: Deployment scripts
- `config/`: Environment-specific configuration files

## Quick Start

### Docker Deployment
```bash
cd deployment_package
./scripts/deploy_docker.sh
```

### Kubernetes Deployment
```bash
cd deployment_package
./scripts/deploy_k8s.sh
```

## Services

The deployment includes the following services:
"""
        
        for config in configs:
            readme_content += f"- **{config.service_name}** ({config.service_type.value}): Port {config.port}\n"
        
        readme_content += """
## Health Checks

All services include health checks accessible at `/health` endpoint.

## Monitoring

Metrics are exported and can be accessed through the monitoring service.

## Configuration

Environment-specific configurations are in the `config/` directory.
Modify these files to customize deployment settings.
"""
        
        with open(output_path / "README.md", 'w') as f:
            f.write(readme_content)
        
        self.logger.info(f"Deployment package created: {output_path}")
        return str(output_path)
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get status of deployed service."""
        for result in self.deployment_history:
            if result.deployment_id == deployment_id:
                return {
                    "deployment_id": deployment_id,
                    "success": result.success,
                    "service_url": result.service_url,
                    "error_message": result.error_message,
                    "logs": result.logs[-10:]  # Last 10 log entries
                }
        
        return {"deployment_id": deployment_id, "status": "not_found"}


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Deployment Manager")
    parser.add_argument("--action", choices=["deploy", "package", "status"], 
                       required=True, help="Action to perform")
    parser.add_argument("--config", help="Deployment configuration file")
    parser.add_argument("--service", help="Service name")
    parser.add_argument("--target", choices=["local", "docker", "kubernetes"], 
                       default="docker", help="Deployment target")
    parser.add_argument("--environment", choices=["dev", "staging", "prod"], 
                       default="dev", help="Environment")
    parser.add_argument("--output", help="Output directory for package")
    
    args = parser.parse_args()
    
    manager = ProductionDeploymentManager()
    
    if args.action == "deploy":
        if not args.service:
            print("Service name required for deployment")
            exit(1)
        
        config = DeploymentConfig(
            target=DeploymentTarget(args.target),
            environment=args.environment,
            service_name=args.service,
            service_type=ServiceType.API,  # Default to API
            version="latest"
        )
        
        result = manager.deploy_service(config)
        
        if result.success:
            print(f"✅ Deployment successful!")
            print(f"Service URL: {result.service_url}")
            print(f"Deployment ID: {result.deployment_id}")
        else:
            print(f"❌ Deployment failed: {result.error_message}")
            
    elif args.action == "package":
        # Create sample configurations
        configs = [
            DeploymentConfig(
                target=DeploymentTarget.DOCKER,
                environment=args.environment,
                service_name="anomaly-api",
                service_type=ServiceType.API,
                port=8000
            ),
            DeploymentConfig(
                target=DeploymentTarget.DOCKER,
                environment=args.environment,
                service_name="anomaly-worker",
                service_type=ServiceType.WORKER,
                port=8001
            ),
            DeploymentConfig(
                target=DeploymentTarget.DOCKER,
                environment=args.environment,
                service_name="anomaly-monitoring",
                service_type=ServiceType.MONITORING,
                port=8002
            )
        ]
        
        output_dir = args.output or "deployment_package"
        package_path = manager.create_deployment_package(configs, output_dir)
        
        print(f"✅ Deployment package created: {package_path}")
        print(f"Run `cd {package_path} && ./scripts/deploy_docker.sh` to deploy")
        
    elif args.action == "status":
        if not args.service:
            print("Service deployment ID required for status")
            exit(1)
        
        status = manager.get_deployment_status(args.service)
        print(json.dumps(status, indent=2))
    
    print("Deployment management complete")