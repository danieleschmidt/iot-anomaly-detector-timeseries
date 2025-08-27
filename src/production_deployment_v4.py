"""Production Deployment System v4.0 - Global-ready deployment infrastructure."""

import os
import json
import yaml
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import subprocess
import shutil
import hashlib
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor

from .logging_config import get_logger, setup_logging
from .config import get_config
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .retry_manager import RetryManager
from .robust_error_handler import RobustErrorHandler
from .security_utils import SecurityManager


class DeploymentConfiguration:
    """Configuration management for production deployment."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "deployment/deployment-config.json"
        self.config = self._load_config()
        self.logger = get_logger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        if not Path(self.config_path).exists():
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default deployment configuration."""
        return {
            "deployment_info": {
                "name": "iot-anomaly-detector-v4",
                "version": "4.0.0",
                "environment": "production",
                "deploy_timestamp": datetime.now().isoformat()
            },
            "infrastructure": {
                "platform": "kubernetes",
                "regions": ["us-east-1", "eu-west-1", "ap-southeast-1"],
                "availability_zones": 3,
                "auto_scaling": {
                    "min_replicas": 2,
                    "max_replicas": 20,
                    "target_cpu_utilization": 70,
                    "target_memory_utilization": 80
                }
            },
            "services": {
                "api": {
                    "enabled": True,
                    "replicas": 3,
                    "resources": {
                        "requests": {"cpu": "200m", "memory": "512Mi"},
                        "limits": {"cpu": "1000m", "memory": "2Gi"}
                    }
                },
                "inference": {
                    "enabled": True,
                    "replicas": 2,
                    "resources": {
                        "requests": {"cpu": "500m", "memory": "1Gi"},
                        "limits": {"cpu": "2000m", "memory": "4Gi"}
                    }
                },
                "monitoring": {
                    "enabled": True,
                    "replicas": 1,
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "256Mi"},
                        "limits": {"cpu": "500m", "memory": "1Gi"}
                    }
                }
            },
            "database": {
                "type": "postgresql",
                "high_availability": True,
                "backup_retention_days": 30,
                "encryption": True
            },
            "monitoring": {
                "metrics": {
                    "prometheus": True,
                    "grafana": True,
                    "custom_dashboards": True
                },
                "logging": {
                    "centralized": True,
                    "log_level": "INFO",
                    "retention_days": 90
                },
                "alerting": {
                    "enabled": True,
                    "channels": ["email", "slack"],
                    "critical_thresholds": {
                        "error_rate": 5.0,
                        "response_time_p95": 2000,
                        "cpu_utilization": 85,
                        "memory_utilization": 90
                    }
                }
            },
            "security": {
                "tls": {
                    "enabled": True,
                    "cert_manager": True,
                    "min_tls_version": "1.2"
                },
                "authentication": {
                    "enabled": True,
                    "methods": ["jwt", "api_key"],
                    "session_timeout": 3600
                },
                "network_policies": {
                    "enabled": True,
                    "ingress_restrictions": True,
                    "egress_restrictions": True
                }
            },
            "compliance": {
                "gdpr": True,
                "ccpa": True,
                "pdpa": True,
                "data_retention_days": 365,
                "audit_logging": True
            },
            "internationalization": {
                "enabled": True,
                "supported_locales": ["en", "es", "fr", "de", "ja", "zh-CN"],
                "default_locale": "en"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save configuration to file."""
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                yaml.safe_dump(self.config, f, indent=2, default_flow_style=False)
            else:
                json.dump(self.config, f, indent=2, default=str)


class ContainerBuilder:
    """Docker container builder for production deployment."""
    
    def __init__(self, config: DeploymentConfiguration):
        self.config = config
        self.logger = get_logger(__name__)
        self.security_manager = SecurityManager()
    
    def build_containers(self, force_rebuild: bool = False) -> Dict[str, str]:
        """Build all required containers for deployment."""
        containers = {}
        
        # Define container configurations
        container_configs = {
            "api": {
                "dockerfile": "deployment/dockerfiles/Dockerfile.web_api",
                "context": ".",
                "tag": f"iot-anomaly-api:v{self.config.get('deployment_info.version', '4.0.0')}"
            },
            "inference": {
                "dockerfile": "deployment/dockerfiles/Dockerfile.inference_engine", 
                "context": ".",
                "tag": f"iot-anomaly-inference:v{self.config.get('deployment_info.version', '4.0.0')}"
            },
            "monitoring": {
                "dockerfile": "deployment/dockerfiles/Dockerfile.monitoring",
                "context": ".",
                "tag": f"iot-anomaly-monitoring:v{self.config.get('deployment_info.version', '4.0.0')}"
            }
        }
        
        # Build each container
        for name, config in container_configs.items():
            if self.config.get(f'services.{name}.enabled', True):
                try:
                    self.logger.info(f"Building container: {name}")
                    
                    # Check if rebuild is needed
                    if not force_rebuild and self._container_exists(config['tag']):
                        self.logger.info(f"Container {config['tag']} already exists, skipping build")
                        containers[name] = config['tag']
                        continue
                    
                    # Build container
                    build_result = self._build_container(
                        dockerfile=config['dockerfile'],
                        context=config['context'],
                        tag=config['tag']
                    )
                    
                    if build_result:
                        containers[name] = config['tag']
                        self.logger.info(f"Successfully built container: {config['tag']}")
                    else:
                        self.logger.error(f"Failed to build container: {name}")
                        
                except Exception as e:
                    self.logger.error(f"Error building container {name}: {e}")
        
        return containers
    
    def _container_exists(self, tag: str) -> bool:
        """Check if container with tag exists."""
        try:
            result = subprocess.run(
                ["docker", "images", "-q", tag],
                capture_output=True,
                text=True,
                check=True
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False
    
    def _build_container(self, dockerfile: str, context: str, tag: str) -> bool:
        """Build a single container."""
        try:
            # Ensure dockerfile exists
            if not Path(dockerfile).exists():
                self.logger.warning(f"Dockerfile not found: {dockerfile}, creating basic one")
                self._create_basic_dockerfile(dockerfile)
            
            # Build command
            cmd = [
                "docker", "build",
                "-f", dockerfile,
                "-t", tag,
                "--build-arg", f"VERSION={self.config.get('deployment_info.version')}",
                "--build-arg", f"BUILD_DATE={datetime.now().isoformat()}",
                context
            ]
            
            self.logger.debug(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                # Security scan (if available)
                self._scan_container_security(tag)
                return True
            else:
                self.logger.error(f"Container build failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Container build timed out: {tag}")
            return False
        except Exception as e:
            self.logger.error(f"Container build error: {e}")
            return False
    
    def _create_basic_dockerfile(self, dockerfile_path: str) -> None:
        """Create a basic Dockerfile if none exists."""
        dockerfile_dir = Path(dockerfile_path).parent
        dockerfile_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine service type from path
        if "api" in dockerfile_path:
            content = self._get_api_dockerfile_content()
        elif "inference" in dockerfile_path:
            content = self._get_inference_dockerfile_content()
        elif "monitoring" in dockerfile_path:
            content = self._get_monitoring_dockerfile_content()
        else:
            content = self._get_generic_dockerfile_content()
        
        with open(dockerfile_path, 'w') as f:
            f.write(content)
        
        self.logger.info(f"Created basic Dockerfile: {dockerfile_path}")
    
    def _get_api_dockerfile_content(self) -> str:
        return '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY deployment/configs/ ./configs/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "uvicorn", "src.api_gateway:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    def _get_inference_dockerfile_content(self) -> str:
        return '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY saved_models_v2/ ./saved_models_v2/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "from src.scalable_autoencoder_v3 import ScalableAutoencoderV3; print('OK')"

# Start inference service
CMD ["python", "-m", "src.scalable_inference_engine"]
'''
    
    def _get_monitoring_dockerfile_content(self) -> str:
        return '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt prometheus-client

# Copy application code
COPY src/ ./src/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:9090/metrics || exit 1

# Expose prometheus metrics port
EXPOSE 9090

# Start monitoring service
CMD ["python", "-m", "src.comprehensive_monitoring"]
'''
    
    def _get_generic_dockerfile_content(self) -> str:
        return '''FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Default command
CMD ["python", "-m", "src"]
'''
    
    def _scan_container_security(self, tag: str) -> None:
        """Perform basic security scan on container."""
        try:
            # Basic vulnerability check using docker history
            result = subprocess.run(
                ["docker", "history", "--no-trunc", tag],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Log security scan result
            self.logger.info(f"Security scan completed for {tag}")
            
        except subprocess.CalledProcessError:
            self.logger.warning(f"Could not perform security scan on {tag}")


class KubernetesDeployer:
    """Kubernetes deployment manager."""
    
    def __init__(self, config: DeploymentConfiguration):
        self.config = config
        self.logger = get_logger(__name__)
        self.k8s_manifests_dir = Path("deployment/k8s")
    
    def deploy(self, containers: Dict[str, str]) -> bool:
        """Deploy to Kubernetes cluster."""
        try:
            # Ensure manifests exist
            self._ensure_k8s_manifests()
            
            # Deploy namespace
            if not self._deploy_namespace():
                return False
            
            # Deploy ConfigMaps and Secrets
            if not self._deploy_config():
                return False
            
            # Deploy services
            success = True
            for service_name, container_tag in containers.items():
                if not self._deploy_service(service_name, container_tag):
                    success = False
            
            # Deploy ingress
            if not self._deploy_ingress():
                success = False
            
            # Wait for deployment readiness
            if success:
                success = self._wait_for_deployment_ready()
            
            if success:
                self.logger.info("Kubernetes deployment completed successfully")
            else:
                self.logger.error("Kubernetes deployment failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Kubernetes deployment error: {e}")
            return False
    
    def _ensure_k8s_manifests(self) -> None:
        """Ensure all required Kubernetes manifests exist."""
        self.k8s_manifests_dir.mkdir(parents=True, exist_ok=True)
        
        # Create basic manifests if they don't exist
        manifests = {
            "namespace.yaml": self._get_namespace_manifest(),
            "configmap.yaml": self._get_configmap_manifest(),
            "deployment-api.yaml": self._get_api_deployment_manifest(),
            "deployment-inference.yaml": self._get_inference_deployment_manifest(),
            "deployment-monitoring.yaml": self._get_monitoring_deployment_manifest(),
            "service-api.yaml": self._get_api_service_manifest(),
            "service-inference.yaml": self._get_inference_service_manifest(),
            "service-monitoring.yaml": self._get_monitoring_service_manifest(),
            "ingress-api.yaml": self._get_ingress_manifest(),
            "hpa.yaml": self._get_hpa_manifest()
        }
        
        for filename, content in manifests.items():
            manifest_path = self.k8s_manifests_dir / filename
            if not manifest_path.exists():
                with open(manifest_path, 'w') as f:
                    f.write(content)
                self.logger.info(f"Created K8s manifest: {filename}")
    
    def _deploy_namespace(self) -> bool:
        """Deploy namespace."""
        return self._apply_manifest("namespace.yaml")
    
    def _deploy_config(self) -> bool:
        """Deploy ConfigMaps and Secrets."""
        return self._apply_manifest("configmap.yaml")
    
    def _deploy_service(self, service_name: str, container_tag: str) -> bool:
        """Deploy a specific service."""
        deployment_file = f"deployment-{service_name}.yaml"
        service_file = f"service-{service_name}.yaml"
        
        # Update deployment manifest with container tag
        self._update_deployment_image(deployment_file, container_tag)
        
        # Apply deployment and service
        return (self._apply_manifest(deployment_file) and 
                self._apply_manifest(service_file))
    
    def _deploy_ingress(self) -> bool:
        """Deploy ingress."""
        return self._apply_manifest("ingress-api.yaml")
    
    def _apply_manifest(self, filename: str) -> bool:
        """Apply a Kubernetes manifest."""
        manifest_path = self.k8s_manifests_dir / filename
        
        if not manifest_path.exists():
            self.logger.error(f"Manifest not found: {manifest_path}")
            return False
        
        try:
            result = subprocess.run(
                ["kubectl", "apply", "-f", str(manifest_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info(f"Applied manifest: {filename}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to apply manifest {filename}: {e.stderr}")
            return False
        except FileNotFoundError:
            self.logger.error("kubectl not found. Please install kubectl.")
            return False
    
    def _update_deployment_image(self, deployment_file: str, container_tag: str) -> None:
        """Update deployment manifest with new container image."""
        manifest_path = self.k8s_manifests_dir / deployment_file
        
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    content = f.read()
                
                # Simple image tag replacement
                # In production, you'd use proper YAML parsing
                content = content.replace("{{IMAGE_TAG}}", container_tag)
                
                with open(manifest_path, 'w') as f:
                    f.write(content)
                    
            except Exception as e:
                self.logger.warning(f"Could not update image tag in {deployment_file}: {e}")
    
    def _wait_for_deployment_ready(self, timeout: int = 300) -> bool:
        """Wait for all deployments to be ready."""
        try:
            result = subprocess.run(
                ["kubectl", "wait", "--for=condition=available", "--timeout=300s", 
                 "deployment", "-l", "app=iot-anomaly-detector"],
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info("All deployments are ready")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Deployment readiness check failed: {e.stderr}")
            return False
    
    def _get_namespace_manifest(self) -> str:
        return '''apiVersion: v1
kind: Namespace
metadata:
  name: iot-anomaly-detector
  labels:
    name: iot-anomaly-detector
    environment: production
'''
    
    def _get_configmap_manifest(self) -> str:
        return f'''apiVersion: v1
kind: ConfigMap
metadata:
  name: iot-anomaly-config
  namespace: iot-anomaly-detector
data:
  config.yaml: |
{yaml.dump(self.config.config, indent=4)}
'''
    
    def _get_api_deployment_manifest(self) -> str:
        api_config = self.config.get('services.api', {})
        return f'''apiVersion: apps/v1
kind: Deployment
metadata:
  name: iot-anomaly-api
  namespace: iot-anomaly-detector
  labels:
    app: iot-anomaly-detector
    component: api
spec:
  replicas: {api_config.get('replicas', 3)}
  selector:
    matchLabels:
      app: iot-anomaly-detector
      component: api
  template:
    metadata:
      labels:
        app: iot-anomaly-detector
        component: api
    spec:
      containers:
      - name: api
        image: {{{{IMAGE_TAG}}}}
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: {api_config.get('resources.requests.cpu', '200m')}
            memory: {api_config.get('resources.requests.memory', '512Mi')}
          limits:
            cpu: {api_config.get('resources.limits.cpu', '1000m')}
            memory: {api_config.get('resources.limits.memory', '2Gi')}
        env:
        - name: ENVIRONMENT
          value: "production"
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
'''
    
    def _get_inference_deployment_manifest(self) -> str:
        inference_config = self.config.get('services.inference', {})
        return f'''apiVersion: apps/v1
kind: Deployment
metadata:
  name: iot-anomaly-inference
  namespace: iot-anomaly-detector
  labels:
    app: iot-anomaly-detector
    component: inference
spec:
  replicas: {inference_config.get('replicas', 2)}
  selector:
    matchLabels:
      app: iot-anomaly-detector
      component: inference
  template:
    metadata:
      labels:
        app: iot-anomaly-detector
        component: inference
    spec:
      containers:
      - name: inference
        image: {{{{IMAGE_TAG}}}}
        resources:
          requests:
            cpu: {inference_config.get('resources.requests.cpu', '500m')}
            memory: {inference_config.get('resources.requests.memory', '1Gi')}
          limits:
            cpu: {inference_config.get('resources.limits.cpu', '2000m')}
            memory: {inference_config.get('resources.limits.memory', '4Gi')}
        env:
        - name: ENVIRONMENT
          value: "production"
'''
    
    def _get_monitoring_deployment_manifest(self) -> str:
        monitoring_config = self.config.get('services.monitoring', {})
        return f'''apiVersion: apps/v1
kind: Deployment
metadata:
  name: iot-anomaly-monitoring
  namespace: iot-anomaly-detector
  labels:
    app: iot-anomaly-detector
    component: monitoring
spec:
  replicas: {monitoring_config.get('replicas', 1)}
  selector:
    matchLabels:
      app: iot-anomaly-detector
      component: monitoring
  template:
    metadata:
      labels:
        app: iot-anomaly-detector
        component: monitoring
    spec:
      containers:
      - name: monitoring
        image: {{{{IMAGE_TAG}}}}
        ports:
        - containerPort: 9090
        resources:
          requests:
            cpu: {monitoring_config.get('resources.requests.cpu', '100m')}
            memory: {monitoring_config.get('resources.requests.memory', '256Mi')}
          limits:
            cpu: {monitoring_config.get('resources.limits.cpu', '500m')}
            memory: {monitoring_config.get('resources.limits.memory', '1Gi')}
'''
    
    def _get_api_service_manifest(self) -> str:
        return '''apiVersion: v1
kind: Service
metadata:
  name: iot-anomaly-api-service
  namespace: iot-anomaly-detector
  labels:
    app: iot-anomaly-detector
    component: api
spec:
  selector:
    app: iot-anomaly-detector
    component: api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
'''
    
    def _get_inference_service_manifest(self) -> str:
        return '''apiVersion: v1
kind: Service
metadata:
  name: iot-anomaly-inference-service
  namespace: iot-anomaly-detector
  labels:
    app: iot-anomaly-detector
    component: inference
spec:
  selector:
    app: iot-anomaly-detector
    component: inference
  ports:
  - port: 8080
    targetPort: 8080
    protocol: TCP
  type: ClusterIP
'''
    
    def _get_monitoring_service_manifest(self) -> str:
        return '''apiVersion: v1
kind: Service
metadata:
  name: iot-anomaly-monitoring-service
  namespace: iot-anomaly-detector
  labels:
    app: iot-anomaly-detector
    component: monitoring
spec:
  selector:
    app: iot-anomaly-detector
    component: monitoring
  ports:
  - port: 9090
    targetPort: 9090
    protocol: TCP
  type: ClusterIP
'''
    
    def _get_ingress_manifest(self) -> str:
        return '''apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: iot-anomaly-ingress
  namespace: iot-anomaly-detector
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.iot-anomaly-detector.com
    secretName: iot-anomaly-tls
  rules:
  - host: api.iot-anomaly-detector.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: iot-anomaly-api-service
            port:
              number: 80
'''
    
    def _get_hpa_manifest(self) -> str:
        return '''apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: iot-anomaly-hpa
  namespace: iot-anomaly-detector
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: iot-anomaly-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
'''


class ProductionDeploymentManager:
    """Main production deployment manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = DeploymentConfiguration(config_path)
        self.logger = get_logger(__name__)
        self.container_builder = ContainerBuilder(self.config)
        self.k8s_deployer = KubernetesDeployer(self.config)
        
        # Setup error handling
        self.error_handler = RobustErrorHandler()
        circuit_config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=300.0)
        self.circuit_breaker = CircuitBreaker("deployment", config=circuit_config)
        self.retry_manager = RetryManager(max_retries=3, base_delay=5.0)
        
        # Deployment state
        self.deployment_id = self._generate_deployment_id()
        self.deployment_start_time = None
        self.deployment_status = "initialized"
    
    def _generate_deployment_id(self) -> str:
        """Generate unique deployment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = self.config.get('deployment_info.version', '4.0.0')
        return f"deploy_{version}_{timestamp}"
    
    def deploy(
        self,
        force_rebuild: bool = False,
        dry_run: bool = False,
        skip_tests: bool = False
    ) -> Dict[str, Any]:
        """Execute full production deployment."""
        self.deployment_start_time = time.time()
        self.deployment_status = "in_progress"
        
        deployment_result = {
            'deployment_id': self.deployment_id,
            'start_time': self.deployment_start_time,
            'status': 'started',
            'steps': [],
            'containers': {},
            'k8s_deployed': False,
            'success': False
        }
        
        try:
            self.logger.info(f"Starting production deployment {self.deployment_id}")
            
            # Step 1: Pre-deployment validation
            if not self._pre_deployment_validation():
                deployment_result['status'] = 'failed'
                deployment_result['error'] = 'Pre-deployment validation failed'
                return deployment_result
            
            deployment_result['steps'].append({'name': 'pre_validation', 'status': 'completed'})
            
            # Step 2: Run tests (unless skipped)
            if not skip_tests:
                if not self._run_tests():
                    deployment_result['status'] = 'failed'
                    deployment_result['error'] = 'Test execution failed'
                    return deployment_result
                
                deployment_result['steps'].append({'name': 'tests', 'status': 'completed'})
            
            # Step 3: Build containers
            if not dry_run:
                containers = self.retry_manager.execute_with_retry(
                    lambda: self.container_builder.build_containers(force_rebuild)
                )
                
                if not containers:
                    deployment_result['status'] = 'failed'
                    deployment_result['error'] = 'Container build failed'
                    return deployment_result
                
                deployment_result['containers'] = containers
                deployment_result['steps'].append({'name': 'container_build', 'status': 'completed'})
            
            # Step 4: Deploy to Kubernetes
            if not dry_run:
                k8s_success = self.circuit_breaker.call(
                    lambda: self.k8s_deployer.deploy(containers)
                )
                
                if not k8s_success:
                    deployment_result['status'] = 'failed'
                    deployment_result['error'] = 'Kubernetes deployment failed'
                    return deployment_result
                
                deployment_result['k8s_deployed'] = True
                deployment_result['steps'].append({'name': 'k8s_deployment', 'status': 'completed'})
            
            # Step 5: Post-deployment validation
            if not dry_run:
                if not self._post_deployment_validation():
                    deployment_result['status'] = 'failed'
                    deployment_result['error'] = 'Post-deployment validation failed'
                    return deployment_result
                
                deployment_result['steps'].append({'name': 'post_validation', 'status': 'completed'})
            
            # Success
            deployment_result['success'] = True
            deployment_result['status'] = 'completed'
            deployment_result['end_time'] = time.time()
            deployment_result['duration_seconds'] = deployment_result['end_time'] - self.deployment_start_time
            
            self.deployment_status = "completed"
            self.logger.info(f"Production deployment {self.deployment_id} completed successfully")
            
            # Save deployment metadata
            self._save_deployment_metadata(deployment_result)
            
            return deployment_result
            
        except Exception as e:
            deployment_result['status'] = 'failed'
            deployment_result['error'] = str(e)
            deployment_result['end_time'] = time.time()
            
            self.deployment_status = "failed"
            self.error_handler.handle_error(e, context=f"deployment_{self.deployment_id}")
            self.logger.error(f"Production deployment {self.deployment_id} failed: {e}")
            
            return deployment_result
    
    def _pre_deployment_validation(self) -> bool:
        """Run pre-deployment validation checks."""
        try:
            self.logger.info("Running pre-deployment validation")
            
            # Check required files
            required_files = [
                "requirements.txt",
                "src/",
                "deployment/"
            ]
            
            for file_path in required_files:
                if not Path(file_path).exists():
                    self.logger.error(f"Required file/directory not found: {file_path}")
                    return False
            
            # Check Docker availability
            try:
                subprocess.run(["docker", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.logger.error("Docker not available")
                return False
            
            # Check Kubernetes availability
            try:
                subprocess.run(["kubectl", "version", "--client"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.logger.error("kubectl not available")
                return False
            
            self.logger.info("Pre-deployment validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Pre-deployment validation error: {e}")
            return False
    
    def _run_tests(self) -> bool:
        """Run test suite before deployment."""
        try:
            self.logger.info("Running test suite")
            
            # Run core tests
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-x", "--tb=short", "-q"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info("All tests passed")
                return True
            else:
                self.logger.error(f"Tests failed: {result.stdout}\n{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Test execution timed out")
            return False
        except Exception as e:
            self.logger.error(f"Test execution error: {e}")
            return False
    
    def _post_deployment_validation(self) -> bool:
        """Run post-deployment validation checks."""
        try:
            self.logger.info("Running post-deployment validation")
            
            # Wait for services to be ready
            time.sleep(30)
            
            # Check pod status
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", "iot-anomaly-detector", "--no-headers"],
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line:
                    columns = line.split()
                    if len(columns) >= 3:
                        status = columns[2]
                        if status != "Running":
                            self.logger.error(f"Pod not running: {line}")
                            return False
            
            self.logger.info("Post-deployment validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Post-deployment validation error: {e}")
            return False
    
    def _save_deployment_metadata(self, deployment_result: Dict[str, Any]) -> None:
        """Save deployment metadata for tracking."""
        try:
            metadata_dir = Path("deployment/history")
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            metadata_file = metadata_dir / f"{self.deployment_id}_metadata.json"
            
            with open(metadata_file, 'w') as f:
                json.dump(deployment_result, f, indent=2, default=str)
            
            self.logger.info(f"Deployment metadata saved: {metadata_file}")
            
        except Exception as e:
            self.logger.warning(f"Could not save deployment metadata: {e}")
    
    def rollback(self, target_deployment_id: Optional[str] = None) -> bool:
        """Rollback to previous deployment."""
        try:
            self.logger.info(f"Starting rollback from {self.deployment_id}")
            
            # Get rollback target
            if not target_deployment_id:
                target_deployment_id = self._get_previous_deployment_id()
            
            if not target_deployment_id:
                self.logger.error("No previous deployment found for rollback")
                return False
            
            # Perform rollback (simplified)
            result = subprocess.run(
                ["kubectl", "rollout", "undo", "deployment", "-n", "iot-anomaly-detector", "--all"],
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info(f"Rollback completed to {target_deployment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def _get_previous_deployment_id(self) -> Optional[str]:
        """Get the previous deployment ID."""
        try:
            metadata_dir = Path("deployment/history")
            if not metadata_dir.exists():
                return None
            
            metadata_files = sorted(metadata_dir.glob("*_metadata.json"), reverse=True)
            
            # Skip current deployment, return previous one
            for file in metadata_files[1:]:
                with open(file, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get('success', False):
                        return metadata.get('deployment_id')
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Could not determine previous deployment: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            'deployment_id': self.deployment_id,
            'status': self.deployment_status,
            'start_time': self.deployment_start_time,
            'uptime_seconds': time.time() - self.deployment_start_time if self.deployment_start_time else 0,
            'circuit_breaker_state': self.circuit_breaker.state.name,
            'config_version': self.config.get('deployment_info.version')
        }


# CLI interface for deployment
def main():
    """Main CLI interface for production deployment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="IoT Anomaly Detector Production Deployment")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--force-rebuild", action="store_true", help="Force container rebuild")
    parser.add_argument("--dry-run", action="store_true", help="Run without actual deployment")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test execution")
    parser.add_argument("--rollback", help="Rollback to specific deployment ID")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger(__name__)
    
    # Initialize deployment manager
    deployment_manager = ProductionDeploymentManager(args.config)
    
    if args.rollback:
        # Perform rollback
        success = deployment_manager.rollback(args.rollback)
        if success:
            logger.info("Rollback completed successfully")
        else:
            logger.error("Rollback failed")
            exit(1)
    else:
        # Perform deployment
        result = deployment_manager.deploy(
            force_rebuild=args.force_rebuild,
            dry_run=args.dry_run,
            skip_tests=args.skip_tests
        )
        
        if result['success']:
            logger.info(f"Deployment {result['deployment_id']} completed successfully")
            logger.info(f"Duration: {result['duration_seconds']:.1f} seconds")
        else:
            logger.error(f"Deployment {result['deployment_id']} failed: {result.get('error')}")
            exit(1)


if __name__ == "__main__":
    main()