"""
Robust Deployment Framework for Generation 2 System
Enterprise-grade reliability, fault tolerance, and operational excellence
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import psutil
from pydantic import BaseModel, validator

from .generation_1_autonomous_core import AutonomousAnomalyCore
from .real_time_inference_engine import RealTimeInferenceEngine
from .adaptive_learning_system import AdaptiveLearningSystem
from .logging_config import setup_logging


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ServiceState(Enum):
    """Service lifecycle states."""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    RECOVERING = "recovering"


@dataclass
class SystemMetrics:
    """Comprehensive system metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_connections: int = 0
    inference_latency_ms: float = 0.0
    throughput_per_sec: float = 0.0
    error_rate: float = 0.0
    queue_depth: int = 0
    active_models: int = 0
    uptime_seconds: float = 0.0
    health_score: float = 1.0


@dataclass
class AlertConfig:
    """Configuration for system alerts."""
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    error_rate_threshold: float = 0.05
    latency_threshold_ms: float = 1000.0
    queue_depth_threshold: int = 1000
    health_score_threshold: float = 0.7
    alert_cooldown_seconds: float = 300.0


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RobustCircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 3,
        adaptive: bool = True
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.adaptive = adaptive
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state_change_time = time.time()
        
        # Adaptive parameters
        self.historical_success_rate = 0.95
        self.adaptation_factor = 0.1
        
        self.logger = setup_logging(__name__)
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time < self.timeout:
                raise CircuitBreakerOpenError("Circuit breaker is open")
            else:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.logger.info("Circuit breaker transitioning to half-open")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._record_success()
            return result
            
        except Exception as e:
            await self._record_failure()
            raise
    
    async def _record_success(self) -> None:
        """Record successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.logger.info("Circuit breaker closed after successful recovery")
        
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)  # Gradually reduce failure count
        
        # Adaptive threshold adjustment
        if self.adaptive:
            self.historical_success_rate = 0.99 * self.historical_success_rate + 0.01
            self._adjust_thresholds()
    
    async def _record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.state_change_time = time.time()
                self.logger.error(f"Circuit breaker opened after {self.failure_count} failures")
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.state_change_time = time.time()
            self.logger.warning("Circuit breaker reopened during half-open state")
        
        # Adaptive threshold adjustment
        if self.adaptive:
            self.historical_success_rate = 0.99 * self.historical_success_rate
            self._adjust_thresholds()
    
    def _adjust_thresholds(self) -> None:
        """Adjust thresholds based on historical performance."""
        if self.historical_success_rate < 0.9:
            # Lower thresholds when system is struggling
            self.failure_threshold = max(3, int(self.failure_threshold * 0.9))
            self.timeout = min(300, self.timeout * 1.1)
        elif self.historical_success_rate > 0.98:
            # Raise thresholds when system is performing well
            self.failure_threshold = min(10, int(self.failure_threshold * 1.1))
            self.timeout = max(30, self.timeout * 0.9)


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class GracefulShutdownHandler:
    """Handle graceful shutdown of system components."""
    
    def __init__(self):
        self.shutdown_callbacks: List[Callable] = []
        self.is_shutting_down = False
        self.shutdown_timeout = 30.0
        self.logger = setup_logging(__name__)
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def register_shutdown_callback(self, callback: Callable) -> None:
        """Register callback for graceful shutdown."""
        self.shutdown_callbacks.append(callback)
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        asyncio.create_task(self.graceful_shutdown())
    
    async def graceful_shutdown(self) -> None:
        """Execute graceful shutdown sequence."""
        if self.is_shutting_down:
            return
        
        self.is_shutting_down = True
        self.logger.info("Starting graceful shutdown")
        
        try:
            # Execute shutdown callbacks with timeout
            shutdown_tasks = []
            for callback in self.shutdown_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    task = asyncio.create_task(callback())
                else:
                    task = asyncio.create_task(asyncio.to_thread(callback))
                shutdown_tasks.append(task)
            
            # Wait for all shutdown tasks with timeout
            await asyncio.wait_for(
                asyncio.gather(*shutdown_tasks, return_exceptions=True),
                timeout=self.shutdown_timeout
            )
            
            self.logger.info("Graceful shutdown completed successfully")
            
        except asyncio.TimeoutError:
            self.logger.warning("Shutdown timeout exceeded, forcing exit")
        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {str(e)}")
        
        finally:
            # Force exit if still running
            sys.exit(0)


class HealthMonitor:
    """Comprehensive health monitoring and alerting."""
    
    def __init__(
        self,
        alert_config: AlertConfig,
        check_interval: float = 30.0,
        metrics_retention: int = 1440  # 24 hours at 1-minute intervals
    ):
        self.alert_config = alert_config
        self.check_interval = check_interval
        self.metrics_retention = metrics_retention
        
        self.metrics_history: List[SystemMetrics] = []
        self.alert_history: Dict[str, float] = {}  # Last alert time by type
        self.health_status = HealthStatus.UNKNOWN
        
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        self.logger = setup_logging(__name__)
    
    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Health monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main health monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = await self._collect_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.metrics_retention:
                    self.metrics_history.pop(0)
                
                # Assess health status
                self.health_status = self._assess_health(metrics)
                
                # Check for alerts
                await self._check_alerts(metrics)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {str(e)}")
                await asyncio.sleep(self.check_interval)
    
    async def _collect_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network connections
            try:
                connections = len(psutil.net_connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                connections = 0
            
            # Calculate health score
            health_score = self._calculate_health_score(
                cpu_percent, memory.percent, disk.percent
            )
            
            return SystemMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_connections=connections,
                health_score=health_score,
                uptime_seconds=time.time() - psutil.boot_time()
            )
            
        except Exception as e:
            self.logger.error(f"Metrics collection error: {str(e)}")
            return SystemMetrics()
    
    def _calculate_health_score(
        self, 
        cpu_percent: float, 
        memory_percent: float, 
        disk_percent: float
    ) -> float:
        """Calculate overall health score (0-1)."""
        # Weighted health score
        cpu_score = max(0, 1 - (cpu_percent / 100) ** 2)
        memory_score = max(0, 1 - (memory_percent / 100) ** 2)
        disk_score = max(0, 1 - (disk_percent / 100) ** 2)
        
        # Weighted average
        health_score = (0.4 * cpu_score + 0.4 * memory_score + 0.2 * disk_score)
        return max(0, min(1, health_score))
    
    def _assess_health(self, metrics: SystemMetrics) -> HealthStatus:
        """Assess overall system health status."""
        if metrics.health_score >= 0.9:
            return HealthStatus.HEALTHY
        elif metrics.health_score >= 0.7:
            return HealthStatus.DEGRADED
        elif metrics.health_score >= 0.4:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL
    
    async def _check_alerts(self, metrics: SystemMetrics) -> None:
        """Check for alert conditions."""
        current_time = time.time()
        alerts = []
        
        # CPU alert
        if metrics.cpu_usage_percent > self.alert_config.cpu_threshold:
            if self._should_send_alert("cpu_high", current_time):
                alerts.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        
        # Memory alert
        if metrics.memory_usage_percent > self.alert_config.memory_threshold:
            if self._should_send_alert("memory_high", current_time):
                alerts.append(f"High memory usage: {metrics.memory_usage_percent:.1f}%")
        
        # Health score alert
        if metrics.health_score < self.alert_config.health_score_threshold:
            if self._should_send_alert("health_low", current_time):
                alerts.append(f"Low health score: {metrics.health_score:.2f}")
        
        # Send alerts
        for alert_message in alerts:
            await self._send_alert(alert_message)
    
    def _should_send_alert(self, alert_type: str, current_time: float) -> bool:
        """Check if alert should be sent based on cooldown."""
        last_alert_time = self.alert_history.get(alert_type, 0)
        if current_time - last_alert_time >= self.alert_config.alert_cooldown_seconds:
            self.alert_history[alert_type] = current_time
            return True
        return False
    
    async def _send_alert(self, message: str) -> None:
        """Send alert notification."""
        self.logger.warning(f"ALERT: {message}")
        # In production, this would integrate with alerting systems
        # like PagerDuty, Slack, email, etc.
    
    def get_current_health(self) -> Dict[str, Any]:
        """Get current health status and metrics."""
        if not self.metrics_history:
            return {"status": "unknown", "metrics": None}
        
        latest_metrics = self.metrics_history[-1]
        
        return {
            "status": self.health_status.value,
            "health_score": latest_metrics.health_score,
            "metrics": {
                "cpu_usage_percent": latest_metrics.cpu_usage_percent,
                "memory_usage_percent": latest_metrics.memory_usage_percent,
                "disk_usage_percent": latest_metrics.disk_usage_percent,
                "uptime_seconds": latest_metrics.uptime_seconds,
                "timestamp": latest_metrics.timestamp
            },
            "trend": self._calculate_health_trend()
        }
    
    def _calculate_health_trend(self) -> str:
        """Calculate health trend over recent history."""
        if len(self.metrics_history) < 5:
            return "unknown"
        
        recent_scores = [m.health_score for m in self.metrics_history[-5:]]
        
        if recent_scores[-1] > recent_scores[0] + 0.1:
            return "improving"
        elif recent_scores[-1] < recent_scores[0] - 0.1:
            return "declining"
        else:
            return "stable"


class RobustDeploymentFramework:
    """
    Comprehensive deployment framework with enterprise-grade reliability.
    
    Features:
    - Circuit breaker protection
    - Graceful shutdown handling
    - Health monitoring and alerting
    - Service lifecycle management
    - Error recovery and retry logic
    - Performance monitoring
    - Resource management
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        alert_config: Optional[AlertConfig] = None
    ):
        # Load configuration
        self.config = self._load_config(config_path)
        self.alert_config = alert_config or AlertConfig()
        
        # Core components
        self.anomaly_core: Optional[AutonomousAnomalyCore] = None
        self.inference_engine: Optional[RealTimeInferenceEngine] = None
        self.learning_system: Optional[AdaptiveLearningSystem] = None
        
        # Reliability components
        self.circuit_breaker = RobustCircuitBreaker()
        self.shutdown_handler = GracefulShutdownHandler()
        self.health_monitor = HealthMonitor(self.alert_config)
        
        # State management
        self.service_state = ServiceState.STOPPED
        self.start_time = 0.0
        self.error_count = 0
        
        self.logger = setup_logging(__name__)
        
        # Register shutdown callbacks
        self.shutdown_handler.register_shutdown_callback(self.graceful_shutdown)
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load system configuration."""
        default_config = {
            "model": {
                "window_size": 30,
                "latent_dim": 16,
                "ensemble_size": 3
            },
            "inference": {
                "batch_size": 32,
                "max_queue_size": 10000,
                "processing_timeout": 5.0
            },
            "learning": {
                "feedback_buffer_size": 10000,
                "adaptation_threshold": 0.1,
                "min_samples_for_adaptation": 100
            },
            "deployment": {
                "max_startup_time": 300.0,
                "health_check_interval": 30.0,
                "auto_recovery": True
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    user_config = json.load(f)
                    # Merge configurations
                    self._deep_update(default_config, user_config)
            except Exception as e:
                logging.warning(f"Failed to load config from {config_path}: {str(e)}")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    async def initialize_system(self) -> bool:
        """Initialize all system components."""
        self.service_state = ServiceState.INITIALIZING
        self.logger.info("Initializing robust deployment framework")
        
        try:
            # Initialize core anomaly detection system
            self.anomaly_core = AutonomousAnomalyCore(
                window_size=self.config["model"]["window_size"],
                latent_dim=self.config["model"]["latent_dim"],
                ensemble_size=self.config["model"]["ensemble_size"]
            )
            
            # Initialize real-time inference engine
            self.inference_engine = RealTimeInferenceEngine(
                core_model=self.anomaly_core,
                batch_size=self.config["inference"]["batch_size"],
                max_queue_size=self.config["inference"]["max_queue_size"],
                processing_timeout=self.config["inference"]["processing_timeout"]
            )
            
            # Initialize adaptive learning system
            self.learning_system = AdaptiveLearningSystem(
                core_model=self.anomaly_core,
                feedback_buffer_size=self.config["learning"]["feedback_buffer_size"],
                adaptation_threshold=self.config["learning"]["adaptation_threshold"],
                min_samples_for_adaptation=self.config["learning"]["min_samples_for_adaptation"]
            )
            
            self.logger.info("System components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {str(e)}")
            self.service_state = ServiceState.ERROR
            return False
    
    async def start_system(self) -> bool:
        """Start all system services."""
        if self.service_state != ServiceState.INITIALIZING and self.service_state != ServiceState.STOPPED:
            self.logger.warning(f"Cannot start system from state: {self.service_state}")
            return False
        
        self.service_state = ServiceState.STARTING
        self.start_time = time.time()
        self.logger.info("Starting system services")
        
        try:
            # Start health monitoring first
            await self.health_monitor.start_monitoring()
            
            # Start inference engine
            await self.inference_engine.start()
            
            # System is now running
            self.service_state = ServiceState.RUNNING
            self.logger.info("All system services started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {str(e)}")
            self.service_state = ServiceState.ERROR
            await self._handle_startup_failure()
            return False
    
    async def _handle_startup_failure(self) -> None:
        """Handle startup failure with recovery attempts."""
        if self.config["deployment"]["auto_recovery"]:
            self.logger.info("Attempting automatic recovery from startup failure")
            
            # Stop any partially started services
            await self._partial_shutdown()
            
            # Wait before retry
            await asyncio.sleep(10.0)
            
            # Retry initialization and startup
            if await self.initialize_system():
                await self.start_system()
    
    async def _partial_shutdown(self) -> None:
        """Shutdown any partially started services."""
        try:
            if self.inference_engine:
                await self.inference_engine.stop()
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()
        except Exception as e:
            self.logger.error(f"Error during partial shutdown: {str(e)}")
    
    async def graceful_shutdown(self) -> None:
        """Gracefully shutdown all services."""
        self.service_state = ServiceState.STOPPING
        self.logger.info("Starting graceful shutdown of all services")
        
        shutdown_tasks = []
        
        # Stop inference engine
        if self.inference_engine:
            shutdown_tasks.append(self.inference_engine.stop())
        
        # Stop health monitoring
        if self.health_monitor:
            shutdown_tasks.append(self.health_monitor.stop_monitoring())
        
        # Execute all shutdown tasks
        try:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            self.service_state = ServiceState.STOPPED
            self.logger.info("Graceful shutdown completed successfully")
        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {str(e)}")
    
    async def process_with_reliability(
        self,
        data: pd.DataFrame,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[List]:
        """Process data with reliability guarantees."""
        for attempt in range(max_retries + 1):
            try:
                # Use circuit breaker protection
                result = await self.circuit_breaker.call(
                    self.inference_engine.core_model.predict_anomaly,
                    data
                )
                return result
                
            except CircuitBreakerOpenError:
                self.logger.warning("Circuit breaker open, request rejected")
                raise
                
            except Exception as e:
                self.error_count += 1
                self.logger.warning(f"Processing attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error(f"Processing failed after {max_retries + 1} attempts")
                    raise
        
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_info = self.health_monitor.get_current_health()
        
        return {
            "service_state": self.service_state.value,
            "uptime_seconds": time.time() - self.start_time if self.start_time > 0 else 0,
            "health_status": health_info["status"],
            "health_score": health_info.get("health_score", 0),
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "error_count": self.error_count,
            "components": {
                "anomaly_core": "initialized" if self.anomaly_core else "not_initialized",
                "inference_engine": "running" if self.inference_engine and self.inference_engine.is_running else "stopped",
                "learning_system": "active" if self.learning_system else "inactive",
                "health_monitor": "active" if self.health_monitor.monitoring_active else "inactive"
            },
            "metrics": health_info.get("metrics", {}),
            "config": {
                "auto_recovery": self.config["deployment"]["auto_recovery"],
                "health_check_interval": self.config["deployment"]["health_check_interval"]
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_info = self.health_monitor.get_current_health()
        
        # Additional service-specific health checks
        service_checks = {
            "anomaly_core": self.anomaly_core.is_trained if self.anomaly_core else False,
            "inference_engine": self.inference_engine.is_running if self.inference_engine else False,
            "learning_system": not self.learning_system.is_learning_active if self.learning_system else True
        }
        
        overall_healthy = (
            self.service_state == ServiceState.RUNNING and
            health_info["status"] in ["healthy", "degraded"] and
            all(service_checks.values())
        )
        
        return {
            "healthy": overall_healthy,
            "status": health_info["status"],
            "service_state": self.service_state.value,
            "service_checks": service_checks,
            "health_score": health_info.get("health_score", 0),
            "uptime": time.time() - self.start_time if self.start_time > 0 else 0,
            "timestamp": time.time()
        }


# Production deployment utilities

async def deploy_production_system(
    config_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    training_data_path: Optional[Path] = None
) -> RobustDeploymentFramework:
    """Deploy production-ready system with full reliability features."""
    
    # Create deployment framework
    framework = RobustDeploymentFramework(config_path)
    
    # Initialize system
    if not await framework.initialize_system():
        raise RuntimeError("Failed to initialize deployment framework")
    
    # Load or train model if paths provided
    if model_path and model_path.exists():
        framework.anomaly_core = AutonomousAnomalyCore.load_state(model_path)
        logging.info(f"Model loaded from {model_path}")
    elif training_data_path and training_data_path.exists():
        training_data = pd.read_csv(training_data_path)
        await framework.anomaly_core.train_ensemble(training_data)
        logging.info(f"Model trained on data from {training_data_path}")
    
    # Start system services
    if not await framework.start_system():
        raise RuntimeError("Failed to start system services")
    
    logging.info("Production system deployed successfully")
    return framework


if __name__ == "__main__":
    # Example production deployment
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy robust anomaly detection system")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--model", type=Path, help="Pre-trained model path")
    parser.add_argument("--training-data", type=Path, help="Training data path")
    
    args = parser.parse_args()
    
    async def main():
        try:
            framework = await deploy_production_system(
                config_path=args.config,
                model_path=args.model,
                training_data_path=args.training_data
            )
            
            # Keep system running
            while framework.service_state == ServiceState.RUNNING:
                status = framework.get_system_status()
                logging.info(f"System status: {status['health_status']} "
                           f"(uptime: {status['uptime_seconds']:.0f}s)")
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            logging.info("Shutdown requested by user")
        except Exception as e:
            logging.error(f"System error: {str(e)}")
            raise
    
    asyncio.run(main())