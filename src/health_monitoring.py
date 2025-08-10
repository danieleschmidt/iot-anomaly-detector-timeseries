#!/usr/bin/env python3
"""
Comprehensive Health Monitoring System for IoT Anomaly Detection Platform
Provides real-time health checks, dependency validation, and system diagnostics.
"""

import asyncio
import json
import logging
import os
import platform
import psutil
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


class HealthStatus(Enum):
    """Health check status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result"""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    metadata: Dict[str, Any]


@dataclass
class SystemHealth:
    """Complete system health report"""
    overall_status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    checks: List[HealthCheck]
    summary: Dict[str, int]


class HealthMonitor:
    """Comprehensive health monitoring system"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config.yaml")
        self.start_time = time.time()
        self.logger = self._setup_logging()
        
        # Health thresholds
        self.cpu_threshold = 85.0
        self.memory_threshold = 85.0
        self.disk_threshold = 90.0
        self.model_latency_threshold = 1000  # ms
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for health monitoring"""
        logger = logging.getLogger("health_monitor")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def check_system_resources(self) -> HealthCheck:
        """Monitor system resource utilization"""
        start_time = time.time()
        
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory utilization  
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk utilization
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Determine status
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent > self.cpu_threshold:
                status = HealthStatus.CRITICAL
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                
            if memory_percent > self.memory_threshold:
                status = HealthStatus.CRITICAL
                issues.append(f"High memory usage: {memory_percent:.1f}%")
                
            if disk_percent > self.disk_threshold:
                status = HealthStatus.WARNING
                issues.append(f"Low disk space: {disk_percent:.1f}% used")
            
            message = "System resources healthy" if not issues else "; ".join(issues)
            
            metadata = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3)
            }
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Failed to check system resources: {str(e)}"
            metadata = {"error": str(e)}
            
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheck(
            name="system_resources",
            status=status,
            message=message,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            metadata=metadata
        )
    
    async def check_model_availability(self) -> HealthCheck:
        """Check if ML models are available and functional"""
        start_time = time.time()
        
        try:
            from src.model_manager import ModelManager
            
            model_manager = ModelManager()
            
            # Check if default model exists
            default_model_path = Path("saved_models/autoencoder.h5")
            if not default_model_path.exists():
                return HealthCheck(
                    name="model_availability",
                    status=HealthStatus.WARNING,
                    message="Default model not found",
                    timestamp=datetime.now(),
                    duration_ms=(time.time() - start_time) * 1000,
                    metadata={"model_path": str(default_model_path)}
                )
            
            # Try loading the model
            model = tf.keras.models.load_model(str(default_model_path))
            
            # Quick inference test with dummy data
            dummy_input = np.random.random((1, 30, 3))  # Typical window shape
            inference_start = time.time()
            _ = model.predict(dummy_input, verbose=0)
            inference_time = (time.time() - inference_start) * 1000
            
            # Check inference latency
            status = HealthStatus.HEALTHY
            if inference_time > self.model_latency_threshold:
                status = HealthStatus.WARNING
                
            message = f"Model functional, inference: {inference_time:.2f}ms"
            
            metadata = {
                "model_path": str(default_model_path),
                "inference_time_ms": inference_time,
                "model_input_shape": list(model.input.shape),
                "model_output_shape": list(model.output.shape)
            }
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Model check failed: {str(e)}"
            metadata = {"error": str(e)}
            
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheck(
            name="model_availability", 
            status=status,
            message=message,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            metadata=metadata
        )
    
    async def check_data_pipeline(self) -> HealthCheck:
        """Verify data preprocessing pipeline functionality"""
        start_time = time.time()
        
        try:
            from src.data_preprocessor import DataPreprocessor
            
            # Test with synthetic data
            test_data = pd.DataFrame({
                'sensor_1': np.random.normal(0, 1, 100),
                'sensor_2': np.random.normal(0, 1, 100), 
                'sensor_3': np.random.normal(0, 1, 100)
            })
            
            preprocessor = DataPreprocessor()
            
            # Test preprocessing
            processed_data = preprocessor.fit_transform(test_data)
            
            if processed_data is None or len(processed_data) == 0:
                raise ValueError("Preprocessing returned empty result")
                
            status = HealthStatus.HEALTHY
            message = "Data pipeline functional"
            
            metadata = {
                "input_shape": test_data.shape,
                "output_shape": processed_data.shape,
                "preprocessing_successful": True
            }
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Data pipeline check failed: {str(e)}"
            metadata = {"error": str(e)}
            
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheck(
            name="data_pipeline",
            status=status, 
            message=message,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            metadata=metadata
        )
    
    async def check_dependencies(self) -> HealthCheck:
        """Verify critical dependencies and versions"""
        start_time = time.time()
        
        try:
            dependencies = {
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__, 
                "sklearn": __import__("sklearn").__version__,
                "psutil": psutil.__version__
            }
            
            # Check Python version
            python_version = platform.python_version()
            
            status = HealthStatus.HEALTHY
            message = "All dependencies available"
            
            metadata = {
                "python_version": python_version,
                "dependencies": dependencies,
                "platform": platform.platform()
            }
            
        except Exception as e:
            status = HealthStatus.CRITICAL  
            message = f"Dependency check failed: {str(e)}"
            metadata = {"error": str(e)}
            
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheck(
            name="dependencies",
            status=status,
            message=message,
            timestamp=datetime.now(), 
            duration_ms=duration_ms,
            metadata=metadata
        )
    
    async def check_storage_access(self) -> HealthCheck:
        """Verify file system access for models and data"""
        start_time = time.time()
        
        try:
            required_dirs = [
                Path("saved_models"),
                Path("data/raw"),
                Path("data/processed")
            ]
            
            issues = []
            for dir_path in required_dirs:
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    
                # Test write access
                test_file = dir_path / ".health_check"
                try:
                    test_file.write_text("health_check")
                    test_file.unlink()
                except OSError:
                    issues.append(f"No write access to {dir_path}")
            
            status = HealthStatus.HEALTHY if not issues else HealthStatus.CRITICAL
            message = "Storage accessible" if not issues else "; ".join(issues)
            
            metadata = {
                "required_directories": [str(d) for d in required_dirs],
                "write_access": len(issues) == 0
            }
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Storage check failed: {str(e)}"
            metadata = {"error": str(e)}
            
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheck(
            name="storage_access",
            status=status,
            message=message,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            metadata=metadata
        )
    
    async def check_network_connectivity(self) -> HealthCheck:
        """Verify network connectivity for distributed operations"""
        start_time = time.time()
        
        try:
            # Test local network connectivity
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('8.8.8.8', 53))  # Google DNS
            sock.close()
            
            if result == 0:
                status = HealthStatus.HEALTHY
                message = "Network connectivity available"
                metadata = {"connectivity": True}
            else:
                status = HealthStatus.WARNING
                message = "Limited network connectivity"  
                metadata = {"connectivity": False}
                
        except Exception as e:
            status = HealthStatus.WARNING
            message = f"Network check failed: {str(e)}"
            metadata = {"error": str(e), "connectivity": False}
            
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheck(
            name="network_connectivity",
            status=status,
            message=message, 
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            metadata=metadata
        )
    
    async def run_all_checks(self) -> SystemHealth:
        """Execute all health checks and generate comprehensive report"""
        self.logger.info("Starting comprehensive health check")
        
        # Run all checks concurrently
        check_tasks = [
            self.check_system_resources(),
            self.check_model_availability(),
            self.check_data_pipeline(), 
            self.check_dependencies(),
            self.check_storage_access(),
            self.check_network_connectivity()
        ]
        
        checks = await asyncio.gather(*check_tasks)
        
        # Calculate overall status
        status_counts = {}
        for check in checks:
            status_counts[check.status] = status_counts.get(check.status, 0) + 1
        
        # Determine overall status
        if status_counts.get(HealthStatus.CRITICAL, 0) > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts.get(HealthStatus.WARNING, 0) > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
            
        uptime = time.time() - self.start_time
        
        summary = {status.value: count for status, count in status_counts.items()}
        
        health_report = SystemHealth(
            overall_status=overall_status,
            timestamp=datetime.now(),
            uptime_seconds=uptime,
            checks=checks,
            summary=summary
        )
        
        self.logger.info(f"Health check completed - Overall: {overall_status.value}")
        return health_report
    
    def export_health_report(self, health: SystemHealth, output_path: Path) -> None:
        """Export health report to JSON file"""
        try:
            # Convert to serializable format
            report_data = {
                "overall_status": health.overall_status.value,
                "timestamp": health.timestamp.isoformat(),
                "uptime_seconds": health.uptime_seconds,
                "summary": health.summary,
                "checks": [
                    {
                        "name": check.name,
                        "status": check.status.value,
                        "message": check.message,
                        "timestamp": check.timestamp.isoformat(),
                        "duration_ms": check.duration_ms,
                        "metadata": check.metadata
                    } for check in health.checks
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            self.logger.info(f"Health report exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export health report: {e}")


async def main():
    """Main entry point for health monitoring"""
    monitor = HealthMonitor()
    health = await monitor.run_all_checks()
    
    # Print summary
    print(f"\n🏥 System Health Report - {health.overall_status.value.upper()}")
    print(f"⏱️  Uptime: {health.uptime_seconds:.1f}s")
    print(f"📊 Summary: {health.summary}")
    
    print("\n📋 Detailed Results:")
    for check in health.checks:
        status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}.get(
            check.status.value, "❓"
        )
        print(f"{status_emoji} {check.name}: {check.message} ({check.duration_ms:.1f}ms)")
    
    # Export report
    output_path = Path("health_report.json")
    monitor.export_health_report(health, output_path)
    
    return health.overall_status == HealthStatus.HEALTHY


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)