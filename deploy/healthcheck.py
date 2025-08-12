#!/usr/bin/env python3
"""Health check script for IoT Anomaly Detection API."""

import os
import sys
import requests
import time
from pathlib import Path


def check_api_health():
    """Check if the API is responding to health checks."""
    host = os.environ.get('HOST', '0.0.0.0')
    port = os.environ.get('PORT', '8080')
    
    # Use localhost for internal health checks
    if host == '0.0.0.0':
        host = 'localhost'
    
    health_url = f"http://{host}:{port}/health"
    
    try:
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get('status') == 'healthy':
                return True, "API health check passed"
            else:
                return False, f"API reported unhealthy status: {health_data.get('status')}"
        else:
            return False, f"Health endpoint returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to API health endpoint"
    except requests.exceptions.Timeout:
        return False, "Health check request timed out"
    except Exception as e:
        return False, f"Health check error: {str(e)}"


def check_model_availability():
    """Check if required models are available."""
    model_path = Path(os.environ.get('MODEL_PATH', '/app/models'))
    
    if not model_path.exists():
        return False, f"Model path {model_path} does not exist"
    
    # Check for model files
    model_files = list(model_path.glob('*.h5')) + list(model_path.glob('*.pkl'))
    
    if not model_files:
        return False, "No model files found in model directory"
    
    return True, f"Found {len(model_files)} model files"


def check_memory_usage():
    """Check if memory usage is within acceptable limits."""
    try:
        import psutil
        
        # Get current process memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # Alert if memory usage is above 90%
        if memory_percent > 90:
            return False, f"High memory usage: {memory_percent:.1f}%"
        
        return True, f"Memory usage: {memory_percent:.1f}%"
        
    except ImportError:
        return True, "Memory check skipped (psutil not available)"
    except Exception as e:
        return False, f"Memory check error: {str(e)}"


def check_disk_space():
    """Check available disk space."""
    try:
        import shutil
        
        # Check disk space in model directory
        model_path = os.environ.get('MODEL_PATH', '/app/models')
        total, used, free = shutil.disk_usage(model_path)
        
        # Convert to GB
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        usage_percent = (used / total) * 100
        
        # Alert if less than 1GB free or usage > 95%
        if free_gb < 1.0 or usage_percent > 95:
            return False, f"Low disk space: {free_gb:.1f}GB free ({usage_percent:.1f}% used)"
        
        return True, f"Disk space: {free_gb:.1f}GB free"
        
    except Exception as e:
        return False, f"Disk space check error: {str(e)}"


def check_cache_connectivity():
    """Check cache/Redis connectivity if configured."""
    redis_url = os.environ.get('REDIS_URL')
    
    if not redis_url:
        return True, "Cache not configured"
    
    try:
        import redis
        
        client = redis.from_url(redis_url, socket_connect_timeout=3)
        client.ping()
        
        return True, "Cache connectivity OK"
        
    except ImportError:
        return True, "Redis check skipped (redis not available)"
    except Exception as e:
        return False, f"Cache connectivity error: {str(e)}"


def main():
    """Run all health checks."""
    print("IoT Anomaly Detection API Health Check")
    print("=" * 40)
    
    checks = [
        ("API Health", check_api_health),
        ("Model Availability", check_model_availability),
        ("Memory Usage", check_memory_usage),
        ("Disk Space", check_disk_space),
        ("Cache Connectivity", check_cache_connectivity),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            passed, message = check_func()
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{check_name:20} | {status} | {message}")
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"{check_name:20} | ✗ ERROR | {str(e)}")
            all_passed = False
    
    print("=" * 40)
    
    if all_passed:
        print("Overall Status: HEALTHY ✓")
        sys.exit(0)
    else:
        print("Overall Status: UNHEALTHY ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()