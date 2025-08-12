#!/bin/bash

# Production startup script for IoT Anomaly Detection API

set -e

echo "Starting IoT Anomaly Detection API..."

# Environment variables with defaults
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}
WORKERS=${WORKERS:-4}
MODEL_PATH=${MODEL_PATH:-/app/models}
LOG_LEVEL=${LOG_LEVEL:-INFO}
TIMEOUT=${TIMEOUT:-300}
KEEPALIVE=${KEEPALIVE:-2}

# Create directories if they don't exist
mkdir -p /app/logs /app/cache "$MODEL_PATH"

# Pre-flight checks
echo "Performing pre-flight checks..."

# Check if required directories exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path $MODEL_PATH does not exist"
    exit 1
fi

# Check if Python can import the application
python -c "from src.model_serving_api import app; print('✓ Application import successful')"

# Check database/cache connectivity if configured
if [ -n "$REDIS_URL" ]; then
    echo "Testing Redis connectivity..."
    python -c "
import redis
import os
try:
    r = redis.from_url(os.environ.get('REDIS_URL'))
    r.ping()
    print('✓ Redis connection successful')
except Exception as e:
    print(f'⚠ Redis connection failed: {e}')
    print('Continuing without distributed cache...')
"
fi

# Initialize models if needed
echo "Initializing models..."
python -c "
import os
from pathlib import Path
from src.resilient_anomaly_pipeline import ResilientAnomalyPipeline

model_path = Path(os.environ.get('MODEL_PATH', '/app/models'))
if not any(model_path.glob('*.h5')) and not any(model_path.glob('*.pkl')):
    print('No trained models found. Models should be loaded separately.')
else:
    print('✓ Models found in model directory')
"

# Set up logging
export LOGURU_LEVEL="$LOG_LEVEL"

# Calculate optimal worker count if not specified
if [ "$WORKERS" = "auto" ]; then
    WORKERS=$(python -c "import os; print(min(8, max(1, os.cpu_count())))")
    echo "Auto-detected $WORKERS workers"
fi

echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  Model Path: $MODEL_PATH"
echo "  Log Level: $LOG_LEVEL"
echo "  Timeout: $TIMEOUT"

# Start the application with Gunicorn
echo "Starting Gunicorn server..."

exec gunicorn \
    --bind "$HOST:$PORT" \
    --workers "$WORKERS" \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout "$TIMEOUT" \
    --keepalive "$KEEPALIVE" \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --log-level "$LOG_LEVEL" \
    --access-logfile /app/logs/access.log \
    --error-logfile /app/logs/error.log \
    --capture-output \
    --enable-stdio-inheritance \
    --worker-tmp-dir /dev/shm \
    --worker-connections 1000 \
    src.model_serving_api:app