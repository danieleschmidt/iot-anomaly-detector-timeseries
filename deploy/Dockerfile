# Multi-stage production Dockerfile for IoT Anomaly Detection System
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn uvicorn[standard]

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./

# Install the package
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app/src ./src
COPY --from=builder /app/pyproject.toml ./

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/cache /tmp && \
    chown -R appuser:appuser /app /tmp

# Copy startup scripts
COPY deploy/scripts/ ./scripts/
RUN chmod +x ./scripts/*.sh

# Health check script
COPY deploy/healthcheck.py ./
RUN chmod +x healthcheck.py

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8080 9090

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models \
    LOG_LEVEL=INFO \
    WORKERS=4 \
    HOST=0.0.0.0 \
    PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python healthcheck.py || exit 1

# Default command
CMD ["./scripts/start.sh"]

# Development stage
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy test files
COPY tests/ ./tests/

# Switch to non-root user
USER appuser

# Development command
CMD ["python", "-m", "src.model_serving_api", "--reload"]

# Training stage for model preparation
FROM builder as training

# Install training-specific dependencies
RUN pip install --no-cache-dir tensorboard jupyter

# Copy training scripts
COPY scripts/ ./scripts/
COPY notebooks/ ./notebooks/

# Training command
CMD ["python", "-m", "src.train_autoencoder"]