# Multi-stage build for IoT Anomaly Detection System
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy source code
COPY --chown=appuser:appuser . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data/processed /app/saved_models /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "src.model_serving_api"]

# Development stage
FROM base as development

USER root

# Install development dependencies
RUN pip install -r requirements-dev.txt

# Install pre-commit hooks
RUN git config --global --add safe.directory /app
USER appuser

# Override command for development
CMD ["python", "-m", "src.model_serving_api", "--reload", "--debug"]

# Production stage
FROM base as production

# Remove development files
RUN rm -rf tests/ docs/ scripts/ .git/

# Set production environment
ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    DEBUG=false

# Use gunicorn for production
RUN pip install gunicorn

# Production command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "src.model_serving_api:app"]

# Testing stage
FROM base as testing

USER root

# Install test dependencies
RUN pip install -r requirements-dev.txt

# Copy test files
COPY tests/ tests/

USER appuser

# Run tests
CMD ["pytest", "-v", "--cov=src", "--cov-report=html", "--cov-report=term"]

# Security scanning stage
FROM base as security

USER root

# Install security tools
RUN pip install bandit safety pip-audit

USER appuser

# Run security scans
CMD ["sh", "-c", "bandit -r src/ && safety check && pip-audit"]