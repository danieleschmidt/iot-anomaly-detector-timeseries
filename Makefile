.PHONY: test lint security integration setup install clean format type-check coverage build docker-build docker-run docker-test docker-security quality ci-test dev-setup dev-check monitoring-up db-up train-model serve help

# Setup development environment
setup:
	@echo "Setting up development environment..."
	./scripts/setup.sh
	@echo "Development environment setup complete!"

# Install package in development mode
install:
	pip install -e .

# Clean build artifacts and cache
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Clean complete!"

# Linting with ruff
lint:
	@echo "Running linting..."
	ruff check .
	@echo "Linting complete!"

# Auto-fix linting issues
lint-fix:
	@echo "Auto-fixing linting issues..."
	ruff check --fix .
	@echo "Auto-fix complete!"

# Format code with black
format:
	@echo "Formatting code..."
	black .
	isort .
	@echo "Formatting complete!"

# Type checking with mypy
type-check:
	@echo "Running type checks..."
	mypy src/ || true
	@echo "Type checking complete!"

# Security checks
security:
	@echo "Running security checks..."
	bandit -r src
	@echo "Security checks complete!"

# Run tests
test:
	@echo "Running tests..."
	./scripts/test.sh
	@echo "Tests complete!"

# Run tests with coverage
coverage:
	@echo "Running tests with coverage..."
	pytest --cov=src --cov-report=html --cov-report=term --cov-report=xml
	@echo "Coverage report generated!"

# Run integration tests
integration:
	@echo "Running integration tests..."
	pytest -m integration
	@echo "Integration tests complete!"

# Run all quality checks
quality: lint type-check security test
	@echo "All quality checks complete!"

# Build package
build: clean
	@echo "Building package..."
	python -m build
	@echo "Build complete!"

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t iot-anomaly-detector:latest .
	@echo "Docker build complete!"

docker-build-dev:
	@echo "Building development Docker image..."
	docker build --target development -t iot-anomaly-detector:dev .
	@echo "Development Docker build complete!"

docker-run:
	@echo "Starting services with docker-compose..."
	docker-compose up -d
	@echo "Services started!"

docker-stop:
	@echo "Stopping services..."
	docker-compose down
	@echo "Services stopped!"

docker-test:
	@echo "Running tests in Docker..."
	docker-compose --profile testing run --rm test
	@echo "Docker tests complete!"

docker-security:
	@echo "Running security scans in Docker..."
	docker-compose --profile security run --rm security
	@echo "Docker security scans complete!"

# Development workflow
dev-setup: setup install
	@echo "Development setup complete!"

dev-check: lint type-check test
	@echo "Development checks complete!"

dev-full: clean dev-check security coverage
	@echo "Full development pipeline complete!"

# CI/CD helpers
ci-install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

ci-test: lint type-check security test

ci-build: ci-install ci-test build

# Monitoring setup
monitoring-up:
	@echo "Starting monitoring stack..."
	docker-compose up -d prometheus grafana
	@echo "Monitoring stack started at http://localhost:3000"

monitoring-down:
	@echo "Stopping monitoring stack..."
	docker-compose stop prometheus grafana
	@echo "Monitoring stack stopped!"

# Database management
db-up:
	@echo "Starting database..."
	docker-compose up -d postgres redis
	@echo "Database services started!"

db-down:
	@echo "Stopping database..."
	docker-compose stop postgres redis
	@echo "Database services stopped!"

# Model training
train-model:
	@echo "Training model..."
	python -m src.train_autoencoder --epochs 5 --window-size 30 --latent-dim 16
	@echo "Model training complete!"

# Generate sample data
generate-data:
	@echo "Generating sample data..."
	python -m src.generate_data --num-samples 1000 --num-features 3 --output-path data/raw/sensor_data.csv
	@echo "Data generation complete!"

# API server
serve:
	@echo "Starting API server..."
	python -m src.model_serving_api
	@echo "API server stopped!"

serve-dev:
	@echo "Starting development API server..."
	python -m src.model_serving_api --reload --debug
	@echo "Development API server stopped!"

# Help
help:
	@echo "Available targets:"
	@echo "  setup          - Setup development environment"
	@echo "  install        - Install package in development mode"
	@echo "  clean          - Clean build artifacts"
	@echo "  lint           - Run linting"
	@echo "  format         - Format code"
	@echo "  type-check     - Run type checking"
	@echo "  test           - Run tests"
	@echo "  coverage       - Run tests with coverage"
	@echo "  security       - Run security checks"
	@echo "  quality        - Run all quality checks"
	@echo "  build          - Build package"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Start services with Docker"
	@echo "  serve          - Start API server"
	@echo "  train-model    - Train ML model"
	@echo "  generate-data  - Generate sample data"
	@echo "  help           - Show this help message"
