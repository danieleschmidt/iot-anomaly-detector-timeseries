# Development Guide

This guide covers everything you need to know to contribute to the IoT Anomaly Detection system.

## Table of Contents

- [Quick Start](#quick-start)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.8+ (recommended: 3.12)
- Git
- Docker (optional, for containerized development)
- Visual Studio Code (recommended)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/terragonlabs/iot-anomaly-detector.git
cd iot-anomaly-detector

# Setup development environment
make dev-setup

# Or manually:
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

### 2. Verify Installation

```bash
# Run tests
make test

# Check code quality
make lint
make type-check

# Generate sample data and train a model
make generate-data
make train-model
```

### 3. Start Development Server

```bash
# Start API server
make serve-dev

# Or with Docker
make docker-run
```

## Development Environment

### Local Development

The project supports multiple development environments:

#### Option 1: Native Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make setup
```

#### Option 2: Docker Development

```bash
# Build development image
make docker-build-dev

# Start development environment
make docker-run
```

#### Option 3: VS Code Dev Containers

Open the project in VS Code and select "Reopen in Container" when prompted. This provides a fully configured development environment.

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key development settings:
- `ENVIRONMENT=development`
- `LOG_LEVEL=DEBUG`
- `DEBUG=true`

## Project Structure

```
iot-anomaly-detector/
├── src/                           # Source code
│   ├── data_preprocessor.py       # Data preprocessing
│   ├── autoencoder_model.py       # Model definitions
│   ├── train_autoencoder.py       # Training logic
│   ├── anomaly_detector.py        # Detection logic
│   ├── model_serving_api.py       # API server
│   └── ...
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   ├── performance/               # Performance tests
│   └── security/                  # Security tests
├── docs/                          # Documentation
├── data/                          # Data directory
│   ├── raw/                       # Raw data
│   └── processed/                 # Processed data
├── saved_models/                  # Trained models
├── scripts/                       # Utility scripts
├── monitoring/                    # Monitoring configs
├── .devcontainer/                 # Dev container config
├── .github/                       # GitHub workflows
└── docker-compose.yml             # Development services
```

### Module Organization

- **Data Processing**: `data_preprocessor.py`, `data_validator.py`
- **Model Training**: `train_autoencoder.py`, `model_manager.py`
- **Inference**: `anomaly_detector.py`, `streaming_processor.py`
- **API**: `model_serving_api.py`
- **Utilities**: `config.py`, `security_utils.py`, `logging_config.py`

## Development Workflow

### 1. Creating a Feature

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# ... edit files ...

# Run quality checks
make dev-check

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

### 2. Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test changes
- `chore`: Build/tooling changes

Examples:
```
feat(detector): add streaming anomaly detection
fix(api): handle missing model files gracefully
docs(readme): update installation instructions
test(integration): add end-to-end pipeline tests
```

### 3. Code Review Process

1. Create pull request with descriptive title and body
2. Ensure all CI checks pass
3. Request review from maintainers
4. Address review feedback
5. Squash merge when approved

## Code Style and Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- Line length: 88 characters (Black default)
- Use type hints for all functions
- Docstrings for all public functions/classes
- Import organization with `isort`

### Code Quality Tools

```bash
# Format code
make format

# Lint code
make lint

# Type checking
make type-check

# Security scanning
make security
```

### Pre-commit Hooks

Pre-commit hooks automatically run quality checks:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Documentation Style

- Use Google-style docstrings
- Include type information
- Provide examples for complex functions
- Keep README and docs up to date

Example:
```python
def detect_anomalies(
    data: pd.DataFrame,
    threshold: float = 0.5,
    window_size: int = 30
) -> pd.Series:
    """Detect anomalies in time series data.
    
    Args:
        data: Input time series data with sensor readings
        threshold: Anomaly threshold (0.0 to 1.0)
        window_size: Size of sliding window for detection
        
    Returns:
        Binary series indicating anomalies (1) or normal (0)
        
    Example:
        >>> data = pd.DataFrame({'sensor_0': [1, 2, 3, 100, 4, 5]})
        >>> anomalies = detect_anomalies(data, threshold=0.5)
        >>> print(anomalies.sum())  # Number of anomalies detected
    """
```

## Testing

### Test Organization

- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test performance requirements
- **Security Tests**: Test security features

### Running Tests

```bash
# All tests
make test

# Specific test types
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
pytest tests/security/

# With coverage
make coverage

# Performance benchmarks
pytest tests/performance/ --benchmark-json=results.json
```

### Writing Tests

#### Unit Test Example

```python
import pytest
import pandas as pd
from src.data_preprocessor import DataPreprocessor

class TestDataPreprocessor:
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'sensor_0': [1, 2, 3, 4, 5],
            'sensor_1': [5, 4, 3, 2, 1]
        })
    
    def test_fit_transform_basic(self, sample_data):
        preprocessor = DataPreprocessor(window_size=3)
        X, y = preprocessor.fit_transform(sample_data)
        
        assert X.shape[1] == 3  # window_size
        assert X.shape[2] == 2  # n_features
        assert len(X) > 0
```

#### Integration Test Example

```python
@pytest.mark.integration
def test_end_to_end_pipeline(tmp_path):
    # Generate data
    data = generate_sample_data(n_samples=1000)
    
    # Train model
    model, scaler = train_model(data, epochs=5)
    
    # Detect anomalies
    detector = AnomalyDetector(model, scaler)
    anomalies = detector.predict(data)
    
    assert len(anomalies) == len(data)
    assert anomalies.dtype == bool
```

### Test Configuration

Test settings in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: unit tests",
    "integration: integration tests", 
    "performance: performance tests",
    "slow: slow running tests"
]
```

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install -r requirements-dev.txt

# Build documentation (if using Sphinx)
make docs-build

# Serve locally
make docs-serve
```

### Documentation Guidelines

1. **README**: Keep updated with latest features
2. **API Documentation**: Document all public functions
3. **Tutorials**: Provide step-by-step guides
4. **Architecture**: Maintain architecture diagrams
5. **Changelog**: Document all changes

### Documentation Structure

```
docs/
├── ARCHITECTURE.md          # System architecture
├── API.md                   # API documentation
├── DEVELOPMENT.md           # This file
├── DEPLOYMENT.md            # Deployment guide
├── ROADMAP.md               # Project roadmap
├── tutorials/               # Step-by-step tutorials
├── api/                     # API reference
└── examples/                # Code examples
```

## Performance Considerations

### Performance Requirements

- Inference latency: < 100ms per window
- Training time: < 30 minutes for typical datasets
- Memory usage: < 2GB for standard deployments
- Throughput: > 1000 predictions per second

### Optimization Guidelines

1. **Vectorization**: Use NumPy/Pandas vectorized operations
2. **Caching**: Cache expensive computations
3. **Batching**: Process data in batches
4. **Memory**: Monitor memory usage with large datasets
5. **Profiling**: Use `cProfile` and `memory_profiler`

### Performance Testing

```bash
# Run performance tests
pytest tests/performance/ -v

# Profile code
python -m cProfile -o profile.stats script.py

# Memory profiling
mprof run python script.py
mprof plot
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure package is installed in development mode
pip install -e .

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### Test Failures
```bash
# Run specific test with verbose output
pytest tests/test_specific.py::test_function -v -s

# Debug with pdb
pytest tests/test_specific.py::test_function --pdb
```

#### Docker Issues
```bash
# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up

# Check logs
docker-compose logs app
```

#### Performance Issues
```bash
# Profile slow code
python -m cProfile -s cumulative slow_script.py

# Check memory usage
python -m memory_profiler memory_intensive_script.py
```

### Development Tips

1. **Use IDE Integration**: Configure VS Code/PyCharm for best experience
2. **Test Early**: Write tests alongside code
3. **Small Commits**: Make frequent, small commits
4. **Documentation**: Update docs with code changes
5. **Performance**: Profile before optimizing

### Getting Help

1. **Documentation**: Check existing docs first
2. **Issues**: Search existing GitHub issues
3. **Discussions**: Use GitHub Discussions for questions
4. **Code Review**: Ask for early feedback on complex changes

## Advanced Topics

### Custom Models

To add a new model architecture:

1. Create model class in `src/models/`
2. Implement required interface methods
3. Add tests in `tests/unit/models/`
4. Update configuration options
5. Document usage examples

### Custom Preprocessors

To add a new preprocessing method:

1. Extend `DataPreprocessor` class
2. Implement transformation logic
3. Add validation and error handling
4. Write comprehensive tests
5. Update documentation

### Performance Monitoring

Monitor development performance:

```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f}s")
        return result
    return wrapper
```

---

Happy coding! 🚀