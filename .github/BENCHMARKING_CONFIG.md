# Performance Benchmarking Automation Configuration

## Overview

This repository has comprehensive performance benchmarking capabilities that can be automated via GitHub Actions. The benchmarking infrastructure is already implemented but requires workflow activation.

## Existing Benchmarking Infrastructure

### 1. Performance Benchmarking Module
- **Location**: `benchmarks/performance_benchmarks.py`
- **Features**: 
  - Model training performance benchmarking
  - Inference performance measurement
  - Memory usage tracking
  - GPU utilization monitoring (when available)

### 2. Performance Monitoring System
- **Location**: `src/performance_monitor_cli.py`
- **Features**:
  - Real-time performance metrics
  - Performance scoring (0-100 scale)
  - Automated performance issue detection
  - Metrics export (JSON/CSV)

### 3. Profiling Tools Configuration
- **py-spy**: Configured in `pyproject.toml` [tool.py-spy]
- **memray**: Configured in `pyproject.toml` [tool.memray]
- **Requirements**: Listed in `requirements-profiling.txt`

## Automated Benchmarking Workflow Configuration

### Required GitHub Actions Workflow

Create `.github/workflows/benchmarks.yml`:

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run benchmarks weekly on Sundays at 3 AM UTC
    - cron: '0 3 * * 0'

jobs:
  benchmark:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.11]
        
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-profiling.txt
        
    - name: Run performance benchmarks
      run: |
        python -m benchmarks.performance_benchmarks --output benchmarks_${{ matrix.python-version }}.json
        
    - name: Generate performance report
      run: |
        python -m src.performance_monitor_cli --report --format json --output performance_report_${{ matrix.python-version }}.json
        
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results-${{ matrix.python-version }}
        path: |
          benchmarks_${{ matrix.python-version }}.json
          performance_report_${{ matrix.python-version }}.json
          
    - name: Performance regression check
      run: |
        # Compare with baseline if available
        python scripts/check_performance_regression.py benchmarks_${{ matrix.python-version }}.json
```

### Continuous Benchmarking Integration

```yaml
# Add to existing CI workflow
- name: Performance regression test
  run: |
    python -m pytest tests/performance/ -v --benchmark-only
    python -m benchmarks.performance_benchmarks --quick-test
```

### Benchmark Comparison Workflow

```yaml
name: Benchmark Comparison

on:
  pull_request:
    paths:
      - 'src/**'
      - 'benchmarks/**'

jobs:
  compare-benchmarks:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for comparison
        
    - name: Benchmark current branch
      run: |
        python -m benchmarks.performance_benchmarks --output current_benchmarks.json
        
    - name: Checkout main branch
      run: |
        git checkout main
        python -m benchmarks.performance_benchmarks --output main_benchmarks.json
        
    - name: Compare performance
      run: |
        python scripts/compare_benchmarks.py main_benchmarks.json current_benchmarks.json
```

## Required Script Creation

### 1. Performance Regression Checker

Create `scripts/check_performance_regression.py`:

```python
#!/usr/bin/env python3
"""Performance regression detection script."""

import json
import sys
from pathlib import Path
from typing import Dict, Any

def check_regression(benchmark_file: Path, threshold: float = 0.1) -> bool:
    """Check for performance regression against baseline."""
    if not benchmark_file.exists():
        print(f"Benchmark file not found: {benchmark_file}")
        return False
        
    # Load current benchmarks
    with open(benchmark_file) as f:
        benchmarks = json.load(f)
    
    # Load baseline if available
    baseline_file = Path("baseline_benchmarks.json")
    if not baseline_file.exists():
        print("No baseline found, saving current as baseline")
        with open(baseline_file, 'w') as f:
            json.dump(benchmarks, f, indent=2)
        return True
        
    with open(baseline_file) as f:
        baseline = json.load(f)
    
    # Compare performance metrics
    regressions = []
    for test_name, current_metrics in benchmarks.items():
        if test_name in baseline:
            baseline_metrics = baseline[test_name]
            
            # Check timing regression
            if 'execution_time' in current_metrics and 'execution_time' in baseline_metrics:
                current_time = current_metrics['execution_time']
                baseline_time = baseline_metrics['execution_time']
                regression = (current_time - baseline_time) / baseline_time
                
                if regression > threshold:
                    regressions.append(f"{test_name}: {regression:.2%} slower")
    
    if regressions:
        print("Performance regressions detected:")
        for regression in regressions:
            print(f"  - {regression}")
        return False
    
    print("No performance regressions detected")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_performance_regression.py <benchmark_file>")
        sys.exit(1)
        
    benchmark_file = Path(sys.argv[1])
    success = check_regression(benchmark_file)
    sys.exit(0 if success else 1)
```

### 2. Benchmark Comparison Script

Create `scripts/compare_benchmarks.py`:

```python
#!/usr/bin/env python3
"""Compare benchmark results between branches."""

import json
import sys
from pathlib import Path
from typing import Dict, Any

def compare_benchmarks(baseline_file: Path, current_file: Path) -> None:
    """Compare benchmark results and generate report."""
    with open(baseline_file) as f:
        baseline = json.load(f)
    with open(current_file) as f:
        current = json.load(f)
    
    print("## Performance Comparison Report")
    print(f"Baseline: {baseline_file}")
    print(f"Current: {current_file}")
    print()
    
    for test_name in sorted(set(baseline.keys()) | set(current.keys())):
        if test_name in both baseline and current:
            baseline_time = baseline[test_name].get('execution_time', 0)
            current_time = current[test_name].get('execution_time', 0)
            
            if baseline_time > 0:
                change = (current_time - baseline_time) / baseline_time
                status = "🔴" if change > 0.1 else "🟡" if change > 0.05 else "🟢"
                print(f"{status} {test_name}: {change:+.2%} ({current_time:.3f}s vs {baseline_time:.3f}s)")
            else:
                print(f"ℹ️ {test_name}: {current_time:.3f}s (new test)")
        elif test_name in baseline:
            print(f"❌ {test_name}: Test removed")
        else:
            print(f"➕ {test_name}: New test ({current[test_name].get('execution_time', 0):.3f}s)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_benchmarks.py <baseline_file> <current_file>")
        sys.exit(1)
        
    baseline_file = Path(sys.argv[1])
    current_file = Path(sys.argv[2])
    compare_benchmarks(baseline_file, current_file)
```

## Configuration Files

### 1. Benchmark Configuration

Create `.benchmarkrc`:
```json
{
  "benchmark_timeout": 300,
  "memory_limit": "2GB", 
  "warmup_runs": 3,
  "measurement_runs": 10,
  "output_format": "json",
  "include_memory_usage": true,
  "include_gpu_metrics": true,
  "regression_threshold": 0.1
}
```

### 2. Performance Baseline

Create `performance_baseline.json`:
```json
{
  "model_training": {
    "target_time": 30.0,
    "max_memory_mb": 1024
  },
  "inference_batch_100": {
    "target_time": 0.5,
    "max_memory_mb": 256
  },
  "data_preprocessing": {
    "target_time": 2.0,
    "max_memory_mb": 512
  }
}
```

## Integration with Existing Tools

### 1. Makefile Integration

Add to existing `Makefile`:
```makefile
.PHONY: benchmark benchmark-quick benchmark-compare

benchmark:
	python -m benchmarks.performance_benchmarks --output benchmarks.json
	python -m src.performance_monitor_cli --report --format json

benchmark-quick:
	python -m benchmarks.performance_benchmarks --quick-test

benchmark-compare:
	@if [ -f baseline_benchmarks.json ]; then \
		python scripts/compare_benchmarks.py baseline_benchmarks.json benchmarks.json; \
	else \
		echo "No baseline found. Run 'make benchmark-baseline' first"; \
	fi

benchmark-baseline:
	python -m benchmarks.performance_benchmarks --output baseline_benchmarks.json
	@echo "Baseline benchmarks saved"
```

### 2. Pre-commit Hook Integration

Add to `.pre-commit-config.yaml`:
```yaml
- repo: local
  hooks:
  - id: performance-regression-check
    name: Performance regression check
    entry: python scripts/check_performance_regression.py
    language: python
    files: '^(src/|benchmarks/)'
    pass_filenames: false
    args: ['benchmarks.json']
```

## Usage Instructions

### Manual Benchmarking
```bash
# Run full benchmark suite
make benchmark

# Quick performance check
make benchmark-quick

# Compare with baseline
make benchmark-compare

# Set new baseline
make benchmark-baseline
```

### Automated Benchmarking
1. **Activate GitHub Actions**: Copy workflow files to `.github/workflows/`
2. **Configure secrets**: Set up any required API tokens
3. **Initial baseline**: Run benchmarks on main branch to establish baseline
4. **Monitor results**: Check Actions tab for benchmark results and regressions

## Benefits

- **Automated Performance Monitoring**: Continuous performance tracking
- **Regression Prevention**: Early detection of performance issues
- **Historical Tracking**: Performance trends over time
- **Multi-platform Testing**: Cross-platform performance validation
- **Integration Ready**: Works with existing development workflow

## Next Steps

1. Create `.github/workflows/` directory
2. Copy benchmark workflow configurations
3. Create required scripts in `scripts/` directory
4. Run initial baseline benchmarks
5. Monitor performance in subsequent commits

---

**Note**: This configuration builds on the existing comprehensive performance monitoring infrastructure already implemented in this repository.