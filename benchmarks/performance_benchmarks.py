#!/usr/bin/env python3
"""
Advanced Performance Benchmarking Suite for IoT Anomaly Detector

Provides comprehensive performance testing and profiling capabilities
for measuring system performance across different scenarios.
"""

import time
import psutil
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager
import tracemalloc
import cProfile
import pstats
from io import StringIO

from src.data_preprocessor import DataPreprocessor
from src.anomaly_detector import AnomalyDetector
from src.autoencoder_model import create_autoencoder
from src.generate_data import generate_synthetic_data


@dataclass
class BenchmarkResult:
    """Container for benchmark results with metrics."""
    name: str
    execution_time: float
    memory_peak: float
    memory_current: float
    cpu_percent: float
    throughput: float
    metadata: Dict[str, Any]


class PerformanceBenchmarker:
    """Advanced performance benchmarking suite."""
    
    def __init__(self, output_dir: Path = Path("benchmarks/results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
    @contextmanager
    def measure_performance(self, name: str, **metadata):
        """Context manager for measuring performance metrics."""
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            # Calculate metrics
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            current, peak = tracemalloc.get_traced_memory()
            peak_memory = peak / 1024 / 1024  # MB
            tracemalloc.stop()
            
            cpu_percent = process.cpu_percent()
            
            # Calculate throughput if data size provided
            throughput = 0.0
            if 'data_size' in metadata:
                throughput = metadata['data_size'] / execution_time
                
            result = BenchmarkResult(
                name=name,
                execution_time=execution_time,
                memory_peak=peak_memory,
                memory_current=current_memory,
                cpu_percent=cpu_percent - start_cpu,
                throughput=throughput,
                metadata=metadata
            )
            
            self.results.append(result)
            
    def benchmark_data_preprocessing(self, data_sizes: List[int] = None):
        """Benchmark data preprocessing performance."""
        if data_sizes is None:
            data_sizes = [1000, 5000, 10000, 50000]
            
        preprocessor = DataPreprocessor()
        
        for size in data_sizes:
            # Generate test data
            data = generate_synthetic_data(
                num_samples=size,
                num_features=5,
                seed=42
            )
            
            with self.measure_performance(
                f"data_preprocessing_{size}",
                data_size=size,
                component="preprocessing"
            ):
                processed = preprocessor.fit_transform(data)
                
    def benchmark_model_training(self, data_sizes: List[int] = None):
        """Benchmark model training performance."""
        if data_sizes is None:
            data_sizes = [1000, 5000, 10000]
            
        for size in data_sizes:
            data = generate_synthetic_data(
                num_samples=size,
                num_features=5,
                seed=42
            )
            
            preprocessor = DataPreprocessor()
            processed_data = preprocessor.fit_transform(data)
            
            with self.measure_performance(
                f"model_training_{size}",
                data_size=size,
                component="training"
            ):
                model = create_autoencoder(
                    input_dim=processed_data.shape[1],
                    latent_dim=16
                )
                model.compile(optimizer='adam', loss='mse')
                model.fit(
                    processed_data, processed_data,
                    epochs=5,
                    batch_size=32,
                    verbose=0
                )
                
    def benchmark_anomaly_detection(self, data_sizes: List[int] = None):
        """Benchmark anomaly detection performance."""
        if data_sizes is None:
            data_sizes = [1000, 5000, 10000, 50000]
            
        # Pre-train a model
        train_data = generate_synthetic_data(5000, 5, seed=42)
        preprocessor = DataPreprocessor()
        processed_train = preprocessor.fit_transform(train_data)
        
        model = create_autoencoder(processed_train.shape[1], 16)
        model.compile(optimizer='adam', loss='mse')
        model.fit(processed_train, processed_train, epochs=5, verbose=0)
        
        for size in data_sizes:
            test_data = generate_synthetic_data(size, 5, seed=123)
            
            with self.measure_performance(
                f"anomaly_detection_{size}",
                data_size=size,
                component="detection"
            ):
                processed_test = preprocessor.transform(test_data)
                predictions = model.predict(processed_test, verbose=0)
                reconstruction_errors = np.mean(
                    np.square(processed_test - predictions), axis=1
                )
                threshold = np.percentile(reconstruction_errors, 95)
                anomalies = reconstruction_errors > threshold
                
    def benchmark_memory_efficiency(self):
        """Benchmark memory efficiency with large datasets."""
        sizes = [10000, 50000, 100000]
        
        for size in sizes:
            with self.measure_performance(
                f"memory_efficiency_{size}",
                data_size=size,
                component="memory"
            ):
                # Simulate large dataset processing
                data = generate_synthetic_data(size, 10, seed=42)
                preprocessor = DataPreprocessor()
                
                # Process in chunks to test memory efficiency
                chunk_size = 1000
                results = []
                
                for i in range(0, len(data), chunk_size):
                    chunk = data.iloc[i:i+chunk_size]
                    processed_chunk = preprocessor.transform(chunk)
                    results.append(processed_chunk)
                    
                final_result = np.vstack(results)
                
    def profile_function(self, func, *args, **kwargs):
        """Profile a specific function and save results."""
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        # Save profiling results
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        output = StringIO()
        stats.print_stats(output)
        
        profile_path = self.output_dir / f"profile_{func.__name__}.txt"
        with open(profile_path, 'w') as f:
            f.write(output.getvalue())
            
        return result
        
    def run_all_benchmarks(self):
        """Run complete benchmark suite."""
        print("🚀 Starting comprehensive performance benchmarks...")
        
        print("📊 Benchmarking data preprocessing...")
        self.benchmark_data_preprocessing()
        
        print("🤖 Benchmarking model training...")
        self.benchmark_model_training()
        
        print("🔍 Benchmarking anomaly detection...")
        self.benchmark_anomaly_detection()
        
        print("💾 Benchmarking memory efficiency...")
        self.benchmark_memory_efficiency()
        
        print("✅ All benchmarks completed!")
        
    def generate_report(self) -> pd.DataFrame:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return pd.DataFrame()
            
        data = []
        for result in self.results:
            row = {
                'benchmark': result.name,
                'execution_time_s': result.execution_time,
                'memory_peak_mb': result.memory_peak,
                'memory_current_mb': result.memory_current,
                'cpu_percent': result.cpu_percent,
                'throughput_items_per_s': result.throughput,
                'component': result.metadata.get('component', 'unknown'),
                'data_size': result.metadata.get('data_size', 0)
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Save report
        report_path = self.output_dir / "benchmark_report.csv"
        df.to_csv(report_path, index=False)
        
        # Generate summary statistics
        summary = df.groupby('component').agg({
            'execution_time_s': ['mean', 'std', 'min', 'max'],
            'memory_peak_mb': ['mean', 'std', 'min', 'max'],
            'throughput_items_per_s': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        summary_path = self.output_dir / "benchmark_summary.csv"
        summary.to_csv(summary_path)
        
        return df
        
    def compare_with_baseline(self, baseline_path: Path):
        """Compare current results with baseline performance."""
        if not baseline_path.exists():
            print(f"⚠️  No baseline found at {baseline_path}")
            return
            
        baseline_df = pd.read_csv(baseline_path)
        current_df = self.generate_report()
        
        comparison = []
        for _, current in current_df.iterrows():
            baseline_row = baseline_df[
                baseline_df['benchmark'] == current['benchmark']
            ]
            
            if not baseline_row.empty:
                baseline_time = baseline_row.iloc[0]['execution_time_s']
                current_time = current['execution_time_s']
                
                performance_change = (
                    (current_time - baseline_time) / baseline_time * 100
                )
                
                comparison.append({
                    'benchmark': current['benchmark'],
                    'current_time_s': current_time,
                    'baseline_time_s': baseline_time,
                    'performance_change_percent': performance_change,
                    'status': '🟢 IMPROVED' if performance_change < -5 else 
                             '🔴 REGRESSED' if performance_change > 5 else 
                             '🟡 SIMILAR'
                })
                
        comparison_df = pd.DataFrame(comparison)
        comparison_path = self.output_dir / "performance_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        return comparison_df


@pytest.mark.performance
def test_benchmark_suite():
    """Test the benchmark suite execution."""
    benchmarker = PerformanceBenchmarker()
    
    # Run smaller benchmarks for testing
    benchmarker.benchmark_data_preprocessing([100, 500])
    benchmarker.benchmark_anomaly_detection([100, 500])
    
    report = benchmarker.generate_report()
    assert len(report) > 0
    assert 'execution_time_s' in report.columns
    assert 'memory_peak_mb' in report.columns


if __name__ == "__main__":
    benchmarker = PerformanceBenchmarker()
    benchmarker.run_all_benchmarks()
    report = benchmarker.generate_report()
    
    print("\n📈 Benchmark Summary:")
    print(report.groupby('component')['execution_time_s'].describe())