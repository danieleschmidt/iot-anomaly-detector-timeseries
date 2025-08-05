"""
Quantum Pipeline CLI

Command-line interface for quantum-inspired ML pipeline optimization.
Provides easy access to quantum task planning features for ML workflows.
"""

import argparse
import json
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from .quantum_ml_integration import QuantumMLPipelineOptimizer, MLTask
from .quantum_inspired.task_representation import TaskPriority
from .task_planning.task_scheduler import SchedulingAlgorithm
from .quantum_inspired.quantum_optimization_base import OptimizationObjective


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save optimization results to file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")


def create_sample_config() -> Dict[str, Any]:
    """Create sample configuration for ML pipeline optimization."""
    return {
        "system_resources": {
            "max_cpu_cores": 8,
            "max_gpu_memory": 8.0,
            "max_system_memory": 32.0
        },
        "optimization": {
            "algorithm": "quantum_annealing",
            "objective": "minimize_makespan",
            "algorithm_params": {
                "max_iterations": 500,
                "initial_temperature": 100.0,
                "final_temperature": 0.01
            }
        },
        "ml_tasks": [
            {
                "task_id": "preprocess_data",
                "task_type": "preprocessing",
                "data_path": "data/raw/sensor_data.csv",
                "expected_duration": 300,
                "priority": "HIGH",
                "memory_estimate": 2.0,
                "cpu_cores": 2,
                "gpu_required": false,
                "dependencies": []
            },
            {
                "task_id": "train_model_1",
                "task_type": "training",
                "model_path": "models/autoencoder_1.h5",
                "data_path": "data/processed/train_data.csv",
                "epochs": 50,
                "batch_size": 32,
                "expected_duration": 3600,
                "priority": "HIGH",
                "memory_estimate": 6.0,
                "cpu_cores": 4,
                "gpu_required": true,
                "dependencies": ["preprocess_data"]
            },
            {
                "task_id": "evaluate_model_1",
                "task_type": "evaluation",
                "model_path": "models/autoencoder_1.h5",
                "data_path": "data/processed/test_data.csv",
                "expected_duration": 300,
                "priority": "MEDIUM",
                "memory_estimate": 2.0,
                "cpu_cores": 2,
                "gpu_required": false,
                "dependencies": ["train_model_1"]
            }
        ]
    }


def parse_ml_tasks(tasks_config: List[Dict[str, Any]]) -> List[MLTask]:
    """Parse ML tasks from configuration."""
    ml_tasks = []
    
    for task_config in tasks_config:
        # Parse priority
        priority_str = task_config.get("priority", "MEDIUM").upper()
        priority = getattr(TaskPriority, priority_str, TaskPriority.MEDIUM)
        
        ml_task = MLTask(
            task_id=task_config["task_id"],
            task_type=task_config["task_type"],
            model_path=task_config.get("model_path"),
            data_path=task_config.get("data_path"),
            batch_size=task_config.get("batch_size", 32),
            epochs=task_config.get("epochs", 1),
            gpu_required=task_config.get("gpu_required", False),
            memory_estimate=task_config.get("memory_estimate", 1.0),
            cpu_cores=task_config.get("cpu_cores", 1),
            expected_duration=task_config.get("expected_duration", 300),
            priority=priority,
            dependencies=task_config.get("dependencies", []),
            metadata=task_config.get("metadata", {})
        )
        ml_tasks.append(ml_task)
    
    return ml_tasks


def cmd_optimize(args) -> None:
    """Run ML pipeline optimization."""
    setup_logging(args.verbose)
    
    # Load configuration
    config = load_config(args.config)
    
    # Parse system resources
    resources = config.get("system_resources", {})
    optimizer = QuantumMLPipelineOptimizer(
        max_cpu_cores=resources.get("max_cpu_cores", 8),
        max_gpu_memory=resources.get("max_gpu_memory", 8.0),
        max_system_memory=resources.get("max_system_memory", 32.0)
    )
    
    # Parse optimization settings
    opt_config = config.get("optimization", {})
    algorithm = SchedulingAlgorithm(opt_config.get("algorithm", "quantum_annealing"))
    objective = OptimizationObjective(opt_config.get("objective", "minimize_makespan"))
    
    # Parse ML tasks
    ml_tasks = parse_ml_tasks(config["ml_tasks"])
    
    print(f"Optimizing pipeline with {len(ml_tasks)} ML tasks")
    print(f"Algorithm: {algorithm.value}")
    print(f"Objective: {objective.value}")
    
    # Run optimization
    start_time = time.time()
    result = optimizer.optimize_ml_pipeline(
        ml_tasks=ml_tasks,
        objective=objective,
        algorithm=algorithm
    )
    optimization_time = time.time() - start_time
    
    # Display results
    print(f"\nOptimization completed in {optimization_time:.2f} seconds")
    print(f"Optimized makespan: {result.makespan:.2f} seconds")
    print(f"Quality score: {result.quality_metrics.get('overall_quality', 0):.3f}")
    print(f"Resource utilization:")
    for resource, utilization in result.resource_utilization.items():
        print(f"  {resource}: {utilization:.1%}")
    
    # Save results if output path specified
    if args.output:
        results_data = {
            "optimization_time": optimization_time,
            "scheduling_result": result.to_dict(),
            "ml_pipeline_stats": result.metadata.get("ml_pipeline", {}),
            "recommendations": optimizer.get_pipeline_recommendations(ml_tasks)
        }
        save_results(results_data, args.output)


def cmd_create_config(args) -> None:
    """Create sample configuration file."""
    config = create_sample_config()
    
    try:
        with open(args.output, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Sample configuration saved to {args.output}")
        print("Edit this file to customize your ML pipeline optimization.")
    except Exception as e:
        print(f"Error creating config file: {e}")
        sys.exit(1)


def cmd_analyze(args) -> None:
    """Analyze ML pipeline configuration."""
    setup_logging(args.verbose)
    
    config = load_config(args.config)
    ml_tasks = parse_ml_tasks(config["ml_tasks"])
    
    # Create optimizer for analysis
    resources = config.get("system_resources", {})
    optimizer = QuantumMLPipelineOptimizer(
        max_cpu_cores=resources.get("max_cpu_cores", 8),
        max_gpu_memory=resources.get("max_gpu_memory", 8.0),
        max_system_memory=resources.get("max_system_memory", 32.0)
    )
    
    # Get recommendations
    recommendations = optimizer.get_pipeline_recommendations(ml_tasks)
    
    # Display analysis
    print(f"Pipeline Analysis for {len(ml_tasks)} tasks:")
    print(f"{'='*50}")
    
    # Task type distribution
    task_types = {}
    for task in ml_tasks:
        task_types[task.task_type] = task_types.get(task.task_type, 0) + 1
    
    print("Task Type Distribution:")
    for task_type, count in task_types.items():
        print(f"  {task_type}: {count}")
    
    # Resource requirements
    gpu_tasks = sum(1 for task in ml_tasks if task.gpu_required)
    total_memory = sum(task.memory_estimate for task in ml_tasks)
    total_duration = sum(task.expected_duration for task in ml_tasks)
    
    print(f"\nResource Requirements:")
    print(f"  GPU tasks: {gpu_tasks}/{len(ml_tasks)}")
    print(f"  Total memory estimate: {total_memory:.1f} GB")
    print(f"  Total sequential duration: {total_duration/3600:.1f} hours")
    
    # Dependencies
    tasks_with_deps = sum(1 for task in ml_tasks if task.dependencies)
    print(f"  Tasks with dependencies: {tasks_with_deps}/{len(ml_tasks)}")
    
    # Recommendations
    print(f"\nRecommendations:")
    print(f"{'='*50}")
    
    for category, recommendations_list in recommendations.items():
        if recommendations_list:
            print(f"{category.replace('_', ' ').title()}:")
            for rec in recommendations_list:
                print(f"  • {rec}")
    
    if args.output:
        analysis_data = {
            "task_count": len(ml_tasks),
            "task_types": task_types,
            "resource_summary": {
                "gpu_tasks": gpu_tasks,
                "total_memory_gb": total_memory,
                "total_duration_hours": total_duration / 3600,
                "tasks_with_dependencies": tasks_with_deps
            },
            "recommendations": recommendations
        }
        save_results(analysis_data, args.output)


def cmd_benchmark(args) -> None:
    """Benchmark different optimization algorithms."""
    setup_logging(args.verbose)
    
    config = load_config(args.config)
    ml_tasks = parse_ml_tasks(config["ml_tasks"])
    
    # Create optimizer
    resources = config.get("system_resources", {})
    optimizer = QuantumMLPipelineOptimizer(
        max_cpu_cores=resources.get("max_cpu_cores", 8),
        max_gpu_memory=resources.get("max_gpu_memory", 8.0),
        max_system_memory=resources.get("max_system_memory", 32.0)
    )
    
    # Algorithms to benchmark
    algorithms = [
        SchedulingAlgorithm.QUANTUM_ANNEALING,
        SchedulingAlgorithm.QAOA,
        SchedulingAlgorithm.HYBRID
    ]
    
    benchmark_results = []
    
    print(f"Benchmarking {len(algorithms)} algorithms on {len(ml_tasks)} tasks")
    print(f"{'='*60}")
    
    for algorithm in algorithms:
        print(f"\nTesting {algorithm.value}...")
        
        start_time = time.time()
        try:
            result = optimizer.optimize_ml_pipeline(
                ml_tasks=ml_tasks,
                algorithm=algorithm
            )
            execution_time = time.time() - start_time
            
            benchmark_result = {
                "algorithm": algorithm.value,
                "execution_time": execution_time,
                "makespan": result.makespan,
                "quality_score": result.quality_metrics.get("overall_quality", 0),
                "converged": result.metadata.get("optimization_converged", False),
                "iterations": result.metadata.get("optimization_iterations", 0),
                "success": True
            }
            
            print(f"  Execution time: {execution_time:.2f}s")
            print(f"  Makespan: {result.makespan:.2f}s")
            print(f"  Quality: {result.quality_metrics.get('overall_quality', 0):.3f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            benchmark_result = {
                "algorithm": algorithm.value,
                "execution_time": 0,
                "makespan": float('inf'),
                "quality_score": 0,
                "success": False,
                "error": str(e)
            }
        
        benchmark_results.append(benchmark_result)
    
    # Find best algorithm
    successful_results = [r for r in benchmark_results if r["success"]]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x["makespan"])
        print(f"\nBest Algorithm: {best_result['algorithm']}")
        print(f"Best Makespan: {best_result['makespan']:.2f}s")
    
    # Save benchmark results
    if args.output:
        benchmark_data = {
            "benchmark_timestamp": time.time(),
            "task_count": len(ml_tasks),
            "results": benchmark_results,
            "best_algorithm": best_result["algorithm"] if successful_results else None
        }
        save_results(benchmark_data, args.output)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Quantum-Inspired ML Pipeline Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample configuration
  python -m src.quantum_pipeline_cli create-config --output config.json
  
  # Optimize ML pipeline
  python -m src.quantum_pipeline_cli optimize --config config.json --output results.json
  
  # Analyze pipeline configuration
  python -m src.quantum_pipeline_cli analyze --config config.json
  
  # Benchmark algorithms
  python -m src.quantum_pipeline_cli benchmark --config config.json --output benchmark.json
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize ML pipeline")
    optimize_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Configuration file path"
    )
    optimize_parser.add_argument(
        "--output", "-o",
        help="Output file for results (JSON)"
    )
    
    # Create config command
    config_parser = subparsers.add_parser("create-config", help="Create sample configuration")
    config_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output configuration file path"
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze ML pipeline")
    analyze_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Configuration file path"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        help="Output file for analysis results (JSON)"
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark algorithms")
    benchmark_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Configuration file path"
    )
    benchmark_parser.add_argument(
        "--output", "-o",
        help="Output file for benchmark results (JSON)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "create-config":
        cmd_create_config(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()