"""
Quantum-Inspired ML Pipeline Example

Demonstrates how to use the quantum-inspired task planning system to optimize
machine learning pipelines including data preprocessing, model training,
hyperparameter tuning, and evaluation tasks.
"""

import asyncio
import logging
import time
from datetime import timedelta
from pathlib import Path

from src.quantum_ml_integration import QuantumMLPipelineOptimizer, MLTask
from src.quantum_inspired.task_representation import TaskPriority
from src.task_planning.task_scheduler import SchedulingAlgorithm
from src.quantum_inspired.quantum_optimization_base import OptimizationObjective
from src.quantum_performance_optimizer import (
    QuantumPerformanceOptimizer, OptimizationConfig, CacheStrategy, ParallelizationMode
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_anomaly_detection_pipeline():
    """Create a comprehensive anomaly detection ML pipeline."""
    logger.info("Creating anomaly detection ML pipeline...")
    
    # Data preprocessing tasks
    preprocessing_tasks = [
        MLTask(
            task_id="load_sensor_data",
            task_type="data_loading",
            data_path="data/raw/sensor_data.csv",
            expected_duration=120,  # 2 minutes
            priority=TaskPriority.HIGH,
            memory_estimate=2.0,
            cpu_cores=2
        ),
        MLTask(
            task_id="validate_data",
            task_type="validation",
            data_path="data/raw/sensor_data.csv",
            expected_duration=180,  # 3 minutes
            priority=TaskPriority.HIGH,
            memory_estimate=1.5,
            cpu_cores=1,
            dependencies=["load_sensor_data"]
        ),
        MLTask(
            task_id="feature_engineering",
            task_type="preprocessing",
            data_path="data/raw/sensor_data.csv",
            expected_duration=600,  # 10 minutes
            priority=TaskPriority.HIGH,
            memory_estimate=4.0,
            cpu_cores=4,
            dependencies=["validate_data"]
        ),
        MLTask(
            task_id="data_splitting",
            task_type="preprocessing",
            expected_duration=60,  # 1 minute
            priority=TaskPriority.MEDIUM,
            memory_estimate=2.0,
            cpu_cores=1,
            dependencies=["feature_engineering"]
        )
    ]
    
    # Model training tasks
    training_tasks = [
        MLTask(
            task_id="train_lstm_autoencoder",
            task_type="training",
            model_path="models/lstm_autoencoder.h5",
            data_path="data/processed/train_data.csv",
            epochs=100,
            batch_size=64,
            expected_duration=3600,  # 1 hour
            priority=TaskPriority.CRITICAL,
            gpu_required=True,
            memory_estimate=8.0,
            cpu_cores=4,
            dependencies=["data_splitting"]
        ),
        MLTask(
            task_id="train_isolation_forest",
            task_type="training",
            model_path="models/isolation_forest.pkl",
            data_path="data/processed/train_data.csv",
            expected_duration=1200,  # 20 minutes
            priority=TaskPriority.HIGH,
            gpu_required=False,
            memory_estimate=4.0,
            cpu_cores=6,
            dependencies=["data_splitting"]
        ),
        MLTask(
            task_id="train_one_class_svm",
            task_type="training",
            model_path="models/one_class_svm.pkl",
            data_path="data/processed/train_data.csv",
            expected_duration=2400,  # 40 minutes
            priority=TaskPriority.MEDIUM,
            gpu_required=False,
            memory_estimate=6.0,
            cpu_cores=4,
            dependencies=["data_splitting"]
        )
    ]
    
    # Hyperparameter tuning tasks
    tuning_tasks = [
        MLTask(
            task_id="tune_lstm_hyperparams",
            task_type="hyperparameter_tuning",
            model_path="models/lstm_autoencoder.h5",
            expected_duration=7200,  # 2 hours
            priority=TaskPriority.MEDIUM,
            gpu_required=True,
            memory_estimate=10.0,
            cpu_cores=4,
            dependencies=["train_lstm_autoencoder"]
        ),
        MLTask(
            task_id="tune_isolation_forest_hyperparams",
            task_type="hyperparameter_tuning",
            model_path="models/isolation_forest.pkl",
            expected_duration=3600,  # 1 hour
            priority=TaskPriority.LOW,
            gpu_required=False,
            memory_estimate=6.0,
            cpu_cores=8,
            dependencies=["train_isolation_forest"]
        )
    ]
    
    # Model evaluation tasks
    evaluation_tasks = [
        MLTask(
            task_id="evaluate_lstm_autoencoder",
            task_type="evaluation",
            model_path="models/lstm_autoencoder.h5",
            data_path="data/processed/test_data.csv",
            expected_duration=300,  # 5 minutes
            priority=TaskPriority.HIGH,
            gpu_required=True,
            memory_estimate=4.0,
            cpu_cores=2,
            dependencies=["tune_lstm_hyperparams"]
        ),
        MLTask(
            task_id="evaluate_isolation_forest",
            task_type="evaluation",
            model_path="models/isolation_forest.pkl",
            data_path="data/processed/test_data.csv",
            expected_duration=180,  # 3 minutes
            priority=TaskPriority.HIGH,
            gpu_required=False,
            memory_estimate=2.0,
            cpu_cores=2,
            dependencies=["tune_isolation_forest_hyperparams"]
        ),
        MLTask(
            task_id="evaluate_one_class_svm",
            task_type="evaluation",
            model_path="models/one_class_svm.pkl",
            data_path="data/processed/test_data.csv",
            expected_duration=240,  # 4 minutes
            priority=TaskPriority.MEDIUM,
            gpu_required=False,
            memory_estimate=3.0,
            cpu_cores=2,
            dependencies=["train_one_class_svm"]
        )
    ]
    
    # Model comparison and ensemble
    final_tasks = [
        MLTask(
            task_id="compare_models",
            task_type="analysis",
            expected_duration=600,  # 10 minutes
            priority=TaskPriority.HIGH,
            memory_estimate=2.0,
            cpu_cores=2,
            dependencies=[
                "evaluate_lstm_autoencoder",
                "evaluate_isolation_forest", 
                "evaluate_one_class_svm"
            ]
        ),
        MLTask(
            task_id="create_ensemble",
            task_type="ensemble",
            expected_duration=900,  # 15 minutes
            priority=TaskPriority.CRITICAL,
            gpu_required=True,
            memory_estimate=8.0,
            cpu_cores=4,
            dependencies=["compare_models"]
        ),
        MLTask(
            task_id="final_evaluation",
            task_type="evaluation",
            expected_duration=300,  # 5 minutes
            priority=TaskPriority.CRITICAL,
            gpu_required=True,
            memory_estimate=4.0,
            cpu_cores=2,
            dependencies=["create_ensemble"]
        )
    ]
    
    # Combine all tasks
    all_tasks = (preprocessing_tasks + training_tasks + 
                tuning_tasks + evaluation_tasks + final_tasks)
    
    logger.info(f"Created pipeline with {len(all_tasks)} tasks")
    return all_tasks


def demonstrate_basic_optimization():
    """Demonstrate basic quantum-inspired optimization."""
    logger.info("=== Basic Quantum-Inspired Optimization ===")
    
    # Create ML pipeline optimizer
    ml_optimizer = QuantumMLPipelineOptimizer(
        max_cpu_cores=16,
        max_gpu_memory=12.0,
        max_system_memory=64.0,
        default_algorithm=SchedulingAlgorithm.QUANTUM_ANNEALING
    )
    
    # Create tasks
    ml_tasks = create_anomaly_detection_pipeline()
    
    # Optimize with quantum annealing  
    logger.info("Optimizing with Quantum Annealing...")
    start_time = time.time()
    
    result_annealing = ml_optimizer.optimize_ml_pipeline(
        ml_tasks=ml_tasks,
        objective=OptimizationObjective.MINIMIZE_MAKESPAN,
        algorithm=SchedulingAlgorithm.QUANTUM_ANNEALING
    )
    
    annealing_time = time.time() - start_time
    
    logger.info(f"Quantum Annealing Results:")
    logger.info(f"  Optimization time: {annealing_time:.2f}s")
    logger.info(f"  Makespan: {result_annealing.makespan/3600:.2f} hours")
    logger.info(f"  Quality score: {result_annealing.quality_metrics['overall_quality']:.3f}")
    logger.info(f"  Resource utilization: {result_annealing.resource_utilization}")
    
    # Optimize with QAOA
    logger.info("\nOptimizing with QAOA...")
    start_time = time.time()
    
    result_qaoa = ml_optimizer.optimize_ml_pipeline(
        ml_tasks=ml_tasks,
        objective=OptimizationObjective.MINIMIZE_MAKESPAN,
        algorithm=SchedulingAlgorithm.QAOA
    )
    
    qaoa_time = time.time() - start_time
    
    logger.info(f"QAOA Results:")
    logger.info(f"  Optimization time: {qaoa_time:.2f}s")
    logger.info(f"  Makespan: {result_qaoa.makespan/3600:.2f} hours")
    logger.info(f"  Quality score: {result_qaoa.quality_metrics['overall_quality']:.3f}")
    logger.info(f"  Resource utilization: {result_qaoa.resource_utilization}")
    
    # Compare results
    logger.info("\nComparison:")
    makespan_improvement = (result_annealing.makespan - result_qaoa.makespan) / result_annealing.makespan * 100
    logger.info(f"  Makespan improvement (QAOA vs Annealing): {makespan_improvement:.1f}%")
    
    return result_annealing, result_qaoa


def demonstrate_performance_optimization():
    """Demonstrate advanced performance optimization features."""
    logger.info("\n=== Advanced Performance Optimization ===")
    
    # Create performance optimizer with advanced features
    config = OptimizationConfig(
        cache_strategy=CacheStrategy.QUANTUM_INSPIRED,
        parallelization_mode=ParallelizationMode.ASYNC,
        cache_size_mb=200,
        max_parallel_workers=4,
        enable_adaptive_algorithms=True,
        enable_quantum_acceleration=True,
        auto_scaling=True
    )
    
    perf_optimizer = QuantumPerformanceOptimizer(config)
    
    # Create ML tasks
    ml_tasks = create_anomaly_detection_pipeline()
    
    # Convert to task graph
    ml_pipeline_optimizer = QuantumMLPipelineOptimizer()
    quantum_tasks = ml_pipeline_optimizer._convert_ml_tasks(ml_tasks)
    
    from src.quantum_inspired.task_representation import TaskGraph
    task_graph = TaskGraph()
    for task in quantum_tasks:
        task_graph.add_task(task)
    
    # First optimization (cache miss)
    logger.info("First optimization (cache miss)...")
    result1, metrics1 = perf_optimizer.optimize(task_graph)
    
    logger.info(f"First optimization:")
    logger.info(f"  Time: {metrics1.optimization_time:.2f}s")
    logger.info(f"  Cache hit ratio: {metrics1.cache_hit_ratio:.3f}")
    logger.info(f"  Quality: {metrics1.solution_quality:.3f}")
    
    # Second optimization (cache hit)
    logger.info("\nSecond optimization (cache hit)...")
    result2, metrics2 = perf_optimizer.optimize(task_graph)
    
    logger.info(f"Second optimization:")
    logger.info(f"  Time: {metrics2.optimization_time:.2f}s")
    logger.info(f"  Cache hit ratio: {metrics2.cache_hit_ratio:.3f}")
    logger.info(f"  Quality: {metrics2.solution_quality:.3f}")
    
    # Performance report
    report = perf_optimizer.get_performance_report()
    logger.info(f"\nPerformance Report:")
    logger.info(f"  Total optimizations: {report['summary']['total_optimizations']}")
    logger.info(f"  Average optimization time: {report['summary']['average_optimization_time']:.2f}s")
    logger.info(f"  Cache utilization: {report['cache_statistics']['utilization']:.1%}")
    logger.info(f"  Algorithm rankings: {report['algorithm_rankings']}")
    
    return perf_optimizer, report


async def demonstrate_async_optimization():
    """Demonstrate asynchronous optimization capabilities."""
    logger.info("\n=== Asynchronous Optimization ===")
    
    # Create performance optimizer
    config = OptimizationConfig(
        parallelization_mode=ParallelizationMode.ASYNC,
        max_parallel_workers=3
    )
    perf_optimizer = QuantumPerformanceOptimizer(config)
    
    # Create multiple smaller task graphs
    task_graphs = []
    for i in range(3):
        from src.quantum_inspired.task_representation import TaskGraph, Task
        task_graph = TaskGraph()
        
        # Create smaller pipeline for each graph
        for j in range(5):
            task = Task(
                name=f"async_task_{i}_{j}",
                estimated_duration=timedelta(minutes=5 + j * 2)
            )
            if j > 0:
                # Add dependency to previous task
                prev_task_id = f"async_task_{i}_{j-1}"
                for existing_task in task_graph.tasks.values():
                    if existing_task.name == prev_task_id:
                        task.add_dependency(existing_task.task_id)
                        break
            task_graph.add_task(task)
        
        task_graphs.append(task_graph)
    
    # Run async optimizations
    logger.info("Running concurrent optimizations...")
    start_time = time.time()
    
    tasks = [perf_optimizer.optimize_async(tg) for tg in task_graphs]
    results = await asyncio.gather(*tasks)
    
    async_time = time.time() - start_time
    
    logger.info(f"Concurrent optimization completed in {async_time:.2f}s")
    
    for i, (result, metrics) in enumerate(results):
        logger.info(f"  Pipeline {i+1}: {result.makespan:.1f}s makespan, "
                   f"{metrics.solution_quality:.3f} quality")
    
    return results


def demonstrate_ml_pipeline_recommendations():
    """Demonstrate ML pipeline analysis and recommendations."""
    logger.info("\n=== ML Pipeline Analysis & Recommendations ===")
    
    ml_optimizer = QuantumMLPipelineOptimizer()
    ml_tasks = create_anomaly_detection_pipeline()
    
    # Get recommendations
    recommendations = ml_optimizer.get_pipeline_recommendations(ml_tasks)
    
    logger.info("Pipeline Analysis:")
    logger.info(f"  Total tasks: {len(ml_tasks)}")
    
    # Count task types
    task_types = {}
    for task in ml_tasks:
        task_types[task.task_type] = task_types.get(task.task_type, 0) + 1
    
    logger.info("  Task distribution:")
    for task_type, count in task_types.items():
        logger.info(f"    {task_type}: {count}")
    
    # Resource analysis
    gpu_tasks = sum(1 for task in ml_tasks if task.gpu_required)
    total_memory = sum(task.memory_estimate for task in ml_tasks)
    
    logger.info(f"  GPU tasks: {gpu_tasks}/{len(ml_tasks)}")
    logger.info(f"  Total memory estimate: {total_memory:.1f} GB")
    
    logger.info("\nRecommendations:")
    for category, rec_list in recommendations.items():
        if rec_list:
            logger.info(f"  {category.replace('_', ' ').title()}:")
            for rec in rec_list:
                logger.info(f"    • {rec}")
    
    return recommendations


def save_results_to_file(results, filename):
    """Save optimization results to JSON file."""
    import json
    from datetime import datetime
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "quantum_annealing_result": results[0].to_dict() if results else None,
        "qaoa_result": results[1].to_dict() if len(results) > 1 else None,
        "performance_comparison": {
            "makespan_improvement_percent": (
                (results[0].makespan - results[1].makespan) / results[0].makespan * 100
                if len(results) > 1 else 0
            )
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    logger.info(f"Results saved to {filename}")


async def main():
    """Main demonstration function."""
    logger.info("🚀 Quantum-Inspired ML Pipeline Optimization Demo")
    logger.info("=" * 60)
    
    try:
        # Basic optimization demonstration
        basic_results = demonstrate_basic_optimization()
        
        # Performance optimization demonstration
        perf_optimizer, report = demonstrate_performance_optimization()
        
        # Async optimization demonstration
        async_results = await demonstrate_async_optimization()
        
        # ML pipeline recommendations
        recommendations = demonstrate_ml_pipeline_recommendations()
        
        # Save results
        save_results_to_file(basic_results, "quantum_ml_optimization_results.json")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Demo completed successfully!")
        logger.info("\nKey Takeaways:")
        logger.info("• Quantum-inspired algorithms provide efficient task scheduling")
        logger.info("• Caching significantly improves repeated optimizations")
        logger.info("• Async optimization enables concurrent pipeline processing")
        logger.info("• Adaptive algorithm selection learns from performance history")
        logger.info("• Performance optimization scales to complex ML workflows")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())