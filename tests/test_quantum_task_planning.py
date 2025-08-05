"""
Comprehensive tests for quantum-inspired task planning functionality.

Tests the core quantum task planning components including task representation,
quantum optimization algorithms, and performance optimization features.
"""

import pytest
import numpy as np
import asyncio
import time
from datetime import timedelta
from typing import List, Dict, Any

from src.quantum_inspired.task_representation import (
    Task, TaskGraph, ResourceConstraint, TaskPriority, ResourceType, ResourceRequirement
)
from src.quantum_inspired.quantum_utils import (
    QuantumState, QuantumRegister, quantum_superposition, quantum_entanglement
)
from src.quantum_inspired.quantum_annealing_planner import (
    QuantumAnnealingPlanner, AnnealingParameters
)
from src.quantum_inspired.qaoa_task_optimizer import (
    QAOATaskOptimizer, QAOAParameters
)
from src.task_planning.task_scheduler import (
    TaskScheduler, SchedulingAlgorithm, SchedulingResult
)
from src.quantum_inspired.quantum_optimization_base import OptimizationObjective
from src.quantum_ml_integration import QuantumMLPipelineOptimizer, MLTask
from src.quantum_performance_optimizer import (
    QuantumPerformanceOptimizer, OptimizationConfig, CacheStrategy, ParallelizationMode
)


class TestQuantumUtils:
    """Test quantum simulation utilities."""
    
    def test_quantum_state_creation(self):
        """Test quantum state creation and normalization."""
        # Test with 2 qubits
        amplitudes = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        state = QuantumState(amplitudes, 2, is_normalized=False)
        
        assert state.num_qubits == 2
        assert len(state.amplitudes) == 4
        assert state.is_normalized
        
        # Check normalization
        norm = np.linalg.norm(state.amplitudes)
        assert abs(norm - 1.0) < 1e-10
    
    def test_quantum_state_measurement(self):
        """Test quantum state measurement."""
        # Create |00⟩ state
        amplitudes = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        state = QuantumState(amplitudes, 2)
        
        # Should always measure 0 (|00⟩)
        for _ in range(10):
            measurement = state.measure()
            assert measurement == 0
        
        # Test specific qubit measurement
        qubit_0_result = state.measure(qubit_index=0)
        assert qubit_0_result == 0
    
    def test_quantum_register_operations(self):
        """Test quantum register operations."""
        register = QuantumRegister(2)
        
        # Initially in |00⟩
        probabilities = register.get_state_probabilities()
        assert "00" in probabilities
        assert probabilities["00"] == 1.0
        
        # Apply Hadamard to first qubit
        register.apply_hadamard(0)
        probabilities = register.get_state_probabilities()
        
        # Should be in superposition |+0⟩ = (|00⟩ + |10⟩)/√2
        assert "00" in probabilities
        assert "10" in probabilities
        assert abs(probabilities["00"] - 0.5) < 1e-10
        assert abs(probabilities["10"] - 0.5) < 1e-10
    
    def test_quantum_superposition(self):
        """Test quantum superposition creation."""
        state = quantum_superposition(3)
        
        assert state.num_qubits == 3
        assert len(state.amplitudes) == 8
        
        # All amplitudes should be equal
        expected_amplitude = 1.0 / np.sqrt(8)
        for amplitude in state.amplitudes:
            assert abs(abs(amplitude) - expected_amplitude) < 1e-10
    
    def test_quantum_entanglement(self):
        """Test quantum entanglement creation."""
        # GHZ state
        state = quantum_entanglement(3, "ghz")
        
        assert state.num_qubits == 3
        
        # Should have only |000⟩ and |111⟩ components
        probabilities = [abs(amp)**2 for amp in state.amplitudes]
        
        # Check that only states 0 (|000⟩) and 7 (|111⟩) have non-zero probability
        assert probabilities[0] > 0.4  # |000⟩
        assert probabilities[7] > 0.4  # |111⟩
        
        for i in range(1, 7):
            assert probabilities[i] < 1e-10


class TestTaskRepresentation:
    """Test task representation components."""
    
    def test_task_creation(self):
        """Test task creation and properties."""
        task = Task(
            name="test_task",
            estimated_duration=timedelta(minutes=30),
            priority=TaskPriority.HIGH
        )
        
        assert task.name == "test_task"
        assert task.estimated_duration == timedelta(minutes=30)
        assert task.priority == TaskPriority.HIGH
        assert task.quantum_weight == 1.0
        assert len(task.dependencies) == 0
    
    def test_task_dependencies(self):
        """Test task dependency management."""
        task1 = Task(name="task1")
        task2 = Task(name="task2")
        
        task2.add_dependency(task1.task_id)
        
        assert task1.task_id in task2.dependencies
        assert task2.is_ready(set()) == False
        assert task2.is_ready({task1.task_id}) == True
    
    def test_resource_requirements(self):
        """Test resource requirement handling."""
        task = Task(name="gpu_task")
        
        gpu_req = ResourceRequirement(
            resource_type=ResourceType.GPU,
            amount=4.0,
            unit="GB"
        )
        
        task.add_resource_requirement(gpu_req)
        
        assert len(task.resource_requirements) == 1
        retrieved_req = task.get_resource_requirement(ResourceType.GPU)
        assert retrieved_req is not None
        assert retrieved_req.amount == 4.0
    
    def test_task_graph_creation(self):
        """Test task graph creation and validation."""
        task_graph = TaskGraph()
        
        # Create tasks
        task1 = Task(name="preprocess")
        task2 = Task(name="train")
        task3 = Task(name="evaluate")
        
        # Add dependencies
        task2.add_dependency(task1.task_id)
        task3.add_dependency(task2.task_id)
        
        # Add to graph
        task_graph.add_task(task1)
        task_graph.add_task(task2)
        task_graph.add_task(task3)
        
        assert len(task_graph.tasks) == 3
        
        # Test validation
        validation_errors = task_graph.validate_graph()
        assert len(validation_errors) == 0
        
        # Test critical path
        critical_path = task_graph.get_critical_path()
        assert len(critical_path) == 3
    
    def test_resource_constraints(self):
        """Test resource constraint management."""
        constraint = ResourceConstraint(
            resource_type=ResourceType.MEMORY,
            total_capacity=16.0,
            available_capacity=16.0,
            unit="GB"
        )
        
        assert constraint.can_allocate(8.0) == True
        assert constraint.can_allocate(20.0) == False
        
        # Test allocation
        success = constraint.allocate(8.0)
        assert success == True
        assert constraint.available_capacity == 8.0
        
        # Test deallocation
        constraint.deallocate(4.0)
        assert constraint.available_capacity == 12.0


class TestQuantumAnnealing:
    """Test quantum annealing planner."""
    
    def test_annealing_planner_creation(self):
        """Test annealing planner initialization."""
        params = AnnealingParameters(
            max_iterations=100,
            initial_temperature=50.0,
            final_temperature=0.1
        )
        
        planner = QuantumAnnealingPlanner(
            objective=OptimizationObjective.MINIMIZE_MAKESPAN,
            parameters=params
        )
        
        assert planner.objective == OptimizationObjective.MINIMIZE_MAKESPAN
        assert planner.annealing_params.max_iterations == 100
    
    def test_simple_task_optimization(self):
        """Test optimization of simple task graph."""
        # Create simple task graph
        task_graph = TaskGraph()
        
        task1 = Task(name="task1", estimated_duration=timedelta(minutes=10))
        task2 = Task(name="task2", estimated_duration=timedelta(minutes=15))
        task2.add_dependency(task1.task_id)
        
        task_graph.add_task(task1)
        task_graph.add_task(task2)
        
        # Create planner with minimal iterations for testing
        params = AnnealingParameters(max_iterations=10)
        planner = QuantumAnnealingPlanner(parameters=params)
        
        # Optimize
        result = planner.optimize(task_graph)
        
        assert result.solution is not None
        assert "schedule" in result.solution
        assert len(result.solution["schedule"]) == 2
        assert result.execution_time > 0
        
        # Check dependency constraint
        schedule = result.solution["schedule"]
        task1_completion = schedule[task1.task_id] + task1.estimated_duration.total_seconds()
        task2_start = schedule[task2.task_id]
        assert task2_start >= task1_completion - 1  # Allow small numerical error
    
    def test_temperature_scheduling(self):
        """Test annealing temperature scheduling."""
        params = AnnealingParameters(
            max_iterations=100,
            initial_temperature=100.0,
            final_temperature=1.0,
            cooling_schedule="exponential"
        )
        
        planner = QuantumAnnealingPlanner(parameters=params)
        
        # Test temperature calculation
        temp_0 = planner._get_annealing_temperature(0)
        temp_50 = planner._get_annealing_temperature(50)
        temp_99 = planner._get_annealing_temperature(99)
        
        assert temp_0 == 100.0
        assert temp_50 < temp_0
        assert temp_99 < temp_50
        assert temp_99 >= 1.0  # Should approach final temperature


class TestQAOAOptimizer:
    """Test QAOA task optimizer."""
    
    def test_qaoa_optimizer_creation(self):
        """Test QAOA optimizer initialization."""
        params = QAOAParameters(
            max_iterations=50,
            circuit_depth=3,
            measurement_shots=100
        )
        
        optimizer = QAOATaskOptimizer(parameters=params)
        
        assert optimizer.qaoa_params.circuit_depth == 3
        assert optimizer.qaoa_params.measurement_shots == 100
    
    def test_hamiltonian_building(self):
        """Test Hamiltonian construction."""
        task_graph = TaskGraph()
        
        task1 = Task(name="task1", estimated_duration=timedelta(minutes=10))
        task2 = Task(name="task2", estimated_duration=timedelta(minutes=15))
        task2.add_dependency(task1.task_id)
        
        task_graph.add_task(task1)
        task_graph.add_task(task2)
        
        params = QAOAParameters(max_iterations=10)
        optimizer = QAOATaskOptimizer(parameters=params)
        
        optimizer._build_hamiltonian(task_graph)
        
        assert len(optimizer.hamiltonian_terms) > 0
        
        # Should have cost terms and constraint terms
        cost_terms = [term for term in optimizer.hamiltonian_terms if term["type"] == "linear"]
        constraint_terms = [term for term in optimizer.hamiltonian_terms if term["type"] == "quadratic"]
        
        assert len(cost_terms) > 0
        assert len(constraint_terms) > 0
    
    def test_parameter_initialization(self):
        """Test variational parameter initialization."""
        params = QAOAParameters(circuit_depth=5)
        optimizer = QAOATaskOptimizer(parameters=params)
        
        optimizer._initialize_parameters()
        
        # Should have 2 * circuit_depth parameters (beta and gamma)
        assert len(optimizer.variational_parameters) == 10
        
        # All parameters should be within bounds
        for param in optimizer.variational_parameters:
            assert -np.pi <= param <= np.pi


class TestTaskScheduler:
    """Test main task scheduler interface."""
    
    def test_scheduler_creation(self):
        """Test task scheduler initialization."""
        scheduler = TaskScheduler(
            default_algorithm=SchedulingAlgorithm.QUANTUM_ANNEALING,
            default_objective=OptimizationObjective.MINIMIZE_MAKESPAN
        )
        
        assert scheduler.default_algorithm == SchedulingAlgorithm.QUANTUM_ANNEALING
        assert scheduler.default_objective == OptimizationObjective.MINIMIZE_MAKESPAN
        assert len(scheduler.scheduling_history) == 0
    
    def test_task_scheduling(self):
        """Test basic task scheduling functionality."""
        scheduler = TaskScheduler()
        
        # Create test tasks
        tasks = [
            Task(name="task1", estimated_duration=timedelta(minutes=10)),
            Task(name="task2", estimated_duration=timedelta(minutes=15))
        ]
        
        # Add dependency
        tasks[1].add_dependency(tasks[0].task_id)
        
        # Schedule tasks
        result = scheduler.schedule_tasks(
            tasks=tasks,
            algorithm=SchedulingAlgorithm.QUANTUM_ANNEALING
        )
        
        assert isinstance(result, SchedulingResult)
        assert len(result.schedule) == 2
        assert result.makespan > 0
        assert result.execution_time > 0
        assert "overall_quality" in result.quality_metrics
    
    def test_hybrid_scheduling(self):
        """Test hybrid scheduling approach."""
        scheduler = TaskScheduler()
        
        tasks = [
            Task(name="task1", estimated_duration=timedelta(minutes=5)),
            Task(name="task2", estimated_duration=timedelta(minutes=8)),
            Task(name="task3", estimated_duration=timedelta(minutes=12))
        ]
        
        result = scheduler.schedule_tasks(
            tasks=tasks,
            algorithm=SchedulingAlgorithm.HYBRID
        )
        
        assert isinstance(result, SchedulingResult)
        assert result.algorithm_used == SchedulingAlgorithm.HYBRID
        assert "hybrid_results" in result.metadata


class TestMLIntegration:
    """Test ML pipeline integration."""
    
    def test_ml_task_creation(self):
        """Test ML task creation."""
        ml_task = MLTask(
            task_id="train_model",
            task_type="training",
            model_path="models/autoencoder.h5",
            epochs=50,
            batch_size=32,
            gpu_required=True,
            memory_estimate=6.0
        )
        
        assert ml_task.task_id == "train_model"
        assert ml_task.task_type == "training"
        assert ml_task.gpu_required == True
        assert ml_task.memory_estimate == 6.0
    
    def test_ml_pipeline_optimizer(self):
        """Test ML pipeline optimizer."""
        optimizer = QuantumMLPipelineOptimizer(
            max_cpu_cores=4,
            max_gpu_memory=8.0,
            max_system_memory=16.0
        )
        
        # Create ML tasks
        ml_tasks = [
            MLTask(
                task_id="preprocess",
                task_type="preprocessing",
                expected_duration=300,
                memory_estimate=2.0
            ),
            MLTask(
                task_id="train",
                task_type="training",
                expected_duration=1800,
                gpu_required=True,
                memory_estimate=6.0,
                dependencies=["preprocess"]
            )
        ]
        
        result = optimizer.optimize_ml_pipeline(ml_tasks)
        
        assert isinstance(result, SchedulingResult)
        assert len(result.schedule) == 2
        assert "ml_pipeline" in result.metadata
    
    def test_training_pipeline_creation(self):
        """Test training pipeline creation."""
        optimizer = QuantumMLPipelineOptimizer()
        
        model_configs = [
            {"model_path": "model1.h5", "epochs": 30, "batch_size": 32},
            {"model_path": "model2.h5", "epochs": 50, "batch_size": 64}
        ]
        
        tasks = optimizer.create_training_pipeline(
            model_configs=model_configs,
            data_preprocessing_required=True,
            hyperparameter_tuning=False
        )
        
        # Should have preprocessing + 2 training + 2 evaluation tasks
        assert len(tasks) == 5
        
        # Check task types
        task_types = [task.task_type for task in tasks]
        assert "preprocessing" in task_types
        assert task_types.count("training") == 2
        assert task_types.count("evaluation") == 2
    
    def test_inference_pipeline_creation(self):
        """Test inference pipeline creation."""
        optimizer = QuantumMLPipelineOptimizer()
        
        model_paths = ["model1.h5", "model2.h5"]
        data_batches = ["batch1.csv", "batch2.csv"]
        
        tasks = optimizer.create_inference_pipeline(
            model_paths=model_paths,
            data_batches=data_batches,
            real_time=False
        )
        
        # Should have 2 models × 2 batches = 4 inference tasks
        assert len(tasks) == 4
        assert all(task.task_type == "inference" for task in tasks)


class TestPerformanceOptimizer:
    """Test quantum performance optimizer."""
    
    def test_performance_optimizer_creation(self):
        """Test performance optimizer initialization."""
        config = OptimizationConfig(
            cache_strategy=CacheStrategy.QUANTUM_INSPIRED,
            parallelization_mode=ParallelizationMode.ASYNC,
            cache_size_mb=50
        )
        
        optimizer = QuantumPerformanceOptimizer(config)
        
        assert optimizer.config.cache_strategy == CacheStrategy.QUANTUM_INSPIRED
        assert optimizer.config.cache_size_mb == 50
        assert optimizer.cache is not None
        assert optimizer.adaptive_selector is not None
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        optimizer = QuantumPerformanceOptimizer()
        
        # Create simple task graph
        task_graph = TaskGraph()
        task = Task(name="test_task")
        task_graph.add_task(task)
        
        # First optimization (cache miss)
        result1, metrics1 = optimizer.optimize(task_graph)
        
        # Second optimization (cache hit)
        result2, metrics2 = optimizer.optimize(task_graph)
        
        # Second call should be faster due to caching
        assert metrics2.optimization_time < metrics1.optimization_time
        assert metrics2.cache_hit_ratio > 0
    
    @pytest.mark.asyncio
    async def test_async_optimization(self):
        """Test asynchronous optimization."""
        optimizer = QuantumPerformanceOptimizer()
        
        task_graph = TaskGraph()
        task = Task(name="async_test_task")
        task_graph.add_task(task)
        
        result, metrics = await optimizer.optimize_async(task_graph)
        
        assert isinstance(result, SchedulingResult)
        assert isinstance(metrics.optimization_time, float)
        assert metrics.optimization_time > 0
    
    def test_batch_optimization(self):
        """Test batch optimization."""
        config = OptimizationConfig(
            parallelization_mode=ParallelizationMode.THREAD_POOL,
            max_parallel_workers=2
        )
        optimizer = QuantumPerformanceOptimizer(config)
        
        # Create multiple task graphs
        task_graphs = []
        for i in range(3):
            task_graph = TaskGraph()
            task = Task(name=f"batch_task_{i}")
            task_graph.add_task(task)
            task_graphs.append(task_graph)
        
        results = optimizer.optimize_batch(task_graphs, max_parallel=2)
        
        assert len(results) == 3
        for result, metrics in results:
            assert isinstance(result, SchedulingResult)
            assert isinstance(metrics.optimization_time, float)
    
    def test_performance_report(self):
        """Test performance report generation."""
        optimizer = QuantumPerformanceOptimizer()
        
        # Run a few optimizations
        task_graph = TaskGraph()
        task = Task(name="report_test_task")
        task_graph.add_task(task)
        
        for _ in range(3):
            optimizer.optimize(task_graph)
        
        report = optimizer.get_performance_report()
        
        assert "summary" in report
        assert "performance_trends" in report
        assert "cache_statistics" in report
        assert report["summary"]["total_optimizations"] == 3


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def test_complete_ml_workflow(self):
        """Test complete ML workflow optimization."""
        # Create ML pipeline optimizer
        ml_optimizer = QuantumMLPipelineOptimizer()
        
        # Create complex training pipeline
        model_configs = [
            {
                "model_path": "autoencoder_lstm.h5",
                "epochs": 100,
                "batch_size": 64,
                "gpu_required": True,
                "training_duration": 3600
            },
            {
                "model_path": "anomaly_detector.h5", 
                "epochs": 75,
                "batch_size": 32,
                "gpu_required": True,
                "training_duration": 2700
            }
        ]
        
        ml_tasks = ml_optimizer.create_training_pipeline(
            model_configs=model_configs,
            data_preprocessing_required=True,
            hyperparameter_tuning=True
        )
        
        # Optimize with performance optimizer
        perf_optimizer = QuantumPerformanceOptimizer()
        
        # Convert ML tasks to task graph
        quantum_tasks = ml_optimizer._convert_ml_tasks(ml_tasks)
        task_graph = TaskGraph()
        for task in quantum_tasks:
            task_graph.add_task(task)
        
        result, metrics = perf_optimizer.optimize(task_graph)
        
        assert isinstance(result, SchedulingResult)
        assert result.makespan > 0
        assert len(result.schedule) == len(ml_tasks)
        assert metrics.solution_quality > 0
    
    def test_quantum_annealing_vs_qaoa(self):
        """Test comparison between quantum annealing and QAOA."""
        # Create moderately complex task graph
        task_graph = TaskGraph()
        
        # Create tasks with dependencies
        tasks = []
        for i in range(5):
            task = Task(
                name=f"task_{i}",
                estimated_duration=timedelta(minutes=10 + i * 5),
                priority=TaskPriority.MEDIUM
            )
            
            # Add some dependencies
            if i > 0:
                task.add_dependency(tasks[i-1].task_id)
            if i > 1:
                task.add_dependency(tasks[i-2].task_id)
            
            tasks.append(task)
            task_graph.add_task(task)
        
        # Add resource constraints
        cpu_constraint = ResourceConstraint(
            resource_type=ResourceType.CPU,
            total_capacity=8.0,
            available_capacity=8.0
        )
        task_graph.add_resource_constraint(cpu_constraint)
        
        # Test both algorithms
        scheduler = TaskScheduler()
        
        result_annealing = scheduler.schedule_tasks(
            tasks=tasks,
            algorithm=SchedulingAlgorithm.QUANTUM_ANNEALING
        )
        
        result_qaoa = scheduler.schedule_tasks(
            tasks=tasks,
            algorithm=SchedulingAlgorithm.QAOA
        )
        
        # Both should produce valid schedules
        assert len(result_annealing.schedule) == 5
        assert len(result_qaoa.schedule) == 5
        
        # Quality should be reasonable
        assert result_annealing.quality_metrics["overall_quality"] > 0.3
        assert result_qaoa.quality_metrics["overall_quality"] > 0.3
    
    def test_adaptive_algorithm_selection(self):
        """Test adaptive algorithm selection."""
        config = OptimizationConfig(enable_adaptive_algorithms=True)
        optimizer = QuantumPerformanceOptimizer(config)
        
        # Create task graph
        task_graph = TaskGraph()
        for i in range(3):
            task = Task(name=f"adaptive_task_{i}")
            task_graph.add_task(task)
        
        # Run multiple optimizations to train adaptive selector
        results = []
        for _ in range(5):
            result, metrics = optimizer.optimize(task_graph)
            results.append((result, metrics))
        
        # Check that adaptive selector has learned something
        rankings = optimizer.adaptive_selector.get_algorithm_rankings()
        assert len(rankings) > 0
        
        # All results should be valid
        for result, metrics in results:
            assert isinstance(result, SchedulingResult)
            assert len(result.schedule) == 3


# Performance benchmarks
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_large_task_graph_optimization(self):
        """Test optimization of large task graphs."""
        # Create large task graph (50 tasks)
        task_graph = TaskGraph()
        tasks = []
        
        for i in range(50):
            task = Task(
                name=f"large_task_{i}",
                estimated_duration=timedelta(minutes=np.random.randint(5, 30)),
                priority=TaskPriority(np.random.randint(1, 4))
            )
            
            # Add random dependencies
            if i > 0 and np.random.random() < 0.3:  # 30% chance of dependency
                dep_idx = np.random.randint(0, i)
                task.add_dependency(tasks[dep_idx].task_id)
            
            tasks.append(task)
            task_graph.add_task(task)
        
        # Add resource constraints
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU]:
            constraint = ResourceConstraint(
                resource_type=resource_type,
                total_capacity=np.random.uniform(10, 100),
                available_capacity=np.random.uniform(10, 100)
            )
            task_graph.add_resource_constraint(constraint)
        
        # Test with performance optimizer
        config = OptimizationConfig(
            cache_strategy=CacheStrategy.QUANTUM_INSPIRED,
            enable_adaptive_algorithms=True
        )
        optimizer = QuantumPerformanceOptimizer(config)
        
        start_time = time.time()
        result, metrics = optimizer.optimize(task_graph)
        end_time = time.time()
        
        # Should complete within reasonable time (< 60 seconds)
        assert end_time - start_time < 60.0
        assert len(result.schedule) == 50
        assert result.quality_metrics["overall_quality"] > 0.0
    
    @pytest.mark.asyncio
    async def test_concurrent_optimizations(self):
        """Test concurrent optimizations."""
        optimizer = QuantumPerformanceOptimizer()
        
        # Create multiple task graphs
        task_graphs = []
        for i in range(5):
            task_graph = TaskGraph()
            for j in range(5):
                task = Task(name=f"concurrent_task_{i}_{j}")
                task_graph.add_task(task)
            task_graphs.append(task_graph)
        
        # Run concurrent optimizations
        start_time = time.time()
        tasks = [optimizer.optimize_async(tg) for tg in task_graphs]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Should be faster than sequential execution
        sequential_time_estimate = len(task_graphs) * 2.0  # Rough estimate
        concurrent_time = end_time - start_time
        
        assert concurrent_time < sequential_time_estimate
        assert len(results) == 5
        
        for result, metrics in results:
            assert isinstance(result, SchedulingResult)
            assert len(result.schedule) == 5


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])