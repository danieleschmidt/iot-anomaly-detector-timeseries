"""
Quantum-ML Pipeline Integration

Integrates quantum-inspired task planning with the existing ML anomaly detection
pipeline. Provides optimized scheduling for ML training, inference, and data
processing tasks.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from .quantum_inspired.task_representation import (
    Task, TaskGraph, ResourceConstraint, TaskPriority, ResourceType, ResourceRequirement
)
from .task_planning.task_scheduler import TaskScheduler, SchedulingAlgorithm, SchedulingResult
from .quantum_inspired.quantum_optimization_base import OptimizationObjective


@dataclass
class MLTask:
    """
    ML-specific task representation.
    
    Attributes:
        task_type: Type of ML task (training, inference, preprocessing, etc.)
        model_path: Path to ML model file
        data_path: Path to data file
        batch_size: Batch size for processing
        epochs: Number of training epochs (if applicable)
        gpu_required: Whether GPU is required
        memory_estimate: Estimated memory usage in GB
        cpu_cores: Number of CPU cores needed
        expected_duration: Expected task duration
        priority: Task priority level
        dependencies: List of dependent task IDs
    """
    task_id: str
    task_type: str
    model_path: Optional[str] = None
    data_path: Optional[str] = None
    batch_size: int = 32
    epochs: int = 1
    gpu_required: bool = False
    memory_estimate: float = 1.0
    cpu_cores: int = 1
    expected_duration: int = 300  # seconds
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


class QuantumMLPipelineOptimizer:
    """
    Quantum-inspired optimizer for ML pipeline tasks.
    
    Optimizes the execution of ML training, inference, and data processing
    tasks using quantum-inspired scheduling algorithms.
    """
    
    def __init__(
        self,
        max_cpu_cores: int = 8,
        max_gpu_memory: float = 8.0,  # GB
        max_system_memory: float = 32.0,  # GB
        default_algorithm: SchedulingAlgorithm = SchedulingAlgorithm.QUANTUM_ANNEALING
    ):
        """
        Initialize ML pipeline optimizer.
        
        Args:
            max_cpu_cores: Maximum available CPU cores
            max_gpu_memory: Maximum GPU memory in GB
            max_system_memory: Maximum system memory in GB
            default_algorithm: Default scheduling algorithm
        """
        self.max_cpu_cores = max_cpu_cores
        self.max_gpu_memory = max_gpu_memory
        self.max_system_memory = max_system_memory
        
        self.task_scheduler = TaskScheduler(default_algorithm=default_algorithm)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # ML-specific resource constraints
        self.resource_constraints = self._create_resource_constraints()
        
        # Performance tracking
        self.execution_history: List[Dict[str, Any]] = []
    
    def _create_resource_constraints(self) -> List[ResourceConstraint]:
        """Create resource constraints for ML workloads."""
        constraints = [
            ResourceConstraint(
                resource_type=ResourceType.CPU,
                total_capacity=float(self.max_cpu_cores),
                available_capacity=float(self.max_cpu_cores),
                unit="cores"
            ),
            ResourceConstraint(
                resource_type=ResourceType.MEMORY,
                total_capacity=self.max_system_memory,
                available_capacity=self.max_system_memory,
                unit="GB"
            ),
            ResourceConstraint(
                resource_type=ResourceType.GPU,
                total_capacity=self.max_gpu_memory,
                available_capacity=self.max_gpu_memory,
                unit="GB"
            )
        ]
        return constraints
    
    def optimize_ml_pipeline(
        self,
        ml_tasks: List[MLTask],
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_MAKESPAN,
        algorithm: Optional[SchedulingAlgorithm] = None
    ) -> SchedulingResult:
        """
        Optimize ML pipeline execution using quantum-inspired scheduling.
        
        Args:
            ml_tasks: List of ML tasks to optimize
            objective: Optimization objective
            algorithm: Scheduling algorithm to use
            
        Returns:
            Optimized scheduling result
        """
        self.logger.info(f"Optimizing ML pipeline with {len(ml_tasks)} tasks")
        
        # Convert ML tasks to quantum task representation
        quantum_tasks = self._convert_ml_tasks(ml_tasks)
        
        # Schedule tasks using quantum algorithms
        result = self.task_scheduler.schedule_tasks(
            tasks=quantum_tasks,
            resource_constraints=self.resource_constraints,
            algorithm=algorithm,
            objective=objective
        )
        
        # Add ML-specific metadata
        result.metadata["ml_pipeline"] = {
            "total_ml_tasks": len(ml_tasks),
            "task_types": self._count_task_types(ml_tasks),
            "gpu_tasks": sum(1 for task in ml_tasks if task.gpu_required),
            "estimated_total_duration": sum(task.expected_duration for task in ml_tasks),
            "optimization_efficiency": self._calculate_optimization_efficiency(result, ml_tasks)
        }
        
        # Store execution history
        self.execution_history.append({
            "timestamp": datetime.now(),
            "num_tasks": len(ml_tasks),
            "makespan": result.makespan,
            "algorithm": result.algorithm_used.value,
            "quality": result.quality_metrics.get("overall_quality", 0)
        })
        
        return result
    
    def _convert_ml_tasks(self, ml_tasks: List[MLTask]) -> List[Task]:
        """Convert ML tasks to quantum task representation."""
        quantum_tasks = []
        
        for ml_task in ml_tasks:
            # Create quantum task
            task = Task(
                task_id=ml_task.task_id,
                name=f"{ml_task.task_type}_{ml_task.task_id}",
                description=f"ML {ml_task.task_type} task",
                priority=ml_task.priority,
                estimated_duration=timedelta(seconds=ml_task.expected_duration),
                duration_uncertainty=0.2,  # 20% uncertainty for ML tasks
                quantum_weight=self._calculate_quantum_weight(ml_task)
            )
            
            # Add dependencies
            for dep_id in ml_task.dependencies:
                task.add_dependency(dep_id)
            
            # Add resource requirements
            self._add_resource_requirements(task, ml_task)
            
            # Add ML-specific metadata
            task.metadata.update({
                "ml_task_type": ml_task.task_type,
                "model_path": ml_task.model_path,
                "data_path": ml_task.data_path,
                "batch_size": ml_task.batch_size,
                "epochs": ml_task.epochs,
                "gpu_required": ml_task.gpu_required
            })
            task.metadata.update(ml_task.metadata)
            
            quantum_tasks.append(task)
        
        return quantum_tasks
    
    def _calculate_quantum_weight(self, ml_task: MLTask) -> float:
        """Calculate quantum weight based on ML task characteristics."""
        weight = 1.0
        
        # Higher weight for GPU tasks (they're more constrained)
        if ml_task.gpu_required:
            weight *= 2.0
        
        # Higher weight for longer tasks
        if ml_task.expected_duration > 1800:  # 30 minutes
            weight *= 1.5
        
        # Higher weight for high-priority tasks
        weight *= ml_task.priority.value / 2.0
        
        # Higher weight for training tasks (usually more important)
        if ml_task.task_type in ["training", "hyperparameter_tuning"]:
            weight *= 1.3
        
        return weight
    
    def _add_resource_requirements(self, task: Task, ml_task: MLTask) -> None:
        """Add resource requirements to quantum task."""
        # CPU requirement
        task.add_resource_requirement(ResourceRequirement(
            resource_type=ResourceType.CPU,
            amount=float(ml_task.cpu_cores),
            unit="cores"
        ))
        
        # Memory requirement
        task.add_resource_requirement(ResourceRequirement(
            resource_type=ResourceType.MEMORY,
            amount=ml_task.memory_estimate,
            unit="GB"
        ))
        
        # GPU requirement (if needed)
        if ml_task.gpu_required:
            gpu_memory = self._calculate_gpu_memory_requirement(ml_task)
            task.add_resource_requirement(ResourceRequirement(
                resource_type=ResourceType.GPU,
                amount=gpu_memory,
                unit="GB"
            ))
    
    def _calculate_gpu_memory_requirement(self, ml_task: MLTask) -> float:
        """Calculate GPU memory requirement based on task characteristics."""
        base_memory = 1.0  # Base GPU memory in GB
        
        # Scale with batch size
        batch_factor = ml_task.batch_size / 32.0  # Normalize to batch size 32
        base_memory *= max(batch_factor, 0.5)
        
        # Training tasks need more memory
        if ml_task.task_type == "training":
            base_memory *= 1.5
        
        # Cap at available GPU memory
        return min(base_memory, self.max_gpu_memory)
    
    def _count_task_types(self, ml_tasks: List[MLTask]) -> Dict[str, int]:
        """Count different types of ML tasks."""
        type_counts = {}
        for task in ml_tasks:
            type_counts[task.task_type] = type_counts.get(task.task_type, 0) + 1
        return type_counts
    
    def _calculate_optimization_efficiency(
        self, 
        result: SchedulingResult, 
        ml_tasks: List[MLTask]
    ) -> float:
        """Calculate optimization efficiency."""
        if not result.schedule:
            return 0.0
        
        # Compare optimized makespan to sequential execution
        sequential_duration = sum(task.expected_duration for task in ml_tasks)
        optimized_duration = result.makespan
        
        if sequential_duration > 0:
            return min(sequential_duration / optimized_duration, 5.0)  # Cap at 5x improvement
        
        return 1.0
    
    def create_training_pipeline(
        self,
        model_configs: List[Dict[str, Any]],
        data_preprocessing_required: bool = True,
        hyperparameter_tuning: bool = False
    ) -> List[MLTask]:
        """
        Create ML training pipeline tasks.
        
        Args:
            model_configs: List of model configurations
            data_preprocessing_required: Whether data preprocessing is needed
            hyperparameter_tuning: Whether to include hyperparameter tuning
            
        Returns:
            List of ML tasks for training pipeline
        """
        tasks = []
        task_id_counter = 0
        
        # Data preprocessing task (if required)
        preprocessing_task_id = None
        if data_preprocessing_required:
            preprocessing_task_id = f"preprocess_{task_id_counter}"
            tasks.append(MLTask(
                task_id=preprocessing_task_id,
                task_type="preprocessing",
                data_path="data/raw/sensor_data.csv",
                expected_duration=300,  # 5 minutes
                priority=TaskPriority.HIGH,
                memory_estimate=2.0,
                cpu_cores=2
            ))
            task_id_counter += 1
        
        # Training tasks for each model
        for i, config in enumerate(model_configs):
            dependencies = [preprocessing_task_id] if preprocessing_task_id else []
            
            # Hyperparameter tuning (if enabled)
            if hyperparameter_tuning:
                tuning_task_id = f"hyperparameter_tuning_{i}_{task_id_counter}"
                tasks.append(MLTask(
                    task_id=tuning_task_id,
                    task_type="hyperparameter_tuning",
                    model_path=config.get("model_path", f"models/model_{i}.h5"),
                    data_path=config.get("data_path", "data/processed/train_data.csv"),
                    epochs=config.get("tuning_epochs", 20),
                    batch_size=config.get("batch_size", 32),
                    expected_duration=config.get("tuning_duration", 1800),  # 30 minutes
                    priority=TaskPriority.MEDIUM,
                    gpu_required=config.get("gpu_required", True),
                    memory_estimate=config.get("memory_estimate", 4.0),
                    dependencies=dependencies
                ))
                dependencies = [tuning_task_id]
                task_id_counter += 1
            
            # Training task
            training_task_id = f"training_{i}_{task_id_counter}"
            tasks.append(MLTask(
                task_id=training_task_id,
                task_type="training",
                model_path=config.get("model_path", f"models/model_{i}.h5"),
                data_path=config.get("data_path", "data/processed/train_data.csv"),
                epochs=config.get("epochs", 50),
                batch_size=config.get("batch_size", 32),
                expected_duration=config.get("training_duration", 3600),  # 1 hour
                priority=TaskPriority.HIGH,
                gpu_required=config.get("gpu_required", True),
                memory_estimate=config.get("memory_estimate", 6.0),
                dependencies=dependencies
            ))
            task_id_counter += 1
            
            # Model evaluation task
            eval_task_id = f"evaluation_{i}_{task_id_counter}"
            tasks.append(MLTask(
                task_id=eval_task_id,
                task_type="evaluation",
                model_path=config.get("model_path", f"models/model_{i}.h5"),
                data_path=config.get("test_data_path", "data/processed/test_data.csv"),
                expected_duration=config.get("eval_duration", 300),  # 5 minutes
                priority=TaskPriority.MEDIUM,
                gpu_required=config.get("gpu_required", False),
                memory_estimate=2.0,
                dependencies=[training_task_id]
            ))
            task_id_counter += 1
        
        return tasks
    
    def create_inference_pipeline(
        self,
        model_paths: List[str],
        data_batches: List[str],
        real_time: bool = False
    ) -> List[MLTask]:
        """
        Create ML inference pipeline tasks.
        
        Args:
            model_paths: List of model file paths
            data_batches: List of data batch paths
            real_time: Whether this is real-time inference
            
        Returns:
            List of ML tasks for inference pipeline
        """
        tasks = []
        task_id_counter = 0
        
        for model_path in model_paths:
            for batch_path in data_batches:
                task_id = f"inference_{task_id_counter}"
                
                # Real-time inference has different characteristics
                if real_time:
                    expected_duration = 30  # 30 seconds
                    priority = TaskPriority.CRITICAL
                    batch_size = 1
                else:
                    expected_duration = 300  # 5 minutes
                    priority = TaskPriority.MEDIUM
                    batch_size = 64
                
                tasks.append(MLTask(
                    task_id=task_id,
                    task_type="inference",
                    model_path=model_path,
                    data_path=batch_path,
                    batch_size=batch_size,
                    expected_duration=expected_duration,
                    priority=priority,
                    gpu_required=not real_time,  # CPU for real-time, GPU for batch
                    memory_estimate=1.0 if real_time else 3.0,
                    cpu_cores=1 if real_time else 2
                ))
                task_id_counter += 1
        
        return tasks
    
    def get_pipeline_recommendations(self, ml_tasks: List[MLTask]) -> Dict[str, Any]:
        """
        Get recommendations for optimizing ML pipeline.
        
        Args:
            ml_tasks: List of ML tasks to analyze
            
        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            "resource_optimization": [],
            "scheduling_optimization": [],
            "performance_optimization": [],
            "cost_optimization": []
        }
        
        # Analyze resource usage
        total_gpu_tasks = sum(1 for task in ml_tasks if task.gpu_required)
        total_cpu_tasks = len(ml_tasks) - total_gpu_tasks
        
        if total_gpu_tasks > 1:
            recommendations["resource_optimization"].append(
                "Consider GPU memory sharing or sequential GPU task execution"
            )
        
        # Analyze task dependencies
        independent_tasks = [task for task in ml_tasks if not task.dependencies]
        if len(independent_tasks) > 3:
            recommendations["scheduling_optimization"].append(
                "High parallelization potential - consider quantum annealing algorithm"
            )
        
        # Analyze task durations
        long_tasks = [task for task in ml_tasks if task.expected_duration > 1800]
        if long_tasks:
            recommendations["performance_optimization"].append(
                f"Consider breaking down {len(long_tasks)} long-running tasks into smaller chunks"
            )
        
        # Memory optimization
        high_memory_tasks = [task for task in ml_tasks if task.memory_estimate > 8.0]
        if high_memory_tasks:
            recommendations["cost_optimization"].append(
                f"Consider memory optimization for {len(high_memory_tasks)} high-memory tasks"
            )
        
        return recommendations
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {"no_data": True}
        
        makespans = [entry["makespan"] for entry in self.execution_history]
        qualities = [entry["quality"] for entry in self.execution_history]
        
        return {
            "total_optimizations": len(self.execution_history),
            "average_makespan": np.mean(makespans),
            "best_makespan": min(makespans),
            "average_quality": np.mean(qualities),
            "best_quality": max(qualities),
            "algorithm_usage": self._count_algorithm_usage(),
            "recent_performance": self.execution_history[-5:] if len(self.execution_history) >= 5 else self.execution_history
        }
    
    def _count_algorithm_usage(self) -> Dict[str, int]:
        """Count algorithm usage in execution history."""
        algorithm_counts = {}
        for entry in self.execution_history:
            alg = entry["algorithm"]
            algorithm_counts[alg] = algorithm_counts.get(alg, 0) + 1
        return algorithm_counts