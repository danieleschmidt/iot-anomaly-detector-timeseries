"""
Task Scheduler

Main interface for quantum-inspired task scheduling. Provides high-level
scheduling functionality with multiple algorithm options and practical
integration capabilities.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta

from ..quantum_inspired.task_representation import Task, TaskGraph, ResourceConstraint
from ..quantum_inspired.quantum_annealing_planner import QuantumAnnealingPlanner, AnnealingParameters
from ..quantum_inspired.qaoa_task_optimizer import QAOATaskOptimizer, QAOAParameters
from ..quantum_inspired.quantum_optimization_base import OptimizationObjective, OptimizationResult


class SchedulingAlgorithm(Enum):
    """Available scheduling algorithms."""
    QUANTUM_ANNEALING = "quantum_annealing"
    QAOA = "qaoa"
    HYBRID = "hybrid"
    CLASSICAL_HEURISTIC = "classical_heuristic"


@dataclass
class SchedulingResult:
    """
    Result of task scheduling operation.
    
    Attributes:
        schedule: Task schedule mapping task_id to start_time
        makespan: Total execution time (completion time of last task)
        resource_utilization: Resource usage statistics
        algorithm_used: Algorithm that produced this result
        execution_time: Time taken for scheduling
        quality_metrics: Quality metrics for the schedule
        metadata: Additional scheduling metadata
    """
    schedule: Dict[str, float]
    makespan: float
    resource_utilization: Dict[str, float]
    algorithm_used: SchedulingAlgorithm
    execution_time: float
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "schedule": self.schedule,
            "makespan": self.makespan,
            "resource_utilization": self.resource_utilization,
            "algorithm_used": self.algorithm_used.value,
            "execution_time": self.execution_time,
            "quality_metrics": self.quality_metrics,
            "metadata": self.metadata
        }
    
    def save_to_file(self, file_path: str) -> None:
        """Save scheduling result to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class TaskScheduler:
    """
    Main task scheduler with quantum-inspired optimization capabilities.
    
    Provides a high-level interface for scheduling tasks using various
    quantum-inspired and classical algorithms.
    """
    
    def __init__(
        self,
        default_algorithm: SchedulingAlgorithm = SchedulingAlgorithm.QUANTUM_ANNEALING,
        default_objective: OptimizationObjective = OptimizationObjective.MINIMIZE_MAKESPAN
    ):
        """
        Initialize task scheduler.
        
        Args:
            default_algorithm: Default scheduling algorithm to use
            default_objective: Default optimization objective
        """
        self.default_algorithm = default_algorithm
        self.default_objective = default_objective
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Algorithm instances
        self._algorithms: Dict[SchedulingAlgorithm, Any] = {}
        self._initialize_algorithms()
        
        # Scheduling history
        self.scheduling_history: List[SchedulingResult] = []
        
    def _initialize_algorithms(self) -> None:
        """Initialize algorithm instances."""
        # Quantum Annealing
        annealing_params = AnnealingParameters(
            max_iterations=500,
            initial_temperature=100.0,
            final_temperature=0.01,
            cooling_schedule="exponential"
        )
        self._algorithms[SchedulingAlgorithm.QUANTUM_ANNEALING] = QuantumAnnealingPlanner(
            objective=self.default_objective,
            parameters=annealing_params
        )
        
        # QAOA
        qaoa_params = QAOAParameters(
            max_iterations=300,
            circuit_depth=3,
            measurement_shots=1000
        )
        self._algorithms[SchedulingAlgorithm.QAOA] = QAOATaskOptimizer(
            objective=self.default_objective,
            parameters=qaoa_params
        )
    
    def schedule_tasks(
        self,
        tasks: List[Task],
        resource_constraints: Optional[List[ResourceConstraint]] = None,
        algorithm: Optional[SchedulingAlgorithm] = None,
        objective: Optional[OptimizationObjective] = None,
        algorithm_params: Optional[Dict[str, Any]] = None
    ) -> SchedulingResult:
        """
        Schedule a list of tasks.
        
        Args:
            tasks: List of tasks to schedule
            resource_constraints: Resource constraints to consider
            algorithm: Scheduling algorithm to use
            objective: Optimization objective
            algorithm_params: Algorithm-specific parameters
            
        Returns:
            Scheduling result with optimized schedule
        """
        start_time = time.time()
        
        # Use defaults if not specified
        algorithm = algorithm or self.default_algorithm
        objective = objective or self.default_objective
        
        self.logger.info(f"Starting task scheduling with {len(tasks)} tasks using {algorithm.value}")
        
        # Build task graph
        task_graph = self._build_task_graph(tasks, resource_constraints)
        
        # Validate task graph
        validation_errors = task_graph.validate_graph()
        if validation_errors:
            self.logger.error(f"Task graph validation failed: {validation_errors}")
            return self._create_error_result(
                algorithm, 
                time.time() - start_time,
                {"validation_errors": validation_errors}
            )
        
        # Select and configure algorithm
        if algorithm == SchedulingAlgorithm.HYBRID:
            result = self._hybrid_scheduling(task_graph, objective, algorithm_params)
        else:
            result = self._single_algorithm_scheduling(
                task_graph, algorithm, objective, algorithm_params
            )
        
        execution_time = time.time() - start_time
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(result.solution, task_graph)
        
        # Create scheduling result
        scheduling_result = SchedulingResult(
            schedule=result.solution.get("schedule", {}),
            makespan=self._calculate_makespan(result.solution.get("schedule", {}), task_graph),
            resource_utilization=self._calculate_resource_utilization(
                result.solution.get("schedule", {}), task_graph
            ),
            algorithm_used=algorithm,
            execution_time=execution_time,
            quality_metrics=quality_metrics,
            metadata={
                "num_tasks": len(tasks),
                "num_resource_constraints": len(resource_constraints) if resource_constraints else 0,
                "objective": objective.value,
                "algorithm_result": result.metadata,
                "optimization_converged": result.converged,
                "optimization_iterations": result.iterations
            }
        )
        
        # Store in history
        self.scheduling_history.append(scheduling_result)
        
        self.logger.info(f"Scheduling completed in {execution_time:.2f}s")
        self.logger.info(f"Makespan: {scheduling_result.makespan:.2f}s")
        self.logger.info(f"Quality score: {quality_metrics.get('overall_quality', 0):.3f}")
        
        return scheduling_result
    
    def _build_task_graph(
        self, 
        tasks: List[Task], 
        resource_constraints: Optional[List[ResourceConstraint]]
    ) -> TaskGraph:
        """Build task graph from tasks and constraints."""
        task_graph = TaskGraph()
        
        # Add tasks
        for task in tasks:
            task_graph.add_task(task)
        
        # Add resource constraints
        if resource_constraints:
            for constraint in resource_constraints:
                task_graph.add_resource_constraint(constraint)
        
        return task_graph
    
    def _single_algorithm_scheduling(
        self,
        task_graph: TaskGraph,
        algorithm: SchedulingAlgorithm,
        objective: OptimizationObjective,
        algorithm_params: Optional[Dict[str, Any]]
    ) -> OptimizationResult:
        """Run single algorithm scheduling."""
        if algorithm not in self._algorithms:
            raise ValueError(f"Algorithm {algorithm} not available")
        
        optimizer = self._algorithms[algorithm]
        
        # Update algorithm parameters if provided
        if algorithm_params:
            self._update_algorithm_params(optimizer, algorithm_params)
        
        # Update objective if different
        if optimizer.objective != objective:
            optimizer.objective = objective
        
        return optimizer.optimize(task_graph)
    
    def _hybrid_scheduling(
        self,
        task_graph: TaskGraph,
        objective: OptimizationObjective,
        algorithm_params: Optional[Dict[str, Any]]
    ) -> OptimizationResult:
        """Run hybrid scheduling using multiple algorithms."""
        self.logger.info("Running hybrid scheduling")
        
        # Run multiple algorithms
        results = []
        algorithms_to_try = [
            SchedulingAlgorithm.QUANTUM_ANNEALING,
            SchedulingAlgorithm.QAOA
        ]
        
        for alg in algorithms_to_try:
            if alg in self._algorithms:
                try:
                    self.logger.info(f"Running {alg.value} for hybrid approach")
                    result = self._single_algorithm_scheduling(
                        task_graph, alg, objective, algorithm_params
                    )
                    results.append((alg, result))
                except Exception as e:
                    self.logger.warning(f"Algorithm {alg.value} failed: {e}")
        
        if not results:
            raise RuntimeError("All algorithms failed in hybrid scheduling")
        
        # Select best result
        best_algorithm, best_result = min(results, key=lambda x: x[1].objective_value)
        
        self.logger.info(f"Hybrid scheduling selected {best_algorithm.value} as best")
        self.logger.info(f"Best objective value: {best_result.objective_value:.6f}")
        
        # Add hybrid metadata
        best_result.metadata["hybrid_results"] = {
            alg.value: {"objective_value": result.objective_value, "converged": result.converged}
            for alg, result in results
        }
        best_result.metadata["selected_algorithm"] = best_algorithm.value
        
        return best_result
    
    def _update_algorithm_params(self, optimizer: Any, params: Dict[str, Any]) -> None:
        """Update algorithm parameters."""
        if hasattr(optimizer, 'parameters'):
            for key, value in params.items():
                if hasattr(optimizer.parameters, key):
                    setattr(optimizer.parameters, key, value)
                else:
                    self.logger.warning(f"Unknown parameter {key} for algorithm")
    
    def _calculate_makespan(self, schedule: Dict[str, float], task_graph: TaskGraph) -> float:
        """Calculate makespan for a schedule."""
        if not schedule or not task_graph.tasks:
            return 0.0
        
        max_completion = 0.0
        for task_id, start_time in schedule.items():
            if task_id in task_graph.tasks:
                task = task_graph.tasks[task_id]
                completion_time = start_time + task.estimated_duration.total_seconds()
                max_completion = max(max_completion, completion_time)
        
        return max_completion
    
    def _calculate_resource_utilization(
        self, 
        schedule: Dict[str, float], 
        task_graph: TaskGraph
    ) -> Dict[str, float]:
        """Calculate resource utilization statistics."""
        utilization = {}
        
        if not schedule or not task_graph.resource_constraints:
            return utilization
        
        # Calculate peak utilization for each resource type
        for resource_type, constraint in task_graph.resource_constraints.items():
            peak_usage = 0.0
            
            # Simple approach: assume worst case overlap
            total_usage = 0.0
            for task_id, start_time in schedule.items():
                if task_id in task_graph.tasks:
                    task = task_graph.tasks[task_id]
                    for req in task.resource_requirements:
                        if req.resource_type == resource_type:
                            total_usage += req.amount
            
            peak_usage = total_usage  # Simplified
            utilization[resource_type.value] = min(
                peak_usage / constraint.total_capacity if constraint.total_capacity > 0 else 0,
                1.0
            )
        
        return utilization
    
    def _calculate_quality_metrics(
        self, 
        solution: Dict[str, Any], 
        task_graph: TaskGraph
    ) -> Dict[str, float]:
        """Calculate quality metrics for the schedule."""
        metrics = {}
        
        if "schedule" not in solution:
            return {"overall_quality": 0.0}
        
        schedule = solution["schedule"]
        
        # Makespan efficiency
        makespan = self._calculate_makespan(schedule, task_graph)
        critical_path_time = self._estimate_critical_path_time(task_graph)
        
        if critical_path_time > 0:
            metrics["makespan_efficiency"] = critical_path_time / makespan
        else:
            metrics["makespan_efficiency"] = 1.0
        
        # Resource utilization balance
        resource_util = self._calculate_resource_utilization(schedule, task_graph)
        if resource_util:
            util_values = list(resource_util.values())
            avg_util = sum(util_values) / len(util_values)
            util_variance = sum((u - avg_util) ** 2 for u in util_values) / len(util_values)
            metrics["resource_balance"] = 1.0 / (1.0 + util_variance)
        else:
            metrics["resource_balance"] = 1.0
        
        # Constraint satisfaction
        validation_errors = task_graph.validate_graph()
        metrics["constraint_satisfaction"] = 1.0 if not validation_errors else 0.0
        
        # Overall quality (weighted combination)
        metrics["overall_quality"] = (
            0.4 * metrics["makespan_efficiency"] +
            0.3 * metrics["resource_balance"] +
            0.3 * metrics["constraint_satisfaction"]
        )
        
        return metrics
    
    def _estimate_critical_path_time(self, task_graph: TaskGraph) -> float:
        """Estimate critical path time through task graph."""
        try:
            critical_path = task_graph.get_critical_path()
            if not critical_path:
                return 0.0
            
            total_time = 0.0
            for task_id in critical_path:
                if task_id in task_graph.tasks:
                    task = task_graph.tasks[task_id]
                    total_time += task.estimated_duration.total_seconds()
            
            return total_time
        except:
            # Fallback: sum of all task durations
            return sum(
                task.estimated_duration.total_seconds() 
                for task in task_graph.tasks.values()
            )
    
    def _create_error_result(
        self, 
        algorithm: SchedulingAlgorithm, 
        execution_time: float,
        error_metadata: Dict[str, Any]
    ) -> SchedulingResult:
        """Create error result for failed scheduling."""
        return SchedulingResult(
            schedule={},
            makespan=float('inf'),
            resource_utilization={},
            algorithm_used=algorithm,
            execution_time=execution_time,
            quality_metrics={"overall_quality": 0.0},
            metadata={
                "error": True,
                **error_metadata
            }
        )
    
    def get_algorithm_info(self, algorithm: SchedulingAlgorithm) -> Dict[str, Any]:
        """Get information about a specific algorithm."""
        if algorithm not in self._algorithms:
            return {"available": False}
        
        optimizer = self._algorithms[algorithm]
        
        info = {
            "available": True,
            "algorithm": algorithm.value,
            "objective": optimizer.objective.value if hasattr(optimizer, 'objective') else None,
            "parameters": {}
        }
        
        if hasattr(optimizer, 'parameters'):
            params = optimizer.parameters
            for attr in dir(params):
                if not attr.startswith('_'):
                    try:
                        value = getattr(params, attr)
                        if not callable(value):
                            info["parameters"][attr] = value
                    except:
                        pass
        
        return info
    
    def get_scheduling_statistics(self) -> Dict[str, Any]:
        """Get statistics about scheduling history."""
        if not self.scheduling_history:
            return {"total_schedules": 0}
        
        total_schedules = len(self.scheduling_history)
        
        # Algorithm usage
        algorithm_counts = {}
        for result in self.scheduling_history:
            alg = result.algorithm_used.value
            algorithm_counts[alg] = algorithm_counts.get(alg, 0) + 1
        
        # Performance statistics
        execution_times = [result.execution_time for result in self.scheduling_history]
        quality_scores = [
            result.quality_metrics.get("overall_quality", 0) 
            for result in self.scheduling_history
        ]
        
        avg_execution_time = sum(execution_times) / len(execution_times)
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        return {
            "total_schedules": total_schedules,
            "algorithm_usage": algorithm_counts,
            "average_execution_time": avg_execution_time,
            "average_quality_score": avg_quality,
            "best_quality_score": max(quality_scores) if quality_scores else 0,
            "worst_quality_score": min(quality_scores) if quality_scores else 0
        }
    
    def clear_history(self) -> None:
        """Clear scheduling history."""
        self.scheduling_history.clear()
        self.logger.info("Scheduling history cleared")
    
    def export_history(self, file_path: str) -> None:
        """Export scheduling history to JSON file."""
        history_data = [result.to_dict() for result in self.scheduling_history]
        
        with open(file_path, 'w') as f:
            json.dump({
                "export_timestamp": datetime.now().isoformat(),
                "total_schedules": len(history_data),
                "scheduling_history": history_data
            }, f, indent=2, default=str)
        
        self.logger.info(f"Exported {len(history_data)} scheduling results to {file_path}")
    
    def schedule_from_config(self, config: Dict[str, Any]) -> SchedulingResult:
        """
        Schedule tasks from configuration dictionary.
        
        Args:
            config: Configuration containing tasks, constraints, and parameters
            
        Returns:
            Scheduling result
        """
        # Parse tasks from config
        tasks = []
        if "tasks" in config:
            for task_data in config["tasks"]:
                task = self._task_from_dict(task_data)
                tasks.append(task)
        
        # Parse resource constraints
        resource_constraints = []
        if "resource_constraints" in config:
            for constraint_data in config["resource_constraints"]:
                constraint = self._constraint_from_dict(constraint_data)
                resource_constraints.append(constraint)
        
        # Parse scheduling parameters
        algorithm = SchedulingAlgorithm(config.get("algorithm", self.default_algorithm.value))
        objective = OptimizationObjective(config.get("objective", self.default_objective.value))
        algorithm_params = config.get("algorithm_params", {})
        
        return self.schedule_tasks(
            tasks=tasks,
            resource_constraints=resource_constraints,
            algorithm=algorithm,
            objective=objective,
            algorithm_params=algorithm_params
        )
    
    def _task_from_dict(self, task_data: Dict[str, Any]) -> Task:
        """Create Task object from dictionary."""
        # This is a simplified implementation
        # In practice, you would handle all Task fields properly
        from ..quantum_inspired.task_representation import TaskPriority, ResourceRequirement, ResourceType
        
        task = Task(
            task_id=task_data.get("task_id", ""),
            name=task_data.get("name", ""),
            description=task_data.get("description", ""),
            estimated_duration=timedelta(seconds=task_data.get("duration", 600))
        )
        
        # Add dependencies
        for dep_id in task_data.get("dependencies", []):
            task.add_dependency(dep_id)
        
        return task
    
    def _constraint_from_dict(self, constraint_data: Dict[str, Any]) -> ResourceConstraint:
        """Create ResourceConstraint object from dictionary."""
        from ..quantum_inspired.task_representation import ResourceType
        
        return ResourceConstraint(
            resource_type=ResourceType(constraint_data["resource_type"]),
            total_capacity=constraint_data["total_capacity"],
            available_capacity=constraint_data.get("available_capacity", constraint_data["total_capacity"]),
            unit=constraint_data.get("unit", ""),
            allocation_policy=constraint_data.get("allocation_policy", "first_fit")
        )