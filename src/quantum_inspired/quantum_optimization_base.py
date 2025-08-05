"""
Quantum Optimization Algorithm Base Class

Provides a common interface and base functionality for quantum-inspired
optimization algorithms used in task planning and scheduling.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import time
import logging
from dataclasses import dataclass
from enum import Enum

from .task_representation import Task, TaskGraph, ResourceConstraint
from .quantum_utils import QuantumState, QuantumRegister


class OptimizationObjective(Enum):
    """Optimization objectives for task planning."""
    MINIMIZE_MAKESPAN = "minimize_makespan"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_RESOURCE_USAGE = "minimize_resource_usage"
    BALANCE_LOAD = "balance_load"
    MINIMIZE_ENERGY = "minimize_energy"


@dataclass
class OptimizationResult:
    """
    Result of quantum-inspired optimization.
    
    Attributes:
        solution: Optimized task schedule or assignment
        objective_value: Value of the optimization objective achieved
        quantum_state: Final quantum state (if applicable)
        execution_time: Time taken for optimization
        iterations: Number of optimization iterations
        converged: Whether the algorithm converged
        metadata: Additional optimization metadata
    """
    solution: Dict[str, Any]
    objective_value: float
    quantum_state: Optional[QuantumState] = None
    execution_time: float = 0.0
    iterations: int = 0
    converged: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class OptimizationParameters:
    """
    Parameters for quantum-inspired optimization algorithms.
    
    Attributes:
        max_iterations: Maximum number of optimization iterations
        convergence_threshold: Threshold for convergence detection
        temperature_schedule: Temperature schedule for annealing algorithms
        quantum_depth: Depth of quantum circuit simulation
        population_size: Population size for genetic algorithms
        mutation_rate: Mutation rate for evolutionary algorithms
        crossover_rate: Crossover rate for genetic algorithms
        learning_rate: Learning rate for gradient-based methods
        regularization: Regularization parameter
        random_seed: Random seed for reproducibility
    """
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    temperature_schedule: str = "exponential"  # exponential, linear, adaptive
    quantum_depth: int = 10
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    learning_rate: float = 0.01
    regularization: float = 0.001
    random_seed: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate optimization parameters."""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not 0 < self.convergence_threshold < 1:
            raise ValueError("convergence_threshold must be between 0 and 1")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("crossover_rate must be between 0 and 1")


class QuantumOptimizationAlgorithm(ABC):
    """
    Abstract base class for quantum-inspired optimization algorithms.
    
    This class provides a common interface for different quantum-inspired
    optimization algorithms used in task planning and scheduling.
    """
    
    def __init__(
        self,
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_MAKESPAN,
        parameters: Optional[OptimizationParameters] = None
    ):
        """
        Initialize quantum optimization algorithm.
        
        Args:
            objective: Optimization objective to pursue
            parameters: Algorithm parameters
        """
        self.objective = objective
        self.parameters = parameters or OptimizationParameters()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set random seed if provided
        if self.parameters.random_seed is not None:
            np.random.seed(self.parameters.random_seed)
        
        # Algorithm state
        self.current_iteration = 0
        self.best_solution: Optional[Dict[str, Any]] = None
        self.best_objective_value = float('inf')
        self.optimization_history: List[float] = []
        self.quantum_register: Optional[QuantumRegister] = None
    
    @abstractmethod
    def optimize(self, task_graph: TaskGraph) -> OptimizationResult:
        """
        Perform quantum-inspired optimization on the task graph.
        
        Args:
            task_graph: Task graph to optimize
            
        Returns:
            Optimization result with solution and metadata
        """
        pass
    
    @abstractmethod
    def evaluate_solution(self, solution: Dict[str, Any], task_graph: TaskGraph) -> float:
        """
        Evaluate the quality of a solution.
        
        Args:
            solution: Solution to evaluate
            task_graph: Task graph context
            
        Returns:
            Objective function value (lower is better for minimization)
        """
        pass
    
    def initialize_quantum_state(self, num_qubits: int) -> None:
        """Initialize quantum register for the optimization."""
        self.quantum_register = QuantumRegister(num_qubits)
        
        # Create initial superposition
        for i in range(num_qubits):
            self.quantum_register.apply_hadamard(i)
    
    def calculate_makespan(self, schedule: Dict[str, float], task_graph: TaskGraph) -> float:
        """
        Calculate makespan (total completion time) for a schedule.
        
        Args:
            schedule: Task schedule mapping task_id to start_time
            task_graph: Task graph with task information
            
        Returns:
            Makespan value
        """
        if not schedule or not task_graph.tasks:
            return 0.0
        
        max_completion_time = 0.0
        for task_id, start_time in schedule.items():
            if task_id in task_graph.tasks:
                task = task_graph.tasks[task_id]
                completion_time = start_time + task.estimated_duration.total_seconds()
                max_completion_time = max(max_completion_time, completion_time)
        
        return max_completion_time
    
    def calculate_resource_usage(
        self, 
        schedule: Dict[str, float], 
        task_graph: TaskGraph
    ) -> Dict[str, float]:
        """
        Calculate resource usage for a schedule.
        
        Args:
            schedule: Task schedule mapping task_id to start_time
            task_graph: Task graph with resource requirements
            
        Returns:
            Dictionary mapping resource types to usage values
        """
        resource_usage = {}
        
        for task_id, start_time in schedule.items():
            if task_id in task_graph.tasks:
                task = task_graph.tasks[task_id]
                duration = task.estimated_duration.total_seconds()
                
                for req in task.resource_requirements:
                    resource_type = req.resource_type.value
                    usage = req.amount * duration
                    resource_usage[resource_type] = resource_usage.get(resource_type, 0) + usage
        
        return resource_usage
    
    def validate_schedule(self, schedule: Dict[str, float], task_graph: TaskGraph) -> List[str]:
        """
        Validate a task schedule for feasibility.
        
        Args:
            schedule: Task schedule to validate
            task_graph: Task graph with constraints
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check all tasks are scheduled
        scheduled_tasks = set(schedule.keys())
        all_tasks = set(task_graph.tasks.keys())
        
        if scheduled_tasks != all_tasks:
            missing = all_tasks - scheduled_tasks
            extra = scheduled_tasks - all_tasks
            if missing:
                errors.append(f"Missing tasks in schedule: {missing}")
            if extra:
                errors.append(f"Extra tasks in schedule: {extra}")
        
        # Check dependency constraints
        for task_id, start_time in schedule.items():
            if task_id in task_graph.tasks:
                task = task_graph.tasks[task_id]
                for dep_id in task.dependencies:
                    if dep_id in schedule:
                        dep_task = task_graph.tasks[dep_id]
                        dep_completion = schedule[dep_id] + dep_task.estimated_duration.total_seconds()
                        if dep_completion > start_time:
                            errors.append(
                                f"Task {task_id} starts before dependency {dep_id} completes"
                            )
        
        return errors
    
    def apply_quantum_mutation(self, solution: Dict[str, Any], mutation_strength: float = 0.1) -> Dict[str, Any]:
        """
        Apply quantum-inspired mutation to a solution.
        
        Args:
            solution: Solution to mutate
            mutation_strength: Strength of mutation (0-1)
            
        Returns:
            Mutated solution
        """
        mutated_solution = solution.copy()
        
        # Simple quantum-inspired mutation: add quantum noise
        if 'schedule' in solution:
            schedule = solution['schedule'].copy()
            for task_id, start_time in schedule.items():
                if np.random.random() < mutation_strength:
                    # Add quantum uncertainty
                    quantum_noise = np.random.normal(0, start_time * 0.1)
                    schedule[task_id] = max(0, start_time + quantum_noise)
            mutated_solution['schedule'] = schedule
        
        return mutated_solution
    
    def quantum_crossover(
        self, 
        solution1: Dict[str, Any], 
        solution2: Dict[str, Any],
        entanglement_strength: float = 0.5
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform quantum-inspired crossover between two solutions.
        
        Args:
            solution1: First parent solution
            solution2: Second parent solution
            entanglement_strength: Strength of quantum entanglement
            
        Returns:
            Tuple of two offspring solutions
        """
        offspring1 = solution1.copy()
        offspring2 = solution2.copy()
        
        if 'schedule' in solution1 and 'schedule' in solution2:
            schedule1 = solution1['schedule']
            schedule2 = solution2['schedule']
            
            new_schedule1 = {}
            new_schedule2 = {}
            
            for task_id in schedule1.keys():
                if task_id in schedule2:
                    if np.random.random() < entanglement_strength:
                        # Quantum entanglement: blend the values
                        alpha = np.random.random()
                        new_schedule1[task_id] = alpha * schedule1[task_id] + (1 - alpha) * schedule2[task_id]
                        new_schedule2[task_id] = (1 - alpha) * schedule1[task_id] + alpha * schedule2[task_id]
                    else:
                        # Classical crossover
                        new_schedule1[task_id] = schedule1[task_id]
                        new_schedule2[task_id] = schedule2[task_id]
            
            offspring1['schedule'] = new_schedule1
            offspring2['schedule'] = new_schedule2
        
        return offspring1, offspring2
    
    def check_convergence(self) -> bool:
        """
        Check if the optimization has converged.
        
        Returns:
            True if converged, False otherwise
        """
        if len(self.optimization_history) < 10:
            return False
        
        # Check if improvement in last 10 iterations is below threshold
        recent_values = self.optimization_history[-10:]
        improvement = (max(recent_values) - min(recent_values)) / max(recent_values)
        
        return improvement < self.parameters.convergence_threshold
    
    def get_temperature(self, iteration: int) -> float:
        """
        Get temperature for simulated annealing schedules.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Temperature value
        """
        max_temp = 100.0
        min_temp = 0.01
        
        if self.parameters.temperature_schedule == "exponential":
            decay_rate = np.log(min_temp / max_temp) / self.parameters.max_iterations
            return max_temp * np.exp(decay_rate * iteration)
        elif self.parameters.temperature_schedule == "linear":
            return max_temp - (max_temp - min_temp) * iteration / self.parameters.max_iterations
        elif self.parameters.temperature_schedule == "adaptive":
            # Adaptive schedule based on improvement
            if len(self.optimization_history) >= 2:
                improvement_rate = (
                    self.optimization_history[-2] - self.optimization_history[-1]
                ) / self.optimization_history[-2]
                if improvement_rate > 0.01:
                    return max_temp * 0.9 ** iteration  # Fast cooling when improving
                else:
                    return max_temp * 0.99 ** iteration  # Slow cooling when stuck
            else:
                return max_temp * 0.95 ** iteration
        else:
            return max_temp * 0.95 ** iteration
    
    def log_progress(self, iteration: int, objective_value: float, additional_info: str = "") -> None:
        """
        Log optimization progress.
        
        Args:
            iteration: Current iteration
            objective_value: Current objective value
            additional_info: Additional information to log
        """
        self.logger.info(
            f"Iteration {iteration}: Objective = {objective_value:.6f} {additional_info}"
        )
        
        # Update history
        self.optimization_history.append(objective_value)
        
        # Update best solution tracking
        if objective_value < self.best_objective_value:
            self.best_objective_value = objective_value
            self.logger.info(f"New best objective: {objective_value:.6f}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics.
        
        Returns:
            Dictionary with optimization statistics
        """
        return {
            "total_iterations": self.current_iteration,
            "best_objective_value": self.best_objective_value,
            "final_objective_value": self.optimization_history[-1] if self.optimization_history else None,
            "improvement": (
                (self.optimization_history[0] - self.optimization_history[-1]) / self.optimization_history[0]
                if len(self.optimization_history) > 1 else 0.0
            ),
            "converged": self.check_convergence(),
            "optimization_history": self.optimization_history.copy()
        }