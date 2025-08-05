"""
Quantum Annealing Task Planner

Implements quantum annealing simulation for optimizing task scheduling and
resource allocation. Uses simulated annealing with quantum-inspired 
neighborhood exploration and energy landscapes.
"""

import numpy as np
import random
import math
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import copy
import logging

from .quantum_optimization_base import (
    QuantumOptimizationAlgorithm,
    OptimizationObjective,
    OptimizationResult,
    OptimizationParameters
)
from .task_representation import Task, TaskGraph, ResourceConstraint, TaskStatus
from .quantum_utils import QuantumState, quantum_superposition, measure_quantum_state


@dataclass
class AnnealingParameters(OptimizationParameters):
    """
    Extended parameters for quantum annealing algorithm.
    
    Attributes:
        initial_temperature: Starting temperature for annealing
        final_temperature: Final temperature for annealing
        cooling_schedule: Cooling schedule type
        tunnel_probability: Quantum tunneling probability
        energy_precision: Precision for energy calculations
        restart_temperature: Temperature threshold for restart
        max_restarts: Maximum number of restarts allowed
    """
    initial_temperature: float = 100.0
    final_temperature: float = 0.01
    cooling_schedule: str = "exponential"  # exponential, linear, logarithmic
    tunnel_probability: float = 0.1
    energy_precision: float = 1e-6
    restart_temperature: float = 1.0
    max_restarts: int = 3
    
    def __post_init__(self) -> None:
        """Validate annealing parameters."""
        super().__post_init__()
        if self.initial_temperature <= self.final_temperature:
            raise ValueError("Initial temperature must be greater than final temperature")
        if not 0 <= self.tunnel_probability <= 1:
            raise ValueError("Tunnel probability must be between 0 and 1")


class QuantumAnnealingPlanner(QuantumOptimizationAlgorithm):
    """
    Quantum annealing planner for task scheduling optimization.
    
    This planner uses simulated quantum annealing to find optimal task schedules
    by exploring the solution space through quantum-inspired transitions and
    tunneling effects.
    """
    
    def __init__(
        self,
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_MAKESPAN,
        parameters: Optional[AnnealingParameters] = None
    ):
        """
        Initialize quantum annealing planner.
        
        Args:
            objective: Optimization objective
            parameters: Annealing parameters
        """
        if parameters is None:
            parameters = AnnealingParameters()
        
        super().__init__(objective, parameters)
        self.annealing_params = parameters
        
        # Annealing state
        self.current_energy = float('inf')
        self.best_energy = float('inf')
        self.temperature_history: List[float] = []
        self.energy_history: List[float] = []
        self.quantum_tunneling_events = 0
        self.restarts_performed = 0
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def optimize(self, task_graph: TaskGraph) -> OptimizationResult:
        """
        Perform quantum annealing optimization on task graph.
        
        Args:
            task_graph: Task graph to optimize
            
        Returns:
            Optimization result with best schedule found
        """
        start_time = time.time()
        
        # Validate task graph
        validation_errors = task_graph.validate_graph()
        if validation_errors:
            self.logger.error(f"Task graph validation failed: {validation_errors}")
            return OptimizationResult(
                solution={},
                objective_value=float('inf'),
                execution_time=time.time() - start_time,
                converged=False,
                metadata={"errors": validation_errors}
            )
        
        # Initialize quantum state for task graph
        num_tasks = len(task_graph.tasks)
        if num_tasks == 0:
            return OptimizationResult(
                solution={"schedule": {}},
                objective_value=0.0,
                execution_time=time.time() - start_time,
                converged=True
            )
        
        # Create initial solution
        current_solution = self._create_initial_solution(task_graph)
        self.current_energy = self.evaluate_solution(current_solution, task_graph)
        self.best_solution = copy.deepcopy(current_solution)
        self.best_energy = self.current_energy
        
        self.logger.info(f"Starting quantum annealing with {num_tasks} tasks")
        self.logger.info(f"Initial energy: {self.current_energy:.6f}")
        
        # Main annealing loop
        converged = False
        
        for restart in range(self.annealing_params.max_restarts + 1):
            if restart > 0:
                self.logger.info(f"Performing restart {restart}")
                current_solution = self._create_initial_solution(task_graph)
                self.current_energy = self.evaluate_solution(current_solution, task_graph)
                self.restarts_performed += 1
            
            converged = self._annealing_loop(current_solution, task_graph)
            
            if converged:
                break
        
        execution_time = time.time() - start_time
        
        # Create final quantum state representation
        final_quantum_state = self._create_solution_quantum_state(self.best_solution, task_graph)
        
        result = OptimizationResult(
            solution=self.best_solution,
            objective_value=self.best_energy,
            quantum_state=final_quantum_state,
            execution_time=execution_time,
            iterations=self.current_iteration,
            converged=converged,
            metadata={
                "quantum_tunneling_events": self.quantum_tunneling_events,
                "restarts_performed": self.restarts_performed,
                "temperature_history": self.temperature_history.copy(),
                "energy_history": self.energy_history.copy(),
                "final_temperature": self.temperature_history[-1] if self.temperature_history else 0,
                "algorithm": "quantum_annealing",
                "objective": self.objective.value
            }
        )
        
        self.logger.info(f"Optimization completed in {execution_time:.2f}s")
        self.logger.info(f"Best energy: {self.best_energy:.6f}")
        self.logger.info(f"Quantum tunneling events: {self.quantum_tunneling_events}")
        
        return result
    
    def _annealing_loop(self, current_solution: Dict[str, Any], task_graph: TaskGraph) -> bool:
        """
        Main annealing optimization loop.
        
        Args:
            current_solution: Current solution to improve
            task_graph: Task graph being optimized
            
        Returns:
            True if converged, False otherwise
        """
        for iteration in range(self.parameters.max_iterations):
            self.current_iteration = iteration
            
            # Calculate current temperature
            temperature = self._get_annealing_temperature(iteration)
            self.temperature_history.append(temperature)
            
            # Generate neighboring solution
            neighbor_solution = self._generate_neighbor(current_solution, task_graph)
            neighbor_energy = self.evaluate_solution(neighbor_solution, task_graph)
            
            # Calculate energy difference
            delta_energy = neighbor_energy - self.current_energy
            
            # Quantum annealing acceptance criterion
            accepted = False
            
            if delta_energy < 0:
                # Always accept better solutions
                accepted = True
            elif temperature > self.annealing_params.energy_precision:
                # Quantum tunneling or thermal acceptance
                if random.random() < self.annealing_params.tunnel_probability:
                    # Quantum tunneling through energy barriers
                    accepted = True
                    self.quantum_tunneling_events += 1
                    self.logger.debug(f"Quantum tunneling at iteration {iteration}")
                else:
                    # Classical thermal acceptance
                    acceptance_probability = math.exp(-delta_energy / temperature)
                    accepted = random.random() < acceptance_probability
            
            # Update current solution
            if accepted:
                current_solution = neighbor_solution
                self.current_energy = neighbor_energy
                
                # Update best solution
                if neighbor_energy < self.best_energy:
                    self.best_solution = copy.deepcopy(neighbor_solution)
                    self.best_energy = neighbor_energy
                    self.logger.debug(f"New best energy: {self.best_energy:.6f}")
            
            # Record progress
            self.energy_history.append(self.current_energy)
            self.log_progress(iteration, self.current_energy, f"T={temperature:.4f}")
            
            # Check convergence
            if temperature < self.annealing_params.final_temperature:
                self.logger.info(f"Reached final temperature at iteration {iteration}")
                return True
            
            if self.check_convergence():
                self.logger.info(f"Converged at iteration {iteration}")
                return True
            
            # Check for restart condition
            if (temperature < self.annealing_params.restart_temperature and 
                len(self.energy_history) > 100 and
                min(self.energy_history[-100:]) > self.best_energy * 1.1):
                self.logger.info(f"Triggering restart at iteration {iteration}")
                break
        
        return False
    
    def _get_annealing_temperature(self, iteration: int) -> float:
        """
        Calculate annealing temperature based on cooling schedule.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Temperature value
        """
        progress = iteration / self.parameters.max_iterations
        initial_temp = self.annealing_params.initial_temperature
        final_temp = self.annealing_params.final_temperature
        
        if self.annealing_params.cooling_schedule == "exponential":
            decay_rate = math.log(final_temp / initial_temp)
            return initial_temp * math.exp(decay_rate * progress)
        elif self.annealing_params.cooling_schedule == "linear":
            return initial_temp - (initial_temp - final_temp) * progress
        elif self.annealing_params.cooling_schedule == "logarithmic":
            if iteration == 0:
                return initial_temp
            return initial_temp / math.log(iteration + 2)
        else:
            # Default exponential
            return initial_temp * (final_temp / initial_temp) ** progress
    
    def _create_initial_solution(self, task_graph: TaskGraph) -> Dict[str, Any]:
        """
        Create initial solution using quantum-inspired heuristics.
        
        Args:
            task_graph: Task graph to schedule
            
        Returns:
            Initial solution dictionary
        """
        schedule = {}
        resource_usage = {rt: 0.0 for rt in task_graph.resource_constraints.keys()}
        
        # Get topological ordering
        try:
            import networkx as nx
            task_order = list(nx.topological_sort(task_graph.graph))
        except:
            # Fallback: order by dependencies
            task_order = self._dependency_based_ordering(task_graph)
        
        # Schedule tasks in topological order with quantum-inspired randomization
        for task_id in task_order:
            if task_id not in task_graph.tasks:
                continue
                
            task = task_graph.tasks[task_id]
            
            # Calculate earliest start time based on dependencies
            earliest_start = 0.0
            for dep_id in task.dependencies:
                if dep_id in schedule:
                    dep_task = task_graph.tasks[dep_id]
                    dep_completion = schedule[dep_id] + dep_task.estimated_duration.total_seconds()
                    earliest_start = max(earliest_start, dep_completion)
            
            # Add quantum uncertainty to start time
            uncertainty_range = task.duration_uncertainty * task.estimated_duration.total_seconds()
            quantum_noise = random.uniform(-uncertainty_range, uncertainty_range)
            
            start_time = max(0, earliest_start + quantum_noise)
            schedule[task_id] = start_time
        
        return {
            "schedule": schedule,
            "resource_usage": resource_usage,
            "metadata": {
                "creation_method": "quantum_heuristic",
                "task_order": task_order
            }
        }
    
    def _dependency_based_ordering(self, task_graph: TaskGraph) -> List[str]:
        """
        Create task ordering based on dependency structure.
        
        Args:
            task_graph: Task graph
            
        Returns:
            List of task IDs in dependency order
        """
        ordered_tasks = []
        remaining_tasks = set(task_graph.tasks.keys())
        completed_tasks = set()
        
        while remaining_tasks:
            # Find tasks with no unmet dependencies
            ready_tasks = []
            for task_id in remaining_tasks:
                task = task_graph.tasks[task_id]
                if task.dependencies.issubset(completed_tasks):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                # Handle cycles or missing dependencies
                ready_tasks = [list(remaining_tasks)[0]]
            
            # Select next task (with quantum-inspired randomization)
            if len(ready_tasks) > 1:
                # Weight by priority and quantum weight
                weights = []
                for task_id in ready_tasks:
                    task = task_graph.tasks[task_id]
                    weight = task.priority.value * task.quantum_weight
                    weights.append(weight)
                
                # Add quantum randomization
                weights = np.array(weights)
                weights = weights + np.random.normal(0, 0.1, len(weights))
                selected_idx = np.argmax(weights)
                selected_task = ready_tasks[selected_idx]
            else:
                selected_task = ready_tasks[0]
            
            ordered_tasks.append(selected_task)
            remaining_tasks.remove(selected_task)
            completed_tasks.add(selected_task)
        
        return ordered_tasks
    
    def _generate_neighbor(self, solution: Dict[str, Any], task_graph: TaskGraph) -> Dict[str, Any]:
        """
        Generate neighboring solution using quantum-inspired mutations.
        
        Args:
            solution: Current solution
            task_graph: Task graph context
            
        Returns:
            Neighboring solution
        """
        neighbor = copy.deepcopy(solution)
        schedule = neighbor["schedule"]
        
        # Choose mutation type with quantum-inspired probabilities
        mutation_types = [
            ("time_shift", 0.4),
            ("task_swap", 0.3),
            ("quantum_tunneling", 0.2),
            ("resource_optimization", 0.1)
        ]
        
        # Select mutation type
        rand = random.random()
        cumulative_prob = 0.0
        selected_mutation = "time_shift"
        
        for mutation_type, probability in mutation_types:
            cumulative_prob += probability
            if rand <= cumulative_prob:
                selected_mutation = mutation_type
                break
        
        # Apply selected mutation
        if selected_mutation == "time_shift":
            self._mutate_time_shift(schedule, task_graph)
        elif selected_mutation == "task_swap":
            self._mutate_task_swap(schedule, task_graph)
        elif selected_mutation == "quantum_tunneling":
            self._mutate_quantum_tunneling(schedule, task_graph)
        elif selected_mutation == "resource_optimization":
            self._mutate_resource_optimization(schedule, task_graph)
        
        return neighbor
    
    def _mutate_time_shift(self, schedule: Dict[str, float], task_graph: TaskGraph) -> None:
        """Apply time shift mutation to random task."""
        if not schedule:
            return
        
        task_id = random.choice(list(schedule.keys()))
        task = task_graph.tasks[task_id]
        
        # Calculate valid time range
        min_start = 0.0
        for dep_id in task.dependencies:
            if dep_id in schedule:
                dep_task = task_graph.tasks[dep_id]
                dep_completion = schedule[dep_id] + dep_task.estimated_duration.total_seconds()
                min_start = max(min_start, dep_completion)
        
        # Add quantum-inspired variation
        current_start = schedule[task_id]
        duration = task.estimated_duration.total_seconds()
        max_shift = duration * task.duration_uncertainty
        
        shift = random.uniform(-max_shift, max_shift)
        new_start = max(min_start, current_start + shift)
        
        schedule[task_id] = new_start
    
    def _mutate_task_swap(self, schedule: Dict[str, float], task_graph: TaskGraph) -> None:
        """Swap start times of two compatible tasks."""
        task_ids = list(schedule.keys())
        if len(task_ids) < 2:
            return
        
        # Select two tasks randomly
        task1_id, task2_id = random.sample(task_ids, 2)
        
        # Check if swap is valid (no dependency violations)
        task1 = task_graph.tasks[task1_id]
        task2 = task_graph.tasks[task2_id]
        
        # Simple check: no direct dependency between tasks
        if (task1_id not in task2.dependencies and 
            task2_id not in task1.dependencies):
            # Swap start times
            schedule[task1_id], schedule[task2_id] = schedule[task2_id], schedule[task1_id]
    
    def _mutate_quantum_tunneling(self, schedule: Dict[str, float], task_graph: TaskGraph) -> None:
        """Apply quantum tunneling mutation for exploring distant solutions."""
        if not schedule:
            return
        
        task_id = random.choice(list(schedule.keys()))
        task = task_graph.tasks[task_id]
        
        # Large quantum jump in schedule
        current_start = schedule[task_id]
        duration = task.estimated_duration.total_seconds()
        
        # Tunneling range proportional to task duration and quantum weight
        tunnel_range = duration * task.quantum_weight * 2.0
        tunnel_shift = random.uniform(-tunnel_range, tunnel_range)
        
        new_start = max(0, current_start + tunnel_shift)
        schedule[task_id] = new_start
    
    def _mutate_resource_optimization(self, schedule: Dict[str, float], task_graph: TaskGraph) -> None:
        """Optimize schedule for better resource utilization."""
        if not schedule:
            return
        
        # Find tasks with high resource requirements
        high_resource_tasks = []
        for task_id, task in task_graph.tasks.items():
            if task.resource_requirements:
                total_resource_need = sum(req.amount for req in task.resource_requirements)
                if total_resource_need > 1.0:  # Threshold for "high resource"
                    high_resource_tasks.append(task_id)
        
        if high_resource_tasks:
            task_id = random.choice(high_resource_tasks)
            # Shift to reduce resource conflicts
            current_start = schedule[task_id]
            # Simple shift to earlier or later time
            shift_direction = random.choice([-1, 1])
            shift_amount = random.uniform(60, 300)  # 1-5 minutes
            
            new_start = max(0, current_start + shift_direction * shift_amount)
            schedule[task_id] = new_start
    
    def evaluate_solution(self, solution: Dict[str, Any], task_graph: TaskGraph) -> float:
        """
        Evaluate solution quality using quantum-inspired energy function.
        
        Args:
            solution: Solution to evaluate
            task_graph: Task graph context
            
        Returns:
            Energy value (lower is better)
        """
        if "schedule" not in solution:
            return float('inf')
        
        schedule = solution["schedule"]
        energy = 0.0
        
        # Primary objective component
        if self.objective == OptimizationObjective.MINIMIZE_MAKESPAN:
            makespan = self.calculate_makespan(schedule, task_graph)
            energy += makespan
        elif self.objective == OptimizationObjective.MINIMIZE_RESOURCE_USAGE:
            resource_usage = self.calculate_resource_usage(schedule, task_graph)
            total_usage = sum(resource_usage.values())
            energy += total_usage
        
        # Constraint violation penalties
        constraint_penalty = self._calculate_constraint_penalties(schedule, task_graph)
        energy += constraint_penalty * 1000  # Heavy penalty for violations
        
        # Quantum-inspired energy terms
        quantum_energy = self._calculate_quantum_energy(schedule, task_graph)
        energy += quantum_energy
        
        return energy
    
    def _calculate_constraint_penalties(self, schedule: Dict[str, float], task_graph: TaskGraph) -> float:
        """Calculate penalties for constraint violations."""
        penalty = 0.0
        
        # Dependency constraint violations
        for task_id, start_time in schedule.items():
            if task_id in task_graph.tasks:
                task = task_graph.tasks[task_id]
                for dep_id in task.dependencies:
                    if dep_id in schedule:
                        dep_task = task_graph.tasks[dep_id]
                        dep_completion = schedule[dep_id] + dep_task.estimated_duration.total_seconds()
                        if start_time < dep_completion:
                            penalty += dep_completion - start_time
        
        # Resource constraint violations
        # (Simplified - assumes all tasks run simultaneously for worst case)
        resource_usage = {}
        for task_id, start_time in schedule.items():
            if task_id in task_graph.tasks:
                task = task_graph.tasks[task_id]
                for req in task.resource_requirements:
                    rt = req.resource_type
                    resource_usage[rt] = resource_usage.get(rt, 0) + req.amount
        
        for rt, usage in resource_usage.items():
            if rt in task_graph.resource_constraints:
                constraint = task_graph.resource_constraints[rt]
                if usage > constraint.total_capacity:
                    penalty += (usage - constraint.total_capacity) * 100
        
        return penalty
    
    def _calculate_quantum_energy(self, schedule: Dict[str, float], task_graph: TaskGraph) -> float:
        """Calculate quantum-inspired energy terms."""
        quantum_energy = 0.0
        
        # Entanglement energy: penalize scheduling entangled tasks far apart
        for task_id, start_time in schedule.items():
            if task_id in task_graph.tasks:
                task = task_graph.tasks[task_id]
                for entangled_id in task.entangled_tasks:
                    if entangled_id in schedule:
                        time_separation = abs(start_time - schedule[entangled_id])
                        quantum_energy += time_separation * 0.01  # Small penalty
        
        # Quantum weight influence
        for task_id, start_time in schedule.items():
            if task_id in task_graph.tasks:
                task = task_graph.tasks[task_id]
                # Higher quantum weight tasks prefer earlier scheduling
                quantum_energy += start_time * task.quantum_weight * 0.001
        
        return quantum_energy
    
    def _create_solution_quantum_state(self, solution: Dict[str, Any], task_graph: TaskGraph) -> QuantumState:
        """
        Create quantum state representation of the solution.
        
        Args:
            solution: Solution to represent
            task_graph: Task graph context
            
        Returns:
            Quantum state representing the solution
        """
        num_tasks = len(task_graph.tasks)
        if num_tasks == 0:
            return QuantumState(np.array([1.0]), 0)
        
        # Use limited number of qubits for efficiency
        num_qubits = min(num_tasks, 8)
        
        # Create quantum state based on task scheduling order
        schedule = solution.get("schedule", {})
        sorted_tasks = sorted(schedule.items(), key=lambda x: x[1])  # Sort by start time
        
        # Encode scheduling order in quantum state
        amplitudes = np.zeros(2 ** num_qubits, dtype=complex)
        
        # Simple encoding: binary representation of task order
        for i, (task_id, _) in enumerate(sorted_tasks[:num_qubits]):
            state_index = i % (2 ** num_qubits)
            amplitudes[state_index] = 1.0
        
        # Normalize and add quantum superposition
        if np.sum(np.abs(amplitudes)) > 0:
            state = QuantumState(amplitudes, num_qubits, is_normalized=False)
            state.normalize()
        else:
            # Fallback to uniform superposition
            state = quantum_superposition(num_qubits)
        
        return state