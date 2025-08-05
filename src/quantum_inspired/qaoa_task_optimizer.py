"""
QAOA Task Optimizer

Implements Quantum Approximate Optimization Algorithm (QAOA) simulation
for multi-objective task planning and resource optimization. Uses variational
quantum algorithm principles for complex constraint satisfaction problems.
"""

import numpy as np
import time
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import copy
import logging
from enum import Enum

from .quantum_optimization_base import (
    QuantumOptimizationAlgorithm,
    OptimizationObjective,
    OptimizationResult,
    OptimizationParameters
)
from .task_representation import Task, TaskGraph, ResourceConstraint, TaskStatus
from .quantum_utils import (
    QuantumState,
    QuantumRegister,
    quantum_superposition,
    quantum_entanglement,
    quantum_amplitude_amplification
)


class QAOAObjectiveType(Enum):
    """QAOA-specific optimization objectives."""
    QUADRATIC_ASSIGNMENT = "quadratic_assignment"
    MAX_CUT = "max_cut"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class QAOAParameters(OptimizationParameters):
    """
    Parameters for QAOA optimization algorithm.
    
    Attributes:
        circuit_depth: Number of QAOA layers (p parameter)
        mixer_strength: Strength of mixer Hamiltonian
        cost_function_weight: Weight for cost function in Hamiltonian
        constraint_penalty_weight: Penalty weight for constraint violations
        measurement_shots: Number of quantum measurements per iteration
        parameter_bounds: Bounds for variational parameters
        gradient_method: Method for parameter optimization
        adaptive_depth: Whether to adaptively increase circuit depth
        entanglement_pattern: Pattern for quantum entanglement
    """
    circuit_depth: int = 5
    mixer_strength: float = 1.0
    cost_function_weight: float = 1.0
    constraint_penalty_weight: float = 10.0
    measurement_shots: int = 1000
    parameter_bounds: Tuple[float, float] = (-np.pi, np.pi)
    gradient_method: str = "finite_difference"  # finite_difference, parameter_shift
    adaptive_depth: bool = False
    entanglement_pattern: str = "linear"  # linear, all_to_all, circular
    
    def __post_init__(self) -> None:
        """Validate QAOA parameters."""
        super().__post_init__()
        if self.circuit_depth <= 0:
            raise ValueError("Circuit depth must be positive")
        if self.measurement_shots <= 0:
            raise ValueError("Measurement shots must be positive")
        if len(self.parameter_bounds) != 2 or self.parameter_bounds[0] >= self.parameter_bounds[1]:
            raise ValueError("Invalid parameter bounds")


class QAOATaskOptimizer(QuantumOptimizationAlgorithm):
    """
    QAOA-based task optimizer for multi-objective planning.
    
    This optimizer uses quantum approximate optimization principles to solve
    complex task planning problems with multiple objectives and constraints.
    """
    
    def __init__(
        self,
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_MAKESPAN,
        qaoa_objective: QAOAObjectiveType = QAOAObjectiveType.MULTI_OBJECTIVE,
        parameters: Optional[QAOAParameters] = None
    ):
        """
        Initialize QAOA task optimizer.
        
        Args:
            objective: Primary optimization objective
            qaoa_objective: QAOA-specific objective type
            parameters: QAOA parameters
        """
        if parameters is None:
            parameters = QAOAParameters()
        
        super().__init__(objective, parameters)
        self.qaoa_params = parameters
        self.qaoa_objective = qaoa_objective
        
        # QAOA-specific state
        self.variational_parameters: List[float] = []
        self.hamiltonian_terms: List[Dict[str, Any]] = []
        self.measurement_history: List[Dict[str, int]] = []
        self.energy_landscape: List[float] = []
        self.quantum_circuit_depth = parameters.circuit_depth
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def optimize(self, task_graph: TaskGraph) -> OptimizationResult:
        """
        Perform QAOA optimization on task graph.
        
        Args:
            task_graph: Task graph to optimize
            
        Returns:
            Optimization result with best solution found
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
        
        num_tasks = len(task_graph.tasks)
        if num_tasks == 0:
            return OptimizationResult(
                solution={"schedule": {}},
                objective_value=0.0,
                execution_time=time.time() - start_time,
                converged=True
            )
        
        self.logger.info(f"Starting QAOA optimization with {num_tasks} tasks")
        self.logger.info(f"Circuit depth: {self.quantum_circuit_depth}")
        
        # Initialize quantum system
        num_qubits = min(num_tasks * 2, 16)  # Limit for efficiency
        self.initialize_quantum_state(num_qubits)
        
        # Build problem Hamiltonian
        self._build_hamiltonian(task_graph)
        
        # Initialize variational parameters
        self._initialize_parameters()
        
        # Run QAOA optimization loop
        best_solution = None
        best_energy = float('inf')
        converged = False
        
        for iteration in range(self.parameters.max_iterations):
            self.current_iteration = iteration
            
            # Prepare quantum state with current parameters
            quantum_state = self._prepare_qaoa_state(task_graph)
            
            # Measure quantum state to get candidate solution
            solution = self._measure_solution(quantum_state, task_graph)
            
            # Evaluate solution
            energy = self.evaluate_solution(solution, task_graph)
            self.energy_landscape.append(energy)
            
            # Update best solution
            if energy < best_energy:
                best_solution = copy.deepcopy(solution)
                best_energy = energy
                self.best_solution = best_solution
                self.best_objective_value = best_energy
            
            # Update variational parameters
            self._update_parameters(energy, task_graph)
            
            # Log progress
            self.log_progress(iteration, energy, f"Best: {best_energy:.6f}")
            
            # Check convergence
            if self.check_convergence():
                converged = True
                self.logger.info(f"QAOA converged at iteration {iteration}")
                break
            
            # Adaptive depth adjustment
            if (self.qaoa_params.adaptive_depth and 
                iteration > 0 and iteration % 100 == 0 and
                not self._check_recent_improvement()):
                self.quantum_circuit_depth = min(self.quantum_circuit_depth + 1, 10)
                self.logger.info(f"Increased circuit depth to {self.quantum_circuit_depth}")
        
        execution_time = time.time() - start_time
        
        # Create final quantum state
        final_quantum_state = self._prepare_qaoa_state(task_graph)
        
        result = OptimizationResult(
            solution=best_solution or {"schedule": {}},
            objective_value=best_energy,
            quantum_state=final_quantum_state,
            execution_time=execution_time,
            iterations=self.current_iteration,
            converged=converged,
            metadata={
                "circuit_depth": self.quantum_circuit_depth,
                "num_qubits": num_qubits,
                "hamiltonian_terms": len(self.hamiltonian_terms),
                "measurement_shots": self.qaoa_params.measurement_shots,
                "energy_landscape": self.energy_landscape.copy(),
                "final_parameters": self.variational_parameters.copy(),
                "algorithm": "qaoa",
                "qaoa_objective": self.qaoa_objective.value,
                "entanglement_pattern": self.qaoa_params.entanglement_pattern
            }
        )
        
        self.logger.info(f"QAOA optimization completed in {execution_time:.2f}s")
        self.logger.info(f"Best energy: {best_energy:.6f}")
        self.logger.info(f"Final circuit depth: {self.quantum_circuit_depth}")
        
        return result
    
    def _build_hamiltonian(self, task_graph: TaskGraph) -> None:
        """
        Build problem Hamiltonian for QAOA.
        
        Args:
            task_graph: Task graph to encode in Hamiltonian
        """
        self.hamiltonian_terms = []
        
        # Cost function terms
        self._add_cost_hamiltonian_terms(task_graph)
        
        # Constraint terms
        self._add_constraint_hamiltonian_terms(task_graph)
        
        # Multi-objective terms
        if self.qaoa_objective == QAOAObjectiveType.MULTI_OBJECTIVE:
            self._add_multi_objective_terms(task_graph)
        
        self.logger.info(f"Built Hamiltonian with {len(self.hamiltonian_terms)} terms")
    
    def _add_cost_hamiltonian_terms(self, task_graph: TaskGraph) -> None:
        """Add cost function terms to Hamiltonian."""
        task_ids = list(task_graph.tasks.keys())
        
        if self.objective == OptimizationObjective.MINIMIZE_MAKESPAN:
            # Makespan minimization: penalize late task completions
            for i, task_id in enumerate(task_ids):
                task = task_graph.tasks[task_id]
                duration = task.estimated_duration.total_seconds()
                
                # Linear term for task completion time
                self.hamiltonian_terms.append({
                    "type": "linear",
                    "qubit": i,
                    "coefficient": duration * self.qaoa_params.cost_function_weight,
                    "task_id": task_id
                })
        
        elif self.objective == OptimizationObjective.MINIMIZE_RESOURCE_USAGE:
            # Resource usage minimization
            for i, task_id in enumerate(task_ids):
                task = task_graph.tasks[task_id]
                total_resources = sum(req.amount for req in task.resource_requirements)
                
                self.hamiltonian_terms.append({
                    "type": "linear",
                    "qubit": i,
                    "coefficient": total_resources * self.qaoa_params.cost_function_weight,
                    "task_id": task_id
                })
    
    def _add_constraint_hamiltonian_terms(self, task_graph: TaskGraph) -> None:
        """Add constraint terms to Hamiltonian."""
        task_ids = list(task_graph.tasks.keys())
        task_id_to_index = {task_id: i for i, task_id in enumerate(task_ids)}
        
        # Dependency constraints
        for task_id, task in task_graph.tasks.items():
            task_idx = task_id_to_index[task_id]
            
            for dep_id in task.dependencies:
                if dep_id in task_id_to_index:
                    dep_idx = task_id_to_index[dep_id]
                    
                    # Quadratic term: penalize scheduling task before dependency
                    self.hamiltonian_terms.append({
                        "type": "quadratic",
                        "qubits": [task_idx, dep_idx],
                        "coefficient": self.qaoa_params.constraint_penalty_weight,
                        "constraint": "dependency",
                        "task_id": task_id,
                        "dependency_id": dep_id
                    })
        
        # Resource constraints
        resource_groups = {}
        for task_id, task in task_graph.tasks.items():
            for req in task.resource_requirements:
                rt = req.resource_type
                if rt not in resource_groups:
                    resource_groups[rt] = []
                resource_groups[rt].append((task_id, req.amount))
        
        for resource_type, task_resource_pairs in resource_groups.items():
            if resource_type in task_graph.resource_constraints:
                constraint = task_graph.resource_constraints[resource_type]
                
                # Add quadratic terms for resource conflicts
                for i, (task1_id, amount1) in enumerate(task_resource_pairs):
                    for j, (task2_id, amount2) in enumerate(task_resource_pairs[i+1:], i+1):
                        if (amount1 + amount2 > constraint.total_capacity):
                            task1_idx = task_id_to_index[task1_id]
                            task2_idx = task_id_to_index[task2_id]
                            
                            self.hamiltonian_terms.append({
                                "type": "quadratic",
                                "qubits": [task1_idx, task2_idx],
                                "coefficient": self.qaoa_params.constraint_penalty_weight * (amount1 + amount2),
                                "constraint": "resource",
                                "resource_type": resource_type.value,
                                "task1_id": task1_id,
                                "task2_id": task2_id
                            })
    
    def _add_multi_objective_terms(self, task_graph: TaskGraph) -> None:
        """Add multi-objective optimization terms."""
        task_ids = list(task_graph.tasks.keys())
        
        # Load balancing terms
        for i, task_id in enumerate(task_ids):
            task = task_graph.tasks[task_id]
            
            # Encourage even distribution based on task priority
            priority_weight = task.priority.value / 4.0  # Normalize priority
            quantum_weight = task.quantum_weight
            
            combined_weight = priority_weight * quantum_weight * 0.1
            
            self.hamiltonian_terms.append({
                "type": "linear",
                "qubit": i,
                "coefficient": combined_weight,
                "objective": "load_balancing",
                "task_id": task_id
            })
        
        # Entanglement-based terms
        for task_id, task in task_graph.tasks.items():
            task_idx = task_ids.index(task_id)
            
            for entangled_id in task.entangled_tasks:
                if entangled_id in task_ids:
                    entangled_idx = task_ids.index(entangled_id)
                    
                    # Encourage correlated scheduling of entangled tasks
                    self.hamiltonian_terms.append({
                        "type": "quadratic",
                        "qubits": [task_idx, entangled_idx],
                        "coefficient": -0.5,  # Negative for attraction
                        "constraint": "entanglement",
                        "task_id": task_id,
                        "entangled_id": entangled_id
                    })
    
    def _initialize_parameters(self) -> None:
        """Initialize variational parameters for QAOA."""
        num_params = 2 * self.quantum_circuit_depth  # Beta and gamma parameters
        
        # Initialize with random values in bounds
        param_min, param_max = self.qaoa_params.parameter_bounds
        self.variational_parameters = [
            random.uniform(param_min, param_max) for _ in range(num_params)
        ]
        
        self.logger.debug(f"Initialized {num_params} variational parameters")
    
    def _prepare_qaoa_state(self, task_graph: TaskGraph) -> QuantumState:
        """
        Prepare QAOA quantum state with current parameters.
        
        Args:
            task_graph: Task graph context
            
        Returns:
            Prepared quantum state
        """
        if not self.quantum_register:
            return QuantumState(np.array([1.0]), 0)
        
        num_qubits = self.quantum_register.num_qubits
        
        # Start with uniform superposition
        for i in range(num_qubits):
            self.quantum_register.apply_hadamard(i)
        
        # Apply QAOA layers
        for layer in range(self.quantum_circuit_depth):
            gamma_idx = 2 * layer
            beta_idx = 2 * layer + 1
            
            if gamma_idx < len(self.variational_parameters):
                gamma = self.variational_parameters[gamma_idx]
                # Apply cost Hamiltonian evolution
                self._apply_cost_hamiltonian(gamma)
            
            if beta_idx < len(self.variational_parameters):
                beta = self.variational_parameters[beta_idx]
                # Apply mixer Hamiltonian evolution
                self._apply_mixer_hamiltonian(beta)
        
        return self.quantum_register.state
    
    def _apply_cost_hamiltonian(self, gamma: float) -> None:
        """Apply cost Hamiltonian evolution."""
        # Simplified cost Hamiltonian application
        # In a real quantum system, this would be unitary evolution
        
        for term in self.hamiltonian_terms:
            if term["type"] == "linear":
                qubit = term["qubit"]
                coeff = term["coefficient"]
                
                # Apply rotation proportional to coefficient and gamma
                rotation_angle = gamma * coeff * 0.1  # Scale for stability
                
                # Simulate Z-rotation (phase rotation)
                if qubit < self.quantum_register.num_qubits:
                    # Simple phase application (simplified)
                    phase = np.exp(1j * rotation_angle)
                    
                    # Apply phase to states where qubit is |1⟩
                    for i in range(len(self.quantum_register.state.amplitudes)):
                        if (i >> qubit) & 1:  # Qubit is in |1⟩ state
                            self.quantum_register.state.amplitudes[i] *= phase
            
            elif term["type"] == "quadratic":
                qubits = term["qubits"]
                coeff = term["coefficient"]
                
                if len(qubits) == 2 and all(q < self.quantum_register.num_qubits for q in qubits):
                    # Apply two-qubit interaction
                    rotation_angle = gamma * coeff * 0.05
                    
                    # Apply CNOT gate for entanglement (simplified)
                    if rotation_angle > 0.1:  # Threshold for applying gate
                        self.quantum_register.apply_cnot(qubits[0], qubits[1])
    
    def _apply_mixer_hamiltonian(self, beta: float) -> None:
        """Apply mixer Hamiltonian evolution."""
        # Apply X-rotations to all qubits (mixer Hamiltonian)
        for qubit in range(self.quantum_register.num_qubits):
            rotation_angle = beta * self.qaoa_params.mixer_strength
            
            # Apply Hadamard as approximation of X-rotation
            if abs(rotation_angle) > 0.1:  # Threshold for applying gate
                self.quantum_register.apply_hadamard(qubit)
    
    def _measure_solution(self, quantum_state: QuantumState, task_graph: TaskGraph) -> Dict[str, Any]:
        """
        Measure quantum state to extract solution.
        
        Args:
            quantum_state: Quantum state to measure
            task_graph: Task graph context
            
        Returns:
            Measured solution
        """
        # Perform multiple measurements
        measurements = {}
        for _ in range(self.qaoa_params.measurement_shots):
            measurement = quantum_state.measure()
            bit_string = format(measurement, f'0{quantum_state.num_qubits}b')
            measurements[bit_string] = measurements.get(bit_string, 0) + 1
        
        # Find most frequent measurement
        most_frequent = max(measurements.items(), key=lambda x: x[1])
        bit_string = most_frequent[0]
        
        self.measurement_history.append(measurements)
        
        # Convert bit string to schedule
        schedule = self._decode_solution(bit_string, task_graph)
        
        return {
            "schedule": schedule,
            "measurement": bit_string,
            "measurement_counts": measurements,
            "confidence": most_frequent[1] / self.qaoa_params.measurement_shots
        }
    
    def _decode_solution(self, bit_string: str, task_graph: TaskGraph) -> Dict[str, float]:
        """
        Decode bit string measurement to task schedule.
        
        Args:
            bit_string: Binary measurement result
            task_graph: Task graph context
            
        Returns:
            Task schedule
        """
        schedule = {}
        task_ids = list(task_graph.tasks.keys())
        
        # Simple decoding: bit value determines scheduling priority
        bit_values = [int(b) for b in bit_string]
        
        # Create time slots based on bit patterns
        base_time = 0.0
        time_slot_duration = 3600.0  # 1 hour slots
        
        for i, task_id in enumerate(task_ids):
            if i < len(bit_values):
                # Bit determines time slot preference
                bit_value = bit_values[i]
                
                # Multiple bits can influence scheduling
                slot_index = 0
                for j in range(min(3, len(bit_values) - i)):  # Use up to 3 bits
                    if i + j < len(bit_values):
                        slot_index += bit_values[i + j] * (2 ** j)
                
                # Calculate start time with dependency constraints
                earliest_start = base_time + slot_index * time_slot_duration
                
                # Respect dependencies
                task = task_graph.tasks[task_id]
                for dep_id in task.dependencies:
                    if dep_id in schedule:
                        dep_task = task_graph.tasks[dep_id]
                        dep_completion = schedule[dep_id] + dep_task.estimated_duration.total_seconds()
                        earliest_start = max(earliest_start, dep_completion)
                
                schedule[task_id] = earliest_start
        
        return schedule
    
    def _update_parameters(self, energy: float, task_graph: TaskGraph) -> None:
        """
        Update variational parameters using gradient-based optimization.
        
        Args:
            energy: Current energy value
            task_graph: Task graph context
        """
        if self.qaoa_params.gradient_method == "finite_difference":
            self._update_parameters_finite_difference(energy, task_graph)
        elif self.qaoa_params.gradient_method == "parameter_shift":
            self._update_parameters_parameter_shift(energy, task_graph)
    
    def _update_parameters_finite_difference(self, energy: float, task_graph: TaskGraph) -> None:
        """Update parameters using finite difference gradients."""
        gradients = []
        epsilon = 0.01
        
        for i, param in enumerate(self.variational_parameters):
            # Forward difference
            self.variational_parameters[i] = param + epsilon
            quantum_state_plus = self._prepare_qaoa_state(task_graph)
            solution_plus = self._measure_solution(quantum_state_plus, task_graph)
            energy_plus = self.evaluate_solution(solution_plus, task_graph)
            
            # Backward difference
            self.variational_parameters[i] = param - epsilon
            quantum_state_minus = self._prepare_qaoa_state(task_graph)
            solution_minus = self._measure_solution(quantum_state_minus, task_graph)
            energy_minus = self.evaluate_solution(solution_minus, task_graph)
            
            # Calculate gradient
            gradient = (energy_plus - energy_minus) / (2 * epsilon)
            gradients.append(gradient)
            
            # Restore parameter
            self.variational_parameters[i] = param
        
        # Update parameters using gradient descent
        for i, gradient in enumerate(gradients):
            update = self.parameters.learning_rate * gradient
            new_param = self.variational_parameters[i] - update
            
            # Apply bounds
            param_min, param_max = self.qaoa_params.parameter_bounds
            new_param = max(param_min, min(param_max, new_param))
            
            self.variational_parameters[i] = new_param
    
    def _update_parameters_parameter_shift(self, energy: float, task_graph: TaskGraph) -> None:
        """Update parameters using parameter shift rule (simplified)."""
        # Simplified parameter shift implementation
        shift = np.pi / 4
        
        for i in range(len(self.variational_parameters)):
            # Shift parameter
            original_param = self.variational_parameters[i]
            
            # Measure at shifted parameters
            self.variational_parameters[i] = original_param + shift
            quantum_state_plus = self._prepare_qaoa_state(task_graph)
            solution_plus = self._measure_solution(quantum_state_plus, task_graph)
            energy_plus = self.evaluate_solution(solution_plus, task_graph)
            
            self.variational_parameters[i] = original_param - shift
            quantum_state_minus = self._prepare_qaoa_state(task_graph)
            solution_minus = self._measure_solution(quantum_state_minus, task_graph)
            energy_minus = self.evaluate_solution(solution_minus, task_graph)
            
            # Calculate gradient using parameter shift
            gradient = (energy_plus - energy_minus) / 2
            
            # Update parameter
            update = self.parameters.learning_rate * gradient
            new_param = original_param - update
            
            # Apply bounds
            param_min, param_max = self.qaoa_params.parameter_bounds
            new_param = max(param_min, min(param_max, new_param))
            
            self.variational_parameters[i] = new_param
    
    def _check_recent_improvement(self, window_size: int = 50) -> bool:
        """Check if there has been recent improvement in energy."""
        if len(self.energy_landscape) < window_size:
            return True
        
        recent_energies = self.energy_landscape[-window_size:]
        best_recent = min(recent_energies)
        worst_recent = max(recent_energies)
        
        improvement_ratio = (worst_recent - best_recent) / worst_recent if worst_recent > 0 else 0
        return improvement_ratio > 0.01  # 1% improvement threshold
    
    def evaluate_solution(self, solution: Dict[str, Any], task_graph: TaskGraph) -> float:
        """
        Evaluate solution using QAOA-specific energy function.
        
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
        
        # Evaluate using Hamiltonian terms
        for term in self.hamiltonian_terms:
            if term["type"] == "linear":
                task_id = term["task_id"]
                if task_id in schedule:
                    # Linear term contribution
                    energy += term["coefficient"] * schedule[task_id] * 1e-6  # Scale for time
            
            elif term["type"] == "quadratic":
                task1_id = term.get("task1_id") or term.get("task_id")
                task2_id = term.get("task2_id") or term.get("dependency_id") or term.get("entangled_id")
                
                if task1_id in schedule and task2_id in schedule:
                    # Quadratic term contribution
                    if "constraint" in term:
                        constraint_type = term["constraint"]
                        
                        if constraint_type == "dependency":
                            # Penalty for dependency violations
                            task1 = task_graph.tasks[task1_id]
                            task1_completion = schedule[task1_id] + task1.estimated_duration.total_seconds()
                            
                            if schedule[task2_id] > task1_completion:
                                energy += term["coefficient"]  # Positive penalty
                        
                        elif constraint_type == "resource":
                            # Penalty for resource conflicts (simplified)
                            time_overlap = abs(schedule[task1_id] - schedule[task2_id])
                            if time_overlap < 3600:  # 1 hour threshold
                                energy += term["coefficient"] * (3600 - time_overlap) / 3600
                        
                        elif constraint_type == "entanglement":
                            # Entanglement correlation term
                            time_separation = abs(schedule[task1_id] - schedule[task2_id])
                            energy += term["coefficient"] * time_separation * 1e-6
        
        # Add base objective
        if self.objective == OptimizationObjective.MINIMIZE_MAKESPAN:
            makespan = self.calculate_makespan(schedule, task_graph)
            energy += makespan * 1e-6  # Scale to match Hamiltonian terms
        
        return energy