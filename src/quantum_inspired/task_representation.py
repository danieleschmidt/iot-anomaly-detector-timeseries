"""
Task Representation System

Defines data structures and classes for representing tasks, dependencies,
and constraints in quantum-inspired task planning algorithms.
"""

import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import networkx as nx
import numpy as np


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResourceType(Enum):
    """Types of resources that tasks can consume."""
    CPU = "cpu"
    MEMORY = "memory" 
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"
    CUSTOM = "custom"


@dataclass
class ResourceRequirement:
    """Represents resource requirements for a task."""
    resource_type: ResourceType
    amount: float
    unit: str = ""
    max_amount: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate resource requirements."""
        if self.amount < 0:
            raise ValueError("Resource amount cannot be negative")
        if self.max_amount is not None and self.max_amount < self.amount:
            raise ValueError("Maximum amount cannot be less than required amount")


@dataclass
class Task:
    """
    Represents a computational task with quantum-inspired properties.
    
    Tasks can have dependencies, resource requirements, and quantum-inspired
    properties like superposition (uncertain execution time) and entanglement
    (correlated task outcomes).
    """
    
    # Core task properties
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    
    # Timing properties
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    duration_uncertainty: float = 0.1  # Quantum superposition of execution time
    earliest_start: Optional[datetime] = None
    latest_finish: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_finish: Optional[datetime] = None
    
    # Resource requirements
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    
    # Dependencies
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    # Quantum-inspired properties
    quantum_weight: float = 1.0  # Weight in quantum optimization
    entangled_tasks: Set[str] = field(default_factory=set)  # Correlated tasks
    superposition_states: List[Dict[str, Any]] = field(default_factory=list)
    
    # Custom properties
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self) -> None:
        """Validate task properties."""
        if not self.name:
            self.name = f"Task_{self.task_id[:8]}"
        
        if self.duration_uncertainty < 0 or self.duration_uncertainty > 1:
            raise ValueError("Duration uncertainty must be between 0 and 1")
        
        if self.quantum_weight < 0:
            raise ValueError("Quantum weight cannot be negative")
    
    def add_dependency(self, task_id: str) -> None:
        """Add a task dependency."""
        if task_id == self.task_id:
            raise ValueError("Task cannot depend on itself")
        self.dependencies.add(task_id)
    
    def remove_dependency(self, task_id: str) -> None:
        """Remove a task dependency."""
        self.dependencies.discard(task_id)
    
    def add_resource_requirement(self, resource_req: ResourceRequirement) -> None:
        """Add a resource requirement."""
        self.resource_requirements.append(resource_req)
    
    def get_resource_requirement(self, resource_type: ResourceType) -> Optional[ResourceRequirement]:
        """Get resource requirement by type."""
        for req in self.resource_requirements:
            if req.resource_type == resource_type:
                return req
        return None
    
    def entangle_with(self, task_id: str) -> None:
        """Create quantum entanglement with another task."""
        if task_id != self.task_id:
            self.entangled_tasks.add(task_id)
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute (all dependencies completed)."""
        return self.dependencies.issubset(completed_tasks)
    
    def estimate_duration_range(self) -> tuple[timedelta, timedelta]:
        """Get estimated duration range considering uncertainty."""
        base_duration = self.estimated_duration.total_seconds()
        uncertainty_seconds = base_duration * self.duration_uncertainty
        
        min_duration = timedelta(seconds=max(0, base_duration - uncertainty_seconds))
        max_duration = timedelta(seconds=base_duration + uncertainty_seconds)
        
        return min_duration, max_duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "estimated_duration": self.estimated_duration.total_seconds(),
            "duration_uncertainty": self.duration_uncertainty,
            "dependencies": list(self.dependencies),
            "entangled_tasks": list(self.entangled_tasks),
            "quantum_weight": self.quantum_weight,
            "resource_requirements": [
                {
                    "type": req.resource_type.value,
                    "amount": req.amount,
                    "unit": req.unit,
                    "max_amount": req.max_amount
                }
                for req in self.resource_requirements
            ],
            "metadata": self.metadata,
            "tags": list(self.tags)
        }


@dataclass
class ResourceConstraint:
    """Represents constraints on available resources."""
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    unit: str = ""
    allocation_policy: str = "first_fit"  # first_fit, best_fit, worst_fit
    
    def __post_init__(self) -> None:
        """Validate resource constraints."""
        if self.total_capacity < 0 or self.available_capacity < 0:
            raise ValueError("Resource capacity cannot be negative")
        if self.available_capacity > self.total_capacity:
            raise ValueError("Available capacity cannot exceed total capacity")
    
    def can_allocate(self, amount: float) -> bool:
        """Check if resource can be allocated."""
        return self.available_capacity >= amount
    
    def allocate(self, amount: float) -> bool:
        """Allocate resource amount."""
        if self.can_allocate(amount):
            self.available_capacity -= amount
            return True
        return False
    
    def deallocate(self, amount: float) -> None:
        """Deallocate resource amount."""
        self.available_capacity = min(
            self.total_capacity, 
            self.available_capacity + amount
        )


class TaskGraph:
    """
    Represents a directed acyclic graph (DAG) of tasks with quantum-inspired properties.
    
    Supports quantum superposition of execution paths and entanglement between tasks.
    """
    
    def __init__(self):
        """Initialize empty task graph."""
        self.tasks: Dict[str, Task] = {}
        self.graph = nx.DiGraph()
        self.resource_constraints: Dict[ResourceType, ResourceConstraint] = {}
        self.quantum_correlations: Dict[str, Dict[str, float]] = {}
    
    def add_task(self, task: Task) -> None:
        """Add task to the graph."""
        self.tasks[task.task_id] = task
        self.graph.add_node(task.task_id, task=task)
        
        # Add dependency edges
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                self.graph.add_edge(dep_id, task.task_id)
    
    def remove_task(self, task_id: str) -> None:
        """Remove task from the graph."""
        if task_id in self.tasks:
            # Update dependent tasks
            for dependent_id in self.tasks[task_id].dependents:
                if dependent_id in self.tasks:
                    self.tasks[dependent_id].dependencies.discard(task_id)
            
            # Remove from graph and tasks
            self.graph.remove_node(task_id)
            del self.tasks[task_id]
    
    def add_dependency(self, from_task_id: str, to_task_id: str) -> None:
        """Add dependency between tasks."""
        if from_task_id in self.tasks and to_task_id in self.tasks:
            self.tasks[to_task_id].add_dependency(from_task_id)
            self.tasks[from_task_id].dependents.add(to_task_id)
            self.graph.add_edge(from_task_id, to_task_id)
            
            # Check for cycles
            if not nx.is_directed_acyclic_graph(self.graph):
                # Remove the edge that created the cycle
                self.graph.remove_edge(from_task_id, to_task_id)
                self.tasks[to_task_id].remove_dependency(from_task_id)
                self.tasks[from_task_id].dependents.discard(to_task_id)
                raise ValueError("Adding dependency would create a cycle")
    
    def add_resource_constraint(self, constraint: ResourceConstraint) -> None:
        """Add resource constraint."""
        self.resource_constraints[constraint.resource_type] = constraint
    
    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[Task]:
        """Get tasks that are ready to execute."""
        ready_tasks = []
        for task in self.tasks.values():
            if (task.status == TaskStatus.PENDING and 
                task.is_ready(completed_tasks)):
                ready_tasks.append(task)
        return ready_tasks
    
    def get_critical_path(self) -> List[str]:
        """Calculate critical path through the task graph."""
        if not self.tasks:
            return []
        
        # Calculate longest path (critical path)
        try:
            # Add weights based on estimated duration
            weighted_graph = self.graph.copy()
            for task_id, task in self.tasks.items():
                weighted_graph.nodes[task_id]['weight'] = task.estimated_duration.total_seconds()
            
            # Find longest path
            longest_path = nx.dag_longest_path(weighted_graph, weight='weight')
            return longest_path
        except nx.NetworkXError:
            return []
    
    def get_parallel_paths(self) -> List[List[str]]:
        """Identify parallel execution paths."""
        paths = []
        
        # Find all simple paths from sources to sinks
        sources = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        sinks = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        
        for source in sources:
            for sink in sinks:
                for path in nx.all_simple_paths(self.graph, source, sink):
                    paths.append(path)
        
        return paths
    
    def create_quantum_superposition_state(self) -> np.ndarray:
        """
        Create quantum superposition state representing all possible execution orders.
        
        Returns:
            Quantum state vector representing task execution superposition
        """
        num_tasks = len(self.tasks)
        if num_tasks == 0:
            return np.array([1.0])
        
        # Simple superposition of valid topological orderings
        try:
            orderings = list(nx.all_topological_sorts(self.graph))
            if not orderings:
                # Fallback to single ordering
                ordering = list(nx.topological_sort(self.graph))
                orderings = [ordering]
            
            # Create uniform superposition over valid orderings
            state_size = min(len(orderings), 2**min(num_tasks, 10))  # Limit for efficiency
            amplitudes = np.ones(state_size) / np.sqrt(state_size)
            
            return amplitudes
        except nx.NetworkXError:
            # Graph has cycles, return single state
            return np.array([1.0])
    
    def calculate_entanglement_matrix(self) -> np.ndarray:
        """
        Calculate quantum entanglement matrix between tasks.
        
        Returns:
            Matrix representing entanglement correlations
        """
        task_ids = list(self.tasks.keys())
        n = len(task_ids)
        entanglement_matrix = np.zeros((n, n))
        
        for i, task_id_i in enumerate(task_ids):
            task_i = self.tasks[task_id_i]
            for j, task_id_j in enumerate(task_ids):
                if i != j:
                    # Base entanglement on dependencies and explicit entanglement
                    entanglement = 0.0
                    
                    # Dependency-based entanglement
                    if task_id_j in task_i.dependencies or task_id_i in task_i.dependents:
                        entanglement += 0.5
                    
                    # Explicit entanglement
                    if task_id_j in task_i.entangled_tasks:
                        entanglement += 0.3
                    
                    # Resource-based entanglement
                    task_j = self.tasks[task_id_j]
                    shared_resources = set(req.resource_type for req in task_i.resource_requirements) & \
                                     set(req.resource_type for req in task_j.resource_requirements)
                    if shared_resources:
                        entanglement += 0.2 * len(shared_resources)
                    
                    entanglement_matrix[i, j] = min(entanglement, 1.0)
        
        return entanglement_matrix
    
    def validate_graph(self) -> List[str]:
        """
        Validate task graph and return list of issues.
        
        Returns:
            List of validation issues found
        """
        issues = []
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            issues.append("Task graph contains cycles")
        
        # Check for orphaned dependencies
        for task_id, task in self.tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    issues.append(f"Task {task_id} depends on non-existent task {dep_id}")
        
        # Check resource constraints
        for task in self.tasks.values():
            for req in task.resource_requirements:
                if req.resource_type in self.resource_constraints:
                    constraint = self.resource_constraints[req.resource_type]
                    if req.amount > constraint.total_capacity:
                        issues.append(
                            f"Task {task.task_id} requires more {req.resource_type.value} "
                            f"than total capacity"
                        )
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task graph to dictionary representation."""
        return {
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "resource_constraints": {
                rt.value: {
                    "total_capacity": constraint.total_capacity,
                    "available_capacity": constraint.available_capacity, 
                    "unit": constraint.unit,
                    "allocation_policy": constraint.allocation_policy
                }
                for rt, constraint in self.resource_constraints.items()
            },
            "graph_info": {
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
                "is_dag": nx.is_directed_acyclic_graph(self.graph)
            }
        }