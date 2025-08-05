"""
Task Planning Module

Provides high-level interfaces for quantum-inspired task planning and scheduling.
Integrates quantum optimization algorithms with practical task management functionality.

Key Components:
- TaskScheduler: Main interface for task scheduling
- ResourceAllocator: Resource management and allocation
- ConstraintSolver: Constraint satisfaction and optimization
- PlanningOptimizer: Integration with quantum algorithms

Authors: Terragon Labs
License: MIT
"""

from .task_scheduler import TaskScheduler, SchedulingAlgorithm, SchedulingResult

# Note: Additional components can be implemented as needed
# from .resource_allocator import ResourceAllocator, AllocationStrategy, AllocationResult
# from .constraint_solver import ConstraintSolver, ConstraintType, ConstraintSolution
# from .planning_optimizer import PlanningOptimizer, OptimizationStrategy

__version__ = "1.0.0"
__all__ = [
    "TaskScheduler",
    "SchedulingAlgorithm", 
    "SchedulingResult",
    # "ResourceAllocator",
    # "AllocationStrategy",
    # "AllocationResult",
    # "ConstraintSolver",
    # "ConstraintType",
    # "ConstraintSolution",
    # "PlanningOptimizer",
    # "OptimizationStrategy",
]