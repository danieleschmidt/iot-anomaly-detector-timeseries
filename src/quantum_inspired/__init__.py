"""
Quantum-Inspired Task Planning Module

This module implements quantum-inspired algorithms for optimizing task planning,
scheduling, and resource allocation. It provides classical simulations of
quantum optimization techniques for practical task management scenarios.

Key Features:
- Quantum Annealing Simulation for combinatorial optimization
- QAOA-inspired multi-objective optimization
- VQE-inspired dependency resolution
- Quantum-inspired genetic algorithms for long-term planning

Authors: Terragon Labs
License: MIT
"""

from .quantum_utils import (
    QuantumState,
    QuantumRegister,
    quantum_superposition,
    quantum_entanglement,
    measure_quantum_state,
)
from .quantum_annealing_planner import QuantumAnnealingPlanner
from .task_representation import Task, TaskGraph, ResourceConstraint
from .quantum_optimization_base import QuantumOptimizationAlgorithm

__version__ = "1.0.0"
__all__ = [
    "QuantumState",
    "QuantumRegister", 
    "quantum_superposition",
    "quantum_entanglement",
    "measure_quantum_state",
    "QuantumAnnealingPlanner",
    "Task",
    "TaskGraph", 
    "ResourceConstraint",
    "QuantumOptimizationAlgorithm",
]