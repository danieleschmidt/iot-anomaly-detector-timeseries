"""
Quantum Simulation Utilities

Provides basic quantum state simulation capabilities for quantum-inspired
optimization algorithms. Implements classical simulation of quantum states,
superposition, entanglement, and measurement operations.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import random
from enum import Enum


class QuantumBasisState(Enum):
    """Quantum basis states for qubits."""
    ZERO = 0
    ONE = 1


@dataclass
class QuantumState:
    """
    Represents a quantum state as a classical simulation.
    
    Attributes:
        amplitudes: Complex probability amplitudes for each basis state
        num_qubits: Number of qubits in the system
        is_normalized: Whether the state is properly normalized
    """
    amplitudes: np.ndarray
    num_qubits: int
    is_normalized: bool = True
    
    def __post_init__(self) -> None:
        """Validate and normalize the quantum state."""
        if len(self.amplitudes) != 2 ** self.num_qubits:
            raise ValueError(
                f"Amplitude array size {len(self.amplitudes)} does not match "
                f"2^{self.num_qubits} = {2**self.num_qubits}"
            )
        
        if not self.is_normalized:
            self.normalize()
    
    def normalize(self) -> None:
        """Normalize the quantum state."""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes = self.amplitudes / norm
            self.is_normalized = True
    
    def measure(self, qubit_index: Optional[int] = None) -> int:
        """
        Measure the quantum state and collapse to a classical state.
        
        Args:
            qubit_index: Specific qubit to measure (None for full measurement)
            
        Returns:
            Measured classical state (integer representation)
        """
        probabilities = np.abs(self.amplitudes) ** 2
        
        if qubit_index is not None:
            # Measure specific qubit
            prob_zero = sum(
                probabilities[i] for i in range(len(probabilities))
                if not (i >> qubit_index) & 1
            )
            return 0 if random.random() < prob_zero else 1
        else:
            # Measure entire system
            return np.random.choice(len(probabilities), p=probabilities)
    
    def get_probability(self, state: int) -> float:
        """Get probability of measuring a specific state."""
        return float(np.abs(self.amplitudes[state]) ** 2)


class QuantumRegister:
    """
    Quantum register for managing multiple qubits and quantum operations.
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize quantum register with specified number of qubits.
        
        Args:
            num_qubits: Number of qubits in the register
        """
        self.num_qubits = num_qubits
        self.state = self._initialize_zero_state()
        self.measurement_history: List[int] = []
    
    def _initialize_zero_state(self) -> QuantumState:
        """Initialize all qubits in |0⟩ state."""
        amplitudes = np.zeros(2 ** self.num_qubits, dtype=complex)
        amplitudes[0] = 1.0  # |00...0⟩ state
        return QuantumState(amplitudes, self.num_qubits)
    
    def apply_hadamard(self, qubit_index: int) -> None:
        """Apply Hadamard gate to create superposition."""
        if qubit_index >= self.num_qubits:
            raise ValueError(f"Qubit index {qubit_index} out of range")
            
        new_amplitudes = np.zeros_like(self.state.amplitudes)
        
        for i in range(len(self.state.amplitudes)):
            # Check if qubit at qubit_index is 0 or 1
            if (i >> qubit_index) & 1 == 0:
                # Qubit is 0: |0⟩ → (|0⟩ + |1⟩)/√2
                j = i | (1 << qubit_index)  # Flip the qubit
                new_amplitudes[i] += self.state.amplitudes[i] / np.sqrt(2)
                new_amplitudes[j] += self.state.amplitudes[i] / np.sqrt(2)
            else:
                # Qubit is 1: |1⟩ → (|0⟩ - |1⟩)/√2
                j = i & ~(1 << qubit_index)  # Flip the qubit
                new_amplitudes[j] += self.state.amplitudes[i] / np.sqrt(2)
                new_amplitudes[i] -= self.state.amplitudes[i] / np.sqrt(2)
        
        self.state.amplitudes = new_amplitudes
    
    def apply_cnot(self, control_qubit: int, target_qubit: int) -> None:
        """Apply CNOT gate for entanglement."""
        if control_qubit >= self.num_qubits or target_qubit >= self.num_qubits:
            raise ValueError("Qubit indices out of range")
            
        new_amplitudes = np.copy(self.state.amplitudes)
        
        for i in range(len(self.state.amplitudes)):
            # If control qubit is 1, flip target qubit
            if (i >> control_qubit) & 1:
                j = i ^ (1 << target_qubit)  # Flip target qubit
                new_amplitudes[i], new_amplitudes[j] = new_amplitudes[j], new_amplitudes[i]
        
        self.state.amplitudes = new_amplitudes
    
    def measure_all(self) -> int:
        """Measure all qubits and return classical state."""
        result = self.state.measure()
        self.measurement_history.append(result)
        return result
    
    def measure_qubit(self, qubit_index: int) -> int:
        """Measure specific qubit."""
        return self.state.measure(qubit_index)
    
    def get_state_probabilities(self) -> Dict[str, float]:
        """Get probabilities for all possible states."""
        probabilities = {}
        for i in range(2 ** self.num_qubits):
            state_str = format(i, f'0{self.num_qubits}b')
            prob = self.state.get_probability(i)
            if prob > 1e-10:  # Only include non-negligible probabilities
                probabilities[state_str] = prob
        return probabilities


def quantum_superposition(num_qubits: int) -> QuantumState:
    """
    Create uniform superposition state |+⟩^⊗n = (|0⟩ + |1⟩)^⊗n / 2^(n/2).
    
    Args:
        num_qubits: Number of qubits
        
    Returns:
        Quantum state in uniform superposition
    """
    num_states = 2 ** num_qubits
    amplitudes = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    return QuantumState(amplitudes, num_qubits)


def quantum_entanglement(num_qubits: int, entanglement_pattern: str = "ghz") -> QuantumState:
    """
    Create entangled quantum states.
    
    Args:
        num_qubits: Number of qubits
        entanglement_pattern: Type of entanglement ("ghz", "bell", "cluster")
        
    Returns:
        Entangled quantum state
    """
    amplitudes = np.zeros(2 ** num_qubits, dtype=complex)
    
    if entanglement_pattern == "ghz":
        # GHZ state: (|00...0⟩ + |11...1⟩) / √2
        amplitudes[0] = 1.0 / np.sqrt(2)  # |00...0⟩
        amplitudes[-1] = 1.0 / np.sqrt(2)  # |11...1⟩
    elif entanglement_pattern == "bell" and num_qubits == 2:
        # Bell state: (|00⟩ + |11⟩) / √2
        amplitudes[0] = 1.0 / np.sqrt(2)  # |00⟩
        amplitudes[3] = 1.0 / np.sqrt(2)  # |11⟩
    elif entanglement_pattern == "cluster":
        # Simple cluster state approximation
        for i in range(0, 2 ** num_qubits, 2):
            amplitudes[i] = 1.0 / np.sqrt(2 ** (num_qubits - 1))
    else:
        # Default to superposition
        return quantum_superposition(num_qubits)
    
    return QuantumState(amplitudes, num_qubits)


def measure_quantum_state(state: QuantumState, num_measurements: int = 1000) -> Dict[str, int]:
    """
    Perform multiple measurements on a quantum state for statistical analysis.
    
    Args:
        state: Quantum state to measure
        num_measurements: Number of measurements to perform
        
    Returns:
        Dictionary with measurement outcomes and their frequencies
    """
    measurements = {}
    
    for _ in range(num_measurements):
        result = state.measure()
        state_str = format(result, f'0{state.num_qubits}b')
        measurements[state_str] = measurements.get(state_str, 0) + 1
    
    return measurements


def quantum_phase_estimation(eigenvalue: float, precision_bits: int = 4) -> QuantumState:
    """
    Simplified quantum phase estimation for optimization algorithms.
    
    Args:
        eigenvalue: Eigenvalue to estimate (between 0 and 1)
        precision_bits: Number of precision bits
        
    Returns:
        Quantum state encoding the phase
    """
    # Classical simulation of quantum phase estimation
    estimated_phase = int(eigenvalue * (2 ** precision_bits)) % (2 ** precision_bits)
    
    amplitudes = np.zeros(2 ** precision_bits, dtype=complex)
    # Add some quantum uncertainty around the estimated phase
    for i in range(max(0, estimated_phase - 1), min(2 ** precision_bits, estimated_phase + 2)):
        weight = 1.0 - abs(i - estimated_phase) * 0.3
        amplitudes[i] = weight
    
    state = QuantumState(amplitudes, precision_bits, is_normalized=False)
    state.normalize()
    return state


def quantum_amplitude_amplification(
    state: QuantumState, 
    target_states: List[int], 
    iterations: int = 1
) -> QuantumState:
    """
    Simplified quantum amplitude amplification to boost target state probabilities.
    
    Args:
        state: Initial quantum state
        target_states: List of target state indices to amplify
        iterations: Number of amplification iterations
        
    Returns:
        Quantum state with amplified target states
    """
    amplitudes = np.copy(state.amplitudes)
    
    for _ in range(iterations):
        # Amplify target states
        for target in target_states:
            if target < len(amplitudes):
                amplitudes[target] *= 1.1  # Simple amplification
        
        # Suppress non-target states slightly
        for i in range(len(amplitudes)):
            if i not in target_states:
                amplitudes[i] *= 0.95
    
    result_state = QuantumState(amplitudes, state.num_qubits, is_normalized=False)
    result_state.normalize()
    return result_state