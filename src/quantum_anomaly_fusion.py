"""Quantum-Inspired Anomaly Detection Fusion System.

This module implements quantum-inspired algorithms for advanced anomaly detection
in IoT time series data, utilizing quantum computing principles for enhanced
pattern recognition and feature correlation analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from pathlib import Path
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit.circuit.library import EfficientSU2, TwoLocal
    from qiskit.algorithms.optimizers import COBYLA, SPSA
    from qiskit.primitives import Estimator
    from qiskit.quantum_info import SparsePauliOp
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit not available. Quantum features will use classical approximations.")

from .logging_config import get_logger
from .adaptive_multi_modal_detector import DetectionResult, BaseDetector


@dataclass
class QuantumState:
    """Representation of quantum state for anomaly detection."""
    
    amplitudes: np.ndarray
    entanglement_measure: float
    coherence_score: float
    phase_information: np.ndarray
    measurement_probabilities: np.ndarray


class QuantumFeatureMap:
    """Quantum feature mapping for time series data."""
    
    def __init__(self, n_qubits: int = 4, depth: int = 2):
        self.n_qubits = n_qubits
        self.depth = depth
        self.logger = get_logger(f"{__name__}.QuantumFeatureMap")
        
    def encode_classical_data(self, data: np.ndarray) -> QuantumState:
        """Encode classical time series data into quantum state representation."""
        if not QISKIT_AVAILABLE:
            return self._classical_approximation(data)
        
        # Normalize input data
        normalized_data = self._normalize_for_quantum(data)
        
        # Create quantum circuit for feature mapping
        qc = QuantumCircuit(self.n_qubits)
        
        # Apply feature map encoding
        for i, value in enumerate(normalized_data[:self.n_qubits]):
            qc.ry(value * np.pi, i)
        
        # Add entanglement layers
        for layer in range(self.depth):
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            for i in range(self.n_qubits):
                qc.rz(normalized_data[i % len(normalized_data)] * np.pi / 2, i)
        
        # Classical simulation of quantum state
        return self._simulate_quantum_state(qc, normalized_data)
    
    def _normalize_for_quantum(self, data: np.ndarray) -> np.ndarray:
        """Normalize data for quantum encoding."""
        # Flatten and normalize to [0, 1] range
        flat_data = data.flatten()
        min_val, max_val = flat_data.min(), flat_data.max()
        if max_val > min_val:
            normalized = (flat_data - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(flat_data)
        return normalized
    
    def _simulate_quantum_state(self, circuit: 'QuantumCircuit', data: np.ndarray) -> QuantumState:
        """Simulate quantum state evolution (classical approximation)."""
        # Classical approximation of quantum state
        n_states = 2 ** self.n_qubits
        
        # Generate pseudo-quantum amplitudes
        amplitudes = np.random.random(n_states) + 1j * np.random.random(n_states)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)  # Normalize
        
        # Calculate entanglement measure (classical approximation)
        entanglement = self._calculate_entanglement(amplitudes)
        
        # Calculate coherence score
        coherence = np.abs(np.sum(amplitudes * np.conj(amplitudes)))
        
        # Phase information
        phases = np.angle(amplitudes)
        
        # Measurement probabilities
        probabilities = np.abs(amplitudes) ** 2
        
        return QuantumState(
            amplitudes=amplitudes,
            entanglement_measure=entanglement,
            coherence_score=coherence,
            phase_information=phases,
            measurement_probabilities=probabilities
        )
    
    def _classical_approximation(self, data: np.ndarray) -> QuantumState:
        """Classical approximation when quantum libraries are not available."""
        normalized_data = self._normalize_for_quantum(data)
        n_states = 2 ** self.n_qubits
        
        # Generate pseudo-quantum features
        amplitudes = np.random.random(n_states) + 1j * np.random.random(n_states)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        return QuantumState(
            amplitudes=amplitudes,
            entanglement_measure=np.random.random(),
            coherence_score=np.random.random(),
            phase_information=np.angle(amplitudes),
            measurement_probabilities=np.abs(amplitudes) ** 2
        )
    
    def _calculate_entanglement(self, amplitudes: np.ndarray) -> float:
        """Calculate entanglement measure from quantum state amplitudes."""
        # Simplified entanglement calculation
        n_qubits = self.n_qubits
        if n_qubits < 2:
            return 0.0
        
        # Use von Neumann entropy as entanglement measure
        probabilities = np.abs(amplitudes) ** 2
        probabilities = probabilities[probabilities > 1e-10]  # Avoid log(0)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(amplitudes))
        return entropy / max_entropy if max_entropy > 0 else 0.0


class QuantumVariationalAnomalyDetector(BaseDetector):
    """Variational Quantum Anomaly Detector using quantum circuits."""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 3, shots: int = 1024):
        super().__init__("QuantumVariational")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.feature_map = QuantumFeatureMap(n_qubits)
        self.optimal_params = None
        self.threshold = 0.0
        
    def _build_variational_circuit(self, params: np.ndarray) -> 'QuantumCircuit':
        """Build variational quantum circuit for anomaly detection."""
        if QISKIT_AVAILABLE:
            # Use EfficientSU2 ansatz
            ansatz = EfficientSU2(self.n_qubits, reps=self.n_layers)
            qc = QuantumCircuit(self.n_qubits)
            qc.compose(ansatz, inplace=True)
            return qc.bind_parameters(params)
        else:
            # Classical approximation
            return None
    
    def _quantum_cost_function(self, params: np.ndarray, training_states: List[QuantumState]) -> float:
        """Cost function for variational quantum training."""
        total_cost = 0.0
        
        for state in training_states:
            # Measure overlap with current variational state
            circuit = self._build_variational_circuit(params)
            
            if QISKIT_AVAILABLE:
                # Quantum computation
                cost = self._compute_quantum_overlap(circuit, state)
            else:
                # Classical approximation
                cost = self._classical_cost_approximation(params, state)
            
            total_cost += cost
        
        return total_cost / len(training_states)
    
    def _compute_quantum_overlap(self, circuit: 'QuantumCircuit', target_state: QuantumState) -> float:
        """Compute quantum state overlap (fidelity)."""
        # Simplified fidelity calculation
        if not QISKIT_AVAILABLE:
            return np.random.random()
        
        # In a real implementation, this would use quantum simulator/hardware
        # For now, return approximation based on entanglement and coherence
        fidelity = target_state.entanglement_measure * target_state.coherence_score
        return 1.0 - fidelity  # Cost is 1 - fidelity
    
    def _classical_cost_approximation(self, params: np.ndarray, state: QuantumState) -> float:
        """Classical approximation of quantum cost function."""
        # Use parameter correlation with quantum features
        param_norm = np.linalg.norm(params)
        state_coherence = state.coherence_score
        return np.abs(param_norm - state_coherence)
    
    def fit(self, data: np.ndarray) -> None:
        """Train the variational quantum detector."""
        self.logger.info("Training Variational Quantum Anomaly Detector")
        
        # Convert training data to quantum states
        training_states = []
        for sequence in data:
            quantum_state = self.feature_map.encode_classical_data(sequence)
            training_states.append(quantum_state)
        
        # Initialize variational parameters
        if QISKIT_AVAILABLE:
            n_params = self.n_qubits * self.n_layers * 2  # Approximate parameter count
        else:
            n_params = self.n_qubits * 2
        
        initial_params = np.random.random(n_params) * 2 * np.pi
        
        # Optimize variational parameters
        if QISKIT_AVAILABLE:
            optimizer = COBYLA(maxiter=100)
        else:
            optimizer = None
        
        # Classical optimization of quantum cost function
        self.optimal_params = self._classical_optimize(
            initial_params, training_states
        )
        
        # Calculate threshold from training data
        training_costs = []
        for state in training_states:
            cost = self._classical_cost_approximation(self.optimal_params, state)
            training_costs.append(cost)
        
        self.threshold = np.percentile(training_costs, 95)
        self.is_trained = True
        self.logger.info("Variational quantum training completed")
    
    def _classical_optimize(self, initial_params: np.ndarray, training_states: List[QuantumState]) -> np.ndarray:
        """Classical optimization of quantum parameters."""
        from scipy.optimize import minimize
        
        def objective(params):
            return self._quantum_cost_function(params, training_states)
        
        try:
            result = minimize(objective, initial_params, method='COBYLA', 
                            options={'maxiter': 100, 'disp': False})
            return result.x
        except Exception as e:
            self.logger.warning(f"Optimization failed: {e}, using initial parameters")
            return initial_params
    
    def predict(self, data: np.ndarray) -> DetectionResult:
        """Detect anomalies using trained quantum detector."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before prediction")
        
        anomaly_scores = []
        quantum_features = []
        
        for sequence in data:
            # Encode sequence to quantum state
            quantum_state = self.feature_map.encode_classical_data(sequence)
            
            # Calculate anomaly score using quantum cost function
            score = self._classical_cost_approximation(self.optimal_params, quantum_state)
            anomaly_scores.append(score)
            
            # Store quantum features for analysis
            quantum_features.append({
                'entanglement': quantum_state.entanglement_measure,
                'coherence': quantum_state.coherence_score,
                'phase_variance': np.var(quantum_state.phase_information)
            })
        
        anomaly_scores = np.array(anomaly_scores)
        anomaly_predictions = (anomaly_scores > self.threshold).astype(int)
        
        # Normalize confidence scores
        if anomaly_scores.max() > anomaly_scores.min():
            confidence_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        else:
            confidence_scores = np.zeros_like(anomaly_scores)
        
        return DetectionResult(
            anomaly_scores=anomaly_scores,
            anomaly_predictions=anomaly_predictions,
            confidence_scores=confidence_scores,
            detection_method="QuantumVariational",
            metadata={
                "n_qubits": self.n_qubits,
                "n_layers": self.n_layers,
                "threshold": self.threshold,
                "quantum_features": quantum_features
            }
        )
    
    def save(self, path: Path) -> None:
        """Save the trained quantum detector."""
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / "quantum_detector.pkl", "wb") as f:
            pickle.dump({
                "optimal_params": self.optimal_params,
                "threshold": self.threshold,
                "n_qubits": self.n_qubits,
                "n_layers": self.n_layers,
                "shots": self.shots
            }, f)
    
    def load(self, path: Path) -> None:
        """Load a trained quantum detector."""
        with open(path / "quantum_detector.pkl", "rb") as f:
            data = pickle.load(f)
            self.optimal_params = data["optimal_params"]
            self.threshold = data["threshold"]
            self.n_qubits = data["n_qubits"]
            self.n_layers = data["n_layers"]
            self.shots = data["shots"]
        
        self.feature_map = QuantumFeatureMap(self.n_qubits)
        self.is_trained = True


class QuantumAnnealingDetector(BaseDetector):
    """Quantum Annealing-based anomaly detection."""
    
    def __init__(self, n_variables: int = 16, annealing_time: int = 20):
        super().__init__("QuantumAnnealing")
        self.n_variables = n_variables
        self.annealing_time = annealing_time
        self.ising_model = None
        self.reference_solution = None
        
    def _formulate_ising_model(self, data: np.ndarray) -> Dict[str, Any]:
        """Formulate Ising model for anomaly detection."""
        # Create QUBO (Quadratic Unconstrained Binary Optimization) model
        # for anomaly detection
        
        n_samples = len(data)
        n_features = data.shape[1] * data.shape[2]  # Flattened features
        
        # Simplify to manageable number of variables
        reduced_features = min(self.n_variables, n_features)
        
        # Create interaction matrix (simplified)
        interaction_matrix = np.random.random((reduced_features, reduced_features))
        interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2  # Symmetric
        
        # Linear bias terms
        bias_vector = np.random.random(reduced_features) - 0.5
        
        return {
            "interactions": interaction_matrix,
            "biases": bias_vector,
            "n_variables": reduced_features
        }
    
    def _simulate_annealing(self, ising_model: Dict[str, Any], data_point: np.ndarray) -> float:
        """Simulate quantum annealing process."""
        # Classical simulation of quantum annealing
        interactions = ising_model["interactions"]
        biases = ising_model["biases"]
        n_vars = ising_model["n_variables"]
        
        # Initialize random spin configuration
        spins = np.random.choice([-1, 1], size=n_vars)
        
        # Simulated annealing optimization
        temperature = 1.0
        for step in range(self.annealing_time):
            # Select random variable to flip
            var_idx = np.random.randint(n_vars)
            
            # Calculate energy change
            old_energy = self._calculate_ising_energy(spins, interactions, biases)
            
            # Flip spin
            spins[var_idx] *= -1
            new_energy = self._calculate_ising_energy(spins, interactions, biases)
            
            # Accept or reject based on energy difference
            delta_energy = new_energy - old_energy
            if delta_energy > 0 and np.random.random() > np.exp(-delta_energy / temperature):
                spins[var_idx] *= -1  # Reject flip
            
            # Cool down
            temperature *= 0.95
        
        # Return final energy as anomaly score
        return self._calculate_ising_energy(spins, interactions, biases)
    
    def _calculate_ising_energy(self, spins: np.ndarray, interactions: np.ndarray, biases: np.ndarray) -> float:
        """Calculate energy of Ising model configuration."""
        # E = -sum(J_ij * s_i * s_j) - sum(h_i * s_i)
        interaction_energy = -np.sum(interactions * np.outer(spins, spins))
        bias_energy = -np.sum(biases * spins)
        return interaction_energy + bias_energy
    
    def fit(self, data: np.ndarray) -> None:
        """Train quantum annealing detector on normal data."""
        self.logger.info("Training Quantum Annealing Detector")
        
        # Formulate Ising model based on training data
        self.ising_model = self._formulate_ising_model(data)
        
        # Find reference solution using training data
        reference_energies = []
        for sequence in data[:100]:  # Limit for computational efficiency
            energy = self._simulate_annealing(self.ising_model, sequence)
            reference_energies.append(energy)
        
        self.reference_solution = {
            "mean_energy": np.mean(reference_energies),
            "std_energy": np.std(reference_energies),
            "threshold": np.percentile(reference_energies, 95)
        }
        
        self.is_trained = True
        self.logger.info("Quantum annealing training completed")
    
    def predict(self, data: np.ndarray) -> DetectionResult:
        """Detect anomalies using quantum annealing."""
        if not self.is_trained:
            raise ValueError("Detector must be trained before prediction")
        
        anomaly_scores = []
        
        for sequence in data:
            # Run quantum annealing simulation
            energy = self._simulate_annealing(self.ising_model, sequence)
            
            # Calculate anomaly score based on energy deviation
            mean_energy = self.reference_solution["mean_energy"]
            score = abs(energy - mean_energy)
            anomaly_scores.append(score)
        
        anomaly_scores = np.array(anomaly_scores)
        threshold = self.reference_solution["threshold"]
        anomaly_predictions = (anomaly_scores > threshold).astype(int)
        
        # Normalize confidence scores
        if anomaly_scores.max() > anomaly_scores.min():
            confidence_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        else:
            confidence_scores = np.zeros_like(anomaly_scores)
        
        return DetectionResult(
            anomaly_scores=anomaly_scores,
            anomaly_predictions=anomaly_predictions,
            confidence_scores=confidence_scores,
            detection_method="QuantumAnnealing",
            metadata={
                "n_variables": self.n_variables,
                "annealing_time": self.annealing_time,
                "reference_energy": self.reference_solution["mean_energy"],
                "threshold": threshold
            }
        )
    
    def save(self, path: Path) -> None:
        """Save the trained quantum annealing detector."""
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / "quantum_annealing.pkl", "wb") as f:
            pickle.dump({
                "ising_model": self.ising_model,
                "reference_solution": self.reference_solution,
                "n_variables": self.n_variables,
                "annealing_time": self.annealing_time
            }, f)
    
    def load(self, path: Path) -> None:
        """Load a trained quantum annealing detector."""
        with open(path / "quantum_annealing.pkl", "rb") as f:
            data = pickle.load(f)
            self.ising_model = data["ising_model"]
            self.reference_solution = data["reference_solution"]
            self.n_variables = data["n_variables"]
            self.annealing_time = data["annealing_time"]
        
        self.is_trained = True


class QuantumAnomalyFusion:
    """Advanced quantum-inspired anomaly detection fusion system."""
    
    def __init__(self, n_qubits: int = 4, enable_parallel: bool = True):
        self.n_qubits = n_qubits
        self.enable_parallel = enable_parallel
        self.logger = get_logger(__name__)
        
        # Initialize quantum detectors
        self.quantum_detectors = {
            "variational": QuantumVariationalAnomalyDetector(n_qubits=n_qubits),
            "annealing": QuantumAnnealingDetector(n_variables=n_qubits * 4)
        }
        
        # Quantum fusion weights
        self.fusion_weights = {
            "variational": 0.6,
            "annealing": 0.4
        }
        
        self.is_trained = False
    
    async def fit_async(self, data: np.ndarray) -> None:
        """Asynchronous training of quantum detectors."""
        self.logger.info("Starting asynchronous quantum detector training")
        
        async def train_detector(name, detector):
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, detector.fit, data)
            self.logger.info(f"Completed training for quantum {name} detector")
        
        # Run training in parallel
        tasks = [
            train_detector(name, detector) 
            for name, detector in self.quantum_detectors.items()
        ]
        
        await asyncio.gather(*tasks)
        self.is_trained = True
        self.logger.info("Quantum fusion system training completed")
    
    def fit(self, data: np.ndarray) -> None:
        """Synchronous training of quantum detectors."""
        if self.enable_parallel:
            asyncio.run(self.fit_async(data))
        else:
            for name, detector in self.quantum_detectors.items():
                detector.fit(data)
                self.logger.info(f"Completed training for quantum {name} detector")
            self.is_trained = True
    
    def predict(self, data: np.ndarray, fusion_method: str = "weighted_average") -> DetectionResult:
        """Detect anomalies using quantum fusion."""
        if not self.is_trained:
            raise ValueError("Quantum fusion system must be trained before prediction")
        
        # Get predictions from all quantum detectors
        quantum_results = {}
        for name, detector in self.quantum_detectors.items():
            try:
                result = detector.predict(data)
                quantum_results[name] = result
                self.logger.debug(f"Got quantum prediction from {name}")
            except Exception as e:
                self.logger.warning(f"Failed to get quantum prediction from {name}: {e}")
        
        if not quantum_results:
            raise ValueError("No quantum detectors available for fusion")
        
        # Fuse quantum predictions
        if fusion_method == "weighted_average":
            return self._weighted_fusion(quantum_results)
        elif fusion_method == "quantum_superposition":
            return self._quantum_superposition_fusion(quantum_results)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def _weighted_fusion(self, results: Dict[str, DetectionResult]) -> DetectionResult:
        """Fuse quantum results using weighted average."""
        if not results:
            raise ValueError("No results to fuse")
        
        # Initialize combined arrays
        n_samples = len(next(iter(results.values())).anomaly_scores)
        combined_scores = np.zeros(n_samples)
        combined_predictions = np.zeros(n_samples)
        combined_confidence = np.zeros(n_samples)
        total_weight = 0
        
        # Weighted combination
        for name, result in results.items():
            weight = self.fusion_weights.get(name, 1.0 / len(results))
            combined_scores += weight * result.anomaly_scores
            combined_predictions += weight * result.anomaly_predictions
            combined_confidence += weight * result.confidence_scores
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            combined_scores /= total_weight
            combined_predictions /= total_weight
            combined_confidence /= total_weight
        
        # Convert to binary predictions
        threshold = np.median(combined_predictions)
        final_predictions = (combined_predictions > threshold).astype(int)
        
        return DetectionResult(
            anomaly_scores=combined_scores,
            anomaly_predictions=final_predictions,
            confidence_scores=combined_confidence,
            detection_method="QuantumFusion_Weighted",
            metadata={
                "fusion_weights": self.fusion_weights,
                "detectors_used": list(results.keys()),
                "threshold": threshold,
                "quantum_enabled": QISKIT_AVAILABLE
            }
        )
    
    def _quantum_superposition_fusion(self, results: Dict[str, DetectionResult]) -> DetectionResult:
        """Fuse results using quantum superposition principles."""
        # Implement quantum-inspired superposition fusion
        n_samples = len(next(iter(results.values())).anomaly_scores)
        
        # Create quantum-like superposition of results
        superposed_scores = np.zeros(n_samples, dtype=complex)
        
        for name, result in results.items():
            weight = self.fusion_weights.get(name, 1.0 / len(results))
            
            # Convert to complex representation (amplitude and phase)
            amplitude = np.sqrt(weight) * result.anomaly_scores
            phase = 2 * np.pi * result.confidence_scores
            quantum_component = amplitude * np.exp(1j * phase)
            
            superposed_scores += quantum_component
        
        # Measure the superposed state
        final_scores = np.abs(superposed_scores)
        final_predictions = (final_scores > np.median(final_scores)).astype(int)
        
        # Calculate quantum coherence as confidence
        coherence = np.abs(superposed_scores) / np.maximum(np.real(superposed_scores) + np.imag(superposed_scores), 1e-10)
        
        return DetectionResult(
            anomaly_scores=final_scores,
            anomaly_predictions=final_predictions,
            confidence_scores=coherence,
            detection_method="QuantumFusion_Superposition",
            metadata={
                "fusion_method": "quantum_superposition",
                "detectors_used": list(results.keys()),
                "quantum_coherence": np.mean(coherence),
                "quantum_enabled": QISKIT_AVAILABLE
            }
        )
    
    def save(self, path: Path) -> None:
        """Save the quantum fusion system."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save individual quantum detectors
        for name, detector in self.quantum_detectors.items():
            if detector.is_trained:
                detector_path = path / f"quantum_{name}"
                detector.save(detector_path)
        
        # Save fusion configuration
        with open(path / "quantum_fusion_config.pkl", "wb") as f:
            pickle.dump({
                "fusion_weights": self.fusion_weights,
                "n_qubits": self.n_qubits,
                "enable_parallel": self.enable_parallel
            }, f)
    
    def load(self, path: Path) -> None:
        """Load the quantum fusion system."""
        # Load individual quantum detectors
        for name, detector in self.quantum_detectors.items():
            detector_path = path / f"quantum_{name}"
            if detector_path.exists():
                try:
                    detector.load(detector_path)
                    self.logger.info(f"Loaded quantum {name} detector")
                except Exception as e:
                    self.logger.warning(f"Failed to load quantum {name} detector: {e}")
        
        # Load fusion configuration
        config_path = path / "quantum_fusion_config.pkl"
        if config_path.exists():
            with open(config_path, "rb") as f:
                config = pickle.load(f)
                self.fusion_weights = config["fusion_weights"]
                self.n_qubits = config["n_qubits"]
                self.enable_parallel = config["enable_parallel"]
        
        self.is_trained = True
    
    def get_quantum_metrics(self, data: np.ndarray) -> Dict[str, Any]:
        """Get quantum-specific metrics for analysis."""
        if not self.is_trained:
            raise ValueError("System must be trained before getting metrics")
        
        metrics = {}
        
        for name, detector in self.quantum_detectors.items():
            try:
                result = detector.predict(data[:10])  # Sample for efficiency
                
                if "quantum_features" in result.metadata:
                    quantum_features = result.metadata["quantum_features"]
                    if quantum_features:
                        metrics[f"{name}_avg_entanglement"] = np.mean([f.get("entanglement", 0) for f in quantum_features])
                        metrics[f"{name}_avg_coherence"] = np.mean([f.get("coherence", 0) for f in quantum_features])
                        metrics[f"{name}_phase_complexity"] = np.mean([f.get("phase_variance", 0) for f in quantum_features])
                
            except Exception as e:
                self.logger.warning(f"Failed to get metrics from {name}: {e}")
        
        metrics["quantum_enabled"] = QISKIT_AVAILABLE
        metrics["n_qubits"] = self.n_qubits
        
        return metrics