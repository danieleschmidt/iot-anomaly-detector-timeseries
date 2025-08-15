"""Neuromorphic Spike-Based Processing for IoT Anomaly Detection.

Advanced neuromorphic computing implementation that mimics brain-like processing
for ultra-low power anomaly detection in edge IoT devices. Uses spiking neural
networks (SNNs) and event-driven processing for energy-efficient computation.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import pickle
import json
import time
import threading
from collections import deque, defaultdict
import heapq
import warnings

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    import matplotlib.pyplot as plt
    from scipy import signal, stats
    from sklearn.preprocessing import StandardScaler
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    warnings.warn("Neuromorphic dependencies not available. Using simplified implementations.")

from .logging_config import get_logger
from .data_preprocessor import DataPreprocessor


@dataclass
class SpikeEvent:
    """Represents a spike event in neuromorphic processing."""
    
    timestamp: float
    neuron_id: int
    layer_id: int
    spike_amplitude: float = 1.0
    spike_type: str = "excitatory"  # "excitatory" or "inhibitory"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynapseConnection:
    """Represents a synaptic connection between neurons."""
    
    pre_neuron_id: int
    post_neuron_id: int
    weight: float
    delay: float  # Propagation delay in ms
    plasticity_rule: str = "STDP"  # Spike-timing dependent plasticity
    last_update_time: float = 0.0
    connection_strength: float = 1.0
    is_plastic: bool = True


@dataclass
class NeuronState:
    """State of a spiking neuron."""
    
    neuron_id: int
    layer_id: int
    membrane_potential: float = 0.0
    threshold: float = 1.0
    reset_potential: float = 0.0
    refractory_period: float = 2.0  # ms
    last_spike_time: float = -np.inf
    tau_membrane: float = 20.0  # Membrane time constant
    tau_synaptic: float = 5.0   # Synaptic time constant
    leak_conductance: float = 0.1
    spike_count: int = 0
    adaptation_current: float = 0.0
    is_refractory: bool = False


class NeuronModel(ABC):
    """Abstract base class for neuron models."""
    
    @abstractmethod
    def update_membrane_potential(
        self, 
        neuron_state: NeuronState, 
        input_current: float, 
        dt: float
    ) -> float:
        """Update membrane potential based on input current."""
        pass
    
    @abstractmethod
    def check_spike_condition(self, neuron_state: NeuronState) -> bool:
        """Check if neuron should generate a spike."""
        pass
    
    @abstractmethod
    def reset_neuron(self, neuron_state: NeuronState) -> None:
        """Reset neuron state after spike."""
        pass


class LeakyIntegrateFireNeuron(NeuronModel):
    """Leaky Integrate-and-Fire (LIF) neuron model."""
    
    def __init__(self, adaptation_strength: float = 0.02):
        self.adaptation_strength = adaptation_strength
        self.logger = get_logger(__name__)
    
    def update_membrane_potential(
        self, 
        neuron_state: NeuronState, 
        input_current: float, 
        dt: float
    ) -> float:
        """Update membrane potential using LIF dynamics."""
        try:
            # Skip update if in refractory period
            current_time = time.time() * 1000  # Convert to ms
            if current_time - neuron_state.last_spike_time < neuron_state.refractory_period:
                neuron_state.is_refractory = True
                return neuron_state.reset_potential
            
            neuron_state.is_refractory = False
            
            # LIF equation: C * dV/dt = -g_L * (V - V_L) + I
            # Simplified: tau * dV/dt = -(V - V_rest) + R*I
            leak_current = -neuron_state.leak_conductance * (
                neuron_state.membrane_potential - neuron_state.reset_potential
            )
            
            # Adaptation current (spike-frequency adaptation)
            adaptation_decay = np.exp(-dt / (neuron_state.tau_membrane * 5))
            neuron_state.adaptation_current *= adaptation_decay
            
            total_current = input_current + leak_current - neuron_state.adaptation_current
            
            # Update membrane potential
            voltage_change = (total_current / neuron_state.tau_membrane) * dt
            neuron_state.membrane_potential += voltage_change
            
            # Ensure membrane potential doesn't go below reset
            neuron_state.membrane_potential = max(
                neuron_state.membrane_potential,
                neuron_state.reset_potential
            )
            
            return neuron_state.membrane_potential
            
        except Exception as e:
            self.logger.error(f"Error updating membrane potential: {str(e)}")
            return neuron_state.membrane_potential
    
    def check_spike_condition(self, neuron_state: NeuronState) -> bool:
        """Check if membrane potential crosses threshold."""
        return (neuron_state.membrane_potential >= neuron_state.threshold and 
                not neuron_state.is_refractory)
    
    def reset_neuron(self, neuron_state: NeuronState) -> None:
        """Reset neuron after spike and add adaptation."""
        neuron_state.membrane_potential = neuron_state.reset_potential
        neuron_state.last_spike_time = time.time() * 1000
        neuron_state.spike_count += 1
        
        # Add adaptation current (makes neuron less likely to spike)
        neuron_state.adaptation_current += self.adaptation_strength


class AdaptiveExponentialNeuron(NeuronModel):
    """Adaptive Exponential Integrate-and-Fire (AdEx) neuron model."""
    
    def __init__(self, delta_t: float = 2.0, v_spike: float = 20.0):
        self.delta_t = delta_t  # Sharpness of exponential approach to spike
        self.v_spike = v_spike  # Spike detection threshold
        self.logger = get_logger(__name__)
    
    def update_membrane_potential(
        self, 
        neuron_state: NeuronState, 
        input_current: float, 
        dt: float
    ) -> float:
        """Update membrane potential using AdEx dynamics."""
        try:
            current_time = time.time() * 1000
            if current_time - neuron_state.last_spike_time < neuron_state.refractory_period:
                neuron_state.is_refractory = True
                return neuron_state.reset_potential
            
            neuron_state.is_refractory = False
            
            # AdEx equation with exponential term
            V = neuron_state.membrane_potential
            V_rest = neuron_state.reset_potential
            
            # Exponential term
            exp_term = self.delta_t * np.exp((V - neuron_state.threshold) / self.delta_t)
            
            # Leak and input currents
            leak_current = -neuron_state.leak_conductance * (V - V_rest)
            
            # Adaptation current decay
            adaptation_decay = np.exp(-dt / (neuron_state.tau_membrane * 3))
            neuron_state.adaptation_current *= adaptation_decay
            
            # Total current
            total_current = (
                leak_current + 
                exp_term + 
                input_current - 
                neuron_state.adaptation_current
            )
            
            # Update membrane potential
            dV_dt = total_current / neuron_state.tau_membrane
            neuron_state.membrane_potential += dV_dt * dt
            
            # Prevent numerical explosion
            if neuron_state.membrane_potential > self.v_spike * 2:
                neuron_state.membrane_potential = self.v_spike * 2
            
            return neuron_state.membrane_potential
            
        except Exception as e:
            self.logger.error(f"Error in AdEx update: {str(e)}")
            return neuron_state.membrane_potential
    
    def check_spike_condition(self, neuron_state: NeuronState) -> bool:
        """Check if membrane potential crosses spike threshold."""
        return (neuron_state.membrane_potential >= self.v_spike and 
                not neuron_state.is_refractory)
    
    def reset_neuron(self, neuron_state: NeuronState) -> None:
        """Reset neuron with stronger adaptation."""
        neuron_state.membrane_potential = neuron_state.reset_potential
        neuron_state.last_spike_time = time.time() * 1000
        neuron_state.spike_count += 1
        
        # Stronger adaptation for AdEx
        neuron_state.adaptation_current += 0.1


class SynapticPlasticityRule(ABC):
    """Abstract base class for synaptic plasticity rules."""
    
    @abstractmethod
    def update_weight(
        self, 
        synapse: SynapseConnection,
        pre_spike_time: float,
        post_spike_time: float,
        current_time: float
    ) -> float:
        """Update synaptic weight based on spike timing."""
        pass


class STDPPlasticityRule(SynapticPlasticityRule):
    """Spike-Timing Dependent Plasticity (STDP) rule."""
    
    def __init__(
        self, 
        A_plus: float = 0.01, 
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_max: float = 5.0,
        w_min: float = 0.0
    ):
        self.A_plus = A_plus    # LTP amplitude
        self.A_minus = A_minus  # LTD amplitude  
        self.tau_plus = tau_plus   # LTP time constant
        self.tau_minus = tau_minus # LTD time constant
        self.w_max = w_max      # Maximum weight
        self.w_min = w_min      # Minimum weight
    
    def update_weight(
        self, 
        synapse: SynapseConnection,
        pre_spike_time: float,
        post_spike_time: float,
        current_time: float
    ) -> float:
        """Update weight using STDP rule."""
        try:
            if not synapse.is_plastic:
                return synapse.weight
            
            delta_t = post_spike_time - pre_spike_time
            
            if abs(delta_t) > 100:  # Ignore very large time differences
                return synapse.weight
            
            # STDP weight update
            if delta_t > 0:
                # Post before pre -> LTP (potentiation)
                weight_change = self.A_plus * np.exp(-delta_t / self.tau_plus)
            else:
                # Pre before post -> LTD (depression)
                weight_change = -self.A_minus * np.exp(delta_t / self.tau_minus)
            
            # Update weight with bounds
            new_weight = synapse.weight + weight_change
            new_weight = np.clip(new_weight, self.w_min, self.w_max)
            
            synapse.weight = new_weight
            synapse.last_update_time = current_time
            
            return new_weight
            
        except Exception as e:
            return synapse.weight


class HomeostaticPlasticityRule(SynapticPlasticityRule):
    """Homeostatic plasticity for maintaining stable firing rates."""
    
    def __init__(
        self, 
        target_rate: float = 5.0,  # Target firing rate in Hz
        learning_rate: float = 0.001,
        time_window: float = 1000.0  # Time window in ms
    ):
        self.target_rate = target_rate
        self.learning_rate = learning_rate
        self.time_window = time_window
        self.neuron_rates = defaultdict(list)
    
    def update_weight(
        self, 
        synapse: SynapseConnection,
        pre_spike_time: float,
        post_spike_time: float,
        current_time: float
    ) -> float:
        """Update weight based on homeostatic scaling."""
        try:
            # Track post-synaptic neuron firing rate
            post_neuron_id = synapse.post_neuron_id
            self.neuron_rates[post_neuron_id].append(current_time)
            
            # Remove old spikes outside time window
            cutoff_time = current_time - self.time_window
            self.neuron_rates[post_neuron_id] = [
                t for t in self.neuron_rates[post_neuron_id] if t > cutoff_time
            ]
            
            # Calculate current firing rate
            current_rate = len(self.neuron_rates[post_neuron_id]) / (self.time_window / 1000.0)
            
            # Homeostatic scaling
            rate_error = self.target_rate - current_rate
            scaling_factor = 1.0 + self.learning_rate * rate_error
            
            # Update weight
            new_weight = synapse.weight * scaling_factor
            new_weight = np.clip(new_weight, 0.0, 10.0)  # Reasonable bounds
            
            synapse.weight = new_weight
            return new_weight
            
        except Exception as e:
            return synapse.weight


class EventDrivenProcessor:
    """Event-driven processing engine for spiking neural networks."""
    
    def __init__(self, time_resolution: float = 0.1):
        self.time_resolution = time_resolution  # ms
        self.event_queue = []  # Priority queue for events
        self.current_time = 0.0
        self.processing_stats = {
            'events_processed': 0,
            'spikes_generated': 0,
            'computation_time': 0.0
        }
        self.logger = get_logger(__name__)
    
    def schedule_event(self, event: SpikeEvent, delay: float = 0.0) -> None:
        """Schedule a spike event for processing."""
        event_time = self.current_time + delay
        heapq.heappush(self.event_queue, (event_time, event))
    
    def process_events_until(self, end_time: float) -> List[SpikeEvent]:
        """Process all events up to specified time."""
        processed_events = []
        
        while (self.event_queue and 
               self.event_queue[0][0] <= end_time):
            
            event_time, event = heapq.heappop(self.event_queue)
            self.current_time = event_time
            
            processed_events.append(event)
            self.processing_stats['events_processed'] += 1
            
            # Log significant events
            if len(processed_events) % 1000 == 0:
                self.logger.debug(f"Processed {len(processed_events)} events")
        
        return processed_events
    
    def get_next_event_time(self) -> Optional[float]:
        """Get timestamp of next scheduled event."""
        if self.event_queue:
            return self.event_queue[0][0]
        return None
    
    def clear_old_events(self, before_time: float) -> None:
        """Clear events scheduled before specified time."""
        self.event_queue = [
            (t, e) for t, e in self.event_queue if t >= before_time
        ]
        heapq.heapify(self.event_queue)


class SpikingNeuralNetwork:
    """Complete spiking neural network for neuromorphic processing."""
    
    def __init__(
        self,
        architecture: List[int],
        neuron_model: Optional[NeuronModel] = None,
        plasticity_rule: Optional[SynapticPlasticityRule] = None,
        time_step: float = 0.1,
        simulation_time: float = 1000.0
    ):
        """Initialize spiking neural network.
        
        Args:
            architecture: List of neurons per layer [input, hidden1, hidden2, ..., output]
            neuron_model: Neuron model to use for all neurons
            plasticity_rule: Synaptic plasticity rule
            time_step: Simulation time step in ms
            simulation_time: Total simulation time in ms
        """
        self.architecture = architecture
        self.neuron_model = neuron_model or LeakyIntegrateFireNeuron()
        self.plasticity_rule = plasticity_rule or STDPPlasticityRule()
        self.time_step = time_step
        self.simulation_time = simulation_time
        
        # Network components
        self.neurons: Dict[int, NeuronState] = {}
        self.synapses: Dict[Tuple[int, int], SynapseConnection] = {}
        self.layer_mapping: Dict[int, int] = {}  # neuron_id -> layer_id
        
        # Processing components
        self.event_processor = EventDrivenProcessor(time_step)
        self.spike_history: Dict[int, List[float]] = defaultdict(list)
        self.membrane_potential_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        
        # Performance tracking
        self.network_stats = {
            'total_spikes': 0,
            'average_firing_rate': 0.0,
            'synaptic_updates': 0,
            'energy_consumption': 0.0
        }
        
        self.logger = get_logger(__name__)
        self._initialize_network()
    
    def _initialize_network(self) -> None:
        """Initialize network topology and connections."""
        try:
            neuron_id = 0
            
            # Create neurons for each layer
            for layer_id, num_neurons in enumerate(self.architecture):
                for _ in range(num_neurons):
                    # Vary neuron parameters slightly for diversity
                    threshold = np.random.normal(1.0, 0.1)
                    tau_membrane = np.random.normal(20.0, 2.0)
                    
                    neuron_state = NeuronState(
                        neuron_id=neuron_id,
                        layer_id=layer_id,
                        threshold=max(0.5, threshold),
                        tau_membrane=max(10.0, tau_membrane),
                        refractory_period=np.random.normal(2.0, 0.2)
                    )
                    
                    self.neurons[neuron_id] = neuron_state
                    self.layer_mapping[neuron_id] = layer_id
                    neuron_id += 1
            
            # Create synaptic connections (fully connected between adjacent layers)
            self._create_synaptic_connections()
            
            self.logger.info(
                f"Initialized SNN: {len(self.neurons)} neurons, "
                f"{len(self.synapses)} synapses across {len(self.architecture)} layers"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize network: {str(e)}")
            raise
    
    def _create_synaptic_connections(self) -> None:
        """Create synaptic connections between layers."""
        try:
            neuron_id = 0
            
            for layer_id in range(len(self.architecture) - 1):
                # Current layer neurons
                current_layer_start = neuron_id
                current_layer_size = self.architecture[layer_id]
                
                # Next layer neurons  
                next_layer_start = neuron_id + current_layer_size
                next_layer_size = self.architecture[layer_id + 1]
                
                # Create connections between all neurons in adjacent layers
                for pre_id in range(current_layer_start, current_layer_start + current_layer_size):
                    for post_id in range(next_layer_start, next_layer_start + next_layer_size):
                        
                        # Initialize random weight
                        weight = np.random.normal(0.5, 0.2)
                        weight = np.clip(weight, 0.1, 2.0)
                        
                        # Random synaptic delay (biological realism)
                        delay = np.random.uniform(0.5, 5.0)
                        
                        synapse = SynapseConnection(
                            pre_neuron_id=pre_id,
                            post_neuron_id=post_id,
                            weight=weight,
                            delay=delay,
                            plasticity_rule="STDP",
                            is_plastic=True
                        )
                        
                        self.synapses[(pre_id, post_id)] = synapse
                
                neuron_id += current_layer_size
            
        except Exception as e:
            self.logger.error(f"Failed to create synapses: {str(e)}")
            raise
    
    def encode_input_to_spikes(
        self, 
        input_data: np.ndarray,
        encoding_type: str = "rate",
        time_window: float = 100.0
    ) -> List[SpikeEvent]:
        """Convert analog input to spike trains."""
        try:
            spike_events = []
            
            if encoding_type == "rate":
                # Rate-based encoding: higher values -> higher firing rates
                for i, value in enumerate(input_data.flatten()):
                    # Normalize to reasonable firing rate (0-100 Hz)
                    firing_rate = np.clip(value * 50, 0, 100)
                    
                    # Generate Poisson spike train
                    if firing_rate > 0:
                        inter_spike_intervals = np.random.exponential(
                            1000.0 / firing_rate,  # Convert Hz to ms intervals
                            size=int(firing_rate * time_window / 1000)
                        )
                        
                        spike_times = np.cumsum(inter_spike_intervals)
                        spike_times = spike_times[spike_times <= time_window]
                        
                        for spike_time in spike_times:
                            spike_event = SpikeEvent(
                                timestamp=spike_time,
                                neuron_id=i,
                                layer_id=0,
                                spike_amplitude=1.0,
                                metadata={'encoding': 'rate', 'original_value': value}
                            )
                            spike_events.append(spike_event)
            
            elif encoding_type == "temporal":
                # Temporal encoding: value determines spike timing
                for i, value in enumerate(input_data.flatten()):
                    # Map value to spike time (0-1 -> 0-time_window ms)
                    spike_time = np.clip(value, 0, 1) * time_window
                    
                    spike_event = SpikeEvent(
                        timestamp=spike_time,
                        neuron_id=i,
                        layer_id=0,
                        spike_amplitude=1.0,
                        metadata={'encoding': 'temporal', 'original_value': value}
                    )
                    spike_events.append(spike_event)
            
            elif encoding_type == "population":
                # Population vector encoding
                for i, value in enumerate(input_data.flatten()):
                    # Each input connects to multiple neurons with Gaussian response
                    neurons_per_input = min(5, self.architecture[0] // len(input_data.flatten()))
                    
                    for j in range(neurons_per_input):
                        neuron_id = i * neurons_per_input + j
                        if neuron_id >= self.architecture[0]:
                            break
                        
                        # Gaussian response curve
                        preferred_value = j / (neurons_per_input - 1) if neurons_per_input > 1 else 0.5
                        response = np.exp(-((value - preferred_value) ** 2) / (2 * 0.1 ** 2))
                        
                        # Convert to spike rate
                        firing_rate = response * 50  # Max 50 Hz
                        
                        if firing_rate > 1:
                            inter_spike_intervals = np.random.exponential(
                                1000.0 / firing_rate,
                                size=int(firing_rate * time_window / 1000)
                            )
                            
                            spike_times = np.cumsum(inter_spike_intervals)
                            spike_times = spike_times[spike_times <= time_window]
                            
                            for spike_time in spike_times:
                                spike_event = SpikeEvent(
                                    timestamp=spike_time,
                                    neuron_id=neuron_id,
                                    layer_id=0,
                                    spike_amplitude=1.0,
                                    metadata={
                                        'encoding': 'population',
                                        'original_value': value,
                                        'preferred_value': preferred_value
                                    }
                                )
                                spike_events.append(spike_event)
            
            self.logger.debug(f"Encoded input to {len(spike_events)} spike events using {encoding_type}")
            return sorted(spike_events, key=lambda x: x.timestamp)
            
        except Exception as e:
            self.logger.error(f"Failed to encode input: {str(e)}")
            return []
    
    def simulate_network(
        self, 
        input_spikes: List[SpikeEvent],
        simulation_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Run network simulation with given input spikes."""
        try:
            sim_time = simulation_time or self.simulation_time
            start_time = time.time()
            
            # Reset network state
            self._reset_network_state()
            
            # Schedule input spikes
            for spike in input_spikes:
                self.event_processor.schedule_event(spike)
            
            # Main simulation loop
            current_time = 0.0
            output_spikes = []
            
            while current_time < sim_time:
                # Process events at current time
                events = self.event_processor.process_events_until(
                    current_time + self.time_step
                )
                
                # Process each spike event
                for event in events:
                    self._process_spike_event(event)
                    
                    # Collect output spikes
                    if self.layer_mapping[event.neuron_id] == len(self.architecture) - 1:
                        output_spikes.append(event)
                
                # Update all neuron membrane potentials
                self._update_membrane_potentials(current_time)
                
                # Check for new spikes
                new_spikes = self._check_for_spikes(current_time)
                for spike in new_spikes:
                    self.event_processor.schedule_event(spike)
                
                current_time += self.time_step
            
            # Calculate results
            simulation_results = {
                'output_spikes': output_spikes,
                'total_spikes': self.network_stats['total_spikes'],
                'simulation_time': sim_time,
                'computation_time': time.time() - start_time,
                'firing_rates': self._calculate_firing_rates(sim_time),
                'synaptic_changes': self.network_stats['synaptic_updates'],
                'energy_estimate': self._estimate_energy_consumption()
            }
            
            self.logger.info(
                f"Simulation completed: {len(output_spikes)} output spikes, "
                f"{self.network_stats['total_spikes']} total spikes"
            )
            
            return simulation_results
            
        except Exception as e:
            self.logger.error(f"Network simulation failed: {str(e)}")
            raise
    
    def _reset_network_state(self) -> None:
        """Reset all neurons to resting state."""
        for neuron in self.neurons.values():
            neuron.membrane_potential = neuron.reset_potential
            neuron.last_spike_time = -np.inf
            neuron.spike_count = 0
            neuron.adaptation_current = 0.0
            neuron.is_refractory = False
        
        self.spike_history.clear()
        self.membrane_potential_history.clear()
        self.network_stats['total_spikes'] = 0
        self.network_stats['synaptic_updates'] = 0
    
    def _process_spike_event(self, event: SpikeEvent) -> None:
        """Process a single spike event."""
        try:
            # Record spike
            self.spike_history[event.neuron_id].append(event.timestamp)
            self.network_stats['total_spikes'] += 1
            
            # Find all outgoing synapses
            outgoing_synapses = [
                synapse for (pre_id, post_id), synapse in self.synapses.items()
                if pre_id == event.neuron_id
            ]
            
            # Propagate spike through synapses
            for synapse in outgoing_synapses:
                # Calculate synaptic current
                synaptic_current = synapse.weight * event.spike_amplitude
                
                # Apply to post-synaptic neuron (with delay)
                post_neuron = self.neurons[synapse.post_neuron_id]
                
                # Update synaptic plasticity
                if synapse.is_plastic:
                    self._update_synaptic_plasticity(
                        synapse, 
                        event.timestamp,
                        post_neuron.last_spike_time
                    )
                
                # Add synaptic current to neuron (simplified)
                post_neuron.membrane_potential += synaptic_current * 0.1
                
        except Exception as e:
            self.logger.error(f"Error processing spike event: {str(e)}")
    
    def _update_membrane_potentials(self, current_time: float) -> None:
        """Update membrane potentials for all neurons."""
        try:
            for neuron in self.neurons.values():
                # Calculate input current (simplified)
                input_current = 0.0
                
                # Update using neuron model
                new_potential = self.neuron_model.update_membrane_potential(
                    neuron, input_current, self.time_step
                )
                
                # Record membrane potential history
                self.membrane_potential_history[neuron.neuron_id].append(
                    (current_time, new_potential)
                )
                
                # Keep history manageable
                if len(self.membrane_potential_history[neuron.neuron_id]) > 1000:
                    self.membrane_potential_history[neuron.neuron_id].pop(0)
                
        except Exception as e:
            self.logger.error(f"Error updating membrane potentials: {str(e)}")
    
    def _check_for_spikes(self, current_time: float) -> List[SpikeEvent]:
        """Check all neurons for spike generation."""
        new_spikes = []
        
        try:
            for neuron in self.neurons.values():
                if self.neuron_model.check_spike_condition(neuron):
                    # Generate spike
                    spike = SpikeEvent(
                        timestamp=current_time,
                        neuron_id=neuron.neuron_id,
                        layer_id=neuron.layer_id,
                        spike_amplitude=1.0
                    )
                    new_spikes.append(spike)
                    
                    # Reset neuron
                    self.neuron_model.reset_neuron(neuron)
                    
                    # Update energy consumption
                    self.network_stats['energy_consumption'] += 1.0  # Simplified
        
        except Exception as e:
            self.logger.error(f"Error checking for spikes: {str(e)}")
        
        return new_spikes
    
    def _update_synaptic_plasticity(
        self, 
        synapse: SynapseConnection,
        pre_spike_time: float,
        post_spike_time: float
    ) -> None:
        """Update synaptic weights based on plasticity rule."""
        try:
            if post_spike_time != -np.inf:  # Only if post-synaptic neuron has spiked
                new_weight = self.plasticity_rule.update_weight(
                    synapse,
                    pre_spike_time,
                    post_spike_time,
                    pre_spike_time  # Current time
                )
                
                self.network_stats['synaptic_updates'] += 1
                
        except Exception as e:
            self.logger.error(f"Error updating synaptic plasticity: {str(e)}")
    
    def _calculate_firing_rates(self, simulation_time: float) -> Dict[int, float]:
        """Calculate firing rates for all neurons."""
        firing_rates = {}
        
        try:
            for neuron_id, spike_times in self.spike_history.items():
                firing_rate = len(spike_times) / (simulation_time / 1000.0)  # Hz
                firing_rates[neuron_id] = firing_rate
            
            # Fill in zeros for neurons that didn't spike
            for neuron_id in self.neurons:
                if neuron_id not in firing_rates:
                    firing_rates[neuron_id] = 0.0
                    
        except Exception as e:
            self.logger.error(f"Error calculating firing rates: {str(e)}")
        
        return firing_rates
    
    def _estimate_energy_consumption(self) -> Dict[str, float]:
        """Estimate energy consumption of the network."""
        try:
            # Simplified energy model
            spike_energy = self.network_stats['total_spikes'] * 1.0  # pJ per spike
            leak_energy = len(self.neurons) * self.simulation_time * 0.01  # Leakage
            synaptic_energy = self.network_stats['synaptic_updates'] * 0.5  # Synaptic updates
            
            total_energy = spike_energy + leak_energy + synaptic_energy
            
            return {
                'total_energy_pj': total_energy,
                'spike_energy_pj': spike_energy,
                'leak_energy_pj': leak_energy,
                'synaptic_energy_pj': synaptic_energy,
                'energy_per_spike_pj': total_energy / max(1, self.network_stats['total_spikes'])
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating energy: {str(e)}")
            return {'total_energy_pj': 0.0}


class NeuromorphicAnomalyDetector:
    """Complete neuromorphic anomaly detection system."""
    
    def __init__(
        self,
        input_features: int = 5,
        hidden_layers: List[int] = [20, 10],
        encoding_type: str = "rate",
        simulation_time: float = 100.0,
        detection_threshold: float = 0.7
    ):
        """Initialize neuromorphic anomaly detector.
        
        Args:
            input_features: Number of input features
            hidden_layers: List of neurons in hidden layers
            encoding_type: Input encoding method ('rate', 'temporal', 'population')
            simulation_time: Simulation time per sample in ms
            detection_threshold: Anomaly detection threshold
        """
        self.input_features = input_features
        self.encoding_type = encoding_type
        self.simulation_time = simulation_time
        self.detection_threshold = detection_threshold
        
        # Create network architecture
        architecture = [input_features] + hidden_layers + [1]  # Single output neuron
        
        # Initialize components
        self.snn = SpikingNeuralNetwork(
            architecture=architecture,
            neuron_model=LeakyIntegrateFireNeuron(adaptation_strength=0.02),
            plasticity_rule=STDPPlasticityRule(A_plus=0.01, A_minus=0.012),
            simulation_time=simulation_time
        )
        
        self.preprocessor = DataPreprocessor()
        self.is_trained = False
        
        # Training history
        self.training_history = {
            'normal_responses': [],
            'anomaly_responses': [],
            'energy_consumption': [],
            'adaptation_progress': []
        }
        
        self.logger = get_logger(__name__)
        self.logger.info(f"Initialized neuromorphic anomaly detector: {architecture}")
    
    def train_unsupervised(
        self, 
        normal_data: np.ndarray,
        training_epochs: int = 10,
        adaptation_rate: float = 0.01
    ) -> Dict[str, Any]:
        """Train the neuromorphic detector on normal data."""
        try:
            start_time = time.time()
            
            # Preprocess data
            normal_data_scaled = self.preprocessor.fit_transform(normal_data)
            
            self.logger.info(f"Training neuromorphic detector on {len(normal_data)} normal samples")
            
            normal_responses = []
            
            for epoch in range(training_epochs):
                epoch_responses = []
                epoch_energy = 0.0
                
                for sample_idx, sample in enumerate(normal_data_scaled):
                    # Encode sample to spikes
                    input_spikes = self.snn.encode_input_to_spikes(
                        sample,
                        encoding_type=self.encoding_type,
                        time_window=self.simulation_time
                    )
                    
                    # Run simulation
                    sim_results = self.snn.simulate_network(
                        input_spikes,
                        simulation_time=self.simulation_time
                    )
                    
                    # Extract response (output firing rate)
                    output_neuron_id = max(self.snn.neurons.keys())  # Last neuron is output
                    output_firing_rate = sim_results['firing_rates'].get(output_neuron_id, 0.0)
                    
                    epoch_responses.append(output_firing_rate)
                    epoch_energy += sim_results['energy_estimate']['total_energy_pj']
                    
                    if (sample_idx + 1) % 100 == 0:
                        self.logger.debug(f"Epoch {epoch+1}, Sample {sample_idx+1}/{len(normal_data_scaled)}")
                
                normal_responses.extend(epoch_responses)
                
                # Log epoch statistics
                mean_response = np.mean(epoch_responses)
                std_response = np.std(epoch_responses)
                
                self.logger.info(
                    f"Epoch {epoch+1}: mean_response={mean_response:.3f}, "
                    f"std_response={std_response:.3f}, "
                    f"energy={epoch_energy:.1f}pJ"
                )
                
                # Store training history
                self.training_history['normal_responses'].append(epoch_responses)
                self.training_history['energy_consumption'].append(epoch_energy)
            
            # Calculate detection threshold based on normal responses
            normal_responses_flat = [r for epoch in self.training_history['normal_responses'] for r in epoch]
            
            # Use statistical threshold (mean + k*std)
            mean_normal = np.mean(normal_responses_flat)
            std_normal = np.std(normal_responses_flat)
            
            # Adaptive threshold based on response distribution
            self.detection_threshold = mean_normal + 2.0 * std_normal
            
            training_time = time.time() - start_time
            
            # Training summary
            training_summary = {
                'training_time': training_time,
                'samples_processed': len(normal_data) * training_epochs,
                'mean_normal_response': mean_normal,
                'std_normal_response': std_normal,
                'detection_threshold': self.detection_threshold,
                'total_energy_pj': sum(self.training_history['energy_consumption']),
                'energy_per_sample_pj': sum(self.training_history['energy_consumption']) / (len(normal_data) * training_epochs),
                'synaptic_adaptations': sum(result.get('synaptic_changes', 0) for result in []),
                'final_firing_rates': self.snn._calculate_firing_rates(self.simulation_time)
            }
            
            self.is_trained = True
            self.logger.info(f"Training completed: {training_summary}")
            
            return training_summary
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect anomalies using neuromorphic processing."""
        try:
            if not self.is_trained:
                self.logger.warning("Detector not trained, using default threshold")
            
            start_time = time.time()
            
            # Preprocess data
            X_scaled = self.preprocessor.transform(X)
            
            predictions = []
            confidence_scores = []
            total_energy = 0.0
            
            for sample_idx, sample in enumerate(X_scaled):
                # Encode to spikes
                input_spikes = self.snn.encode_input_to_spikes(
                    sample,
                    encoding_type=self.encoding_type,
                    time_window=self.simulation_time
                )
                
                # Simulate network
                sim_results = self.snn.simulate_network(
                    input_spikes,
                    simulation_time=self.simulation_time
                )
                
                # Extract output response
                output_neuron_id = max(self.snn.neurons.keys())
                output_firing_rate = sim_results['firing_rates'].get(output_neuron_id, 0.0)
                
                # Determine anomaly
                is_anomaly = output_firing_rate > self.detection_threshold
                predictions.append(int(is_anomaly))
                
                # Confidence based on distance from threshold
                confidence = abs(output_firing_rate - self.detection_threshold) / self.detection_threshold
                confidence_scores.append(min(confidence, 1.0))
                
                total_energy += sim_results['energy_estimate']['total_energy_pj']
                
                if (sample_idx + 1) % 50 == 0:
                    self.logger.debug(f"Processed {sample_idx+1}/{len(X_scaled)} samples")
            
            inference_time = time.time() - start_time
            
            # Prediction metadata
            metadata = {
                'inference_time': inference_time,
                'samples_processed': len(X),
                'total_energy_pj': total_energy,
                'energy_per_sample_pj': total_energy / len(X),
                'detection_threshold': self.detection_threshold,
                'mean_confidence': np.mean(confidence_scores),
                'neuromorphic_processing': True,
                'encoding_type': self.encoding_type
            }
            
            self.logger.info(
                f"Neuromorphic inference completed: {np.sum(predictions)} anomalies detected, "
                f"energy={total_energy:.1f}pJ"
            )
            
            return np.array(predictions), metadata
            
        except Exception as e:
            self.logger.error(f"Neuromorphic prediction failed: {str(e)}")
            raise
    
    def get_network_visualization_data(self) -> Dict[str, Any]:
        """Get data for visualizing network activity."""
        try:
            # Spike raster data
            spike_raster = {}
            for neuron_id, spike_times in self.snn.spike_history.items():
                spike_raster[neuron_id] = spike_times
            
            # Membrane potential traces
            membrane_traces = {}
            for neuron_id, trace in self.snn.membrane_potential_history.items():
                if trace:  # Only include neurons with recorded activity
                    times, potentials = zip(*trace) if trace else ([], [])
                    membrane_traces[neuron_id] = {'times': times, 'potentials': potentials}
            
            # Network connectivity
            connections = []
            for (pre_id, post_id), synapse in self.snn.synapses.items():
                connections.append({
                    'pre_neuron': pre_id,
                    'post_neuron': post_id,
                    'weight': synapse.weight,
                    'delay': synapse.delay
                })
            
            # Layer information
            layer_info = {}
            for layer_id in range(len(self.snn.architecture)):
                neurons_in_layer = [
                    nid for nid, lid in self.snn.layer_mapping.items() if lid == layer_id
                ]
                layer_info[layer_id] = {
                    'size': len(neurons_in_layer),
                    'neurons': neurons_in_layer,
                    'type': 'input' if layer_id == 0 else 'output' if layer_id == len(self.snn.architecture) - 1 else 'hidden'
                }
            
            return {
                'spike_raster': spike_raster,
                'membrane_traces': membrane_traces,
                'connections': connections,
                'layer_info': layer_info,
                'network_stats': self.snn.network_stats,
                'architecture': self.snn.architecture
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get visualization data: {str(e)}")
            return {}
    
    def save_neuromorphic_model(self, filepath: str) -> None:
        """Save neuromorphic model state."""
        try:
            model_state = {
                'input_features': self.input_features,
                'encoding_type': self.encoding_type,
                'simulation_time': self.simulation_time,
                'detection_threshold': self.detection_threshold,
                'is_trained': self.is_trained,
                'architecture': self.snn.architecture,
                'training_history': self.training_history,
                'preprocessor_state': {
                    'mean_': getattr(self.preprocessor.scaler, 'mean_', None),
                    'scale_': getattr(self.preprocessor.scaler, 'scale_', None)
                },
                'neuron_states': {
                    nid: {
                        'threshold': neuron.threshold,
                        'tau_membrane': neuron.tau_membrane,
                        'refractory_period': neuron.refractory_period
                    }
                    for nid, neuron in self.snn.neurons.items()
                },
                'synaptic_weights': {
                    (pre, post): synapse.weight
                    for (pre, post), synapse in self.snn.synapses.items()
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_state, f)
            
            self.logger.info(f"Saved neuromorphic model to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save neuromorphic model: {str(e)}")
            raise
    
    def load_neuromorphic_model(self, filepath: str) -> None:
        """Load neuromorphic model state."""
        try:
            with open(filepath, 'rb') as f:
                model_state = pickle.load(f)
            
            # Restore basic parameters
            self.input_features = model_state['input_features']
            self.encoding_type = model_state['encoding_type']
            self.simulation_time = model_state['simulation_time']
            self.detection_threshold = model_state['detection_threshold']
            self.is_trained = model_state['is_trained']
            self.training_history = model_state.get('training_history', {})
            
            # Recreate SNN with loaded architecture
            self.snn = SpikingNeuralNetwork(
                architecture=model_state['architecture'],
                simulation_time=self.simulation_time
            )
            
            # Restore synaptic weights
            for (pre, post), weight in model_state['synaptic_weights'].items():
                if (pre, post) in self.snn.synapses:
                    self.snn.synapses[(pre, post)].weight = weight
            
            # Restore neuron parameters
            for nid, params in model_state['neuron_states'].items():
                if nid in self.snn.neurons:
                    self.snn.neurons[nid].threshold = params['threshold']
                    self.snn.neurons[nid].tau_membrane = params['tau_membrane']
                    self.snn.neurons[nid].refractory_period = params['refractory_period']
            
            # Restore preprocessor
            preprocessor_state = model_state.get('preprocessor_state', {})
            if preprocessor_state.get('mean_') is not None:
                self.preprocessor.scaler.mean_ = preprocessor_state['mean_']
                self.preprocessor.scaler.scale_ = preprocessor_state['scale_']
            
            self.logger.info(f"Loaded neuromorphic model from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load neuromorphic model: {str(e)}")
            raise


def create_optimized_neuromorphic_detector(
    input_features: int,
    target_energy_budget: float = 1000.0,  # pJ per inference
    target_latency: float = 10.0  # ms
) -> NeuromorphicAnomalyDetector:
    """Create neuromorphic detector optimized for energy and latency."""
    
    # Calculate optimal architecture based on constraints
    max_neurons = int(target_energy_budget / 10.0)  # Rough estimate: 10pJ per neuron
    simulation_time = min(target_latency, 50.0)  # Cap simulation time
    
    # Architecture optimization
    if max_neurons > 100:
        hidden_layers = [min(50, input_features * 3), min(20, input_features * 2)]
    elif max_neurons > 50:
        hidden_layers = [min(30, input_features * 2)]
    else:
        hidden_layers = [min(15, input_features)]
    
    # Use rate encoding for efficiency
    detector = NeuromorphicAnomalyDetector(
        input_features=input_features,
        hidden_layers=hidden_layers,
        encoding_type="rate",
        simulation_time=simulation_time,
        detection_threshold=0.7
    )
    
    return detector


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Neuromorphic Anomaly Detection")
    parser.add_argument("--input-features", type=int, default=5)
    parser.add_argument("--simulation-time", type=float, default=100.0)
    parser.add_argument("--encoding", choices=["rate", "temporal", "population"], default="rate")
    parser.add_argument("--output", type=str, default="neuromorphic_model.pkl")
    
    args = parser.parse_args()
    
    # Create detector
    detector = create_optimized_neuromorphic_detector(
        input_features=args.input_features,
        target_energy_budget=2000.0,
        target_latency=args.simulation_time
    )
    
    # Generate synthetic training data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (500, args.input_features))
    
    print("Training neuromorphic detector...")
    training_results = detector.train_unsupervised(
        normal_data,
        training_epochs=3,
        adaptation_rate=0.01
    )
    
    print(f"Training Results:")
    for key, value in training_results.items():
        if isinstance(value, dict):
            continue  # Skip complex nested data
        print(f"  {key}: {value}")
    
    # Test on synthetic anomalies
    test_normal = np.random.normal(0, 1, (50, args.input_features))
    test_anomalies = np.random.normal(0, 3, (10, args.input_features))  # Higher variance
    test_data = np.vstack([test_normal, test_anomalies])
    
    print("\nRunning neuromorphic inference...")
    predictions, metadata = detector.predict(test_data)
    
    print(f"\nInference Results:")
    print(f"  Samples processed: {metadata['samples_processed']}")
    print(f"  Anomalies detected: {np.sum(predictions)}")
    print(f"  Total energy: {metadata['total_energy_pj']:.1f} pJ")
    print(f"  Energy per sample: {metadata['energy_per_sample_pj']:.1f} pJ")
    print(f"  Inference time: {metadata['inference_time']:.3f} s")
    
    # Save model
    detector.save_neuromorphic_model(args.output)
    
    # Get visualization data
    viz_data = detector.get_network_visualization_data()
    print(f"\nNetwork Activity:")
    print(f"  Total spikes: {viz_data['network_stats']['total_spikes']}")
    print(f"  Active neurons: {len([n for n in viz_data['spike_raster'] if viz_data['spike_raster'][n]])}")
    print(f"  Network layers: {len(viz_data['layer_info'])}")