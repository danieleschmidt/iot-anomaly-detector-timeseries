"""Federated Learning System for Distributed IoT Anomaly Detection.

Advanced federated learning implementation enabling collaborative model training
across distributed IoT devices while preserving privacy and optimizing for
edge computing constraints.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import pickle
import json
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from collections import defaultdict

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, clone_model
    from tensorflow.keras import layers, optimizers
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

from .logging_config import get_logger
from .data_preprocessor import DataPreprocessor
from .security_utils import encrypt_data, decrypt_data, generate_keypair


@dataclass
class FederatedNode:
    """Represents a federated learning node (IoT device)."""
    
    node_id: str
    node_type: str  # "edge", "gateway", "cloud"
    computational_capacity: float  # FLOPS rating
    memory_capacity: int  # MB
    bandwidth_capacity: float  # Mbps
    is_active: bool = True
    trust_score: float = 1.0
    local_data_samples: int = 0
    last_update_timestamp: float = field(default_factory=time.time)
    public_key: Optional[str] = None
    private_key: Optional[str] = None


@dataclass
class ModelUpdate:
    """Represents a model update from a federated node."""
    
    node_id: str
    model_weights: Dict[str, np.ndarray]
    gradient_norms: List[float]
    local_loss: float
    sample_count: int
    training_time: float
    update_timestamp: float
    signature: Optional[str] = None
    is_verified: bool = False


@dataclass
class FederatedRound:
    """Represents a complete federated learning round."""
    
    round_number: int
    participating_nodes: List[str]
    global_model_version: str
    aggregated_weights: Dict[str, np.ndarray]
    global_loss: float
    convergence_metric: float
    round_duration: float
    start_timestamp: float
    completion_timestamp: float


class FederatedAggregationStrategy(ABC):
    """Abstract base class for federated aggregation strategies."""
    
    @abstractmethod
    def aggregate(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate model updates from multiple nodes."""
        pass


class FedAvgStrategy(FederatedAggregationStrategy):
    """Federated Averaging (FedAvg) aggregation strategy."""
    
    def __init__(self, weighted: bool = True):
        self.weighted = weighted
        self.logger = get_logger(__name__)
    
    def aggregate(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate using weighted or simple averaging."""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        total_samples = sum(update.sample_count for update in updates)
        aggregated_weights = {}
        
        for layer_name in updates[0].model_weights.keys():
            layer_updates = []
            weights = []
            
            for update in updates:
                layer_updates.append(update.model_weights[layer_name])
                if self.weighted:
                    weights.append(update.sample_count / total_samples)
                else:
                    weights.append(1.0 / len(updates))
            
            # Weighted average of layer weights
            aggregated_weights[layer_name] = np.average(
                layer_updates, axis=0, weights=weights
            )
        
        self.logger.info(f"Aggregated {len(updates)} updates using FedAvg")
        return aggregated_weights


class FedProxStrategy(FederatedAggregationStrategy):
    """Federated Proximal (FedProx) aggregation with regularization."""
    
    def __init__(self, mu: float = 0.01, adaptive_mu: bool = True):
        self.mu = mu  # Proximal term coefficient
        self.adaptive_mu = adaptive_mu
        self.logger = get_logger(__name__)
    
    def aggregate(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate with proximal regularization."""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Adaptive mu based on gradient diversity
        if self.adaptive_mu:
            grad_variance = np.var([np.mean(update.gradient_norms) for update in updates])
            adaptive_mu = self.mu * (1 + grad_variance)
        else:
            adaptive_mu = self.mu
        
        total_samples = sum(update.sample_count for update in updates)
        aggregated_weights = {}
        
        for layer_name in updates[0].model_weights.keys():
            layer_updates = []
            weights = []
            
            for update in updates:
                # Apply proximal regularization
                regularized_weights = update.model_weights[layer_name] * (1 - adaptive_mu)
                layer_updates.append(regularized_weights)
                weights.append(update.sample_count / total_samples)
            
            aggregated_weights[layer_name] = np.average(
                layer_updates, axis=0, weights=weights
            )
        
        self.logger.info(f"Aggregated {len(updates)} updates using FedProx (mu={adaptive_mu:.4f})")
        return aggregated_weights


class FedNestrovStrategy(FederatedAggregationStrategy):
    """Federated Nesterov momentum aggregation."""
    
    def __init__(self, momentum: float = 0.9, learning_rate: float = 0.01):
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.velocity = {}
        self.logger = get_logger(__name__)
    
    def aggregate(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate using Nesterov momentum."""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Standard FedAvg first
        fedavg = FedAvgStrategy(weighted=True)
        avg_weights = fedavg.aggregate(updates)
        
        # Apply Nesterov momentum
        aggregated_weights = {}
        for layer_name, weights in avg_weights.items():
            if layer_name not in self.velocity:
                self.velocity[layer_name] = np.zeros_like(weights)
            
            # Update velocity
            self.velocity[layer_name] = (
                self.momentum * self.velocity[layer_name] + 
                self.learning_rate * weights
            )
            
            # Apply Nesterov update
            aggregated_weights[layer_name] = weights + self.momentum * self.velocity[layer_name]
        
        self.logger.info(f"Aggregated {len(updates)} updates using FedNesterov")
        return aggregated_weights


class AdaptiveFederatedStrategy(FederatedAggregationStrategy):
    """Adaptive federated learning strategy with intelligent node selection."""
    
    def __init__(self, trust_threshold: float = 0.7, performance_weight: float = 0.3):
        self.trust_threshold = trust_threshold
        self.performance_weight = performance_weight
        self.node_performance_history = defaultdict(list)
        self.logger = get_logger(__name__)
    
    def aggregate(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Intelligent aggregation with trust and performance weighting."""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Filter by trust score and performance
        filtered_updates = []
        total_samples = 0
        
        for update in updates:
            node_id = update.node_id
            
            # Calculate performance score based on loss improvement
            performance_score = 1.0 / (1.0 + update.local_loss)
            self.node_performance_history[node_id].append(performance_score)
            
            # Keep only last 10 performance scores
            if len(self.node_performance_history[node_id]) > 10:
                self.node_performance_history[node_id].pop(0)
            
            avg_performance = np.mean(self.node_performance_history[node_id])
            
            # Include update if performance is above threshold
            if avg_performance >= self.trust_threshold and update.is_verified:
                filtered_updates.append(update)
                total_samples += update.sample_count
        
        if not filtered_updates:
            self.logger.warning("No trusted updates available, using all updates")
            filtered_updates = updates
            total_samples = sum(update.sample_count for update in updates)
        
        # Weighted aggregation with performance and sample size
        aggregated_weights = {}
        
        for layer_name in filtered_updates[0].model_weights.keys():
            layer_updates = []
            weights = []
            
            for update in filtered_updates:
                node_performance = np.mean(
                    self.node_performance_history[update.node_id]
                )
                
                # Combined weight: sample size + performance
                sample_weight = update.sample_count / total_samples
                perf_weight = node_performance / sum(
                    np.mean(self.node_performance_history[u.node_id]) 
                    for u in filtered_updates
                )
                
                combined_weight = (
                    (1 - self.performance_weight) * sample_weight + 
                    self.performance_weight * perf_weight
                )
                
                layer_updates.append(update.model_weights[layer_name])
                weights.append(combined_weight)
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            aggregated_weights[layer_name] = np.average(
                layer_updates, axis=0, weights=weights
            )
        
        self.logger.info(
            f"Aggregated {len(filtered_updates)}/{len(updates)} trusted updates"
        )
        return aggregated_weights


class FederatedAnomalyLearner:
    """Federated learning system for distributed anomaly detection."""
    
    def __init__(
        self,
        aggregation_strategy: Optional[FederatedAggregationStrategy] = None,
        min_nodes_per_round: int = 3,
        max_rounds: int = 100,
        convergence_threshold: float = 0.001,
        model_compression_ratio: float = 0.1,
        privacy_budget: float = 1.0,
        enable_encryption: bool = True
    ):
        """Initialize federated learning system.
        
        Args:
            aggregation_strategy: Strategy for aggregating model updates
            min_nodes_per_round: Minimum nodes required per round
            max_rounds: Maximum training rounds
            convergence_threshold: Convergence criteria
            model_compression_ratio: Compression for model updates
            privacy_budget: Differential privacy budget
            enable_encryption: Enable secure communication
        """
        self.aggregation_strategy = aggregation_strategy or AdaptiveFederatedStrategy()
        self.min_nodes_per_round = min_nodes_per_round
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.model_compression_ratio = model_compression_ratio
        self.privacy_budget = privacy_budget
        self.enable_encryption = enable_encryption
        
        # System state
        self.nodes: Dict[str, FederatedNode] = {}
        self.global_model: Optional[Model] = None
        self.global_model_version = "v0.0.0"
        self.training_history: List[FederatedRound] = []
        self.current_round = 0
        self.is_training = False
        
        # Performance tracking
        self.convergence_history = []
        self.node_selection_history = []
        
        self.logger = get_logger(__name__)
        self.logger.info("Initialized Federated Anomaly Learning System")
    
    def register_node(
        self, 
        node_id: str, 
        node_type: str = "edge",
        computational_capacity: float = 1.0,
        memory_capacity: int = 512,
        bandwidth_capacity: float = 10.0
    ) -> bool:
        """Register a new federated node."""
        try:
            if self.enable_encryption:
                pub_key, priv_key = generate_keypair()
            else:
                pub_key, priv_key = None, None
            
            node = FederatedNode(
                node_id=node_id,
                node_type=node_type,
                computational_capacity=computational_capacity,
                memory_capacity=memory_capacity,
                bandwidth_capacity=bandwidth_capacity,
                public_key=pub_key,
                private_key=priv_key
            )
            
            self.nodes[node_id] = node
            self.logger.info(f"Registered federated node: {node_id} ({node_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register node {node_id}: {str(e)}")
            return False
    
    def initialize_global_model(self, input_shape: Tuple[int, ...]) -> None:
        """Initialize the global model architecture."""
        try:
            # Lightweight model for federated learning
            inputs = layers.Input(shape=input_shape)
            
            # Encoder
            x = layers.LSTM(32, return_sequences=True)(inputs)
            x = layers.Dropout(0.2)(x)
            x = layers.LSTM(16, return_sequences=False)(x)
            x = layers.Dense(8, activation='relu')(x)
            
            # Decoder
            x = layers.RepeatVector(input_shape[0])(x)
            x = layers.LSTM(16, return_sequences=True)(x)
            x = layers.LSTM(32, return_sequences=True)(x)
            outputs = layers.TimeDistributed(layers.Dense(input_shape[1]))(x)
            
            self.global_model = tf.keras.Model(inputs, outputs)
            self.global_model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            self.global_model_version = "v1.0.0"
            self.logger.info(f"Initialized global model: {self.global_model_version}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize global model: {str(e)}")
            raise
    
    def select_nodes_for_round(
        self, 
        selection_ratio: float = 0.3,
        strategy: str = "adaptive"
    ) -> List[str]:
        """Select nodes for the current training round."""
        active_nodes = [
            node_id for node_id, node in self.nodes.items() 
            if node.is_active and node.local_data_samples > 0
        ]
        
        if len(active_nodes) < self.min_nodes_per_round:
            self.logger.warning(
                f"Insufficient active nodes: {len(active_nodes)} < {self.min_nodes_per_round}"
            )
            return active_nodes
        
        num_selected = max(
            self.min_nodes_per_round,
            int(len(active_nodes) * selection_ratio)
        )
        
        if strategy == "random":
            selected_nodes = np.random.choice(
                active_nodes, 
                size=min(num_selected, len(active_nodes)), 
                replace=False
            ).tolist()
        
        elif strategy == "adaptive":
            # Score-based selection
            node_scores = []
            for node_id in active_nodes:
                node = self.nodes[node_id]
                
                # Combine multiple factors for selection
                data_score = min(node.local_data_samples / 1000, 1.0)  # Normalize to [0,1]
                compute_score = min(node.computational_capacity / 10.0, 1.0)
                trust_score = node.trust_score
                recency_score = max(0, 1 - (time.time() - node.last_update_timestamp) / 3600)
                
                combined_score = (
                    0.3 * data_score + 
                    0.2 * compute_score + 
                    0.3 * trust_score + 
                    0.2 * recency_score
                )
                
                node_scores.append((node_id, combined_score))
            
            # Select top nodes
            node_scores.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = [
                node_id for node_id, _ in node_scores[:num_selected]
            ]
        
        else:  # "balanced"
            # Stratified selection by node type
            node_types = defaultdict(list)
            for node_id in active_nodes:
                node_types[self.nodes[node_id].node_type].append(node_id)
            
            selected_nodes = []
            nodes_per_type = max(1, num_selected // len(node_types))
            
            for node_type, type_nodes in node_types.items():
                selected_count = min(nodes_per_type, len(type_nodes))
                selected_nodes.extend(
                    np.random.choice(type_nodes, size=selected_count, replace=False)
                )
            
            # Fill remaining slots randomly
            remaining = num_selected - len(selected_nodes)
            if remaining > 0:
                remaining_nodes = [
                    n for n in active_nodes if n not in selected_nodes
                ]
                if remaining_nodes:
                    additional = np.random.choice(
                        remaining_nodes, 
                        size=min(remaining, len(remaining_nodes)), 
                        replace=False
                    )
                    selected_nodes.extend(additional)
        
        self.node_selection_history.append({
            'round': self.current_round,
            'strategy': strategy,
            'selected_nodes': selected_nodes,
            'selection_scores': dict(node_scores) if strategy == "adaptive" else {}
        })
        
        self.logger.info(
            f"Selected {len(selected_nodes)} nodes for round {self.current_round}: "
            f"{selected_nodes}"
        )
        
        return selected_nodes
    
    def create_model_update(
        self, 
        node_id: str,
        local_weights: Dict[str, np.ndarray],
        local_loss: float,
        sample_count: int,
        training_time: float
    ) -> ModelUpdate:
        """Create a model update from local training."""
        try:
            # Calculate gradient norms for monitoring
            gradient_norms = []
            for layer_name, weights in local_weights.items():
                grad_norm = np.linalg.norm(weights.flatten())
                gradient_norms.append(grad_norm)
            
            # Create update
            update = ModelUpdate(
                node_id=node_id,
                model_weights=local_weights,
                gradient_norms=gradient_norms,
                local_loss=local_loss,
                sample_count=sample_count,
                training_time=training_time,
                update_timestamp=time.time()
            )
            
            # Add security signature if encryption enabled
            if self.enable_encryption and node_id in self.nodes:
                node = self.nodes[node_id]
                if node.private_key:
                    # Create signature from model weights hash
                    weights_hash = self._hash_model_weights(local_weights)
                    update.signature = self._sign_data(weights_hash, node.private_key)
            
            return update
            
        except Exception as e:
            self.logger.error(f"Failed to create model update for {node_id}: {str(e)}")
            raise
    
    def verify_model_update(self, update: ModelUpdate) -> bool:
        """Verify the integrity of a model update."""
        try:
            if not self.enable_encryption:
                update.is_verified = True
                return True
            
            if update.node_id not in self.nodes:
                self.logger.warning(f"Unknown node {update.node_id}")
                return False
            
            node = self.nodes[update.node_id]
            if not node.public_key or not update.signature:
                self.logger.warning(f"Missing keys/signature for {update.node_id}")
                return False
            
            # Verify signature
            weights_hash = self._hash_model_weights(update.model_weights)
            is_valid = self._verify_signature(
                weights_hash, 
                update.signature, 
                node.public_key
            )
            
            update.is_verified = is_valid
            
            if not is_valid:
                # Reduce trust score for invalid updates
                node.trust_score = max(0.1, node.trust_score * 0.9)
                self.logger.warning(f"Invalid signature from {update.node_id}")
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Failed to verify update from {update.node_id}: {str(e)}")
            return False
    
    def aggregate_updates(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate model updates from multiple nodes."""
        try:
            # Verify all updates first
            verified_updates = []
            for update in updates:
                if self.verify_model_update(update):
                    verified_updates.append(update)
            
            if not verified_updates:
                raise ValueError("No verified updates available for aggregation")
            
            # Apply differential privacy if enabled
            if self.privacy_budget > 0:
                verified_updates = self._apply_differential_privacy(verified_updates)
            
            # Aggregate using selected strategy
            aggregated_weights = self.aggregation_strategy.aggregate(verified_updates)
            
            # Update node trust scores based on participation
            for update in verified_updates:
                node = self.nodes[update.node_id]
                # Reward participation and good performance
                performance_factor = max(0.1, 1.0 / (1.0 + update.local_loss))
                node.trust_score = min(1.0, node.trust_score * 1.02 * performance_factor)
                node.last_update_timestamp = update.update_timestamp
            
            return aggregated_weights
            
        except Exception as e:
            self.logger.error(f"Failed to aggregate updates: {str(e)}")
            raise
    
    def train_federated_round(
        self, 
        selected_nodes: List[str],
        local_epochs: int = 1,
        local_batch_size: int = 32
    ) -> FederatedRound:
        """Execute a complete federated learning round."""
        round_start_time = time.time()
        
        try:
            # Simulate local training (in real implementation, this would be distributed)
            updates = []
            
            for node_id in selected_nodes:
                node = self.nodes[node_id]
                
                # Simulate local training time based on capacity
                training_time = max(
                    1.0, 
                    local_epochs * 10.0 / node.computational_capacity
                )
                
                # Simulate local loss (would be real training result)
                base_loss = 0.1
                capacity_factor = 1.0 / (1.0 + node.computational_capacity)
                data_factor = 1.0 / (1.0 + node.local_data_samples / 100)
                simulated_loss = base_loss * (capacity_factor + data_factor)
                
                # Get current model weights (simplified)
                if self.global_model:
                    local_weights = {}
                    for i, layer in enumerate(self.global_model.layers):
                        if layer.trainable_weights:
                            layer_name = f"layer_{i}_{layer.name}"
                            # Add some random variation to simulate training
                            noise = np.random.normal(0, 0.01, layer.get_weights()[0].shape)
                            local_weights[layer_name] = layer.get_weights()[0] + noise
                    
                    update = self.create_model_update(
                        node_id=node_id,
                        local_weights=local_weights,
                        local_loss=simulated_loss,
                        sample_count=node.local_data_samples,
                        training_time=training_time
                    )
                    
                    updates.append(update)
            
            # Aggregate updates
            if updates:
                aggregated_weights = self.aggregate_updates(updates)
                
                # Update global model
                if self.global_model:
                    # Apply aggregated weights (simplified)
                    for layer_name, weights in aggregated_weights.items():
                        # In real implementation, map back to actual layers
                        pass
                
                # Calculate global metrics
                global_loss = np.mean([update.local_loss for update in updates])
                
                # Calculate convergence metric
                if self.training_history:
                    prev_loss = self.training_history[-1].global_loss
                    convergence_metric = abs(prev_loss - global_loss) / prev_loss
                else:
                    convergence_metric = float('inf')
                
                self.convergence_history.append(convergence_metric)
                
                # Create round record
                round_duration = time.time() - round_start_time
                federated_round = FederatedRound(
                    round_number=self.current_round,
                    participating_nodes=selected_nodes,
                    global_model_version=self.global_model_version,
                    aggregated_weights=aggregated_weights,
                    global_loss=global_loss,
                    convergence_metric=convergence_metric,
                    round_duration=round_duration,
                    start_timestamp=round_start_time,
                    completion_timestamp=time.time()
                )
                
                self.training_history.append(federated_round)
                
                self.logger.info(
                    f"Completed round {self.current_round}: "
                    f"loss={global_loss:.4f}, "
                    f"convergence={convergence_metric:.6f}, "
                    f"duration={round_duration:.2f}s"
                )
                
                return federated_round
            
            else:
                raise ValueError("No valid updates received for aggregation")
                
        except Exception as e:
            self.logger.error(f"Failed to complete federated round {self.current_round}: {str(e)}")
            raise
    
    def train_federated(
        self,
        input_shape: Tuple[int, ...],
        rounds: Optional[int] = None,
        local_epochs: int = 1,
        node_selection_ratio: float = 0.3,
        selection_strategy: str = "adaptive"
    ) -> Dict[str, Any]:
        """Execute complete federated learning training."""
        try:
            self.is_training = True
            start_time = time.time()
            
            # Initialize global model if needed
            if self.global_model is None:
                self.initialize_global_model(input_shape)
            
            max_rounds = rounds or self.max_rounds
            
            self.logger.info(
                f"Starting federated training: {max_rounds} rounds, "
                f"{len(self.nodes)} registered nodes"
            )
            
            converged = False
            
            for round_num in range(max_rounds):
                self.current_round = round_num
                
                # Select nodes for this round
                selected_nodes = self.select_nodes_for_round(
                    selection_ratio=node_selection_ratio,
                    strategy=selection_strategy
                )
                
                if len(selected_nodes) < self.min_nodes_per_round:
                    self.logger.warning(f"Insufficient nodes for round {round_num}")
                    break
                
                # Execute federated round
                federated_round = self.train_federated_round(
                    selected_nodes=selected_nodes,
                    local_epochs=local_epochs
                )
                
                # Check convergence
                if (federated_round.convergence_metric < self.convergence_threshold and 
                    round_num > 5):  # Minimum rounds before convergence
                    self.logger.info(f"Converged after {round_num + 1} rounds")
                    converged = True
                    break
            
            total_duration = time.time() - start_time
            
            # Training summary
            training_summary = {
                'total_rounds': len(self.training_history),
                'total_duration': total_duration,
                'converged': converged,
                'final_loss': self.training_history[-1].global_loss if self.training_history else None,
                'final_convergence': self.training_history[-1].convergence_metric if self.training_history else None,
                'participating_nodes': len(self.nodes),
                'avg_round_duration': np.mean([r.round_duration for r in self.training_history]),
                'model_version': self.global_model_version
            }
            
            self.logger.info(f"Federated training completed: {training_summary}")
            
            return training_summary
            
        except Exception as e:
            self.logger.error(f"Federated training failed: {str(e)}")
            raise
        finally:
            self.is_training = False
    
    def predict_federated(
        self, 
        X: np.ndarray,
        node_id: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform federated inference."""
        try:
            if self.global_model is None:
                raise ValueError("No global model available for inference")
            
            start_time = time.time()
            
            # Use global model for prediction
            predictions = self.global_model.predict(X)
            
            # Calculate reconstruction error for anomaly detection
            reconstruction_error = np.mean(np.square(X - predictions), axis=(1, 2))
            
            # Determine anomalies (simplified threshold)
            threshold = np.percentile(reconstruction_error, 95)
            anomalies = (reconstruction_error > threshold).astype(int)
            
            inference_time = time.time() - start_time
            
            metadata = {
                'inference_time': inference_time,
                'model_version': self.global_model_version,
                'threshold': threshold,
                'max_error': np.max(reconstruction_error),
                'mean_error': np.mean(reconstruction_error),
                'node_id': node_id
            }
            
            return anomalies, metadata
            
        except Exception as e:
            self.logger.error(f"Federated prediction failed: {str(e)}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            active_nodes = sum(1 for node in self.nodes.values() if node.is_active)
            total_samples = sum(node.local_data_samples for node in self.nodes.values())
            avg_trust_score = np.mean([node.trust_score for node in self.nodes.values()])
            
            status = {
                'system_info': {
                    'total_nodes': len(self.nodes),
                    'active_nodes': active_nodes,
                    'is_training': self.is_training,
                    'current_round': self.current_round,
                    'model_version': self.global_model_version
                },
                'data_info': {
                    'total_samples': total_samples,
                    'avg_samples_per_node': total_samples / len(self.nodes) if self.nodes else 0
                },
                'performance_info': {
                    'total_rounds_completed': len(self.training_history),
                    'avg_trust_score': avg_trust_score,
                    'convergence_history': self.convergence_history[-10:],  # Last 10 rounds
                    'last_round_duration': self.training_history[-1].round_duration if self.training_history else None
                },
                'node_details': {
                    node_id: {
                        'type': node.node_type,
                        'active': node.is_active,
                        'trust_score': node.trust_score,
                        'data_samples': node.local_data_samples,
                        'last_update': node.last_update_timestamp
                    }
                    for node_id, node in self.nodes.items()
                }
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {str(e)}")
            return {'error': str(e)}
    
    def save_federated_model(self, filepath: str) -> None:
        """Save the federated learning system state."""
        try:
            system_state = {
                'global_model_version': self.global_model_version,
                'nodes': {node_id: {
                    'node_type': node.node_type,
                    'computational_capacity': node.computational_capacity,
                    'memory_capacity': node.memory_capacity,
                    'bandwidth_capacity': node.bandwidth_capacity,
                    'trust_score': node.trust_score,
                    'local_data_samples': node.local_data_samples,
                    # Don't save keys for security
                } for node_id, node in self.nodes.items()},
                'training_history': [
                    {
                        'round_number': r.round_number,
                        'participating_nodes': r.participating_nodes,
                        'global_loss': r.global_loss,
                        'convergence_metric': r.convergence_metric,
                        'round_duration': r.round_duration
                    }
                    for r in self.training_history
                ],
                'convergence_history': self.convergence_history,
                'current_round': self.current_round
            }
            
            # Save system state
            with open(filepath, 'wb') as f:
                pickle.dump(system_state, f)
            
            # Save global model separately
            if self.global_model:
                model_path = filepath.replace('.pkl', '_model.h5')
                self.global_model.save(model_path)
            
            self.logger.info(f"Saved federated system state to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save federated system: {str(e)}")
            raise
    
    def load_federated_model(self, filepath: str) -> None:
        """Load federated learning system state."""
        try:
            # Load system state
            with open(filepath, 'rb') as f:
                system_state = pickle.load(f)
            
            # Restore basic state
            self.global_model_version = system_state['global_model_version']
            self.training_history = [
                FederatedRound(
                    round_number=r['round_number'],
                    participating_nodes=r['participating_nodes'],
                    global_model_version=self.global_model_version,
                    aggregated_weights={},  # Not saved due to size
                    global_loss=r['global_loss'],
                    convergence_metric=r['convergence_metric'],
                    round_duration=r['round_duration'],
                    start_timestamp=0,  # Historical
                    completion_timestamp=0  # Historical
                )
                for r in system_state.get('training_history', [])
            ]
            self.convergence_history = system_state.get('convergence_history', [])
            self.current_round = system_state.get('current_round', 0)
            
            # Restore nodes (need to re-register for keys)
            self.nodes.clear()
            for node_id, node_data in system_state.get('nodes', {}).items():
                self.register_node(
                    node_id=node_id,
                    node_type=node_data.get('node_type', 'edge'),
                    computational_capacity=node_data.get('computational_capacity', 1.0),
                    memory_capacity=node_data.get('memory_capacity', 512),
                    bandwidth_capacity=node_data.get('bandwidth_capacity', 10.0)
                )
                # Restore additional attributes
                self.nodes[node_id].trust_score = node_data.get('trust_score', 1.0)
                self.nodes[node_id].local_data_samples = node_data.get('local_data_samples', 0)
            
            # Load global model
            model_path = filepath.replace('.pkl', '_model.h5')
            if Path(model_path).exists():
                self.global_model = tf.keras.models.load_model(model_path)
            
            self.logger.info(f"Loaded federated system state from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load federated system: {str(e)}")
            raise
    
    # Utility methods
    def _hash_model_weights(self, weights: Dict[str, np.ndarray]) -> str:
        """Create hash of model weights for integrity checking."""
        weights_str = ""
        for layer_name in sorted(weights.keys()):
            weights_str += f"{layer_name}:{weights[layer_name].tobytes().hex()}"
        return hashlib.sha256(weights_str.encode()).hexdigest()
    
    def _sign_data(self, data: str, private_key: str) -> str:
        """Sign data with private key (simplified implementation)."""
        # In real implementation, use proper cryptographic signing
        return hashlib.sha256(f"{data}{private_key}".encode()).hexdigest()
    
    def _verify_signature(self, data: str, signature: str, public_key: str) -> bool:
        """Verify signature with public key (simplified implementation)."""
        # In real implementation, use proper cryptographic verification
        expected_signature = hashlib.sha256(f"{data}{public_key}".encode()).hexdigest()
        return signature == expected_signature
    
    def _apply_differential_privacy(
        self, 
        updates: List[ModelUpdate]
    ) -> List[ModelUpdate]:
        """Apply differential privacy to model updates."""
        if self.privacy_budget <= 0:
            return updates
        
        try:
            # Simple Gaussian noise mechanism
            noise_scale = 2.0 / self.privacy_budget  # Sensitivity / epsilon
            
            for update in updates:
                for layer_name, weights in update.model_weights.items():
                    # Add calibrated noise
                    noise = np.random.normal(0, noise_scale, weights.shape)
                    update.model_weights[layer_name] = weights + noise
            
            # Reduce privacy budget
            self.privacy_budget = max(0, self.privacy_budget - 0.1)
            
            return updates
            
        except Exception as e:
            self.logger.warning(f"Failed to apply differential privacy: {str(e)}")
            return updates


# Factory functions for different federated learning configurations

def create_edge_federated_system(
    num_edge_nodes: int = 10,
    aggregation_strategy: str = "adaptive"
) -> FederatedAnomalyLearner:
    """Create federated system optimized for edge computing."""
    
    strategy_map = {
        "fedavg": FedAvgStrategy(weighted=True),
        "fedprox": FedProxStrategy(mu=0.01),
        "fednesterov": FedNestrovStrategy(momentum=0.9),
        "adaptive": AdaptiveFederatedStrategy(trust_threshold=0.8)
    }
    
    system = FederatedAnomalyLearner(
        aggregation_strategy=strategy_map.get(aggregation_strategy, AdaptiveFederatedStrategy()),
        min_nodes_per_round=max(3, num_edge_nodes // 4),
        max_rounds=50,  # Fewer rounds for edge
        convergence_threshold=0.005,  # Looser convergence
        model_compression_ratio=0.2,  # Higher compression
        privacy_budget=2.0,
        enable_encryption=True
    )
    
    # Register edge nodes with varying capacities
    for i in range(num_edge_nodes):
        node_type = "edge"
        capacity = np.random.uniform(0.5, 2.0)  # Lower capacity for edge
        memory = np.random.randint(256, 1024)
        bandwidth = np.random.uniform(5.0, 50.0)
        
        system.register_node(
            node_id=f"edge_node_{i:03d}",
            node_type=node_type,
            computational_capacity=capacity,
            memory_capacity=memory,
            bandwidth_capacity=bandwidth
        )
        
        # Simulate local data
        system.nodes[f"edge_node_{i:03d}"].local_data_samples = np.random.randint(100, 1000)
    
    return system


def create_hierarchical_federated_system(
    num_edge_nodes: int = 20,
    num_gateway_nodes: int = 5,
    num_cloud_nodes: int = 2
) -> FederatedAnomalyLearner:
    """Create hierarchical federated system with edge, gateway, and cloud tiers."""
    
    system = FederatedAnomalyLearner(
        aggregation_strategy=AdaptiveFederatedStrategy(trust_threshold=0.7),
        min_nodes_per_round=5,
        max_rounds=100,
        convergence_threshold=0.001,
        model_compression_ratio=0.15,
        privacy_budget=3.0,
        enable_encryption=True
    )
    
    # Edge nodes - limited resources
    for i in range(num_edge_nodes):
        system.register_node(
            node_id=f"edge_{i:03d}",
            node_type="edge",
            computational_capacity=np.random.uniform(0.3, 1.5),
            memory_capacity=np.random.randint(128, 512),
            bandwidth_capacity=np.random.uniform(1.0, 20.0)
        )
        system.nodes[f"edge_{i:03d}"].local_data_samples = np.random.randint(50, 300)
    
    # Gateway nodes - medium resources
    for i in range(num_gateway_nodes):
        system.register_node(
            node_id=f"gateway_{i:03d}",
            node_type="gateway",
            computational_capacity=np.random.uniform(2.0, 5.0),
            memory_capacity=np.random.randint(1024, 4096),
            bandwidth_capacity=np.random.uniform(50.0, 200.0)
        )
        system.nodes[f"gateway_{i:03d}"].local_data_samples = np.random.randint(500, 2000)
    
    # Cloud nodes - high resources
    for i in range(num_cloud_nodes):
        system.register_node(
            node_id=f"cloud_{i:03d}",
            node_type="cloud",
            computational_capacity=np.random.uniform(8.0, 20.0),
            memory_capacity=np.random.randint(8192, 32768),
            bandwidth_capacity=np.random.uniform(500.0, 2000.0)
        )
        system.nodes[f"cloud_{i:03d}"].local_data_samples = np.random.randint(2000, 10000)
    
    return system


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Anomaly Learning System")
    parser.add_argument("--mode", choices=["edge", "hierarchical"], default="edge")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=str, default="federated_model.pkl")
    
    args = parser.parse_args()
    
    # Create federated system
    if args.mode == "edge":
        system = create_edge_federated_system(num_edge_nodes=15)
    else:
        system = create_hierarchical_federated_system()
    
    # Simulate training
    input_shape = (30, 5)  # 30 time steps, 5 features
    
    print("Starting federated training...")
    training_summary = system.train_federated(
        input_shape=input_shape,
        rounds=args.rounds,
        local_epochs=2,
        node_selection_ratio=0.4
    )
    
    print(f"\nTraining Summary:")
    for key, value in training_summary.items():
        print(f"  {key}: {value}")
    
    # Save system
    system.save_federated_model(args.output)
    
    # Show system status
    status = system.get_system_status()
    print(f"\nSystem Status:")
    print(f"  Total Nodes: {status['system_info']['total_nodes']}")
    print(f"  Active Nodes: {status['system_info']['active_nodes']}")
    print(f"  Rounds Completed: {status['performance_info']['total_rounds_completed']}")
    print(f"  Average Trust Score: {status['performance_info']['avg_trust_score']:.3f}")