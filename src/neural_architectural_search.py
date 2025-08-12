"""Neural Architecture Search for Optimal Anomaly Detection Models.

This module implements an advanced Neural Architecture Search (NAS) system
specifically designed for discovering optimal neural network architectures
for time series anomaly detection in IoT environments.
"""

import numpy as np
import random
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

try:
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.models import Model
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. NAS will use simulation mode.")

from .logging_config import get_logger
from .adaptive_multi_modal_detector import DetectionResult


class LayerType(Enum):
    """Supported neural network layer types."""
    LSTM = "lstm"
    GRU = "gru" 
    CONV1D = "conv1d"
    DENSE = "dense"
    ATTENTION = "attention"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    RESIDUAL = "residual"
    TRANSFORMER = "transformer"


class ActivationType(Enum):
    """Supported activation functions."""
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SWISH = "swish"
    GELU = "gelu"


@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    layer_type: LayerType
    units: int
    activation: Optional[ActivationType] = None
    dropout_rate: float = 0.0
    kernel_size: Optional[int] = None
    return_sequences: bool = False
    attention_heads: Optional[int] = None
    use_bias: bool = True
    regularization_l1: float = 0.0
    regularization_l2: float = 0.0


@dataclass
class ArchitectureGenome:
    """Genetic representation of neural network architecture."""
    layers: List[LayerConfig]
    optimizer: str = "adam"
    learning_rate: float = 0.001
    batch_size: int = 32
    loss_function: str = "mse"
    architecture_id: str = field(default_factory=lambda: f"arch_{int(time.time() * 1000)}")
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.architecture_id.startswith("arch_"):
            self.architecture_id = f"arch_{self.architecture_id}"


@dataclass 
class ArchitecturePerformance:
    """Performance metrics for an architecture."""
    architecture_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    inference_time: float
    model_size: int  # Number of parameters
    flops: int  # Floating point operations
    reconstruction_error: float
    stability_score: float
    complexity_penalty: float
    
    @property
    def fitness_score(self) -> float:
        """Calculate overall fitness score."""
        # Multi-objective fitness combining accuracy, efficiency, and simplicity
        accuracy_weight = 0.4
        efficiency_weight = 0.3
        simplicity_weight = 0.3
        
        # Normalize and combine metrics
        accuracy_score = self.f1_score
        efficiency_score = 1.0 / (1.0 + self.training_time + self.inference_time * 10)
        simplicity_score = 1.0 / (1.0 + self.complexity_penalty)
        
        return (accuracy_weight * accuracy_score + 
                efficiency_weight * efficiency_score + 
                simplicity_weight * simplicity_score)


class SearchStrategy(Enum):
    """Neural architecture search strategies."""
    RANDOM_SEARCH = "random_search"
    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY_STRATEGY = "evolutionary_strategy"


class ArchitectureBuilder:
    """Builds neural network models from genome specifications."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.ArchitectureBuilder")
    
    def build_model(self, genome: ArchitectureGenome, input_shape: Tuple[int, int]) -> Optional[Model]:
        """Build TensorFlow model from genome."""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        try:
            inputs = layers.Input(shape=input_shape)
            x = inputs
            
            # Build layers sequentially
            for i, layer_config in enumerate(genome.layers):
                x = self._build_layer(x, layer_config, f"layer_{i}")
            
            # Add reconstruction output layer
            if genome.layers and genome.layers[-1].layer_type in [LayerType.LSTM, LayerType.GRU]:
                # For recurrent layers, add decoder
                x = layers.Dense(input_shape[1], name="reconstruction")(x)
            else:
                # For other layer types, ensure proper output shape
                x = layers.Dense(input_shape[0] * input_shape[1], name="reconstruction_flat")(x)
                x = layers.Reshape(input_shape, name="reconstruction")(x)
            
            model = models.Model(inputs, x)
            
            # Compile model
            optimizer_map = {
                "adam": optimizers.Adam(learning_rate=genome.learning_rate),
                "sgd": optimizers.SGD(learning_rate=genome.learning_rate),
                "rmsprop": optimizers.RMSprop(learning_rate=genome.learning_rate)
            }
            
            model.compile(
                optimizer=optimizer_map.get(genome.optimizer, optimizers.Adam()),
                loss=genome.loss_function,
                metrics=["mae"]
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to build model for {genome.architecture_id}: {e}")
            return None
    
    def _build_layer(self, x, layer_config: LayerConfig, layer_name: str):
        """Build individual layer from configuration."""
        if layer_config.layer_type == LayerType.LSTM:
            x = layers.LSTM(
                units=layer_config.units,
                return_sequences=layer_config.return_sequences,
                activation=layer_config.activation.value if layer_config.activation else "tanh",
                dropout=layer_config.dropout_rate,
                name=f"{layer_name}_lstm"
            )(x)
            
        elif layer_config.layer_type == LayerType.GRU:
            x = layers.GRU(
                units=layer_config.units,
                return_sequences=layer_config.return_sequences,
                activation=layer_config.activation.value if layer_config.activation else "tanh",
                dropout=layer_config.dropout_rate,
                name=f"{layer_name}_gru"
            )(x)
            
        elif layer_config.layer_type == LayerType.CONV1D:
            x = layers.Conv1D(
                filters=layer_config.units,
                kernel_size=layer_config.kernel_size or 3,
                activation=layer_config.activation.value if layer_config.activation else "relu",
                use_bias=layer_config.use_bias,
                name=f"{layer_name}_conv1d"
            )(x)
            
        elif layer_config.layer_type == LayerType.DENSE:
            x = layers.Dense(
                units=layer_config.units,
                activation=layer_config.activation.value if layer_config.activation else "relu",
                use_bias=layer_config.use_bias,
                name=f"{layer_name}_dense"
            )(x)
            
        elif layer_config.layer_type == LayerType.ATTENTION:
            # Multi-head self-attention
            attention = layers.MultiHeadAttention(
                num_heads=layer_config.attention_heads or 4,
                key_dim=layer_config.units // (layer_config.attention_heads or 4),
                name=f"{layer_name}_attention"
            )
            x = attention(x, x)
            
        elif layer_config.layer_type == LayerType.DROPOUT:
            x = layers.Dropout(rate=layer_config.dropout_rate, name=f"{layer_name}_dropout")(x)
            
        elif layer_config.layer_type == LayerType.BATCH_NORM:
            x = layers.BatchNormalization(name=f"{layer_name}_batch_norm")(x)
            
        elif layer_config.layer_type == LayerType.RESIDUAL:
            # Residual connection
            residual = x
            x = layers.Dense(layer_config.units, activation="relu", name=f"{layer_name}_residual_dense")(x)
            if x.shape[-1] == residual.shape[-1]:
                x = layers.Add(name=f"{layer_name}_residual_add")([x, residual])
            
        elif layer_config.layer_type == LayerType.TRANSFORMER:
            # Transformer block (simplified)
            attention = layers.MultiHeadAttention(
                num_heads=layer_config.attention_heads or 4,
                key_dim=layer_config.units // (layer_config.attention_heads or 4),
                name=f"{layer_name}_transformer_attention"
            )
            ffn = tf.keras.Sequential([
                layers.Dense(layer_config.units * 2, activation="relu"),
                layers.Dense(layer_config.units),
            ], name=f"{layer_name}_transformer_ffn")
            
            # Attention block
            attn_output = attention(x, x)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization(name=f"{layer_name}_transformer_norm1")(x)
            
            # Feed forward block
            ffn_output = ffn(x)
            x = layers.Add()([x, ffn_output])
            x = layers.LayerNormalization(name=f"{layer_name}_transformer_norm2")(x)
        
        return x


class GeneticAlgorithmNAS:
    """Genetic Algorithm-based Neural Architecture Search."""
    
    def __init__(self, 
                 population_size: int = 20,
                 generations: int = 10,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_size: int = 2):
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_size = elitism_size
        self.logger = get_logger(f"{__name__}.GeneticAlgorithmNAS")
        
        self.architecture_builder = ArchitectureBuilder()
        self.population: List[ArchitectureGenome] = []
        self.fitness_history: List[List[float]] = []
        self.best_architectures: List[ArchitectureGenome] = []
    
    def initialize_population(self) -> None:
        """Initialize random population of architectures."""
        self.logger.info(f"Initializing population of {self.population_size} architectures")
        
        self.population = []
        for i in range(self.population_size):
            genome = self._create_random_genome()
            self.population.append(genome)
        
        self.logger.info("Population initialization completed")
    
    def _create_random_genome(self) -> ArchitectureGenome:
        """Create a random architecture genome."""
        # Random number of layers (2-8)
        num_layers = random.randint(2, 8)
        
        layers = []
        for i in range(num_layers):
            layer_type = random.choice(list(LayerType))
            
            # Skip certain combinations
            if layer_type == LayerType.DROPOUT and i == 0:
                layer_type = LayerType.LSTM
            
            layer_config = LayerConfig(
                layer_type=layer_type,
                units=random.choice([16, 32, 64, 128, 256]),
                activation=random.choice(list(ActivationType)) if layer_type in [LayerType.DENSE, LayerType.CONV1D] else None,
                dropout_rate=random.uniform(0.0, 0.5) if layer_type == LayerType.DROPOUT else 0.0,
                kernel_size=random.choice([3, 5, 7]) if layer_type == LayerType.CONV1D else None,
                return_sequences=(i < num_layers - 1) if layer_type in [LayerType.LSTM, LayerType.GRU] else False,
                attention_heads=random.choice([2, 4, 8]) if layer_type in [LayerType.ATTENTION, LayerType.TRANSFORMER] else None,
                regularization_l2=random.uniform(0.0, 0.01)
            )
            layers.append(layer_config)
        
        return ArchitectureGenome(
            layers=layers,
            optimizer=random.choice(["adam", "sgd", "rmsprop"]),
            learning_rate=random.choice([0.001, 0.01, 0.0001]),
            batch_size=random.choice([16, 32, 64]),
            loss_function=random.choice(["mse", "mae", "huber"])
        )
    
    def evaluate_architecture(self, genome: ArchitectureGenome, 
                            train_data: np.ndarray, 
                            val_data: np.ndarray,
                            max_epochs: int = 5) -> ArchitecturePerformance:
        """Evaluate architecture performance."""
        try:
            start_time = time.time()
            
            if not TENSORFLOW_AVAILABLE:
                return self._simulate_evaluation(genome)
            
            # Build model
            model = self.architecture_builder.build_model(genome, train_data.shape[1:])
            if model is None:
                return self._create_failed_performance(genome)
            
            # Train model
            early_stopping = callbacks.EarlyStopping(patience=2, restore_best_weights=True)
            
            history = model.fit(
                train_data, train_data,
                validation_data=(val_data, val_data),
                epochs=max_epochs,
                batch_size=genome.batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            training_time = time.time() - start_time
            
            # Evaluate performance
            inference_start = time.time()
            predictions = model.predict(val_data, verbose=0)
            inference_time = (time.time() - inference_start) / len(val_data)
            
            # Calculate metrics
            reconstruction_error = np.mean(np.square(val_data - predictions))
            
            # Mock classification metrics (in real implementation, would need labels)
            accuracy = max(0.5, 1.0 - reconstruction_error)
            precision = accuracy * random.uniform(0.8, 1.0)
            recall = accuracy * random.uniform(0.8, 1.0)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Model complexity
            model_size = model.count_params()
            complexity_penalty = np.log10(model_size) / 10.0
            
            # Stability score (based on training history)
            val_losses = history.history.get('val_loss', [])
            stability_score = 1.0 - (np.std(val_losses) / (np.mean(val_losses) + 1e-8)) if val_losses else 0.5
            
            return ArchitecturePerformance(
                architecture_id=genome.architecture_id,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                training_time=training_time,
                inference_time=inference_time,
                model_size=model_size,
                flops=model_size * 2,  # Rough estimate
                reconstruction_error=reconstruction_error,
                stability_score=stability_score,
                complexity_penalty=complexity_penalty
            )
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate architecture {genome.architecture_id}: {e}")
            return self._create_failed_performance(genome)
    
    def _simulate_evaluation(self, genome: ArchitectureGenome) -> ArchitecturePerformance:
        """Simulate architecture evaluation when TensorFlow is not available."""
        # Calculate complexity score based on architecture
        complexity = sum(layer.units for layer in genome.layers)
        
        # Simulate performance based on architecture characteristics
        base_performance = random.uniform(0.6, 0.9)
        complexity_factor = min(1.0, complexity / 1000.0)
        
        f1_score = base_performance - complexity_factor * 0.1
        training_time = complexity_factor * 30.0 + random.uniform(5, 15)
        inference_time = complexity_factor * 0.01 + random.uniform(0.001, 0.005)
        
        return ArchitecturePerformance(
            architecture_id=genome.architecture_id,
            accuracy=f1_score,
            precision=f1_score * 0.95,
            recall=f1_score * 1.05,
            f1_score=f1_score,
            training_time=training_time,
            inference_time=inference_time,
            model_size=complexity * 100,
            flops=complexity * 200,
            reconstruction_error=1.0 - f1_score,
            stability_score=random.uniform(0.7, 0.95),
            complexity_penalty=complexity_factor
        )
    
    def _create_failed_performance(self, genome: ArchitectureGenome) -> ArchitecturePerformance:
        """Create performance object for failed evaluation."""
        return ArchitecturePerformance(
            architecture_id=genome.architecture_id,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            training_time=1000.0,  # Penalty for failure
            inference_time=1.0,
            model_size=0,
            flops=0,
            reconstruction_error=1.0,
            stability_score=0.0,
            complexity_penalty=1.0
        )
    
    def selection(self, fitness_scores: List[float]) -> List[ArchitectureGenome]:
        """Select parents for reproduction using tournament selection."""
        selected = []
        tournament_size = 3
        
        for _ in range(self.population_size):
            # Tournament selection
            tournament_indices = random.sample(range(len(self.population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(self.population[winner_idx])
        
        return selected
    
    def crossover(self, parent1: ArchitectureGenome, parent2: ArchitectureGenome) -> Tuple[ArchitectureGenome, ArchitectureGenome]:
        """Create offspring through crossover."""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Layer-wise crossover
        min_layers = min(len(parent1.layers), len(parent2.layers))
        crossover_point = random.randint(1, min_layers - 1) if min_layers > 1 else 0
        
        child1_layers = parent1.layers[:crossover_point] + parent2.layers[crossover_point:]
        child2_layers = parent2.layers[:crossover_point] + parent1.layers[crossover_point:]
        
        # Hyperparameter crossover
        child1_optimizer = random.choice([parent1.optimizer, parent2.optimizer])
        child2_optimizer = random.choice([parent1.optimizer, parent2.optimizer])
        
        child1_lr = random.choice([parent1.learning_rate, parent2.learning_rate])
        child2_lr = random.choice([parent1.learning_rate, parent2.learning_rate])
        
        child1 = ArchitectureGenome(
            layers=child1_layers,
            optimizer=child1_optimizer,
            learning_rate=child1_lr,
            batch_size=random.choice([parent1.batch_size, parent2.batch_size]),
            loss_function=random.choice([parent1.loss_function, parent2.loss_function])
        )
        
        child2 = ArchitectureGenome(
            layers=child2_layers,
            optimizer=child2_optimizer,
            learning_rate=child2_lr,
            batch_size=random.choice([parent1.batch_size, parent2.batch_size]),
            loss_function=random.choice([parent1.loss_function, parent2.loss_function])
        )
        
        return child1, child2
    
    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Mutate architecture genome."""
        if random.random() > self.mutation_rate:
            return genome
        
        mutated_layers = []
        for layer_config in genome.layers:
            new_layer = LayerConfig(
                layer_type=layer_config.layer_type,
                units=layer_config.units,
                activation=layer_config.activation,
                dropout_rate=layer_config.dropout_rate,
                kernel_size=layer_config.kernel_size,
                return_sequences=layer_config.return_sequences,
                attention_heads=layer_config.attention_heads,
                use_bias=layer_config.use_bias,
                regularization_l1=layer_config.regularization_l1,
                regularization_l2=layer_config.regularization_l2
            )
            
            # Mutate layer properties
            if random.random() < 0.3:
                new_layer.units = random.choice([16, 32, 64, 128, 256])
            
            if random.random() < 0.2 and new_layer.layer_type in [LayerType.DENSE, LayerType.CONV1D]:
                new_layer.activation = random.choice(list(ActivationType))
            
            if random.random() < 0.1:
                new_layer.regularization_l2 = random.uniform(0.0, 0.01)
            
            mutated_layers.append(new_layer)
        
        # Structural mutations
        if random.random() < 0.1 and len(mutated_layers) < 8:
            # Add layer
            new_layer = LayerConfig(
                layer_type=random.choice(list(LayerType)),
                units=random.choice([16, 32, 64, 128])
            )
            insert_pos = random.randint(0, len(mutated_layers))
            mutated_layers.insert(insert_pos, new_layer)
        
        elif random.random() < 0.1 and len(mutated_layers) > 2:
            # Remove layer
            remove_pos = random.randint(0, len(mutated_layers) - 1)
            mutated_layers.pop(remove_pos)
        
        # Hyperparameter mutations
        optimizer = genome.optimizer
        learning_rate = genome.learning_rate
        batch_size = genome.batch_size
        loss_function = genome.loss_function
        
        if random.random() < 0.1:
            optimizer = random.choice(["adam", "sgd", "rmsprop"])
        
        if random.random() < 0.1:
            learning_rate = random.choice([0.001, 0.01, 0.0001])
        
        if random.random() < 0.1:
            batch_size = random.choice([16, 32, 64])
        
        return ArchitectureGenome(
            layers=mutated_layers,
            optimizer=optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            loss_function=loss_function
        )
    
    def evolve(self, train_data: np.ndarray, val_data: np.ndarray) -> ArchitectureGenome:
        """Run genetic algorithm evolution."""
        self.logger.info(f"Starting NAS evolution for {self.generations} generations")
        
        # Initialize population if not already done
        if not self.population:
            self.initialize_population()
        
        generation_stats = []
        
        for generation in range(self.generations):
            self.logger.info(f"Generation {generation + 1}/{self.generations}")
            
            # Evaluate population
            population_performance = []
            fitness_scores = []
            
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                for genome in self.population:
                    future = executor.submit(self.evaluate_architecture, genome, train_data, val_data)
                    futures[future] = genome
                
                for future in as_completed(futures):
                    genome = futures[future]
                    try:
                        performance = future.result(timeout=300)  # 5 minute timeout
                        population_performance.append(performance)
                        fitness_scores.append(performance.fitness_score)
                    except Exception as e:
                        self.logger.error(f"Evaluation failed for {genome.architecture_id}: {e}")
                        failed_performance = self._create_failed_performance(genome)
                        population_performance.append(failed_performance)
                        fitness_scores.append(0.0)
            
            # Track statistics
            gen_stats = {
                "generation": generation + 1,
                "best_fitness": max(fitness_scores),
                "avg_fitness": np.mean(fitness_scores),
                "std_fitness": np.std(fitness_scores),
                "best_architecture": self.population[np.argmax(fitness_scores)].architecture_id
            }
            generation_stats.append(gen_stats)
            
            self.logger.info(f"Generation {generation + 1} - Best: {gen_stats['best_fitness']:.4f}, "
                           f"Avg: {gen_stats['avg_fitness']:.4f}")
            
            # Store best architectures
            best_idx = np.argmax(fitness_scores)
            if not self.best_architectures or fitness_scores[best_idx] > max(self.fitness_history[-1]) if self.fitness_history else 0:
                self.best_architectures.append(self.population[best_idx])
            
            self.fitness_history.append(fitness_scores)
            
            # Create next generation
            if generation < self.generations - 1:
                # Elitism - keep best individuals
                elite_indices = np.argsort(fitness_scores)[-self.elitism_size:]
                new_population = [self.population[i] for i in elite_indices]
                
                # Generate offspring
                selected_parents = self.selection(fitness_scores)
                
                while len(new_population) < self.population_size:
                    parent1, parent2 = random.sample(selected_parents, 2)
                    child1, child2 = self.crossover(parent1, parent2)
                    
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    
                    new_population.extend([child1, child2])
                
                # Trim to population size
                self.population = new_population[:self.population_size]
        
        # Return best architecture
        best_performance_idx = np.argmax([max(gen_fitness) for gen_fitness in self.fitness_history])
        best_gen_fitness = self.fitness_history[best_performance_idx]
        best_arch_idx = np.argmax(best_gen_fitness)
        
        best_architecture = self.population[best_arch_idx] if best_performance_idx == len(self.fitness_history) - 1 else self.best_architectures[-1]
        
        self.logger.info(f"NAS evolution completed. Best fitness: {max(best_gen_fitness):.4f}")
        
        return best_architecture
    
    def get_evolution_history(self) -> Dict[str, Any]:
        """Get detailed evolution history."""
        return {
            "generations": self.generations,
            "population_size": self.population_size,
            "fitness_history": self.fitness_history,
            "best_architectures": [arch.architecture_id for arch in self.best_architectures],
            "final_population": [arch.architecture_id for arch in self.population]
        }


class NeuralArchitectureSearchEngine:
    """Main NAS engine coordinating different search strategies."""
    
    def __init__(self, 
                 search_strategy: SearchStrategy = SearchStrategy.GENETIC_ALGORITHM,
                 max_search_time: int = 3600,  # 1 hour
                 parallel_evaluations: int = 4):
        
        self.search_strategy = search_strategy
        self.max_search_time = max_search_time
        self.parallel_evaluations = parallel_evaluations
        self.logger = get_logger(__name__)
        
        # Initialize search algorithm
        if search_strategy == SearchStrategy.GENETIC_ALGORITHM:
            self.search_algorithm = GeneticAlgorithmNAS()
        else:
            # For other strategies, use GA as default
            self.search_algorithm = GeneticAlgorithmNAS()
            self.logger.warning(f"Strategy {search_strategy.value} not implemented, using genetic algorithm")
    
    def search(self, train_data: np.ndarray, val_data: np.ndarray) -> Tuple[ArchitectureGenome, Dict[str, Any]]:
        """Execute neural architecture search."""
        self.logger.info(f"Starting NAS with {self.search_strategy.value}")
        
        start_time = time.time()
        
        # Run search
        best_architecture = self.search_algorithm.evolve(train_data, val_data)
        
        search_time = time.time() - start_time
        
        # Collect results
        search_results = {
            "search_strategy": self.search_strategy.value,
            "search_time": search_time,
            "best_architecture": {
                "architecture_id": best_architecture.architecture_id,
                "num_layers": len(best_architecture.layers),
                "layer_types": [layer.layer_type.value for layer in best_architecture.layers],
                "total_parameters": sum(layer.units for layer in best_architecture.layers),
                "optimizer": best_architecture.optimizer,
                "learning_rate": best_architecture.learning_rate,
                "batch_size": best_architecture.batch_size
            },
            "evolution_history": self.search_algorithm.get_evolution_history()
        }
        
        self.logger.info(f"NAS completed in {search_time:.2f}s. Best architecture: {best_architecture.architecture_id}")
        
        return best_architecture, search_results
    
    def save_results(self, architecture: ArchitectureGenome, results: Dict[str, Any], path: Path) -> None:
        """Save NAS results to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save architecture genome
        architecture_data = {
            "architecture_id": architecture.architecture_id,
            "layers": [
                {
                    "layer_type": layer.layer_type.value,
                    "units": layer.units,
                    "activation": layer.activation.value if layer.activation else None,
                    "dropout_rate": layer.dropout_rate,
                    "kernel_size": layer.kernel_size,
                    "return_sequences": layer.return_sequences,
                    "attention_heads": layer.attention_heads,
                    "use_bias": layer.use_bias,
                    "regularization_l1": layer.regularization_l1,
                    "regularization_l2": layer.regularization_l2
                }
                for layer in architecture.layers
            ],
            "optimizer": architecture.optimizer,
            "learning_rate": architecture.learning_rate,
            "batch_size": architecture.batch_size,
            "loss_function": architecture.loss_function
        }
        
        with open(path / "best_architecture.json", "w") as f:
            json.dump(architecture_data, f, indent=2)
        
        # Save search results
        with open(path / "search_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"NAS results saved to {path}")
    
    def load_architecture(self, path: Path) -> ArchitectureGenome:
        """Load architecture from saved results."""
        architecture_path = path / "best_architecture.json"
        
        with open(architecture_path, "r") as f:
            architecture_data = json.load(f)
        
        # Reconstruct layers
        layers = []
        for layer_data in architecture_data["layers"]:
            layer_config = LayerConfig(
                layer_type=LayerType(layer_data["layer_type"]),
                units=layer_data["units"],
                activation=ActivationType(layer_data["activation"]) if layer_data["activation"] else None,
                dropout_rate=layer_data["dropout_rate"],
                kernel_size=layer_data["kernel_size"],
                return_sequences=layer_data["return_sequences"],
                attention_heads=layer_data["attention_heads"],
                use_bias=layer_data["use_bias"],
                regularization_l1=layer_data["regularization_l1"],
                regularization_l2=layer_data["regularization_l2"]
            )
            layers.append(layer_config)
        
        return ArchitectureGenome(
            layers=layers,
            optimizer=architecture_data["optimizer"],
            learning_rate=architecture_data["learning_rate"],
            batch_size=architecture_data["batch_size"],
            loss_function=architecture_data["loss_function"],
            architecture_id=architecture_data["architecture_id"]
        )