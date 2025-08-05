"""
Quantum Performance Optimizer

Advanced performance optimization for quantum-inspired task planning using
adaptive algorithms, caching strategies, and parallel processing capabilities.
"""

import asyncio
import concurrent.futures
import functools
import hashlib
import logging
import pickle
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json
from pathlib import Path

from .quantum_inspired.task_representation import Task, TaskGraph
from .quantum_inspired.quantum_optimization_base import OptimizationResult
from .task_planning.task_scheduler import TaskScheduler, SchedulingAlgorithm, SchedulingResult


class CacheStrategy(Enum):
    """Caching strategies for optimization results."""
    NONE = "none"
    LRU = "lru"
    LFU = "lfu"
    QUANTUM_INSPIRED = "quantum_inspired"


class ParallelizationMode(Enum):
    """Parallelization modes for optimization."""
    SEQUENTIAL = "sequential"
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    ASYNC = "async"
    QUANTUM_PARALLEL = "quantum_parallel"


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization operations."""
    optimization_time: float = 0.0
    cache_hit_ratio: float = 0.0
    parallel_efficiency: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    quantum_operations_per_second: float = 0.0
    convergence_speed: float = 0.0
    solution_quality: float = 0.0
    energy_efficiency: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "optimization_time": self.optimization_time,
            "cache_hit_ratio": self.cache_hit_ratio,
            "parallel_efficiency": self.parallel_efficiency,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_utilization": self.cpu_utilization,
            "quantum_operations_per_second": self.quantum_operations_per_second,
            "convergence_speed": self.convergence_speed,
            "solution_quality": self.solution_quality,
            "energy_efficiency": self.energy_efficiency
        }


@dataclass
class OptimizationConfig:
    """Configuration for quantum performance optimization."""
    cache_strategy: CacheStrategy = CacheStrategy.QUANTUM_INSPIRED
    parallelization_mode: ParallelizationMode = ParallelizationMode.QUANTUM_PARALLEL
    cache_size_mb: int = 100
    max_parallel_workers: int = 4
    enable_adaptive_algorithms: bool = True
    enable_quantum_acceleration: bool = True
    enable_memory_optimization: bool = True
    performance_target_quality: float = 0.95
    performance_target_speed: float = 10.0  # seconds
    auto_scaling: bool = True
    profiling_enabled: bool = False


class QuantumCache:
    """
    Quantum-inspired caching system with probabilistic eviction.
    
    Uses quantum superposition principles to determine cache eviction
    with probabilistic selection based on access patterns and quantum weights.
    """
    
    def __init__(
        self, 
        max_size_mb: int = 100, 
        strategy: CacheStrategy = CacheStrategy.QUANTUM_INSPIRED
    ):
        """
        Initialize quantum cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
            strategy: Caching strategy to use
        """
        self.max_size_mb = max_size_mb
        self.strategy = strategy
        self.cache: Dict[str, Any] = {}
        self.access_counts: Dict[str, int] = {}
        self.access_times: Dict[str, float] = {}
        self.quantum_weights: Dict[str, float] = {}
        self.current_size_mb = 0.0
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _calculate_key(self, task_graph: TaskGraph, config: Dict[str, Any]) -> str:
        """Calculate cache key for task graph and configuration."""
        # Create deterministic hash of task graph and config
        task_data = task_graph.to_dict()
        combined_data = {"tasks": task_data, "config": config}
        
        data_str = json.dumps(combined_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result."""
        if key in self.cache:
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any, quantum_weight: float = 1.0) -> None:
        """Store result in cache."""
        # Calculate size of value
        try:
            value_size_mb = len(pickle.dumps(value)) / (1024 * 1024)
        except:
            value_size_mb = 1.0  # Fallback estimate
        
        # Check if we need to evict
        while (self.current_size_mb + value_size_mb > self.max_size_mb and 
               self.cache):
            self._evict_item()
        
        # Store the value
        self.cache[key] = value
        self.access_counts[key] = 1
        self.access_times[key] = time.time()
        self.quantum_weights[key] = quantum_weight
        self.current_size_mb += value_size_mb
    
    def _evict_item(self) -> None:
        """Evict item based on strategy."""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Least Recently Used
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            self._remove_item(oldest_key)
        
        elif self.strategy == CacheStrategy.LFU:
            # Least Frequently Used
            least_used_key = min(self.access_counts.items(), key=lambda x: x[1])[0]
            self._remove_item(least_used_key)
        
        elif self.strategy == CacheStrategy.QUANTUM_INSPIRED:
            # Quantum-inspired probabilistic eviction
            self._quantum_evict()
    
    def _quantum_evict(self) -> None:
        """Quantum-inspired eviction using superposition principles."""
        if not self.cache:
            return
        
        # Calculate eviction probabilities based on quantum weights
        keys = list(self.cache.keys())
        probabilities = []
        
        current_time = time.time()
        
        for key in keys:
            # Factors influencing eviction probability
            access_recency = current_time - self.access_times.get(key, 0)
            access_frequency = self.access_counts.get(key, 1)
            quantum_weight = self.quantum_weights.get(key, 1.0)
            
            # Lower quantum weight and recent access reduce eviction probability
            # Higher access frequency reduces eviction probability
            eviction_prob = (access_recency / 3600.0) / (quantum_weight * np.log(access_frequency + 1))
            probabilities.append(eviction_prob)
        
        # Normalize probabilities
        if sum(probabilities) > 0:
            probabilities = np.array(probabilities) / sum(probabilities)
            
            # Select item to evict based on quantum superposition
            selected_idx = np.random.choice(len(keys), p=probabilities)
            selected_key = keys[selected_idx]
            
            self._remove_item(selected_key)
    
    def _remove_item(self, key: str) -> None:
        """Remove item from cache."""
        if key in self.cache:
            try:
                item_size_mb = len(pickle.dumps(self.cache[key])) / (1024 * 1024)
                self.current_size_mb -= item_size_mb
            except:
                pass
            
            del self.cache[key]
            self.access_counts.pop(key, None)
            self.access_times.pop(key, None)
            self.quantum_weights.pop(key, None)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "cache_size_mb": self.current_size_mb,
            "max_size_mb": self.max_size_mb,
            "utilization": self.current_size_mb / self.max_size_mb if self.max_size_mb > 0 else 0,
            "strategy": self.strategy.value,
            "total_accesses": sum(self.access_counts.values()),
            "average_quantum_weight": np.mean(list(self.quantum_weights.values())) if self.quantum_weights else 0
        }


class AdaptiveAlgorithmSelector:
    """
    Adaptive algorithm selector using quantum-inspired learning.
    
    Learns from past performance to automatically select the best
    algorithm for specific task graph characteristics.
    """
    
    def __init__(self):
        """Initialize adaptive selector."""
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.algorithm_weights: Dict[str, float] = {
            "quantum_annealing": 1.0,
            "qaoa": 1.0,
            "hybrid": 1.5  # Slightly prefer hybrid initially
        }
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def select_algorithm(
        self, 
        task_graph: TaskGraph, 
        target_quality: float = 0.95
    ) -> SchedulingAlgorithm:
        """
        Select best algorithm for given task graph.
        
        Args:
            task_graph: Task graph to analyze
            target_quality: Target solution quality
            
        Returns:
            Selected scheduling algorithm
        """
        # Extract task graph features
        features = self._extract_features(task_graph)
        
        # Quantum-inspired algorithm selection with exploration
        if np.random.random() < self.exploration_rate:
            # Exploration: select random algorithm
            algorithms = list(SchedulingAlgorithm)
            selected = np.random.choice(algorithms)
            self.logger.debug(f"Exploration: selected {selected.value}")
            return selected
        
        # Exploitation: select best algorithm based on learned weights
        algorithm_scores = {}
        
        for alg_name, weight in self.algorithm_weights.items():
            try:
                algorithm = SchedulingAlgorithm(alg_name)
                
                # Calculate expected performance based on features and history
                expected_score = self._calculate_expected_score(
                    algorithm, features, target_quality
                )
                
                # Apply quantum weight
                algorithm_scores[algorithm] = expected_score * weight
                
            except ValueError:
                continue
        
        if algorithm_scores:
            best_algorithm = max(algorithm_scores.items(), key=lambda x: x[1])[0]
            self.logger.debug(f"Selected {best_algorithm.value} with score {algorithm_scores[best_algorithm]:.3f}")
            return best_algorithm
        
        # Fallback
        return SchedulingAlgorithm.QUANTUM_ANNEALING
    
    def update_performance(
        self, 
        algorithm: SchedulingAlgorithm, 
        metrics: PerformanceMetrics,
        task_graph: TaskGraph
    ) -> None:
        """
        Update algorithm performance history and weights.
        
        Args:
            algorithm: Algorithm that was used
            metrics: Performance metrics achieved
            task_graph: Task graph that was optimized
        """
        alg_name = algorithm.value
        
        # Store performance history
        if alg_name not in self.performance_history:
            self.performance_history[alg_name] = []
        
        self.performance_history[alg_name].append(metrics)
        
        # Limit history size
        if len(self.performance_history[alg_name]) > 100:
            self.performance_history[alg_name] = self.performance_history[alg_name][-100:]
        
        # Update algorithm weights based on performance
        performance_score = self._calculate_performance_score(metrics)
        
        # Quantum-inspired weight update with momentum
        current_weight = self.algorithm_weights.get(alg_name, 1.0)
        weight_update = self.learning_rate * (performance_score - 0.5)  # Center around 0.5
        
        new_weight = current_weight + weight_update
        new_weight = max(0.1, min(2.0, new_weight))  # Clamp weights
        
        self.algorithm_weights[alg_name] = new_weight
        
        self.logger.debug(f"Updated {alg_name} weight: {current_weight:.3f} -> {new_weight:.3f}")
    
    def _extract_features(self, task_graph: TaskGraph) -> Dict[str, float]:
        """Extract features from task graph for algorithm selection."""
        num_tasks = len(task_graph.tasks)
        
        if num_tasks == 0:
            return {"num_tasks": 0, "complexity": 0, "parallelism": 0, "resource_intensity": 0}
        
        # Calculate graph complexity
        num_edges = task_graph.graph.number_of_edges()
        complexity = num_edges / (num_tasks * (num_tasks - 1) / 2) if num_tasks > 1 else 0
        
        # Calculate potential parallelism
        try:
            critical_path = task_graph.get_critical_path()
            critical_path_length = len(critical_path)
            parallelism = num_tasks / critical_path_length if critical_path_length > 0 else 1
        except:
            parallelism = 1.0
        
        # Calculate resource intensity
        total_resources = 0
        for task in task_graph.tasks.values():
            total_resources += len(task.resource_requirements)
        resource_intensity = total_resources / num_tasks if num_tasks > 0 else 0
        
        return {
            "num_tasks": float(num_tasks),
            "complexity": complexity,
            "parallelism": parallelism,
            "resource_intensity": resource_intensity
        }
    
    def _calculate_expected_score(
        self, 
        algorithm: SchedulingAlgorithm, 
        features: Dict[str, float],
        target_quality: float
    ) -> float:
        """Calculate expected performance score for algorithm."""
        alg_name = algorithm.value
        
        if alg_name not in self.performance_history or not self.performance_history[alg_name]:
            return 0.5  # Neutral score for unknown algorithms
        
        recent_metrics = self.performance_history[alg_name][-10:]  # Last 10 runs
        
        # Weight metrics based on task graph features
        score = 0.0
        
        for metrics in recent_metrics:
            # Quality component
            quality_score = min(metrics.solution_quality / target_quality, 1.0) * 0.4
            
            # Speed component
            speed_score = max(0, 1.0 - metrics.optimization_time / 30.0) * 0.3  # 30s target
            
            # Efficiency component
            efficiency_score = (metrics.parallel_efficiency + metrics.energy_efficiency) / 2 * 0.3
            
            score += quality_score + speed_score + efficiency_score
        
        return score / len(recent_metrics)
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score from metrics."""
        # Weighted combination of metrics
        score = (
            0.3 * min(metrics.solution_quality, 1.0) +
            0.2 * max(0, 1.0 - metrics.optimization_time / 60.0) +  # 60s target
            0.2 * metrics.parallel_efficiency +
            0.15 * metrics.cache_hit_ratio +
            0.15 * metrics.energy_efficiency
        )
        
        return max(0.0, min(1.0, score))
    
    def get_algorithm_rankings(self) -> Dict[str, float]:
        """Get current algorithm rankings."""
        return self.algorithm_weights.copy()


class QuantumPerformanceOptimizer:
    """
    Advanced performance optimizer for quantum-inspired task planning.
    
    Provides caching, parallelization, adaptive algorithm selection,
    and performance monitoring for scalable quantum optimization.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize quantum performance optimizer.
        
        Args:
            config: Performance optimization configuration
        """
        self.config = config or OptimizationConfig()
        
        # Initialize components
        self.cache = QuantumCache(
            max_size_mb=self.config.cache_size_mb,
            strategy=self.config.cache_strategy
        )
        
        self.adaptive_selector = AdaptiveAlgorithmSelector()
        
        self.task_scheduler = TaskScheduler()
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.total_cache_requests = 0
        self.cache_hits = 0
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def optimize_async(
        self,
        task_graph: TaskGraph,
        algorithm: Optional[SchedulingAlgorithm] = None,
        target_quality: float = None
    ) -> Tuple[SchedulingResult, PerformanceMetrics]:
        """
        Asynchronous optimization with performance monitoring.
        
        Args:
            task_graph: Task graph to optimize
            algorithm: Algorithm to use (None for adaptive selection)
            target_quality: Target solution quality
            
        Returns:
            Tuple of scheduling result and performance metrics
        """
        start_time = time.time()
        target_quality = target_quality or self.config.performance_target_quality
        
        # Check cache first
        cache_key = self._get_cache_key(task_graph, algorithm, target_quality)
        self.total_cache_requests += 1
        
        cached_result = self.cache.get(cache_key)
        if cached_result and self.config.cache_strategy != CacheStrategy.NONE:
            self.cache_hits += 1
            result, cached_metrics = cached_result
            
            # Update cached result with current timing
            metrics = PerformanceMetrics(
                optimization_time=time.time() - start_time,
                cache_hit_ratio=self.cache_hits / self.total_cache_requests,
                solution_quality=cached_metrics.solution_quality,
                parallel_efficiency=cached_metrics.parallel_efficiency,
                energy_efficiency=cached_metrics.energy_efficiency
            )
            
            self.logger.info(f"Cache hit for optimization (key: {cache_key[:16]}...)")
            return result, metrics
        
        # Select algorithm adaptively if not specified
        if algorithm is None and self.config.enable_adaptive_algorithms:
            algorithm = self.adaptive_selector.select_algorithm(task_graph, target_quality)
            self.logger.info(f"Adaptive algorithm selection: {algorithm.value}")
        elif algorithm is None:
            algorithm = SchedulingAlgorithm.QUANTUM_ANNEALING
        
        # Run optimization
        if self.config.parallelization_mode == ParallelizationMode.ASYNC:
            result = await self._optimize_with_async(task_graph, algorithm)
        else:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._optimize_sync, task_graph, algorithm
            )
        
        optimization_time = time.time() - start_time
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(result, optimization_time)
        
        # Update adaptive selector
        if self.config.enable_adaptive_algorithms:
            self.adaptive_selector.update_performance(algorithm, metrics, task_graph)
        
        # Cache result
        if self.config.cache_strategy != CacheStrategy.NONE:
            quantum_weight = self._calculate_quantum_cache_weight(task_graph, metrics)
            self.cache.put(cache_key, (result, metrics), quantum_weight)
        
        # Store performance history
        self.performance_history.append(metrics)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        return result, metrics
    
    def optimize(
        self,
        task_graph: TaskGraph,
        algorithm: Optional[SchedulingAlgorithm] = None,
        target_quality: float = None
    ) -> Tuple[SchedulingResult, PerformanceMetrics]:
        """
        Synchronous optimization wrapper.
        
        Args:
            task_graph: Task graph to optimize
            algorithm: Algorithm to use
            target_quality: Target solution quality
            
        Returns:
            Tuple of scheduling result and performance metrics
        """
        # Use asyncio for consistent interface
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.optimize_async(task_graph, algorithm, target_quality)
            )
        finally:
            loop.close()
    
    async def _optimize_with_async(
        self, 
        task_graph: TaskGraph, 
        algorithm: SchedulingAlgorithm
    ) -> SchedulingResult:
        """Optimize using async execution."""
        # Create tasks list from task graph
        tasks = list(task_graph.tasks.values())
        resource_constraints = list(task_graph.resource_constraints.values())
        
        # Run optimization in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.task_scheduler.schedule_tasks,
            tasks,
            resource_constraints,
            algorithm
        )
        
        return result
    
    def _optimize_sync(
        self, 
        task_graph: TaskGraph, 
        algorithm: SchedulingAlgorithm
    ) -> SchedulingResult:
        """Synchronous optimization."""
        tasks = list(task_graph.tasks.values())
        resource_constraints = list(task_graph.resource_constraints.values())
        
        return self.task_scheduler.schedule_tasks(
            tasks, resource_constraints, algorithm
        )
    
    def _get_cache_key(
        self, 
        task_graph: TaskGraph, 
        algorithm: Optional[SchedulingAlgorithm],
        target_quality: float
    ) -> str:
        """Generate cache key for optimization request."""
        key_data = {
            "task_graph": task_graph.to_dict(),
            "algorithm": algorithm.value if algorithm else "adaptive",
            "target_quality": target_quality,
            "config_hash": hash(str(self.config))
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _calculate_metrics(
        self, 
        result: SchedulingResult, 
        optimization_time: float
    ) -> PerformanceMetrics:
        """Calculate performance metrics for optimization result."""
        metrics = PerformanceMetrics()
        
        metrics.optimization_time = optimization_time
        metrics.cache_hit_ratio = self.cache_hits / self.total_cache_requests if self.total_cache_requests > 0 else 0
        metrics.solution_quality = result.quality_metrics.get("overall_quality", 0)
        
        # Calculate parallel efficiency (simplified)
        if hasattr(result, 'metadata') and 'optimization_iterations' in result.metadata:
            iterations = result.metadata['optimization_iterations']
            target_time = self.config.performance_target_speed
            if optimization_time > 0:
                metrics.parallel_efficiency = min(target_time / optimization_time, 1.0)
        else:
            metrics.parallel_efficiency = 0.5  # Default
        
        # Calculate quantum operations per second (estimated)
        if optimization_time > 0:
            num_tasks = len(result.schedule)
            estimated_operations = num_tasks * 100  # Rough estimate
            metrics.quantum_operations_per_second = estimated_operations / optimization_time
        
        # Energy efficiency (based on optimization speed and quality)
        if optimization_time > 0 and metrics.solution_quality > 0:
            metrics.energy_efficiency = metrics.solution_quality / optimization_time * 10
            metrics.energy_efficiency = min(metrics.energy_efficiency, 1.0)
        
        # Convergence speed
        if hasattr(result, 'metadata') and result.metadata.get('optimization_converged', False):
            metrics.convergence_speed = 1.0 / optimization_time if optimization_time > 0 else 0
        
        return metrics
    
    def _calculate_quantum_cache_weight(
        self, 
        task_graph: TaskGraph, 
        metrics: PerformanceMetrics
    ) -> float:
        """Calculate quantum weight for cache entry."""
        base_weight = 1.0
        
        # Higher weight for better quality solutions
        quality_weight = metrics.solution_quality * 2.0
        
        # Higher weight for faster optimizations
        speed_weight = metrics.parallel_efficiency
        
        # Higher weight for more complex task graphs
        complexity_weight = len(task_graph.tasks) / 10.0
        
        total_weight = base_weight + quality_weight + speed_weight + complexity_weight
        return min(total_weight, 5.0)  # Cap at 5.0
    
    def optimize_batch(
        self,
        task_graphs: List[TaskGraph],
        max_parallel: Optional[int] = None
    ) -> List[Tuple[SchedulingResult, PerformanceMetrics]]:
        """
        Optimize multiple task graphs in parallel.
        
        Args:
            task_graphs: List of task graphs to optimize
            max_parallel: Maximum parallel optimizations
            
        Returns:
            List of optimization results and metrics
        """
        max_parallel = max_parallel or self.config.max_parallel_workers
        
        if self.config.parallelization_mode == ParallelizationMode.ASYNC:
            return asyncio.run(self._optimize_batch_async(task_graphs, max_parallel))
        else:
            return self._optimize_batch_sync(task_graphs, max_parallel)
    
    async def _optimize_batch_async(
        self,
        task_graphs: List[TaskGraph],
        max_parallel: int
    ) -> List[Tuple[SchedulingResult, PerformanceMetrics]]:
        """Asynchronous batch optimization."""
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def optimize_with_semaphore(task_graph):
            async with semaphore:
                return await self.optimize_async(task_graph)
        
        tasks = [optimize_with_semaphore(tg) for tg in task_graphs]
        return await asyncio.gather(*tasks)
    
    def _optimize_batch_sync(
        self,
        task_graphs: List[TaskGraph],
        max_parallel: int
    ) -> List[Tuple[SchedulingResult, PerformanceMetrics]]:
        """Synchronous batch optimization with thread pool."""
        if self.config.parallelization_mode == ParallelizationMode.THREAD_POOL:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = [executor.submit(self.optimize, tg) for tg in task_graphs]
                return [future.result() for future in concurrent.futures.as_completed(futures)]
        
        elif self.config.parallelization_mode == ParallelizationMode.PROCESS_POOL:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel) as executor:
                futures = [executor.submit(self.optimize, tg) for tg in task_graphs]
                return [future.result() for future in concurrent.futures.as_completed(futures)]
        
        else:
            # Sequential processing
            return [self.optimize(tg) for tg in task_graphs]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.performance_history:
            return {"status": "no_data"}
        
        recent_metrics = self.performance_history[-50:]  # Last 50 optimizations
        
        report = {
            "summary": {
                "total_optimizations": len(self.performance_history),
                "average_optimization_time": np.mean([m.optimization_time for m in recent_metrics]),
                "average_solution_quality": np.mean([m.solution_quality for m in recent_metrics]),
                "cache_hit_ratio": self.cache_hits / self.total_cache_requests if self.total_cache_requests > 0 else 0
            },
            "performance_trends": {
                "optimization_time_trend": [m.optimization_time for m in recent_metrics[-10:]],
                "quality_trend": [m.solution_quality for m in recent_metrics[-10:]],
                "parallel_efficiency_trend": [m.parallel_efficiency for m in recent_metrics[-10:]]
            },
            "cache_statistics": self.cache.get_cache_stats(),
            "algorithm_rankings": self.adaptive_selector.get_algorithm_rankings(),
            "configuration": {
                "cache_strategy": self.config.cache_strategy.value,
                "parallelization_mode": self.config.parallelization_mode.value,
                "adaptive_algorithms": self.config.enable_adaptive_algorithms,
                "max_parallel_workers": self.config.max_parallel_workers
            }
        }
        
        return report
    
    def clear_cache(self) -> None:
        """Clear optimization cache."""
        self.cache = QuantumCache(
            max_size_mb=self.config.cache_size_mb,
            strategy=self.config.cache_strategy
        )
        self.cache_hits = 0
        self.total_cache_requests = 0
        self.logger.info("Optimization cache cleared")
    
    def save_performance_data(self, file_path: str) -> None:
        """Save performance data to file."""
        data = {
            "config": {
                "cache_strategy": self.config.cache_strategy.value,
                "parallelization_mode": self.config.parallelization_mode.value,
                "cache_size_mb": self.config.cache_size_mb,
                "max_parallel_workers": self.config.max_parallel_workers
            },
            "performance_history": [m.to_dict() for m in self.performance_history],
            "cache_stats": self.cache.get_cache_stats(),
            "algorithm_rankings": self.adaptive_selector.get_algorithm_rankings()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Performance data saved to {file_path}")
    
    def load_performance_data(self, file_path: str) -> None:
        """Load performance data from file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Restore performance history
            self.performance_history = [
                PerformanceMetrics(**metrics) for metrics in data.get("performance_history", [])
            ]
            
            # Restore algorithm rankings
            if "algorithm_rankings" in data:
                self.adaptive_selector.algorithm_weights.update(data["algorithm_rankings"])
            
            self.logger.info(f"Performance data loaded from {file_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load performance data: {e}")


# Utility functions for performance optimization

def profile_optimization(func: Callable) -> Callable:
    """Decorator for profiling optimization functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(func.__name__)
        logger.debug(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        
        return result
    return wrapper


async def benchmark_algorithms(
    task_graphs: List[TaskGraph],
    algorithms: List[SchedulingAlgorithm],
    iterations: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different algorithms on task graphs.
    
    Args:
        task_graphs: List of task graphs to test
        algorithms: List of algorithms to benchmark
        iterations: Number of iterations per algorithm
        
    Returns:
        Benchmark results dictionary
    """
    optimizer = QuantumPerformanceOptimizer()
    results = {}
    
    for algorithm in algorithms:
        algorithm_results = {
            "average_time": 0.0,
            "average_quality": 0.0,
            "success_rate": 0.0
        }
        
        total_time = 0.0
        total_quality = 0.0
        successes = 0
        
        for _ in range(iterations):
            for task_graph in task_graphs:
                try:
                    result, metrics = await optimizer.optimize_async(
                        task_graph, algorithm
                    )
                    
                    total_time += metrics.optimization_time
                    total_quality += metrics.solution_quality
                    successes += 1
                    
                except Exception as e:
                    logging.warning(f"Algorithm {algorithm.value} failed: {e}")
        
        total_runs = iterations * len(task_graphs)
        if total_runs > 0:
            algorithm_results["average_time"] = total_time / successes if successes > 0 else float('inf')
            algorithm_results["average_quality"] = total_quality / successes if successes > 0 else 0.0
            algorithm_results["success_rate"] = successes / total_runs
        
        results[algorithm.value] = algorithm_results
    
    return results