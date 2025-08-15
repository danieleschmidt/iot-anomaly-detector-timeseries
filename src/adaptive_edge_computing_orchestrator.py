"""Adaptive Edge Computing Orchestrator for IoT Anomaly Detection.

Advanced edge computing orchestration system that dynamically optimizes
anomaly detection across distributed edge devices, cloud resources, and
hybrid deployments. Features intelligent workload distribution, adaptive
resource allocation, and real-time performance optimization.
"""

import numpy as np
import pandas as pd
import asyncio
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
from pathlib import Path
import pickle
import json
import hashlib
import queue
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings

try:
    import psutil
    import aiohttp
    from aiofiles import open as aio_open
    import asyncio_mqtt
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    warnings.warn("Edge computing dependencies not available. Using simplified implementations.")

from .logging_config import get_logger
from .data_preprocessor import DataPreprocessor


@dataclass
class EdgeDevice:
    """Represents an edge computing device in the network."""
    
    device_id: str
    device_type: str  # "raspberry_pi", "nvidia_jetson", "intel_nuc", "smartphone", "gateway"
    location: Tuple[float, float]  # (latitude, longitude)
    
    # Hardware specifications
    cpu_cores: int
    cpu_frequency: float  # GHz
    memory_total: int  # MB
    storage_total: int  # GB
    gpu_available: bool = False
    gpu_memory: int = 0  # MB
    
    # Network specifications
    bandwidth_up: float  # Mbps
    bandwidth_down: float  # Mbps
    latency_to_cloud: float  # ms
    is_mobile: bool = False
    
    # Current status
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    storage_usage: float = 0.0
    temperature: float = 0.0
    battery_level: float = 100.0
    is_online: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    
    # Capabilities
    supported_models: List[str] = field(default_factory=list)
    max_inference_rate: float = 10.0  # inferences per second
    power_consumption: float = 5.0  # watts
    
    # Performance history
    inference_history: List[Dict[str, Any]] = field(default_factory=list)
    reliability_score: float = 1.0
    avg_inference_time: float = 100.0  # ms
    
    # Edge-specific features
    data_caching_enabled: bool = True
    local_storage_path: Optional[str] = None
    secure_enclave_available: bool = False


@dataclass
class WorkloadRequest:
    """Represents an inference workload request."""
    
    request_id: str
    data: np.ndarray
    model_type: str
    priority: int  # 1-10, higher = more urgent
    max_latency: float  # ms
    accuracy_requirement: float  # 0-1
    privacy_level: int  # 1-5, higher = more private
    timestamp: float = field(default_factory=time.time)
    
    # Metadata
    source_device: Optional[str] = None
    data_size: int = 0
    estimated_compute_time: float = 0.0
    requires_gpu: bool = False
    
    # Results
    assigned_devices: List[str] = field(default_factory=list)
    execution_start_time: Optional[float] = None
    execution_end_time: Optional[float] = None
    result: Optional[Any] = None
    confidence: Optional[float] = None


@dataclass
class ResourceAllocation:
    """Represents resource allocation for a workload."""
    
    device_id: str
    workload_id: str
    allocated_cpu: float  # percentage
    allocated_memory: int  # MB
    allocated_gpu: float = 0.0  # percentage
    estimated_duration: float = 0.0  # ms
    energy_cost: float = 0.0  # joules
    monetary_cost: float = 0.0  # cents
    
    allocation_timestamp: float = field(default_factory=time.time)
    status: str = "pending"  # pending, active, completed, failed


class EdgeComputingStrategy(ABC):
    """Abstract base class for edge computing strategies."""
    
    @abstractmethod
    async def allocate_resources(
        self,
        workload: WorkloadRequest,
        available_devices: List[EdgeDevice]
    ) -> List[ResourceAllocation]:
        """Allocate resources for a workload."""
        pass
    
    @abstractmethod
    def calculate_priority_score(
        self,
        workload: WorkloadRequest,
        device: EdgeDevice
    ) -> float:
        """Calculate priority score for device-workload pair."""
        pass


class LatencyOptimizedStrategy(EdgeComputingStrategy):
    """Strategy optimized for minimal latency."""
    
    def __init__(self, locality_weight: float = 0.7):
        self.locality_weight = locality_weight
        self.logger = get_logger(__name__)
    
    async def allocate_resources(
        self,
        workload: WorkloadRequest,
        available_devices: List[EdgeDevice]
    ) -> List[ResourceAllocation]:
        """Allocate to device with lowest expected latency."""
        try:
            # Filter devices that can handle the workload
            capable_devices = [
                device for device in available_devices
                if (workload.model_type in device.supported_models and
                    device.is_online and
                    device.cpu_usage < 80.0 and
                    device.memory_usage < 80.0)
            ]
            
            if not capable_devices:
                return []
            
            # Calculate latency scores
            device_scores = []
            for device in capable_devices:
                # Base inference time
                base_latency = device.avg_inference_time
                
                # Network latency (if remote)
                network_latency = device.latency_to_cloud if device.device_id != "local" else 0
                
                # Queue latency based on current load
                queue_latency = device.cpu_usage * 10  # Simplified
                
                # Total expected latency
                total_latency = base_latency + network_latency + queue_latency
                
                # Penalize if exceeds requirement
                if total_latency > workload.max_latency:
                    score = 0.1  # Very low score
                else:
                    score = 1000.0 / total_latency  # Higher score for lower latency
                
                device_scores.append((device, score))
            
            # Sort by score (highest first)
            device_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Allocate to best device
            best_device = device_scores[0][0]
            
            allocation = ResourceAllocation(
                device_id=best_device.device_id,
                workload_id=workload.request_id,
                allocated_cpu=min(50.0, 100.0 - best_device.cpu_usage),
                allocated_memory=min(1024, best_device.memory_total - int(best_device.memory_usage * best_device.memory_total / 100)),
                estimated_duration=best_device.avg_inference_time,
                energy_cost=best_device.power_consumption * best_device.avg_inference_time / 1000 / 3600,  # Wh
                status="pending"
            )
            
            self.logger.info(f"Allocated workload {workload.request_id} to {best_device.device_id} (latency-optimized)")
            
            return [allocation]
            
        except Exception as e:
            self.logger.error(f"Failed to allocate resources: {str(e)}")
            return []
    
    def calculate_priority_score(
        self,
        workload: WorkloadRequest,
        device: EdgeDevice
    ) -> float:
        """Calculate priority score based on latency factors."""
        if workload.model_type not in device.supported_models:
            return 0.0
        
        # Base score from device capability
        capability_score = min(device.max_inference_rate / 10.0, 1.0)
        
        # Availability score
        load_factor = (device.cpu_usage + device.memory_usage) / 200.0
        availability_score = max(0.0, 1.0 - load_factor)
        
        # Latency score
        expected_latency = device.avg_inference_time + device.latency_to_cloud
        latency_score = min(workload.max_latency / expected_latency, 1.0) if expected_latency > 0 else 0.0
        
        # Combined score
        total_score = capability_score * availability_score * latency_score
        
        return total_score


class EnergyOptimizedStrategy(EdgeComputingStrategy):
    """Strategy optimized for minimal energy consumption."""
    
    def __init__(self, energy_weight: float = 0.8):
        self.energy_weight = energy_weight
        self.logger = get_logger(__name__)
    
    async def allocate_resources(
        self,
        workload: WorkloadRequest,
        available_devices: List[EdgeDevice]
    ) -> List[ResourceAllocation]:
        """Allocate to device with lowest energy cost."""
        try:
            capable_devices = [
                device for device in available_devices
                if (workload.model_type in device.supported_models and
                    device.is_online and
                    device.battery_level > 20.0)  # Ensure sufficient battery
            ]
            
            if not capable_devices:
                return []
            
            # Calculate energy efficiency scores
            device_scores = []
            for device in capable_devices:
                # Energy per inference
                energy_per_inference = device.power_consumption * device.avg_inference_time / 1000 / 3600  # Wh
                
                # Battery efficiency (prefer devices with higher battery)
                battery_factor = device.battery_level / 100.0 if device.is_mobile else 1.0
                
                # Temperature efficiency (avoid overheated devices)
                temp_factor = max(0.1, 1.0 - max(0, device.temperature - 70) / 30)
                
                # Energy efficiency score
                efficiency = 1.0 / (energy_per_inference + 1e-6) * battery_factor * temp_factor
                
                device_scores.append((device, efficiency))
            
            # Sort by efficiency (highest first)
            device_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Allocate to most efficient device
            best_device = device_scores[0][0]
            
            allocation = ResourceAllocation(
                device_id=best_device.device_id,
                workload_id=workload.request_id,
                allocated_cpu=30.0,  # Conservative allocation for energy saving
                allocated_memory=512,
                estimated_duration=best_device.avg_inference_time * 1.2,  # Slower but more efficient
                energy_cost=best_device.power_consumption * best_device.avg_inference_time / 1000 / 3600,
                status="pending"
            )
            
            self.logger.info(f"Allocated workload {workload.request_id} to {best_device.device_id} (energy-optimized)")
            
            return [allocation]
            
        except Exception as e:
            self.logger.error(f"Failed to allocate resources: {str(e)}")
            return []
    
    def calculate_priority_score(
        self,
        workload: WorkloadRequest,
        device: EdgeDevice
    ) -> float:
        """Calculate priority score based on energy factors."""
        if workload.model_type not in device.supported_models:
            return 0.0
        
        # Energy efficiency score
        energy_per_inference = device.power_consumption * device.avg_inference_time / 1000 / 3600
        energy_score = 1.0 / (energy_per_inference + 1e-6)
        
        # Battery score
        battery_score = device.battery_level / 100.0 if device.is_mobile else 1.0
        
        # Temperature score
        temp_score = max(0.1, 1.0 - max(0, device.temperature - 70) / 30)
        
        # Combined score
        total_score = energy_score * battery_score * temp_score
        
        return total_score


class HybridOptimizedStrategy(EdgeComputingStrategy):
    """Hybrid strategy balancing multiple objectives."""
    
    def __init__(
        self,
        latency_weight: float = 0.4,
        energy_weight: float = 0.3,
        accuracy_weight: float = 0.2,
        cost_weight: float = 0.1
    ):
        self.latency_weight = latency_weight
        self.energy_weight = energy_weight
        self.accuracy_weight = accuracy_weight
        self.cost_weight = cost_weight
        self.logger = get_logger(__name__)
    
    async def allocate_resources(
        self,
        workload: WorkloadRequest,
        available_devices: List[EdgeDevice]
    ) -> List[ResourceAllocation]:
        """Multi-objective resource allocation."""
        try:
            capable_devices = [
                device for device in available_devices
                if (workload.model_type in device.supported_models and
                    device.is_online)
            ]
            
            if not capable_devices:
                return []
            
            # Multi-criteria scoring
            device_scores = []
            for device in capable_devices:
                # Latency score
                expected_latency = device.avg_inference_time + device.latency_to_cloud
                latency_score = min(workload.max_latency / expected_latency, 1.0) if expected_latency > 0 else 0.0
                
                # Energy score
                energy_per_inference = device.power_consumption * device.avg_inference_time / 1000 / 3600
                energy_score = 1.0 / (energy_per_inference + 1e-6)
                energy_score = min(energy_score / 10.0, 1.0)  # Normalize
                
                # Accuracy score (based on device capability)
                accuracy_score = min(device.reliability_score, 1.0)
                
                # Cost score (simplified)
                compute_cost = device.power_consumption * 0.1  # Simplified cost model
                cost_score = 1.0 / (compute_cost + 1.0)
                
                # Availability penalty
                availability_penalty = (device.cpu_usage + device.memory_usage) / 200.0
                availability_score = max(0.0, 1.0 - availability_penalty)
                
                # Weighted combination
                total_score = (
                    self.latency_weight * latency_score +
                    self.energy_weight * energy_score +
                    self.accuracy_weight * accuracy_score +
                    self.cost_weight * cost_score
                ) * availability_score
                
                device_scores.append((device, total_score))
            
            # Sort by score (highest first)
            device_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Check if we should use multiple devices for high-priority tasks
            if workload.priority >= 8 and len(device_scores) > 1:
                # Use top 2 devices for redundancy
                allocations = []
                for i in range(min(2, len(device_scores))):
                    device = device_scores[i][0]
                    allocation = ResourceAllocation(
                        device_id=device.device_id,
                        workload_id=workload.request_id,
                        allocated_cpu=40.0,
                        allocated_memory=768,
                        estimated_duration=device.avg_inference_time,
                        energy_cost=device.power_consumption * device.avg_inference_time / 1000 / 3600,
                        status="pending"
                    )
                    allocations.append(allocation)
                
                self.logger.info(f"Allocated high-priority workload {workload.request_id} to {len(allocations)} devices")
                return allocations
            
            else:
                # Single device allocation
                best_device = device_scores[0][0]
                allocation = ResourceAllocation(
                    device_id=best_device.device_id,
                    workload_id=workload.request_id,
                    allocated_cpu=50.0,
                    allocated_memory=1024,
                    estimated_duration=best_device.avg_inference_time,
                    energy_cost=best_device.power_consumption * best_device.avg_inference_time / 1000 / 3600,
                    status="pending"
                )
                
                self.logger.info(f"Allocated workload {workload.request_id} to {best_device.device_id} (hybrid-optimized)")
                return [allocation]
            
        except Exception as e:
            self.logger.error(f"Failed to allocate resources: {str(e)}")
            return []
    
    def calculate_priority_score(
        self,
        workload: WorkloadRequest,
        device: EdgeDevice
    ) -> float:
        """Multi-objective priority scoring."""
        if workload.model_type not in device.supported_models:
            return 0.0
        
        # Individual component scores
        latency_score = min(workload.max_latency / (device.avg_inference_time + 1), 1.0)
        energy_score = 1.0 / (device.power_consumption + 1.0)
        accuracy_score = device.reliability_score
        cost_score = 1.0 / (device.power_consumption * 0.1 + 1.0)
        
        # Normalize scores
        energy_score = min(energy_score / 5.0, 1.0)
        cost_score = min(cost_score, 1.0)
        
        # Weighted combination
        total_score = (
            self.latency_weight * latency_score +
            self.energy_weight * energy_score +
            self.accuracy_weight * accuracy_score +
            self.cost_weight * cost_score
        )
        
        return total_score


class AdaptiveLoadBalancer:
    """Adaptive load balancer for edge computing requests."""
    
    def __init__(
        self,
        rebalancing_interval: float = 30.0,
        load_threshold: float = 80.0,
        migration_cost_threshold: float = 100.0
    ):
        self.rebalancing_interval = rebalancing_interval
        self.load_threshold = load_threshold
        self.migration_cost_threshold = migration_cost_threshold
        
        # Load tracking
        self.device_loads = defaultdict(float)
        self.workload_assignments = {}  # workload_id -> device_id
        self.device_queues = defaultdict(deque)
        
        # Performance history
        self.load_history = defaultdict(list)
        self.migration_history = []
        
        self.logger = get_logger(__name__)
        self.is_running = False
        self.rebalance_task = None
    
    async def start_rebalancing(self):
        """Start automatic load rebalancing."""
        self.is_running = True
        self.rebalance_task = asyncio.create_task(self._rebalancing_loop())
        self.logger.info("Started adaptive load balancer")
    
    async def stop_rebalancing(self):
        """Stop automatic load rebalancing."""
        self.is_running = False
        if self.rebalance_task:
            self.rebalance_task.cancel()
            try:
                await self.rebalance_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped adaptive load balancer")
    
    async def _rebalancing_loop(self):
        """Main rebalancing loop."""
        try:
            while self.is_running:
                await self._perform_load_balancing()
                await asyncio.sleep(self.rebalancing_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Rebalancing loop error: {str(e)}")
    
    async def _perform_load_balancing(self):
        """Perform load balancing across devices."""
        try:
            # Find overloaded and underloaded devices
            overloaded_devices = []
            underloaded_devices = []
            
            for device_id, load in self.device_loads.items():
                if load > self.load_threshold:
                    overloaded_devices.append((device_id, load))
                elif load < self.load_threshold * 0.3:  # Less than 30%
                    underloaded_devices.append((device_id, load))
            
            if not overloaded_devices or not underloaded_devices:
                return
            
            # Sort by load (most overloaded first, least loaded first)
            overloaded_devices.sort(key=lambda x: x[1], reverse=True)
            underloaded_devices.sort(key=lambda x: x[1])
            
            migrations_performed = 0
            
            for overloaded_id, overload in overloaded_devices:
                if migrations_performed >= 5:  # Limit migrations per cycle
                    break
                
                # Find workloads that can be migrated
                queue = self.device_queues[overloaded_id]
                if not queue:
                    continue
                
                # Try to migrate some workloads
                for underloaded_id, underload in underloaded_devices:
                    if overload - underload < 10:  # Not worth migrating
                        continue
                    
                    # Estimate migration cost
                    migration_cost = self._estimate_migration_cost(overloaded_id, underloaded_id)
                    
                    if migration_cost < self.migration_cost_threshold:
                        # Perform migration
                        workload = queue.pop()
                        self.device_queues[underloaded_id].append(workload)
                        
                        # Update tracking
                        self.workload_assignments[workload] = underloaded_id
                        
                        # Update load estimates
                        self.device_loads[overloaded_id] -= 10  # Simplified
                        self.device_loads[underloaded_id] += 10
                        
                        migrations_performed += 1
                        
                        self.migration_history.append({
                            'timestamp': time.time(),
                            'workload': workload,
                            'from_device': overloaded_id,
                            'to_device': underloaded_id,
                            'migration_cost': migration_cost
                        })
                        
                        self.logger.info(f"Migrated workload from {overloaded_id} to {underloaded_id}")
                        break
            
            if migrations_performed > 0:
                self.logger.info(f"Performed {migrations_performed} workload migrations")
            
        except Exception as e:
            self.logger.error(f"Load balancing failed: {str(e)}")
    
    def _estimate_migration_cost(self, from_device: str, to_device: str) -> float:
        """Estimate cost of migrating workload between devices."""
        # Simplified migration cost model
        base_cost = 50.0  # Base overhead
        
        # Network cost (if different locations)
        network_cost = 10.0 if from_device != to_device else 0.0
        
        # State transfer cost
        state_cost = 20.0
        
        total_cost = base_cost + network_cost + state_cost
        return total_cost
    
    def update_device_load(self, device_id: str, load: float):
        """Update device load information."""
        self.device_loads[device_id] = load
        
        # Keep load history
        self.load_history[device_id].append({
            'timestamp': time.time(),
            'load': load
        })
        
        # Keep only last 100 entries
        if len(self.load_history[device_id]) > 100:
            self.load_history[device_id].pop(0)
    
    def assign_workload(self, workload_id: str, device_id: str):
        """Assign workload to device."""
        self.workload_assignments[workload_id] = device_id
        self.device_queues[device_id].append(workload_id)
    
    def complete_workload(self, workload_id: str):
        """Mark workload as completed."""
        if workload_id in self.workload_assignments:
            device_id = self.workload_assignments.pop(workload_id)
            if workload_id in self.device_queues[device_id]:
                self.device_queues[device_id].remove(workload_id)


class EdgeComputingOrchestrator:
    """Main orchestrator for adaptive edge computing."""
    
    def __init__(
        self,
        strategy: Optional[EdgeComputingStrategy] = None,
        max_concurrent_workloads: int = 100,
        device_monitoring_interval: float = 10.0,
        auto_scaling_enabled: bool = True
    ):
        """Initialize edge computing orchestrator.
        
        Args:
            strategy: Resource allocation strategy
            max_concurrent_workloads: Maximum concurrent workloads
            device_monitoring_interval: Device monitoring frequency (seconds)
            auto_scaling_enabled: Enable automatic scaling
        """
        self.strategy = strategy or HybridOptimizedStrategy()
        self.max_concurrent_workloads = max_concurrent_workloads
        self.device_monitoring_interval = device_monitoring_interval
        self.auto_scaling_enabled = auto_scaling_enabled
        
        # System components
        self.devices: Dict[str, EdgeDevice] = {}
        self.active_workloads: Dict[str, WorkloadRequest] = {}
        self.completed_workloads: Dict[str, WorkloadRequest] = {}
        self.resource_allocations: Dict[str, List[ResourceAllocation]] = {}
        
        # Load balancer
        self.load_balancer = AdaptiveLoadBalancer()
        
        # Monitoring and optimization
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_latency': 0.0,
            'total_energy_consumed': 0.0,
            'device_utilization': defaultdict(list),
            'throughput_history': []
        }
        
        # Thread pools for async execution
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Control flags
        self.is_running = False
        self.monitoring_task = None
        
        self.logger = get_logger(__name__)
        self.logger.info("Initialized Edge Computing Orchestrator")
    
    async def start(self):
        """Start the orchestrator."""
        try:
            self.is_running = True
            
            # Start load balancer
            await self.load_balancer.start_rebalancing()
            
            # Start device monitoring
            self.monitoring_task = asyncio.create_task(self._device_monitoring_loop())
            
            self.logger.info("Edge computing orchestrator started")
            
        except Exception as e:
            self.logger.error(f"Failed to start orchestrator: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the orchestrator."""
        try:
            self.is_running = False
            
            # Stop load balancer
            await self.load_balancer.stop_rebalancing()
            
            # Stop monitoring
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Edge computing orchestrator stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping orchestrator: {str(e)}")
    
    def register_device(self, device: EdgeDevice) -> bool:
        """Register a new edge device."""
        try:
            # Validate device capabilities
            if device.cpu_cores <= 0 or device.memory_total <= 0:
                self.logger.warning(f"Invalid device configuration: {device.device_id}")
                return False
            
            # Initialize device-specific metrics
            device.last_heartbeat = time.time()
            device.inference_history = []
            
            # Add to device registry
            self.devices[device.device_id] = device
            
            # Initialize monitoring
            self.performance_metrics['device_utilization'][device.device_id] = []
            
            self.logger.info(f"Registered edge device: {device.device_id} ({device.device_type})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register device {device.device_id}: {str(e)}")
            return False
    
    def unregister_device(self, device_id: str) -> bool:
        """Unregister an edge device."""
        try:
            if device_id not in self.devices:
                self.logger.warning(f"Device {device_id} not registered")
                return False
            
            # Check for active workloads
            active_workloads_on_device = [
                workload_id for workload_id, allocations in self.resource_allocations.items()
                if any(alloc.device_id == device_id and alloc.status == "active" for alloc in allocations)
            ]
            
            if active_workloads_on_device:
                self.logger.warning(f"Device {device_id} has {len(active_workloads_on_device)} active workloads")
                # Could implement workload migration here
            
            # Remove device
            del self.devices[device_id]
            
            # Clean up monitoring data
            if device_id in self.performance_metrics['device_utilization']:
                del self.performance_metrics['device_utilization'][device_id]
            
            self.logger.info(f"Unregistered edge device: {device_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister device {device_id}: {str(e)}")
            return False
    
    async def submit_workload(self, workload: WorkloadRequest) -> str:
        """Submit a workload for processing."""
        try:
            # Validate workload
            if not workload.request_id:
                workload.request_id = self._generate_workload_id()
            
            if len(self.active_workloads) >= self.max_concurrent_workloads:
                raise RuntimeError("Maximum concurrent workloads exceeded")
            
            # Calculate data size and requirements
            workload.data_size = workload.data.nbytes if hasattr(workload.data, 'nbytes') else 0
            workload.estimated_compute_time = self._estimate_compute_time(workload)
            
            # Find available devices
            available_devices = [
                device for device in self.devices.values()
                if device.is_online and device.cpu_usage < 90.0
            ]
            
            if not available_devices:
                raise RuntimeError("No available devices for workload processing")
            
            # Allocate resources using strategy
            allocations = await self.strategy.allocate_resources(workload, available_devices)
            
            if not allocations:
                raise RuntimeError("Failed to allocate resources for workload")
            
            # Store workload and allocations
            self.active_workloads[workload.request_id] = workload
            self.resource_allocations[workload.request_id] = allocations
            
            # Update device assignments
            workload.assigned_devices = [alloc.device_id for alloc in allocations]
            
            # Update load balancer
            for allocation in allocations:
                self.load_balancer.assign_workload(workload.request_id, allocation.device_id)
            
            # Execute workload asynchronously
            asyncio.create_task(self._execute_workload(workload, allocations))
            
            # Update metrics
            self.performance_metrics['total_requests'] += 1
            
            self.logger.info(f"Submitted workload {workload.request_id} to {len(allocations)} device(s)")
            
            return workload.request_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit workload: {str(e)}")
            self.performance_metrics['failed_requests'] += 1
            raise
    
    async def _execute_workload(
        self,
        workload: WorkloadRequest,
        allocations: List[ResourceAllocation]
    ):
        """Execute workload on allocated devices."""
        try:
            workload.execution_start_time = time.time()
            
            # Update allocation status
            for allocation in allocations:
                allocation.status = "active"
                allocation.allocation_timestamp = time.time()
            
            # Execute on devices (parallel if multiple devices)
            if len(allocations) == 1:
                # Single device execution
                result = await self._execute_on_device(workload, allocations[0])
                workload.result = result['prediction']
                workload.confidence = result.get('confidence', 0.0)
            
            else:
                # Multi-device execution (ensemble)
                tasks = [
                    self._execute_on_device(workload, allocation)
                    for allocation in allocations
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Combine results (majority vote or averaging)
                predictions = []
                confidences = []
                
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.warning(f"Device execution failed: {str(result)}")
                        continue
                    
                    predictions.append(result['prediction'])
                    confidences.append(result.get('confidence', 0.0))
                
                if predictions:
                    # Ensemble prediction (simplified)
                    if isinstance(predictions[0], (int, np.integer)):
                        # Classification: majority vote
                        workload.result = max(set(predictions), key=predictions.count)
                    else:
                        # Regression: average
                        workload.result = np.mean(predictions, axis=0)
                    
                    workload.confidence = np.mean(confidences)
                else:
                    raise RuntimeError("All device executions failed")
            
            workload.execution_end_time = time.time()
            execution_time = (workload.execution_end_time - workload.execution_start_time) * 1000  # ms
            
            # Update allocation status
            for allocation in allocations:
                allocation.status = "completed"
            
            # Move to completed workloads
            self.completed_workloads[workload.request_id] = workload
            if workload.request_id in self.active_workloads:
                del self.active_workloads[workload.request_id]
            
            # Update load balancer
            self.load_balancer.complete_workload(workload.request_id)
            
            # Update metrics
            self.performance_metrics['successful_requests'] += 1
            self._update_performance_metrics(workload, execution_time)
            
            self.logger.info(f"Completed workload {workload.request_id} in {execution_time:.1f}ms")
            
        except Exception as e:
            self.logger.error(f"Workload execution failed {workload.request_id}: {str(e)}")
            
            # Update allocation status
            for allocation in allocations:
                allocation.status = "failed"
            
            # Clean up
            if workload.request_id in self.active_workloads:
                del self.active_workloads[workload.request_id]
            
            self.performance_metrics['failed_requests'] += 1
    
    async def _execute_on_device(
        self,
        workload: WorkloadRequest,
        allocation: ResourceAllocation
    ) -> Dict[str, Any]:
        """Execute workload on a specific device."""
        try:
            device = self.devices[allocation.device_id]
            
            # Simulate inference execution (in real implementation, this would
            # communicate with the actual edge device)
            await asyncio.sleep(allocation.estimated_duration / 1000.0)  # Convert ms to s
            
            # Update device metrics
            device.cpu_usage = min(100.0, device.cpu_usage + allocation.allocated_cpu / 2)
            device.memory_usage = min(100.0, device.memory_usage + allocation.allocated_memory / device.memory_total * 100)
            device.last_heartbeat = time.time()
            
            # Add to inference history
            inference_record = {
                'timestamp': time.time(),
                'workload_type': workload.model_type,
                'execution_time': allocation.estimated_duration,
                'cpu_usage': allocation.allocated_cpu,
                'memory_usage': allocation.allocated_memory,
                'energy_consumed': allocation.energy_cost
            }
            device.inference_history.append(inference_record)
            
            # Keep only last 100 records
            if len(device.inference_history) > 100:
                device.inference_history.pop(0)
            
            # Generate mock prediction result
            if workload.model_type == "anomaly_detection":
                prediction = np.random.choice([0, 1], p=[0.9, 0.1])  # 10% anomaly rate
                confidence = np.random.uniform(0.7, 0.95)
            else:
                prediction = np.random.random(workload.data.shape[0])
                confidence = np.random.uniform(0.6, 0.9)
            
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'device_id': device.device_id,
                'execution_time': allocation.estimated_duration,
                'energy_consumed': allocation.energy_cost
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Device execution failed on {allocation.device_id}: {str(e)}")
            raise
    
    async def _device_monitoring_loop(self):
        """Monitor device health and performance."""
        try:
            while self.is_running:
                await self._update_device_metrics()
                await asyncio.sleep(self.device_monitoring_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Device monitoring error: {str(e)}")
    
    async def _update_device_metrics(self):
        """Update metrics for all devices."""
        try:
            for device in self.devices.values():
                # Simulate device metric updates (in real implementation,
                # this would query actual devices)
                
                # Gradual CPU/memory usage decay when not processing
                device.cpu_usage = max(0.0, device.cpu_usage - 2.0)
                device.memory_usage = max(0.0, device.memory_usage - 1.0)
                
                # Random temperature fluctuation
                device.temperature += np.random.normal(0, 2.0)
                device.temperature = np.clip(device.temperature, 30.0, 90.0)
                
                # Battery drain for mobile devices
                if device.is_mobile and device.battery_level > 0:
                    drain_rate = 0.1 + device.cpu_usage * 0.01  # Higher usage = faster drain
                    device.battery_level = max(0.0, device.battery_level - drain_rate)
                
                # Update reliability score based on recent performance
                if device.inference_history:
                    recent_failures = sum(
                        1 for record in device.inference_history[-10:]
                        if record.get('failed', False)
                    )
                    device.reliability_score = max(0.1, 1.0 - recent_failures * 0.1)
                
                # Update average inference time
                if device.inference_history:
                    recent_times = [
                        record['execution_time'] 
                        for record in device.inference_history[-20:]
                    ]
                    device.avg_inference_time = np.mean(recent_times)
                
                # Update load balancer with current load
                current_load = (device.cpu_usage + device.memory_usage) / 2.0
                self.load_balancer.update_device_load(device.device_id, current_load)
                
                # Store utilization history
                self.performance_metrics['device_utilization'][device.device_id].append({
                    'timestamp': time.time(),
                    'cpu_usage': device.cpu_usage,
                    'memory_usage': device.memory_usage,
                    'temperature': device.temperature,
                    'battery_level': device.battery_level
                })
                
                # Keep only last 100 entries
                if len(self.performance_metrics['device_utilization'][device.device_id]) > 100:
                    self.performance_metrics['device_utilization'][device.device_id].pop(0)
                
                # Check for device health issues
                if device.temperature > 85.0:
                    self.logger.warning(f"Device {device.device_id} overheating: {device.temperature:.1f}°C")
                
                if device.is_mobile and device.battery_level < 10.0:
                    self.logger.warning(f"Device {device.device_id} low battery: {device.battery_level:.1f}%")
                
                if device.cpu_usage > 95.0:
                    self.logger.warning(f"Device {device.device_id} high CPU usage: {device.cpu_usage:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Device metrics update failed: {str(e)}")
    
    def _generate_workload_id(self) -> str:
        """Generate unique workload ID."""
        timestamp = int(time.time() * 1000000)
        random_suffix = np.random.randint(1000, 9999)
        return f"workload_{timestamp}_{random_suffix}"
    
    def _estimate_compute_time(self, workload: WorkloadRequest) -> float:
        """Estimate compute time for workload."""
        # Simplified estimation based on data size and model type
        base_time = {
            'anomaly_detection': 50.0,  # ms
            'classification': 30.0,
            'regression': 40.0,
            'clustering': 100.0
        }.get(workload.model_type, 50.0)
        
        # Scale by data size
        data_factor = max(1.0, workload.data_size / 1000.0)  # 1KB baseline
        
        # Adjust for accuracy requirements
        accuracy_factor = 1.0 + (workload.accuracy_requirement - 0.5) * 0.5
        
        estimated_time = base_time * data_factor * accuracy_factor
        return estimated_time
    
    def _update_performance_metrics(self, workload: WorkloadRequest, execution_time: float):
        """Update system performance metrics."""
        try:
            # Update average latency (exponential moving average)
            alpha = 0.1  # Smoothing factor
            if self.performance_metrics['average_latency'] == 0:
                self.performance_metrics['average_latency'] = execution_time
            else:
                self.performance_metrics['average_latency'] = (
                    alpha * execution_time + 
                    (1 - alpha) * self.performance_metrics['average_latency']
                )
            
            # Update energy consumption
            total_energy = sum(
                allocation.energy_cost 
                for allocation in self.resource_allocations.get(workload.request_id, [])
            )
            self.performance_metrics['total_energy_consumed'] += total_energy
            
            # Update throughput history
            self.performance_metrics['throughput_history'].append({
                'timestamp': time.time(),
                'execution_time': execution_time,
                'energy_consumed': total_energy,
                'devices_used': len(workload.assigned_devices)
            })
            
            # Keep only last 1000 entries
            if len(self.performance_metrics['throughput_history']) > 1000:
                self.performance_metrics['throughput_history'].pop(0)
            
        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {str(e)}")
    
    def get_workload_status(self, workload_id: str) -> Dict[str, Any]:
        """Get status of a specific workload."""
        try:
            # Check active workloads
            if workload_id in self.active_workloads:
                workload = self.active_workloads[workload_id]
                allocations = self.resource_allocations.get(workload_id, [])
                
                return {
                    'workload_id': workload_id,
                    'status': 'active',
                    'assigned_devices': workload.assigned_devices,
                    'start_time': workload.execution_start_time,
                    'estimated_completion': workload.execution_start_time + max(
                        alloc.estimated_duration for alloc in allocations
                    ) / 1000.0 if allocations else None,
                    'progress': self._estimate_progress(workload)
                }
            
            # Check completed workloads
            elif workload_id in self.completed_workloads:
                workload = self.completed_workloads[workload_id]
                execution_time = (workload.execution_end_time - workload.execution_start_time) * 1000
                
                return {
                    'workload_id': workload_id,
                    'status': 'completed',
                    'result': workload.result,
                    'confidence': workload.confidence,
                    'execution_time': execution_time,
                    'assigned_devices': workload.assigned_devices
                }
            
            else:
                return {'workload_id': workload_id, 'status': 'not_found'}
            
        except Exception as e:
            self.logger.error(f"Failed to get workload status: {str(e)}")
            return {'workload_id': workload_id, 'status': 'error', 'error': str(e)}
    
    def _estimate_progress(self, workload: WorkloadRequest) -> float:
        """Estimate workload progress."""
        if not workload.execution_start_time:
            return 0.0
        
        elapsed = time.time() - workload.execution_start_time
        allocations = self.resource_allocations.get(workload.request_id, [])
        
        if not allocations:
            return 0.0
        
        max_estimated_duration = max(alloc.estimated_duration for alloc in allocations) / 1000.0
        progress = min(1.0, elapsed / max_estimated_duration)
        
        return progress
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Device summary
            total_devices = len(self.devices)
            online_devices = sum(1 for d in self.devices.values() if d.is_online)
            avg_cpu_usage = np.mean([d.cpu_usage for d in self.devices.values()]) if self.devices else 0.0
            avg_memory_usage = np.mean([d.memory_usage for d in self.devices.values()]) if self.devices else 0.0
            
            # Workload summary
            active_workloads = len(self.active_workloads)
            completed_workloads = len(self.completed_workloads)
            
            # Performance summary
            success_rate = (
                self.performance_metrics['successful_requests'] / 
                max(1, self.performance_metrics['total_requests'])
            ) * 100
            
            # Recent throughput (last hour)
            one_hour_ago = time.time() - 3600
            recent_throughput = [
                entry for entry in self.performance_metrics['throughput_history']
                if entry['timestamp'] > one_hour_ago
            ]
            
            status = {
                'system_info': {
                    'is_running': self.is_running,
                    'strategy': type(self.strategy).__name__,
                    'auto_scaling_enabled': self.auto_scaling_enabled
                },
                'device_summary': {
                    'total_devices': total_devices,
                    'online_devices': online_devices,
                    'avg_cpu_usage': avg_cpu_usage,
                    'avg_memory_usage': avg_memory_usage
                },
                'workload_summary': {
                    'active_workloads': active_workloads,
                    'completed_workloads': completed_workloads,
                    'max_concurrent': self.max_concurrent_workloads
                },
                'performance_summary': {
                    'total_requests': self.performance_metrics['total_requests'],
                    'success_rate': success_rate,
                    'avg_latency': self.performance_metrics['average_latency'],
                    'total_energy_consumed': self.performance_metrics['total_energy_consumed'],
                    'recent_throughput': len(recent_throughput)
                },
                'load_balancer': {
                    'migrations_performed': len(self.load_balancer.migration_history),
                    'device_queues': {
                        device_id: len(queue) 
                        for device_id, queue in self.load_balancer.device_queues.items()
                    }
                }
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {str(e)}")
            return {'error': str(e)}
    
    def get_device_details(self, device_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific device."""
        try:
            if device_id not in self.devices:
                return {'device_id': device_id, 'status': 'not_found'}
            
            device = self.devices[device_id]
            
            # Recent performance metrics
            recent_utilization = []
            if device_id in self.performance_metrics['device_utilization']:
                recent_utilization = self.performance_metrics['device_utilization'][device_id][-10:]
            
            # Active workloads on this device
            active_workloads_on_device = []
            for workload_id, allocations in self.resource_allocations.items():
                for allocation in allocations:
                    if allocation.device_id == device_id and allocation.status == "active":
                        active_workloads_on_device.append(workload_id)
            
            details = {
                'device_info': asdict(device),
                'recent_utilization': recent_utilization,
                'active_workloads': active_workloads_on_device,
                'inference_history_count': len(device.inference_history),
                'last_seen': device.last_heartbeat,
                'queue_length': len(self.load_balancer.device_queues.get(device_id, []))
            }
            
            return details
            
        except Exception as e:
            self.logger.error(f"Failed to get device details: {str(e)}")
            return {'device_id': device_id, 'error': str(e)}


# Factory functions for different deployment scenarios

def create_iot_edge_orchestrator(
    optimization_target: str = "hybrid"
) -> EdgeComputingOrchestrator:
    """Create orchestrator optimized for IoT edge deployments."""
    
    strategy_map = {
        "latency": LatencyOptimizedStrategy(locality_weight=0.8),
        "energy": EnergyOptimizedStrategy(energy_weight=0.9),
        "hybrid": HybridOptimizedStrategy(
            latency_weight=0.3,
            energy_weight=0.4,
            accuracy_weight=0.2,
            cost_weight=0.1
        )
    }
    
    strategy = strategy_map.get(optimization_target, HybridOptimizedStrategy())
    
    orchestrator = EdgeComputingOrchestrator(
        strategy=strategy,
        max_concurrent_workloads=50,  # Conservative for IoT
        device_monitoring_interval=15.0,  # More frequent monitoring
        auto_scaling_enabled=True
    )
    
    return orchestrator


def create_industrial_orchestrator() -> EdgeComputingOrchestrator:
    """Create orchestrator for industrial IoT deployments."""
    
    # Industrial focus: reliability and accuracy over energy
    strategy = HybridOptimizedStrategy(
        latency_weight=0.4,
        energy_weight=0.1,  # Less important in industrial settings
        accuracy_weight=0.4,  # High accuracy critical
        cost_weight=0.1
    )
    
    orchestrator = EdgeComputingOrchestrator(
        strategy=strategy,
        max_concurrent_workloads=200,  # Higher capacity
        device_monitoring_interval=5.0,  # Very frequent monitoring
        auto_scaling_enabled=True
    )
    
    return orchestrator


def create_mobile_orchestrator() -> EdgeComputingOrchestrator:
    """Create orchestrator for mobile/battery-powered deployments."""
    
    # Mobile focus: energy efficiency is paramount
    strategy = EnergyOptimizedStrategy(energy_weight=0.9)
    
    orchestrator = EdgeComputingOrchestrator(
        strategy=strategy,
        max_concurrent_workloads=20,  # Limited for mobile
        device_monitoring_interval=30.0,  # Less frequent to save energy
        auto_scaling_enabled=True
    )
    
    return orchestrator


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Edge Computing Orchestrator")
    parser.add_argument("--deployment", choices=["iot", "industrial", "mobile"], default="iot")
    parser.add_argument("--optimization", choices=["latency", "energy", "hybrid"], default="hybrid")
    parser.add_argument("--devices", type=int, default=10, help="Number of simulated devices")
    parser.add_argument("--workloads", type=int, default=50, help="Number of test workloads")
    
    args = parser.parse_args()
    
    async def run_example():
        # Create orchestrator based on deployment type
        if args.deployment == "iot":
            orchestrator = create_iot_edge_orchestrator(args.optimization)
        elif args.deployment == "industrial":
            orchestrator = create_industrial_orchestrator()
        else:  # mobile
            orchestrator = create_mobile_orchestrator()
        
        # Start orchestrator
        await orchestrator.start()
        
        # Register simulated devices
        device_types = ["raspberry_pi", "nvidia_jetson", "intel_nuc", "smartphone", "gateway"]
        
        for i in range(args.devices):
            device_type = np.random.choice(device_types)
            
            # Device specs based on type
            if device_type == "raspberry_pi":
                cpu_cores, cpu_freq, memory = 4, 1.5, 4096
                gpu_available, power = False, 5.0
            elif device_type == "nvidia_jetson":
                cpu_cores, cpu_freq, memory = 6, 2.0, 8192
                gpu_available, power = True, 15.0
            elif device_type == "intel_nuc":
                cpu_cores, cpu_freq, memory = 8, 2.5, 16384
                gpu_available, power = True, 25.0
            elif device_type == "smartphone":
                cpu_cores, cpu_freq, memory = 8, 2.8, 6144
                gpu_available, power = 3.0, True  # is_mobile
            else:  # gateway
                cpu_cores, cpu_freq, memory = 4, 2.0, 8192
                gpu_available, power = False, 20.0
            
            device = EdgeDevice(
                device_id=f"{device_type}_{i:03d}",
                device_type=device_type,
                location=(np.random.uniform(40.0, 41.0), np.random.uniform(-74.0, -73.0)),
                cpu_cores=cpu_cores,
                cpu_frequency=cpu_freq,
                memory_total=memory,
                storage_total=np.random.randint(32, 128),
                gpu_available=gpu_available,
                bandwidth_up=np.random.uniform(10, 100),
                bandwidth_down=np.random.uniform(50, 500),
                latency_to_cloud=np.random.uniform(10, 100),
                is_mobile=(device_type == "smartphone"),
                supported_models=["anomaly_detection", "classification"],
                max_inference_rate=np.random.uniform(5, 20),
                power_consumption=power
            )
            
            orchestrator.register_device(device)
        
        print(f"Registered {args.devices} edge devices")
        
        # Submit test workloads
        workload_results = []
        
        for i in range(args.workloads):
            # Generate synthetic workload data
            data = np.random.normal(0, 1, (100, 5))  # 100 samples, 5 features
            
            workload = WorkloadRequest(
                request_id=f"test_workload_{i:03d}",
                data=data,
                model_type="anomaly_detection",
                priority=np.random.randint(1, 11),
                max_latency=np.random.uniform(50, 500),
                accuracy_requirement=np.random.uniform(0.7, 0.95),
                privacy_level=np.random.randint(1, 6)
            )
            
            try:
                workload_id = await orchestrator.submit_workload(workload)
                workload_results.append(workload_id)
                
                if (i + 1) % 10 == 0:
                    print(f"Submitted {i + 1}/{args.workloads} workloads")
                
                # Small delay to avoid overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Failed to submit workload {i}: {str(e)}")
        
        # Wait for workloads to complete
        print("Waiting for workloads to complete...")
        await asyncio.sleep(30)
        
        # Get system status
        status = orchestrator.get_system_status()
        print(f"\nSystem Status:")
        print(f"  Devices: {status['device_summary']['online_devices']}/{status['device_summary']['total_devices']} online")
        print(f"  Workloads: {status['workload_summary']['completed_workloads']} completed, {status['workload_summary']['active_workloads']} active")
        print(f"  Success rate: {status['performance_summary']['success_rate']:.1f}%")
        print(f"  Average latency: {status['performance_summary']['avg_latency']:.1f}ms")
        print(f"  Energy consumed: {status['performance_summary']['total_energy_consumed']:.3f}Wh")
        
        # Check a few workload statuses
        print(f"\nSample Workload Results:")
        for i in range(min(5, len(workload_results))):
            workload_status = orchestrator.get_workload_status(workload_results[i])
            print(f"  {workload_results[i]}: {workload_status['status']}")
            if workload_status['status'] == 'completed':
                print(f"    Execution time: {workload_status['execution_time']:.1f}ms")
                print(f"    Confidence: {workload_status['confidence']:.3f}")
        
        # Stop orchestrator
        await orchestrator.stop()
    
    # Run the example
    asyncio.run(run_example())