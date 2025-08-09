"""Distributed anomaly detection system for large-scale IoT deployments."""

import asyncio
import json
import logging
import time
import threading
import queue
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Union, Set
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import concurrent.futures

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import asyncio
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


class NodeRole(Enum):
    """Roles in distributed system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    EDGE = "edge"
    AGGREGATOR = "aggregator"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class WorkloadType(Enum):
    """Types of workloads."""
    INFERENCE = "inference"
    TRAINING = "training"
    DATA_PROCESSING = "data_processing"
    MODEL_UPDATE = "model_update"
    HEALTH_CHECK = "health_check"


@dataclass
class NodeInfo:
    """Information about a distributed node."""
    node_id: str
    role: NodeRole
    address: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    load_factor: float = 0.0
    last_heartbeat: float = 0.0
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedTask:
    """Task for distributed execution."""
    task_id: str
    task_type: WorkloadType
    priority: int
    data: Dict[str, Any]
    assigned_node: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    assigned_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300


@dataclass
class DistributedResult:
    """Result from distributed processing."""
    task_id: str
    node_id: str
    result_data: Dict[str, Any]
    processing_time_ms: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoadBalancer:
    """Load balancing for distributed tasks."""
    
    def __init__(self):
        self.nodes: Dict[str, NodeInfo] = {}
        self.task_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._lock = threading.RLock()
    
    def register_node(self, node: NodeInfo) -> None:
        """Register a node for load balancing."""
        with self._lock:
            self.nodes[node.node_id] = node
            if node.node_id not in self.task_counts:
                self.task_counts[node.node_id] = 0
    
    def unregister_node(self, node_id: str) -> None:
        """Unregister a node."""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
            if node_id in self.task_counts:
                del self.task_counts[node_id]
            if node_id in self.response_times:
                del self.response_times[node_id]
    
    def select_node(
        self, 
        task_type: WorkloadType, 
        required_capabilities: Optional[List[str]] = None
    ) -> Optional[str]:
        """Select best node for task using weighted round-robin."""
        with self._lock:
            eligible_nodes = []
            
            for node_id, node in self.nodes.items():
                # Check if node is active
                if node.status != "active":
                    continue
                
                # Check heartbeat (node is alive)
                if time.time() - node.last_heartbeat > 30:  # 30 second timeout
                    continue
                
                # Check required capabilities
                if required_capabilities:
                    if not all(cap in node.capabilities for cap in required_capabilities):
                        continue
                
                # Check role compatibility
                if task_type == WorkloadType.INFERENCE and node.role in [NodeRole.WORKER, NodeRole.EDGE]:
                    eligible_nodes.append(node_id)
                elif task_type == WorkloadType.TRAINING and node.role == NodeRole.WORKER:
                    eligible_nodes.append(node_id)
                elif task_type == WorkloadType.DATA_PROCESSING and node.role in [NodeRole.WORKER, NodeRole.AGGREGATOR]:
                    eligible_nodes.append(node_id)
            
            if not eligible_nodes:
                return None
            
            # Calculate selection weights
            node_weights = {}
            for node_id in eligible_nodes:
                node = self.nodes[node_id]
                task_count = self.task_counts[node_id]
                
                # Calculate average response time
                avg_response_time = 1.0  # Default 1 second
                if node_id in self.response_times and self.response_times[node_id]:
                    avg_response_time = sum(self.response_times[node_id]) / len(self.response_times[node_id])
                
                # Weight calculation (lower is better)
                # Factors: current load, response time, load factor
                load_weight = 1.0 / (1.0 + task_count)
                response_weight = 1.0 / (1.0 + avg_response_time)
                capacity_weight = 1.0 - node.load_factor
                
                total_weight = load_weight * response_weight * capacity_weight
                node_weights[node_id] = total_weight
            
            # Select node with highest weight
            selected_node = max(node_weights.keys(), key=lambda n: node_weights[n])
            self.task_counts[selected_node] += 1
            
            return selected_node
    
    def report_task_completion(self, node_id: str, response_time_ms: float) -> None:
        """Report task completion for load balancing."""
        with self._lock:
            if node_id in self.task_counts:
                self.task_counts[node_id] = max(0, self.task_counts[node_id] - 1)
            
            if node_id in self.response_times:
                self.response_times[node_id].append(response_time_ms / 1000.0)  # Convert to seconds
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self._lock:
            return {
                "total_nodes": len(self.nodes),
                "active_nodes": sum(1 for n in self.nodes.values() if n.status == "active"),
                "task_distribution": dict(self.task_counts),
                "average_response_times": {
                    node_id: sum(times) / len(times) if times else 0.0
                    for node_id, times in self.response_times.items()
                }
            }


class DistributedTaskQueue:
    """Distributed task queue with priority and failover."""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.local_queue = queue.PriorityQueue()
        self.pending_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedResult] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit task for distributed execution."""
        with self._lock:
            self.pending_tasks[task.task_id] = task
            
            # Add to queue with priority (lower number = higher priority)
            priority_score = task.priority + task.retry_count
            self.local_queue.put((priority_score, task.created_at, task))
            
            # Store in Redis if available for distributed queuing
            if self.redis_client and REDIS_AVAILABLE:
                try:
                    task_data = asdict(task)
                    task_data['status'] = task.status.value
                    task_data['task_type'] = task.task_type.value
                    
                    self.redis_client.hset(
                        "distributed_tasks",
                        task.task_id,
                        json.dumps(task_data)
                    )
                    
                    self.redis_client.zadd(
                        "task_queue",
                        {task.task_id: priority_score}
                    )
                except Exception as e:
                    self.logger.error(f"Failed to store task in Redis: {e}")
        
        return task.task_id
    
    def get_next_task(self, node_id: str, capabilities: List[str]) -> Optional[DistributedTask]:
        """Get next task for worker node."""
        try:
            # Try local queue first
            if not self.local_queue.empty():
                try:
                    _, _, task = self.local_queue.get_nowait()
                    
                    if self._can_handle_task(task, capabilities):
                        task.assigned_node = node_id
                        task.status = TaskStatus.ASSIGNED
                        task.assigned_at = time.time()
                        return task
                    else:
                        # Put back if can't handle
                        priority_score = task.priority + task.retry_count
                        self.local_queue.put((priority_score, task.created_at, task))
                except queue.Empty:
                    pass
            
            # Try Redis queue if available
            if self.redis_client and REDIS_AVAILABLE:
                return self._get_task_from_redis(node_id, capabilities)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting next task: {e}")
            return None
    
    def _can_handle_task(self, task: DistributedTask, capabilities: List[str]) -> bool:
        """Check if node can handle task based on capabilities."""
        required_caps = task.data.get('required_capabilities', [])
        return all(cap in capabilities for cap in required_caps)
    
    def _get_task_from_redis(self, node_id: str, capabilities: List[str]) -> Optional[DistributedTask]:
        """Get task from Redis distributed queue."""
        try:
            # Get highest priority task
            task_items = self.redis_client.zrange("task_queue", 0, 0, withscores=True)
            
            if not task_items:
                return None
            
            task_id = task_items[0][0].decode('utf-8')
            
            # Get task data
            task_data = self.redis_client.hget("distributed_tasks", task_id)
            if not task_data:
                # Clean up orphaned task
                self.redis_client.zrem("task_queue", task_id)
                return None
            
            task_dict = json.loads(task_data.decode('utf-8'))
            
            # Convert back to DistributedTask
            task = DistributedTask(
                task_id=task_dict['task_id'],
                task_type=WorkloadType(task_dict['task_type']),
                priority=task_dict['priority'],
                data=task_dict['data'],
                status=TaskStatus(task_dict['status']),
                created_at=task_dict['created_at'],
                retry_count=task_dict.get('retry_count', 0),
                max_retries=task_dict.get('max_retries', 3),
                timeout_seconds=task_dict.get('timeout_seconds', 300)
            )
            
            if self._can_handle_task(task, capabilities):
                # Assign task
                task.assigned_node = node_id
                task.status = TaskStatus.ASSIGNED
                task.assigned_at = time.time()
                
                # Remove from queue
                self.redis_client.zrem("task_queue", task_id)
                
                # Update in Redis
                task_dict.update(asdict(task))
                task_dict['status'] = task.status.value
                self.redis_client.hset("distributed_tasks", task_id, json.dumps(task_dict))
                
                return task
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting task from Redis: {e}")
            return None
    
    def complete_task(self, task_id: str, result: DistributedResult) -> None:
        """Mark task as completed."""
        with self._lock:
            if task_id in self.pending_tasks:
                task = self.pending_tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                task.result = result.result_data
                
                self.completed_tasks[task_id] = result
                del self.pending_tasks[task_id]
                
                # Update in Redis if available
                if self.redis_client and REDIS_AVAILABLE:
                    try:
                        self.redis_client.hdel("distributed_tasks", task_id)
                        self.redis_client.hset(
                            "completed_tasks",
                            task_id,
                            json.dumps(asdict(result))
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to update task in Redis: {e}")
    
    def fail_task(self, task_id: str, error_message: str, retry: bool = True) -> None:
        """Mark task as failed and optionally retry."""
        with self._lock:
            if task_id in self.pending_tasks:
                task = self.pending_tasks[task_id]
                task.retry_count += 1
                task.error = error_message
                
                if retry and task.retry_count < task.max_retries:
                    # Retry task
                    task.status = TaskStatus.PENDING
                    task.assigned_node = None
                    task.assigned_at = None
                    
                    # Re-add to queue with higher priority due to retry
                    priority_score = task.priority + task.retry_count
                    self.local_queue.put((priority_score, task.created_at, task))
                else:
                    # Permanently failed
                    task.status = TaskStatus.FAILED
                    del self.pending_tasks[task_id]
                    
                    # Update in Redis
                    if self.redis_client and REDIS_AVAILABLE:
                        try:
                            self.redis_client.hdel("distributed_tasks", task_id)
                            self.redis_client.hset(
                                "failed_tasks",
                                task_id,
                                json.dumps({
                                    "task_id": task_id,
                                    "error": error_message,
                                    "retry_count": task.retry_count,
                                    "failed_at": time.time()
                                })
                            )
                        except Exception as e:
                            self.logger.error(f"Failed to update failed task in Redis: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task."""
        with self._lock:
            if task_id in self.pending_tasks:
                return self.pending_tasks[task_id].status
            elif task_id in self.completed_tasks:
                return TaskStatus.COMPLETED
            
            # Check Redis if available
            if self.redis_client and REDIS_AVAILABLE:
                try:
                    task_data = self.redis_client.hget("distributed_tasks", task_id)
                    if task_data:
                        task_dict = json.loads(task_data.decode('utf-8'))
                        return TaskStatus(task_dict['status'])
                except Exception:
                    pass
            
            return None
    
    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            pending_count = len(self.pending_tasks)
            completed_count = len(self.completed_tasks)
            
            # Count by status
            status_counts = defaultdict(int)
            for task in self.pending_tasks.values():
                status_counts[task.status.value] += 1
            
            return {
                "pending_tasks": pending_count,
                "completed_tasks": completed_count,
                "queue_size": self.local_queue.qsize(),
                "status_distribution": dict(status_counts)
            }


class DistributedAnomalyDetector:
    """Main distributed anomaly detection coordinator."""
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        role: NodeRole = NodeRole.COORDINATOR,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        websocket_port: int = 8766
    ):
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.role = role
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.websocket_port = websocket_port
        
        # Initialize components
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                import redis
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
                self.redis_client.ping()  # Test connection
            except Exception as e:
                logging.warning(f"Redis connection failed: {e}")
        
        self.load_balancer = LoadBalancer()
        self.task_queue = DistributedTaskQueue(self.redis_client)
        
        # Node management
        self.worker_nodes: Dict[str, NodeInfo] = {}
        self.heartbeat_interval = 10  # seconds
        self.node_timeout = 30  # seconds
        
        # Communication
        self.websocket_server = None
        self.connected_clients: Set = set()
        
        # Processing
        self.is_running = False
        self.background_tasks: List[threading.Thread] = []
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Register self
        self.node_info = NodeInfo(
            node_id=self.node_id,
            role=self.role,
            address="localhost",
            port=self.websocket_port,
            capabilities=self._get_node_capabilities(),
            last_heartbeat=time.time()
        )
        
        if self.role == NodeRole.COORDINATOR:
            self.load_balancer.register_node(self.node_info)
    
    def _get_node_capabilities(self) -> List[str]:
        """Get node capabilities based on role and available libraries."""
        capabilities = []
        
        if self.role == NodeRole.COORDINATOR:
            capabilities.extend(["task_coordination", "load_balancing", "monitoring"])
        elif self.role == NodeRole.WORKER:
            capabilities.extend(["inference", "training", "data_processing"])
        elif self.role == NodeRole.EDGE:
            capabilities.extend(["inference", "lightweight_processing"])
        elif self.role == NodeRole.AGGREGATOR:
            capabilities.extend(["data_processing", "aggregation", "storage"])
        
        # Add library-specific capabilities
        if NUMPY_AVAILABLE:
            capabilities.append("numpy_processing")
        
        return capabilities
    
    async def start_coordinator(self) -> None:
        """Start coordinator node."""
        if self.role != NodeRole.COORDINATOR:
            raise ValueError("Only coordinator nodes can start coordination")
        
        self.is_running = True
        
        # Start WebSocket server for node communication
        if WEBSOCKETS_AVAILABLE:
            await self._start_websocket_server()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info(f"Distributed coordinator started on {self.node_id}")
    
    async def _start_websocket_server(self) -> None:
        """Start WebSocket server for node communication."""
        try:
            self.websocket_server = await websockets.serve(
                self._websocket_handler,
                "localhost",
                self.websocket_port
            )
            self.logger.info(f"WebSocket server started on port {self.websocket_port}")
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
    
    async def _websocket_handler(self, websocket, path) -> None:
        """Handle WebSocket connections from worker nodes."""
        self.connected_clients.add(websocket)
        client_id = f"client_{id(websocket)}"
        
        try:
            self.logger.info(f"New worker node connected: {client_id}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_worker_message(websocket, data)
                except json.JSONDecodeError as e:
                    await websocket.send(json.dumps({
                        "error": f"Invalid JSON: {e}"
                    }))
                except Exception as e:
                    self.logger.error(f"Error handling worker message: {e}")
                    await websocket.send(json.dumps({
                        "error": f"Message handling error: {e}"
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Worker node disconnected: {client_id}")
        except Exception as e:
            self.logger.error(f"WebSocket handler error: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def _handle_worker_message(self, websocket, data: Dict[str, Any]) -> None:
        """Handle message from worker node."""
        message_type = data.get("type")
        
        if message_type == "register":
            await self._handle_node_registration(websocket, data)
        elif message_type == "heartbeat":
            await self._handle_heartbeat(websocket, data)
        elif message_type == "task_request":
            await self._handle_task_request(websocket, data)
        elif message_type == "task_result":
            await self._handle_task_result(websocket, data)
        elif message_type == "task_error":
            await self._handle_task_error(websocket, data)
        else:
            await websocket.send(json.dumps({
                "error": f"Unknown message type: {message_type}"
            }))
    
    async def _handle_node_registration(self, websocket, data: Dict[str, Any]) -> None:
        """Handle worker node registration."""
        node_info = NodeInfo(
            node_id=data["node_id"],
            role=NodeRole(data["role"]),
            address=data.get("address", "unknown"),
            port=data.get("port", 0),
            capabilities=data.get("capabilities", []),
            last_heartbeat=time.time()
        )
        
        self.worker_nodes[node_info.node_id] = node_info
        self.load_balancer.register_node(node_info)
        
        await websocket.send(json.dumps({
            "type": "registration_success",
            "coordinator_id": self.node_id
        }))
        
        self.logger.info(f"Registered worker node: {node_info.node_id}")
    
    async def _handle_heartbeat(self, websocket, data: Dict[str, Any]) -> None:
        """Handle heartbeat from worker node."""
        node_id = data["node_id"]
        
        if node_id in self.worker_nodes:
            self.worker_nodes[node_id].last_heartbeat = time.time()
            self.worker_nodes[node_id].load_factor = data.get("load_factor", 0.0)
            
            await websocket.send(json.dumps({
                "type": "heartbeat_ack",
                "coordinator_time": time.time()
            }))
    
    async def _handle_task_request(self, websocket, data: Dict[str, Any]) -> None:
        """Handle task request from worker node."""
        node_id = data["node_id"]
        capabilities = data.get("capabilities", [])
        
        # Get next task for worker
        task = self.task_queue.get_next_task(node_id, capabilities)
        
        if task:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            
            await websocket.send(json.dumps({
                "type": "task_assignment",
                "task": {
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "data": task.data,
                    "timeout_seconds": task.timeout_seconds
                }
            }))
        else:
            await websocket.send(json.dumps({
                "type": "no_tasks",
                "message": "No tasks available"
            }))
    
    async def _handle_task_result(self, websocket, data: Dict[str, Any]) -> None:
        """Handle task completion from worker node."""
        task_id = data["task_id"]
        node_id = data["node_id"]
        result_data = data["result"]
        processing_time = data.get("processing_time_ms", 0.0)
        
        result = DistributedResult(
            task_id=task_id,
            node_id=node_id,
            result_data=result_data,
            processing_time_ms=processing_time,
            timestamp=time.time()
        )
        
        self.task_queue.complete_task(task_id, result)
        self.load_balancer.report_task_completion(node_id, processing_time)
        
        # Update statistics
        self.stats["tasks_completed"] += 1
        self.stats["total_processing_time"] += processing_time
        self.stats["average_processing_time"] = (
            self.stats["total_processing_time"] / self.stats["tasks_completed"]
        )
        
        await websocket.send(json.dumps({
            "type": "task_completion_ack",
            "task_id": task_id
        }))
        
        self.logger.info(f"Task {task_id} completed by {node_id}")
    
    async def _handle_task_error(self, websocket, data: Dict[str, Any]) -> None:
        """Handle task error from worker node."""
        task_id = data["task_id"]
        node_id = data["node_id"]
        error_message = data.get("error", "Unknown error")
        
        self.task_queue.fail_task(task_id, error_message)
        self.stats["tasks_failed"] += 1
        
        await websocket.send(json.dumps({
            "type": "task_error_ack",
            "task_id": task_id
        }))
        
        self.logger.error(f"Task {task_id} failed on {node_id}: {error_message}")
    
    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Node health monitoring
        health_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
        health_thread.start()
        self.background_tasks.append(health_thread)
        
        # Statistics reporting
        stats_thread = threading.Thread(target=self._statistics_reporting_loop, daemon=True)
        stats_thread.start()
        self.background_tasks.append(stats_thread)
    
    def _health_monitoring_loop(self) -> None:
        """Monitor health of worker nodes."""
        while self.is_running:
            try:
                current_time = time.time()
                inactive_nodes = []
                
                for node_id, node in self.worker_nodes.items():
                    if current_time - node.last_heartbeat > self.node_timeout:
                        inactive_nodes.append(node_id)
                        node.status = "inactive"
                
                # Remove inactive nodes
                for node_id in inactive_nodes:
                    self.logger.warning(f"Removing inactive node: {node_id}")
                    self.load_balancer.unregister_node(node_id)
                    del self.worker_nodes[node_id]
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(5)
    
    def _statistics_reporting_loop(self) -> None:
        """Periodic statistics reporting."""
        while self.is_running:
            try:
                # Log statistics
                stats = self.get_system_statistics()
                self.logger.info(f"System stats: {json.dumps(stats, indent=2)}")
                
                # Store in Redis if available
                if self.redis_client:
                    try:
                        self.redis_client.hset(
                            "system_statistics",
                            self.node_id,
                            json.dumps(stats)
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to store statistics in Redis: {e}")
                
                time.sleep(60)  # Report every minute
                
            except Exception as e:
                self.logger.error(f"Statistics reporting error: {e}")
                time.sleep(30)
    
    def submit_anomaly_detection_task(
        self,
        sensor_data: Dict[str, Any],
        priority: int = 1,
        required_capabilities: Optional[List[str]] = None
    ) -> str:
        """Submit anomaly detection task for distributed processing."""
        task = DistributedTask(
            task_id=f"anomaly_{uuid.uuid4().hex[:8]}",
            task_type=WorkloadType.INFERENCE,
            priority=priority,
            data={
                "sensor_data": sensor_data,
                "required_capabilities": required_capabilities or ["numpy_processing", "inference"],
                "model_type": "autoencoder"
            },
            timeout_seconds=60
        )
        
        task_id = self.task_queue.submit_task(task)
        self.stats["tasks_submitted"] += 1
        
        return task_id
    
    def submit_model_training_task(
        self,
        training_data: Dict[str, Any],
        priority: int = 2,
        model_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit model training task for distributed processing."""
        task = DistributedTask(
            task_id=f"training_{uuid.uuid4().hex[:8]}",
            task_type=WorkloadType.TRAINING,
            priority=priority,
            data={
                "training_data": training_data,
                "model_config": model_config or {},
                "required_capabilities": ["numpy_processing", "training"]
            },
            timeout_seconds=3600  # 1 hour for training
        )
        
        task_id = self.task_queue.submit_task(task)
        self.stats["tasks_submitted"] += 1
        
        return task_id
    
    def get_task_result(self, task_id: str) -> Optional[DistributedResult]:
        """Get result of completed task."""
        if task_id in self.task_queue.completed_tasks:
            return self.task_queue.completed_tasks[task_id]
        return None
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "node_info": asdict(self.node_info),
            "worker_nodes": len(self.worker_nodes),
            "connected_clients": len(self.connected_clients),
            "task_statistics": self.stats,
            "queue_statistics": self.task_queue.get_queue_statistics(),
            "load_balancer_statistics": self.load_balancer.get_load_statistics(),
            "timestamp": time.time()
        }
    
    def shutdown(self) -> None:
        """Shutdown distributed system."""
        self.is_running = False
        
        if self.websocket_server:
            self.websocket_server.close()
        
        for thread in self.background_tasks:
            thread.join(timeout=2.0)
        
        self.logger.info(f"Distributed system shutdown: {self.node_id}")


# Factory functions
def create_coordinator(**kwargs) -> DistributedAnomalyDetector:
    """Create coordinator node."""
    return DistributedAnomalyDetector(role=NodeRole.COORDINATOR, **kwargs)


def create_worker(**kwargs) -> DistributedAnomalyDetector:
    """Create worker node."""
    return DistributedAnomalyDetector(role=NodeRole.WORKER, **kwargs)


def create_edge_node(**kwargs) -> DistributedAnomalyDetector:
    """Create edge node."""
    return DistributedAnomalyDetector(role=NodeRole.EDGE, **kwargs)


# Example usage and CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Anomaly Detection")
    parser.add_argument("--role", choices=["coordinator", "worker", "edge"], 
                       default="coordinator", help="Node role")
    parser.add_argument("--node-id", help="Node ID")
    parser.add_argument("--coordinator-host", default="localhost", help="Coordinator host")
    parser.add_argument("--coordinator-port", type=int, default=8766, help="Coordinator port")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--test-tasks", type=int, default=10, help="Number of test tasks")
    
    args = parser.parse_args()
    
    async def main():
        # Create distributed system
        if args.role == "coordinator":
            system = create_coordinator(
                node_id=args.node_id,
                redis_host=args.redis_host,
                redis_port=args.redis_port,
                websocket_port=args.coordinator_port
            )
            
            await system.start_coordinator()
            
            # Submit test tasks
            print(f"Submitting {args.test_tasks} test tasks...")
            for i in range(args.test_tasks):
                sensor_data = {
                    "timestamp": time.time(),
                    "values": [1.0, 2.0, 3.0],
                    "sensor_id": f"sensor_{i:03d}"
                }
                
                task_id = system.submit_anomaly_detection_task(sensor_data)
                print(f"Submitted task: {task_id}")
                
                await asyncio.sleep(0.1)  # Small delay between submissions
            
            # Keep running
            try:
                await asyncio.sleep(300)  # Run for 5 minutes
            except KeyboardInterrupt:
                print("Shutting down coordinator...")
            
            system.shutdown()
        
        else:
            print(f"Worker/Edge node implementation would connect to coordinator at {args.coordinator_host}:{args.coordinator_port}")
            print("Worker nodes would use WebSocket client to connect and process tasks")
    
    # Run the async main function
    asyncio.run(main())