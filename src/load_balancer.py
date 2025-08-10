#!/usr/bin/env python3
"""
Intelligent Load Balancing System for IoT Anomaly Detection Platform
Provides advanced load balancing with health-aware routing, adaptive algorithms, and request affinity.
"""

import asyncio
import hashlib
import random
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
import heapq
import statistics
import socket

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .logging_config import setup_logging
from .circuit_breaker import CircuitBreaker, CircuitState
from .health_monitoring import HealthStatus


class LoadBalancingAlgorithm(Enum):
    """Available load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE = "adaptive"
    POWER_OF_TWO = "power_of_two"


@dataclass
class BackendServer:
    """Backend server configuration"""
    id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    health_check_url: str = "/health"
    is_healthy: bool = True
    last_health_check: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def address(self) -> str:
        """Get server address"""
        return f"{self.host}:{self.port}"
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, BackendServer):
            return False
        return self.id == other.id


@dataclass
class ServerMetrics:
    """Server performance metrics"""
    active_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_updated: float = field(default_factory=time.time)
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return 1.0 - self.error_rate
    
    def update_response_time(self, response_time_ms: float):
        """Update response time metrics"""
        self.response_times.append(response_time_ms)
        if self.response_times:
            self.avg_response_time_ms = statistics.mean(self.response_times)


@dataclass
class Request:
    """Load balancer request"""
    id: str
    client_ip: str
    path: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class HealthChecker:
    """Health checking for backend servers"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.logger = setup_logging(self.__class__.__name__)
        self._check_tasks: Dict[str, asyncio.Task] = {}
        self._is_running = False
    
    async def start_monitoring(self, servers: List[BackendServer]):
        """Start health monitoring for servers"""
        self._is_running = True
        
        for server in servers:
            if server.id not in self._check_tasks:
                task = asyncio.create_task(self._health_check_loop(server))
                self._check_tasks[server.id] = task
        
        self.logger.info(f"Started health monitoring for {len(servers)} servers")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self._is_running = False
        
        for task in self._check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._check_tasks.values(), return_exceptions=True)
        self._check_tasks.clear()
    
    async def _health_check_loop(self, server: BackendServer):
        """Health check loop for a single server"""
        while self._is_running:
            try:
                await self._check_server_health(server)
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check for {server.id}: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_server_health(self, server: BackendServer):
        """Check health of a single server"""
        try:
            # Simple TCP connection check
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(server.host, server.port),
                timeout=5.0
            )
            
            writer.close()
            await writer.wait_closed()
            
            if not server.is_healthy:
                self.logger.info(f"Server {server.id} is back online")
            
            server.is_healthy = True
            server.last_health_check = time.time()
            
        except Exception as e:
            if server.is_healthy:
                self.logger.warning(f"Server {server.id} failed health check: {e}")
            
            server.is_healthy = False
            server.last_health_check = time.time()


class LoadBalancingStrategy(ABC):
    """Abstract base class for load balancing strategies"""
    
    @abstractmethod
    async def select_server(
        self, 
        servers: List[BackendServer],
        metrics: Dict[str, ServerMetrics],
        request: Request
    ) -> Optional[BackendServer]:
        """Select best server for request"""
        pass


class RoundRobinStrategy(LoadBalancingStrategy):
    """Round-robin load balancing"""
    
    def __init__(self):
        self.current_index = 0
        self.lock = asyncio.Lock()
    
    async def select_server(
        self, 
        servers: List[BackendServer],
        metrics: Dict[str, ServerMetrics],
        request: Request
    ) -> Optional[BackendServer]:
        """Select server using round-robin"""
        healthy_servers = [s for s in servers if s.is_healthy]
        
        if not healthy_servers:
            return None
        
        async with self.lock:
            server = healthy_servers[self.current_index % len(healthy_servers)]
            self.current_index += 1
            return server


class WeightedRoundRobinStrategy(LoadBalancingStrategy):
    """Weighted round-robin load balancing"""
    
    def __init__(self):
        self.current_weights: Dict[str, float] = {}
        self.lock = asyncio.Lock()
    
    async def select_server(
        self, 
        servers: List[BackendServer],
        metrics: Dict[str, ServerMetrics],
        request: Request
    ) -> Optional[BackendServer]:
        """Select server using weighted round-robin"""
        healthy_servers = [s for s in servers if s.is_healthy]
        
        if not healthy_servers:
            return None
        
        async with self.lock:
            # Initialize weights
            for server in healthy_servers:
                if server.id not in self.current_weights:
                    self.current_weights[server.id] = server.weight
            
            # Find server with highest current weight
            best_server = max(healthy_servers, key=lambda s: self.current_weights[s.id])
            
            # Decrease current weight and increase others
            total_weight = sum(s.weight for s in healthy_servers)
            self.current_weights[best_server.id] -= total_weight
            
            for server in healthy_servers:
                self.current_weights[server.id] += server.weight
            
            return best_server


class LeastConnectionsStrategy(LoadBalancingStrategy):
    """Least connections load balancing"""
    
    async def select_server(
        self, 
        servers: List[BackendServer],
        metrics: Dict[str, ServerMetrics],
        request: Request
    ) -> Optional[BackendServer]:
        """Select server with least connections"""
        healthy_servers = [s for s in servers if s.is_healthy]
        
        if not healthy_servers:
            return None
        
        # Select server with minimum active connections
        return min(
            healthy_servers,
            key=lambda s: metrics.get(s.id, ServerMetrics()).active_connections
        )


class LeastResponseTimeStrategy(LoadBalancingStrategy):
    """Least response time load balancing"""
    
    async def select_server(
        self, 
        servers: List[BackendServer],
        metrics: Dict[str, ServerMetrics],
        request: Request
    ) -> Optional[BackendServer]:
        """Select server with least average response time"""
        healthy_servers = [s for s in servers if s.is_healthy]
        
        if not healthy_servers:
            return None
        
        # Select server with minimum response time
        return min(
            healthy_servers,
            key=lambda s: metrics.get(s.id, ServerMetrics()).avg_response_time_ms
        )


class ConsistentHashStrategy(LoadBalancingStrategy):
    """Consistent hashing load balancing"""
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.hash_ring: Dict[int, BackendServer] = {}
        self.servers_set = set()
        self.lock = asyncio.Lock()
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    async def _rebuild_ring(self, servers: List[BackendServer]):
        """Rebuild hash ring with current servers"""
        current_servers = set(servers)
        
        if current_servers == self.servers_set:
            return  # No changes needed
        
        self.hash_ring.clear()
        self.servers_set = current_servers.copy()
        
        for server in servers:
            for i in range(self.virtual_nodes):
                virtual_key = f"{server.id}:{i}"
                hash_value = self._hash(virtual_key)
                self.hash_ring[hash_value] = server
    
    async def select_server(
        self, 
        servers: List[BackendServer],
        metrics: Dict[str, ServerMetrics],
        request: Request
    ) -> Optional[BackendServer]:
        """Select server using consistent hashing"""
        healthy_servers = [s for s in servers if s.is_healthy]
        
        if not healthy_servers:
            return None
        
        async with self.lock:
            await self._rebuild_ring(healthy_servers)
            
            if not self.hash_ring:
                return healthy_servers[0]  # Fallback
            
            # Use client IP or session ID for consistency
            hash_key = request.session_id or request.client_ip
            request_hash = self._hash(hash_key)
            
            # Find the first server clockwise from the hash
            sorted_hashes = sorted(self.hash_ring.keys())
            for hash_value in sorted_hashes:
                if hash_value >= request_hash:
                    return self.hash_ring[hash_value]
            
            # Wrap around to the first server
            return self.hash_ring[sorted_hashes[0]]


class AdaptiveStrategy(LoadBalancingStrategy):
    """Adaptive load balancing using multiple factors"""
    
    def __init__(self):
        self.weights = {
            'response_time': 0.4,
            'connections': 0.3,
            'error_rate': 0.2,
            'cpu_usage': 0.1
        }
    
    async def select_server(
        self, 
        servers: List[BackendServer],
        metrics: Dict[str, ServerMetrics],
        request: Request
    ) -> Optional[BackendServer]:
        """Select server using adaptive scoring"""
        healthy_servers = [s for s in servers if s.is_healthy]
        
        if not healthy_servers:
            return None
        
        if len(healthy_servers) == 1:
            return healthy_servers[0]
        
        # Calculate scores for each server
        scores = {}
        
        for server in healthy_servers:
            server_metrics = metrics.get(server.id, ServerMetrics())
            
            # Normalize metrics (lower is better for all metrics)
            response_time_score = server_metrics.avg_response_time_ms
            connections_score = server_metrics.active_connections
            error_rate_score = server_metrics.error_rate * 1000  # Scale up
            cpu_score = server_metrics.cpu_usage
            
            # Weighted composite score
            composite_score = (
                self.weights['response_time'] * response_time_score +
                self.weights['connections'] * connections_score +
                self.weights['error_rate'] * error_rate_score +
                self.weights['cpu_usage'] * cpu_score
            )
            
            scores[server.id] = composite_score
        
        # Select server with lowest score
        best_server = min(healthy_servers, key=lambda s: scores[s.id])
        return best_server


class PowerOfTwoStrategy(LoadBalancingStrategy):
    """Power of two choices load balancing"""
    
    async def select_server(
        self, 
        servers: List[BackendServer],
        metrics: Dict[str, ServerMetrics],
        request: Request
    ) -> Optional[BackendServer]:
        """Select better of two random servers"""
        healthy_servers = [s for s in servers if s.is_healthy]
        
        if not healthy_servers:
            return None
        
        if len(healthy_servers) == 1:
            return healthy_servers[0]
        
        # Choose two random servers
        choices = random.sample(healthy_servers, min(2, len(healthy_servers)))
        
        if len(choices) == 1:
            return choices[0]
        
        # Select the one with fewer connections
        server1_metrics = metrics.get(choices[0].id, ServerMetrics())
        server2_metrics = metrics.get(choices[1].id, ServerMetrics())
        
        if server1_metrics.active_connections <= server2_metrics.active_connections:
            return choices[0]
        else:
            return choices[1]


class IntelligentLoadBalancer:
    """Intelligent load balancer with multiple algorithms and health checking"""
    
    def __init__(
        self,
        algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ADAPTIVE,
        health_check_interval: int = 30,
        enable_session_affinity: bool = True
    ):
        self.algorithm = algorithm
        self.enable_session_affinity = enable_session_affinity
        self.logger = setup_logging(self.__class__.__name__)
        
        # Server management
        self.servers: List[BackendServer] = []
        self.server_metrics: Dict[str, ServerMetrics] = {}
        self.servers_lock = asyncio.Lock()
        
        # Load balancing strategy
        self.strategy = self._create_strategy(algorithm)
        
        # Health checking
        self.health_checker = HealthChecker(health_check_interval)
        
        # Session affinity
        self.session_server_map: Dict[str, str] = {}  # session_id -> server_id
        self.affinity_lock = asyncio.Lock()
        
        # Circuit breakers for servers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Request tracking
        self.request_counter = 0
        self.active_requests: Dict[str, int] = defaultdict(int)  # server_id -> count
        
        # Performance metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.request_history = deque(maxlen=1000)
    
    def _create_strategy(self, algorithm: LoadBalancingAlgorithm) -> LoadBalancingStrategy:
        """Create load balancing strategy"""
        if algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return RoundRobinStrategy()
        elif algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return WeightedRoundRobinStrategy()
        elif algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return LeastConnectionsStrategy()
        elif algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            return LeastResponseTimeStrategy()
        elif algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
            return ConsistentHashStrategy()
        elif algorithm == LoadBalancingAlgorithm.ADAPTIVE:
            return AdaptiveStrategy()
        elif algorithm == LoadBalancingAlgorithm.POWER_OF_TWO:
            return PowerOfTwoStrategy()
        else:
            return AdaptiveStrategy()  # Default fallback
    
    async def add_server(self, server: BackendServer):
        """Add backend server"""
        async with self.servers_lock:
            if server not in self.servers:
                self.servers.append(server)
                self.server_metrics[server.id] = ServerMetrics()
                
                # Create circuit breaker for server
                from .circuit_breaker import CircuitBreakerConfig
                config = CircuitBreakerConfig(
                    failure_threshold=5,
                    timeout_seconds=60.0,
                    success_threshold=3
                )
                self.circuit_breakers[server.id] = CircuitBreaker(
                    name=f"server_{server.id}",
                    config=config
                )
                
                self.logger.info(f"Added server: {server.address}")
                
                # Start health monitoring
                await self.health_checker.start_monitoring([server])
    
    async def remove_server(self, server_id: str):
        """Remove backend server"""
        async with self.servers_lock:
            self.servers = [s for s in self.servers if s.id != server_id]
            if server_id in self.server_metrics:
                del self.server_metrics[server_id]
            if server_id in self.circuit_breakers:
                del self.circuit_breakers[server_id]
            
            self.logger.info(f"Removed server: {server_id}")
    
    async def route_request(self, request: Request) -> Optional[BackendServer]:
        """Route request to appropriate backend server"""
        async with self.servers_lock:
            available_servers = [
                s for s in self.servers 
                if s.is_healthy and self.circuit_breakers[s.id].state != CircuitState.OPEN
            ]
        
        if not available_servers:
            self.logger.warning("No healthy servers available")
            return None
        
        # Check session affinity
        if (self.enable_session_affinity and request.session_id and 
            request.session_id in self.session_server_map):
            
            server_id = self.session_server_map[request.session_id]
            server = next((s for s in available_servers if s.id == server_id), None)
            
            if server:
                self.logger.debug(f"Using session affinity for {request.session_id}")
                return server
        
        # Use load balancing strategy
        selected_server = await self.strategy.select_server(
            available_servers, self.server_metrics, request
        )
        
        if selected_server and self.enable_session_affinity and request.session_id:
            async with self.affinity_lock:
                self.session_server_map[request.session_id] = selected_server.id
        
        return selected_server
    
    async def record_request_result(
        self, 
        server: BackendServer, 
        success: bool, 
        response_time_ms: float,
        request: Request
    ):
        """Record request result for metrics and circuit breaker"""
        # Update server metrics
        if server.id in self.server_metrics:
            metrics = self.server_metrics[server.id]
            metrics.total_requests += 1
            
            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
            
            metrics.update_response_time(response_time_ms)
        
        # Update circuit breaker
        if server.id in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[server.id]
            if success:
                circuit_breaker._record_success()
            else:
                circuit_breaker._record_failure(Exception("Request failed"))
        
        # Update global metrics
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Record in history
        self.request_history.append({
            'timestamp': time.time(),
            'server_id': server.id,
            'success': success,
            'response_time_ms': response_time_ms,
            'request_id': request.id
        })
    
    async def start_request(self, server: BackendServer):
        """Mark start of request processing"""
        self.active_requests[server.id] += 1
        if server.id in self.server_metrics:
            self.server_metrics[server.id].active_connections += 1
    
    async def end_request(self, server: BackendServer):
        """Mark end of request processing"""
        if self.active_requests[server.id] > 0:
            self.active_requests[server.id] -= 1
        
        if server.id in self.server_metrics:
            metrics = self.server_metrics[server.id]
            if metrics.active_connections > 0:
                metrics.active_connections -= 1
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        healthy_servers = len([s for s in self.servers if s.is_healthy])
        
        success_rate = 0.0
        if self.total_requests > 0:
            success_rate = self.successful_requests / self.total_requests
        
        return {
            'total_servers': len(self.servers),
            'healthy_servers': healthy_servers,
            'total_requests': self.total_requests,
            'success_rate': success_rate,
            'algorithm': self.algorithm.value,
            'active_connections': sum(self.active_requests.values()),
            'server_metrics': {
                server.id: {
                    'address': server.address,
                    'healthy': server.is_healthy,
                    'weight': server.weight,
                    'active_connections': self.server_metrics[server.id].active_connections,
                    'total_requests': self.server_metrics[server.id].total_requests,
                    'success_rate': self.server_metrics[server.id].success_rate,
                    'avg_response_time_ms': self.server_metrics[server.id].avg_response_time_ms,
                    'circuit_breaker_state': self.circuit_breakers[server.id].state.value
                } for server in self.servers
            }
        }
    
    async def shutdown(self):
        """Shutdown load balancer"""
        await self.health_checker.stop_monitoring()
        self.logger.info("Load balancer shut down")


# Context manager for request handling
class RequestContext:
    """Context manager for handling load-balanced requests"""
    
    def __init__(self, load_balancer: IntelligentLoadBalancer, request: Request):
        self.load_balancer = load_balancer
        self.request = request
        self.server: Optional[BackendServer] = None
        self.start_time: Optional[float] = None
    
    async def __aenter__(self):
        """Enter request context"""
        self.server = await self.load_balancer.route_request(self.request)
        
        if not self.server:
            raise Exception("No available servers")
        
        await self.load_balancer.start_request(self.server)
        self.start_time = time.time()
        
        return self.server
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit request context"""
        if self.server and self.start_time:
            response_time_ms = (time.time() - self.start_time) * 1000
            success = exc_type is None
            
            await self.load_balancer.record_request_result(
                self.server, success, response_time_ms, self.request
            )
            await self.load_balancer.end_request(self.server)


# Example usage
async def main():
    """Example usage of intelligent load balancer"""
    logger = setup_logging("loadbalancer_example")
    
    # Create load balancer
    load_balancer = IntelligentLoadBalancer(
        algorithm=LoadBalancingAlgorithm.ADAPTIVE,
        enable_session_affinity=True
    )
    
    # Add backend servers
    servers = [
        BackendServer("server1", "localhost", 8001, weight=1.0),
        BackendServer("server2", "localhost", 8002, weight=1.5),
        BackendServer("server3", "localhost", 8003, weight=1.2),
    ]
    
    for server in servers:
        await load_balancer.add_server(server)
    
    # Simulate requests
    logger.info("Starting load balancing simulation...")
    
    for i in range(20):
        request = Request(
            id=f"req_{i}",
            client_ip=f"192.168.1.{i % 10 + 1}",
            path="/api/anomaly/detect",
            session_id=f"session_{i % 5}"
        )
        
        try:
            async with RequestContext(load_balancer, request) as server:
                logger.info(f"Request {i} routed to {server.address}")
                
                # Simulate processing time
                await asyncio.sleep(random.uniform(0.1, 0.5))
                
                # Simulate random failures
                if random.random() < 0.1:
                    raise Exception("Simulated failure")
                
        except Exception as e:
            logger.error(f"Request {i} failed: {e}")
        
        await asyncio.sleep(0.1)
    
    # Print statistics
    stats = load_balancer.get_server_stats()
    logger.info(f"Final stats: {stats}")
    
    await load_balancer.shutdown()


if __name__ == "__main__":
    asyncio.run(main())