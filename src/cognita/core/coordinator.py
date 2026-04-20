"""
Federated Learning Coordinator Module

Orchestrates the federated learning workflow across multiple clients
and servers, managing rounds, client selection, and monitoring.

Author: Cognita Team
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from cognita.core.server import FederatedServer
from cognita.core.fl_config import FLConfig, ExperimentTracker
from cognita.core.client import ClientUpdate
from cognita.utils.logging import get_logger

logger = get_logger(__name__)


class CoordinatorStatus(str, Enum):
    """Coordinator status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class CoordinatorMetrics:
    """Metrics tracked by coordinator."""
    total_rounds: int = 0
    completed_rounds: int = 0
    active_clients: int = 0
    avg_round_time: float = 0.0
    total_time: float = 0.0
    start_time: float = 0.0
    

class FederatedCoordinator:
    """Coordinator for federated learning experiments.
    
    The coordinator manages the overall federated learning workflow,
    including round execution, client management, monitoring, and
    early stopping.
    
    Attributes:
        server: Federated learning server
        config: Federated learning configuration
        tracker: Experiment tracker
        
    Example:
        >>> coordinator = FederatedCoordinator(server, config)
        >>> coordinator.register_clients([client_1, client_2, client_3])
        >>> 
        >>> def on_round_end(round_num, result):
        ...     print(f"Round {round_num}: accuracy={result.metrics['accuracy']:.4f}")
        >>> 
        >>> coordinator.run(callback=on_round_end)
    """
    
    def __init__(
        self,
        server: FederatedServer,
        config: Optional[FLConfig] = None,
    ):
        """Initialize the coordinator.
        
        Args:
            server: Federated learning server
            config: FL configuration
        """
        self.server = server
        self.config = config or FLConfig()
        self.tracker = ExperimentTracker()
        
        self._status = CoordinatorStatus.IDLE
        self._round_times: List[float] = []
        self._start_time: float = 0.0
        self._early_stop_counter: int = 0
        self._best_metrics: Dict[str, float] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._pause_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        
    def register_callback(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """Register a callback for an event.
        
        Args:
            event: Event name (e.g., 'on_round_start', 'on_round_end')
            callback: Callback function
        """
        self._callbacks[event] = callback
        
    def _trigger_callback(
        self,
        event: str,
        *args,
        **kwargs,
    ) -> Any:
        """Trigger a callback.
        
        Args:
            event: Event name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Callback return value
        """
        if event in self._callbacks:
            return self._callbacks[event](*args, **kwargs)
        return None
        
    def register_clients(self, clients: List[Any]) -> None:
        """Register multiple clients with the server.
        
        Args:
            clients: List of client instances
        """
        for client in clients:
            self.server.register_client(client.client_id, client)
            
    def start(self) -> None:
        """Start the coordinator."""
        self._status = CoordinatorStatus.RUNNING
        self._start_time = time.time()
        self._pause_event.set()
        logger.info("Coordinator started")
        
    def pause(self) -> None:
        """Pause the coordinator."""
        self._status = CoordinatorStatus.PAUSED
        self._pause_event.clear()
        logger.info("Coordinator paused")
        
    def resume(self) -> None:
        """Resume the coordinator."""
        self._status = CoordinatorStatus.RUNNING
        self._pause_event.set()
        logger.info("Coordinator resumed")
        
    def stop(self) -> None:
        """Stop the coordinator."""
        self._status = CoordinatorStatus.STOPPED
        self._stop_event.set()
        logger.info("Coordinator stopped")
        
    def _check_early_stopping(
        self,
        metrics: Dict[str, float],
    ) -> bool:
        """Check if early stopping criteria are met.
        
        Args:
            metrics: Current metrics
            
        Returns:
            True if should stop
        """
        if not self._best_metrics:
            for key, value in metrics.items():
                if key not in ['loss', 'training_time']:
                    self._best_metrics[key] = value
            return False
            
        # Check for improvement in main metric (accuracy)
        current_acc = metrics.get('accuracy', 0)
        best_acc = self._best_metrics.get('accuracy', 0)
        
        if current_acc > best_acc + self.config.server.early_stopping_delta:
            self._best_metrics['accuracy'] = current_acc
            self._early_stop_counter = 0
        else:
            self._early_stop_counter += 1
            
        return self._early_stop_counter >= self.config.server.early_stopping_rounds
        
    def run(
        self,
        num_rounds: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> List[Any]:
        """Run federated learning rounds.
        
        Args:
            num_rounds: Number of rounds to run
            callback: Optional per-round callback
            
        Returns:
            List of round results
        """
        self.start()
        num_rounds = num_rounds or self.config.server.num_rounds
        results = []
        
        try:
            for round_num in range(num_rounds):
                # Check if stopped
                if self._status == CoordinatorStatus.STOPPED:
                    break
                    
                # Wait if paused
                while self._status == CoordinatorStatus.PAUSED:
                    time.sleep(0.1)
                    if self._status == CoordinatorStatus.STOPPED:
                        break
                        
                # Trigger round start callback
                self._trigger_callback('on_round_start', round_num)
                
                # Execute round
                round_start = time.time()
                result = self.server.execute_round(round_num)
                round_time = time.time() - round_start
                
                # Track metrics
                self._round_times.append(round_time)
                self.tracker.log_round(
                    round_num,
                    result.metrics,
                    result.participating_clients,
                    round_time,
                )
                
                # Check early stopping
                if self._check_early_stopping(result.metrics):
                    logger.info(f"Early stopping triggered at round {round_num}")
                    results.append(result)
                    break
                    
                results.append(result)
                
                # Per-round callback
                if callback:
                    callback(round_num, result)
                self._trigger_callback('on_round_end', round_num, result)
                
                # Evaluation callback
                if round_num % self.config.server.evaluation_interval == 0:
                    self._trigger_callback('on_evaluation', round_num, result)
                    
        finally:
            self._status = CoordinatorStatus.STOPPED
            
        return results
        
    async def async_run(
        self,
        num_rounds: Optional[int] = None,
    ) -> List[Any]:
        """Run federated learning asynchronously.
        
        Args:
            num_rounds: Number of rounds
            
        Returns:
            List of round results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.run, num_rounds, None
        )
        
    def get_metrics(self) -> CoordinatorMetrics:
        """Get coordinator metrics.
        
        Returns:
            CoordinatorMetrics instance
        """
        return CoordinatorMetrics(
            total_rounds=self.config.server.num_rounds,
            completed_rounds=len(self._round_times),
            active_clients=self.server.num_clients,
            avg_round_time=sum(self._round_times) / max(len(self._round_times), 1),
            total_time=time.time() - self._start_time,
            start_time=self._start_time,
        )
        
    @property
    def status(self) -> CoordinatorStatus:
        """Get coordinator status.
        
        Returns:
            Current status
        """
        return self._status
        
    @property
    def progress(self) -> float:
        """Get progress (0-1).
        
        Returns:
            Progress ratio
        """
        if self.config.server.num_rounds == 0:
            return 0.0
        return len(self._round_times) / self.config.server.num_rounds


class MultiClusterCoordinator(FederatedCoordinator):
    """Coordinator for multi-cluster federated learning.
    
    Manages multiple federated clusters, each with their own server,
    enabling hierarchical federated learning.
    """
    
    def __init__(
        self,
        config: Optional[FLConfig] = None,
    ):
        """Initialize multi-cluster coordinator.
        
        Args:
            config: FL configuration
        """
        self.config = config or FLConfig()
        self.tracker = ExperimentTracker()
        
        self._clusters: Dict[str, FederatedServer] = {}
        self._cluster_configs: Dict[str, FLConfig] = {}
        self._global_server: Optional[FederatedServer] = None
        self._status = CoordinatorStatus.IDLE
        
    def add_cluster(
        self,
        cluster_id: str,
        server: FederatedServer,
        config: Optional[FLConfig] = None,
    ) -> None:
        """Add a cluster to the coordinator.
        
        Args:
            cluster_id: Unique cluster identifier
            server: Cluster's federated server
            config: Cluster-specific configuration
        """
        self._clusters[cluster_id] = server
        self._cluster_configs[cluster_id] = config or self.config.copy()
        
    def set_global_server(self, server: FederatedServer) -> None:
        """Set the global coordination server.
        
        Args:
            server: Global server
        """
        self._global_server = server
        
    def run_hierarchical_round(
        self,
        round_num: int,
    ) -> Dict[str, Any]:
        """Execute hierarchical FL round.
        
        First, each cluster performs local aggregation.
        Then, clusters aggregate at global level.
        
        Args:
            round_num: Round number
            
        Returns:
            Dictionary with results from all clusters
        """
        results = {}
        
        # Local aggregation in each cluster
        for cluster_id, server in self._clusters.items():
            result = server.execute_round(round_num)
            results[cluster_id] = result
            
        # Global aggregation if global server exists
        if self._global_server:
            cluster_updates = []
            for cluster_id, server in self._clusters.items():
                weights = server.get_global_weights()
                if weights:
                    cluster_updates.append(weights)
                    
            if cluster_updates:
                # Average cluster weights at global level
                global_weights = {}
                for key in cluster_updates[0].keys():
                    global_weights[key] = np.mean(
                        [update[key] for update in cluster_updates],
                        axis=0
                    )
                    
                # Broadcast back to clusters
                for server in self._clusters.values():
                    server.receive_global_weights(global_weights)
                    
        return results
        
    def run(
        self,
        num_rounds: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Run hierarchical federated learning.
        
        Args:
            num_rounds: Number of rounds
            callback: Optional per-round callback
            
        Returns:
            List of hierarchical round results
        """
        self.start()
        num_rounds = num_rounds or self.config.server.num_rounds
        results = []
        
        for round_num in range(num_rounds):
            if self._status == CoordinatorStatus.STOPPED:
                break
                
            result = self.run_hierarchical_round(round_num)
            results.append(result)
            
            if callback:
                callback(round_num, result)
                
        self._status = CoordinatorStatus.STOPPED
        return results
