"""
Federated Learning Server Module

Implements the server-side functionality for federated learning,
including client management, model aggregation, and communication
with clients.

Author: Cognita Team
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import threading

import numpy as np
import torch
import torch.nn as nn

from cognita.core.model_manager import ModelManager
from cognita.core.fl_config import ServerConfig, FLConfig, AggregationStrategy
from cognita.core.client import ClientUpdate
from cognita.aggregation.base import BaseAggregator
from cognita.aggregation.fedavg import FedAvgAggregator
from cognita.byzantine import ByzantineResilientAggregator
from cognita.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RoundResult:
    """Container for round aggregation results.
    
    Attributes:
        round_num: Round number
        aggregated_weights: Result of weight aggregation
        metrics: Aggregation metrics
        participating_clients: Number of clients participated
        duration: Time taken for round
    """
    round_num: int
    aggregated_weights: Dict[str, np.ndarray]
    metrics: Dict[str, float]
    participating_clients: int
    duration: float
    client_updates: List[ClientUpdate] = field(default_factory=list)


class FederatedServer:
    """Federated learning server.
    
    The server coordinates the federated learning process by:
    1. Sampling clients each round
    2. Distributing global model
    3. Aggregating client updates
    4. Updating global model
    
    Attributes:
        model: Global model
        config: Server configuration
        aggregator: Weight aggregation strategy
        
    Example:
        >>> server = FederatedServer(
        ...     model=SimpleCNN(),
        ...     aggregator=FedAvgAggregator()
        ... )
        >>> server.register_client(client_1)
        >>> server.register_client(client_2)
        >>> for round_num in range(100):
        ...     server.start_round(round_num)
        ...     server.wait_for_updates()
        ...     server.aggregate_updates()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[ServerConfig] = None,
        fl_config: Optional[FLConfig] = None,
        aggregator: Optional[BaseAggregator] = None,
        device: str = "cpu",
    ):
        """Initialize the federated server.
        
        Args:
            model: Global PyTorch model
            config: Server configuration
            fl_config: Federated learning configuration
            aggregator: Weight aggregation strategy
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.config = config or ServerConfig()
        self.fl_config = fl_config or FLConfig(server=self.config)
        
        self.model_manager = ModelManager(model, device=device)
        
        # Initialize aggregator
        if aggregator is not None:
            self.aggregator = aggregator
        elif self.config.aggregation_strategy == AggregationStrategy.FEDAVG:
            self.aggregator = FedAvgAggregator()
        else:
            self.aggregator = FedAvgAggregator()
            
        # Byzantine resilience
        if self.fl_config.byzantine.enabled:
            self.aggregator = ByzantineResilientAggregator(
                base_aggregator=self.aggregator,
                num_byzantine=self.fl_config.byzantine.num_byzantine,
                method=self.fl_config.byzantine.defense_method,
            )
            
        self._clients: Dict[str, Any] = {}
        self._client_weights: Dict[str, float] = {}
        self._pending_updates: Dict[str, ClientUpdate] = {}
        self._round_history: List[RoundResult] = []
        self._current_round = 0
        self._global_weights: Optional[Dict[str, np.ndarray]] = None
        self._prev_weights: Optional[Dict[str, np.ndarray]] = None
        self._lock = threading.Lock()
        self._round_lock = threading.Lock()
        
        # Initialize global weights
        self._global_weights = self.model_manager.get_weights()
        
    def register_client(
        self,
        client_id: str,
        client: Any,
        weight: float = 1.0,
    ) -> None:
        """Register a client with the server.
        
        Args:
            client_id: Unique client identifier
            client: Client instance
            weight: Client weight for aggregation
        """
        self._clients[client_id] = client
        self._client_weights[client_id] = weight
        logger.info(f"Registered client: {client_id}")
        
    def unregister_client(self, client_id: str) -> bool:
        """Unregister a client.
        
        Args:
            client_id: Client to remove
            
        Returns:
            True if client was removed
        """
        if client_id in self._clients:
            del self._clients[client_id]
            del self._client_weights[client_id]
            return True
        return False
        
    def sample_clients(
        self,
        num_clients: Optional[int] = None,
    ) -> List[str]:
        """Sample clients for the current round.
        
        Args:
            num_clients: Number of clients to sample
            
        Returns:
            List of sampled client IDs
        """
        num_clients = num_clients or self.config.clients_per_round
        
        if len(self._clients) <= num_clients:
            return list(self._clients.keys())
            
        if self.config.sample_strategy == "uniform":
            return self._uniform_sample(num_clients)
        elif self.config.sample_strategy == "stratified":
            return self._stratified_sample(num_clients)
        else:
            return self._uniform_sample(num_clients)
            
    def _uniform_sample(self, num_clients: int) -> List[str]:
        """Sample clients uniformly.
        
        Args:
            num_clients: Number to sample
            
        Returns:
            List of client IDs
        """
        client_ids = list(self._clients.keys())
        indices = np.random.choice(
            len(client_ids),
            size=min(num_clients, len(client_ids)),
            replace=False,
        )
        return [client_ids[i] for i in indices]
        
    def _stratified_sample(self, num_clients: int) -> List[str]:
        """Sample clients stratified by weight.
        
        Args:
            num_clients: Number to sample
            
        Returns:
            List of client IDs
        """
        client_ids = list(self._clients.keys())
        weights = np.array([self._client_weights.get(cid, 1.0) for cid in client_ids])
        weights = weights / weights.sum()
        indices = np.random.choice(
            len(client_ids),
            size=min(num_clients, len(client_ids)),
            replace=False,
            p=weights,
        )
        return [client_ids[i] for i in indices]
        
    def broadcast_weights(self, client_ids: List[str]) -> None:
        """Broadcast global model weights to clients.
        
        Args:
            client_ids: Clients to receive weights
        """
        for client_id in client_ids:
            if client_id in self._clients:
                client = self._clients[client_id]
                client.receive_global_weights(self._global_weights)
                
    def receive_update(self, update: ClientUpdate) -> None:
        """Receive an update from a client.
        
        Args:
            update: Client update
        """
        with self._lock:
            self._pending_updates[update.client_id] = update
            
    def get_pending_updates(self) -> List[ClientUpdate]:
        """Get all pending client updates.
        
        Returns:
            List of pending updates
        """
        with self._lock:
            return list(self._pending_updates.values())
            
    def clear_updates(self) -> None:
        """Clear pending updates."""
        with self._lock:
            self._pending_updates.clear()
            
    def wait_for_updates(
        self,
        client_ids: List[str],
        timeout: Optional[float] = None,
    ) -> List[ClientUpdate]:
        """Wait for client updates.
        
        Args:
            client_ids: Expected client IDs
            timeout: Maximum time to wait
            
        Returns:
            List of received updates
        """
        timeout = timeout or self.config.timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                received_ids = set(self._pending_updates.keys())
                expected_ids = set(client_ids)
                
                # Check if we have minimum required updates
                if len(received_ids & expected_ids) >= self.config.min_clients:
                    return [
                        self._pending_updates[cid]
                        for cid in client_ids
                        if cid in self._pending_updates
                    ]
            time.sleep(0.1)
            
        # Return whatever we have
        with self._lock:
            return [
                self._pending_updates[cid]
                for cid in client_ids
                if cid in self._pending_updates
            ]
            
    def aggregate_updates(
        self,
        updates: List[ClientUpdate],
    ) -> Dict[str, np.ndarray]:
        """Aggregate client updates.
        
        Args:
            updates: List of client updates
            
        Returns:
            Aggregated weights
        """
        if not updates:
            return self._global_weights or {}
            
        # Prepare weight updates with weights
        weight_updates = []
        for update in updates:
            weight = self._client_weights.get(update.client_id, 1.0)
            weight_updates.append((update.weights, weight))
            
        # Store previous weights
        self._prev_weights = self._global_weights.copy() if self._global_weights else {}
        
        # Aggregate
        aggregated_delta = self.aggregator.aggregate(weight_updates)
        
        # Apply delta to previous weights
        if self._prev_weights:
            self._global_weights = self.model_manager.apply_delta(
                self._prev_weights, aggregated_delta
            )
        else:
            self._global_weights = aggregated_delta
            
        # Update model
        self.model_manager.set_weights(self._global_weights)
        
        return self._global_weights
        
    def start_round(self, round_num: int) -> List[str]:
        """Start a new training round.
        
        Args:
            round_num: Round number
            
        Returns:
            List of sampled client IDs
        """
        with self._round_lock:
            self._current_round = round_num
            self.clear_updates()
            
        # Sample clients
        sampled_clients = self.sample_clients()
        
        # Broadcast weights
        self.broadcast_weights(sampled_clients)
        
        logger.info(f"Round {round_num}: Selected {len(sampled_clients)} clients")
        
        return sampled_clients
        
    def execute_round(
        self,
        round_num: int,
    ) -> RoundResult:
        """Execute a complete training round.
        
        Args:
            round_num: Round number
            
        Returns:
            RoundResult with aggregated weights and metrics
        """
        start_time = time.time()
        
        # Start round and get sampled clients
        sampled_clients = self.start_round(round_num)
        
        # Wait for updates
        updates = self.wait_for_updates(sampled_clients)
        
        # Aggregate
        aggregated_weights = self.aggregate_updates(updates)
        
        duration = time.time() - start_time
        
        # Compute metrics
        metrics = self._compute_round_metrics(updates)
        
        result = RoundResult(
            round_num=round_num,
            aggregated_weights=aggregated_weights,
            metrics=metrics,
            participating_clients=len(updates),
            duration=duration,
            client_updates=updates,
        )
        
        self._round_history.append(result)
        
        # Save checkpoint if needed
        if round_num % self.config.checkpoint_interval == 0:
            self.save_checkpoint(
                Path(self.config.model_save_path) / f"round_{round_num}.pt"
            )
            
        return result
        
    def _compute_round_metrics(
        self,
        updates: List[ClientUpdate],
    ) -> Dict[str, float]:
        """Compute aggregated metrics from client updates.
        
        Args:
            updates: List of client updates
            
        Returns:
            Dictionary of metrics
        """
        if not updates:
            return {}
            
        # Average metrics across clients
        metrics = defaultdict(list)
        for update in updates:
            for key, value in update.metrics.items():
                metrics[key].append(value)
                
        return {
            key: float(np.mean(values))
            for key, values in metrics.items()
        }
        
    def run(
        self,
        num_rounds: Optional[int] = None,
        callback: Optional[Callable[[int, RoundResult], None]] = None,
    ) -> List[RoundResult]:
        """Run federated learning for specified rounds.
        
        Args:
            num_rounds: Number of rounds to run
            callback: Optional callback after each round
            
        Returns:
            List of RoundResults
        """
        num_rounds = num_rounds or self.config.num_rounds
        results = []
        
        for round_num in range(num_rounds):
            result = self.execute_round(round_num)
            results.append(result)
            
            if callback:
                callback(round_num, result)
                
            logger.info(
                f"Round {round_num} completed: "
                f"clients={result.participating_clients}, "
                f"duration={result.duration:.2f}s, "
                f"accuracy={result.metrics.get('accuracy', 0):.4f}"
            )
            
        return results
        
    def get_global_weights(self) -> Optional[Dict[str, np.ndarray]]:
        """Get current global model weights.
        
        Returns:
            Global weights dictionary
        """
        return self._global_weights
        
    def get_current_round(self) -> int:
        """Get current round number.
        
        Returns:
            Current round
        """
        return self._current_round
        
    def get_history(self) -> List[RoundResult]:
        """Get round history.
        
        Returns:
            List of RoundResults
        """
        return self._round_history.copy()
        
    def save_checkpoint(
        self,
        path: Union[str, Path],
        include_history: bool = True,
    ) -> None:
        """Save server state checkpoint.
        
        Args:
            path: Path to save checkpoint
            include_history: Include round history
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "round": self._current_round,
            "weights": self._global_weights,
            "client_weights": self._client_weights,
        }
        
        if include_history:
            checkpoint["history"] = [
                {
                    "round_num": r.round_num,
                    "metrics": r.metrics,
                    "participating_clients": r.participating_clients,
                    "duration": r.duration,
                }
                for r in self._round_history
            ]
            
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load server state checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self._current_round = checkpoint.get("round", 0)
        self._global_weights = checkpoint.get("weights")
        self._client_weights = checkpoint.get("client_weights", {})
        
        if self._global_weights:
            self.model_manager.set_weights(self._global_weights)
            
        logger.info(f"Loaded checkpoint from {path}")
        
    @property
    def num_clients(self) -> int:
        """Get number of registered clients.
        
        Returns:
            Number of clients
        """
        return len(self._clients)
        
    @property
    def is_running(self) -> bool:
        """Check if server is running.
        
        Returns:
            Running status
        """
        return self._current_round >= 0


class AsyncFederatedServer(FederatedServer):
    """Asynchronous federated learning server.
    
    This server variant handles client updates asynchronously,
    allowing for varying client response times and network conditions.
    """
    
    def __init__(self, *args, staleness_weight: float = 0.5, **kwargs):
        """Initialize async server.
        
        Args:
            staleness_weight: Weight for stale updates
            *args: Parent class arguments
            **kwargs: Parent class keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.staleness_weight = staleness_weight
        self._client_last_update: Dict[str, int] = {}
        self._accumulated_deltas: Dict[str, np.ndarray] = defaultdict(
            lambda: defaultdict(np.float32)
        )
        
    def receive_update(self, update: ClientUpdate) -> None:
        """Receive and process update asynchronously.
        
        Args:
            update: Client update
        """
        with self._lock:
            client_id = update.client_id
            last_round = self._client_last_update.get(client_id, -1)
            staleness = self._current_round - update.round_num
            
            # Apply staleness weighting
            if staleness > 0:
                staleness_factor = self.staleness_weight ** staleness
                scaled_weights = {
                    k: v * staleness_factor for k, v in update.weights.items()
                }
                update = ClientUpdate(
                    client_id=update.client_id,
                    round_num=update.round_num,
                    weights=scaled_weights,
                    num_samples=update.num_samples,
                    training_time=update.training_time,
                    metrics=update.metrics,
                )
                
            self._pending_updates[client_id] = update
            self._client_last_update[client_id] = update.round_num
            
    async def async_execute_round(self, round_num: int) -> RoundResult:
        """Execute round asynchronously.
        
        Args:
            round_num: Round number
            
        Returns:
            RoundResult
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_round, round_num)


class VerticalFLServer(FederatedServer):
    """Server for vertical federated learning.
    
    Handles embedding aggregation and alignment for vertical FL scenarios
    where clients hold different features of the same entities.
    """
    
    def __init__(self, *args, embedding_dim: int = 128, **kwargs):
        """Initialize vertical FL server.
        
        Args:
            embedding_dim: Dimension of embeddings
            *args: Parent class arguments
            **kwargs: Parent class keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self._client_embeddings: Dict[str, np.ndarray] = {}
        
    def receive_embedding(
        self,
        client_id: str,
        embedding: np.ndarray,
        round_num: int,
    ) -> None:
        """Receive embedding from a client.
        
        Args:
            client_id: Client identifier
            embedding: Client embedding
            round_num: Current round
        """
        self._client_embeddings[client_id] = embedding
        
    def aggregate_embeddings(self) -> np.ndarray:
        """Aggregate client embeddings.
        
        Returns:
            Aggregated embedding
        """
        if not self._client_embeddings:
            return np.zeros(self.embedding_dim)
            
        embeddings = list(self._client_embeddings.values())
        return np.mean(embeddings, axis=0)
    
    def broadcast_embeddings(
        self,
        client_ids: List[str],
        aggregated_embedding: np.ndarray,
    ) -> None:
        """Broadcast aggregated embedding to clients.
        
        Args:
            client_ids: Clients to receive embedding
            aggregated_embedding: Aggregated embedding
        """
        for client_id in client_ids:
            if client_id in self._clients:
                client = self._clients[client_id]
                if hasattr(client, 'receive_embedding'):
                    client.receive_embedding(aggregated_embedding)
                    
    def clear_embeddings(self) -> None:
        """Clear stored embeddings."""
        self._client_embeddings.clear()
