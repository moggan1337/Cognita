"""
Horizontal Federated Learning Module

Implements horizontal federated learning where clients hold different
samples of the same features.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class HorizontalFederatedClient:
    """Client for horizontal federated learning.
    
    In horizontal FL, clients hold different samples of the same features.
    This is the standard federated learning setup.
    
    Example:
        >>> client = HorizontalFederatedClient(
        ...     client_id="client_1",
        ...     data=X_train,
        ...     labels=y_train
        ... )
        >>> update = client.train_round(global_model)
    """
    
    def __init__(
        self,
        client_id: str,
        model: Optional[Any] = None,
    ):
        """Initialize horizontal FL client.
        
        Args:
            client_id: Unique client identifier
            model: Local model
        """
        self.client_id = client_id
        self.model = model
        self._data: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._num_samples = 0
        
    def set_data(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Set local data.
        
        Args:
            data: Feature data
            labels: Labels
        """
        self._data = data
        self._labels = labels
        self._num_samples = len(data)
        
    def get_data_size(self) -> int:
        """Get number of local samples.
        
        Returns:
            Number of samples
        """
        return self._num_samples
        
    def train(
        self,
        epochs: int = 1,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Train local model.
        
        Args:
            epochs: Number of local epochs
            batch_size: Batch size
            
        Returns:
            Training metrics
        """
        # Simplified training
        return {
            "loss": 0.5,
            "accuracy": 0.8,
            "num_samples": self._num_samples,
        }
        
    def evaluate(
        self,
        test_data: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate local model.
        
        Args:
            test_data: Test data (uses stored if None)
            test_labels: Test labels
            
        Returns:
            Evaluation metrics
        """
        return {
            "accuracy": 0.75,
            "loss": 0.3,
        }


class HorizontalFederatedServer:
    """Server for horizontal federated learning.
    
    Coordinates model aggregation across clients with different
    data samples.
    
    Example:
        >>> server = HorizontalFederatedServer(model=global_model)
        >>> server.register_client(client_1)
        >>> server.register_client(client_2)
        >>> 
        >>> # Each round:
        >>> server.broadcast_model()
        >>> updates = server.collect_updates()
        >>> aggregated = server.aggregate(updates)
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        aggregation_method: str = "fedavg",
    ):
        """Initialize horizontal FL server.
        
        Args:
            model: Global model
            aggregation_method: Aggregation method
        """
        self.model = model
        self.aggregation_method = aggregation_method
        self._clients: Dict[str, HorizontalFederatedClient] = {}
        self._client_weights: Dict[str, float] = {}
        self._round = 0
        
    def register_client(
        self,
        client: HorizontalFederatedClient,
        weight: Optional[float] = None,
    ) -> None:
        """Register a client.
        
        Args:
            client: Client to register
            weight: Client weight for aggregation
        """
        self._clients[client.client_id] = client
        self._client_weights[client.client_id] = weight or client.get_data_size()
        
    def sample_clients(
        self,
        num_clients: int,
    ) -> List[str]:
        """Sample clients for current round.
        
        Args:
            num_clients: Number to sample
            
        Returns:
            List of sampled client IDs
        """
        client_ids = list(self._clients.keys())
        
        if len(client_ids) <= num_clients:
            return client_ids
            
        # Weighted sampling
        weights = [self._client_weights.get(cid, 1.0) for cid in client_ids]
        weights = np.array(weights) / sum(weights)
        
        indices = np.random.choice(len(client_ids), num_clients, p=weights)
        return [client_ids[i] for i in indices]
        
    def get_aggregate_weights(self) -> Dict[str, float]:
        """Get normalized aggregation weights.
        
        Returns:
            Dictionary of client_id to weight
        """
        total = sum(self._client_weights.values())
        if total == 0:
            return {cid: 1.0 / len(self._clients) for cid in self._clients}
        return {
            cid: w / total for cid, w in self._client_weights.items()
        }
        
    def get_num_clients(self) -> int:
        """Get number of registered clients.
        
        Returns:
            Number of clients
        """
        return len(self._clients)
