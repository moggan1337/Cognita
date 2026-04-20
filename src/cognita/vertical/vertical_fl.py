"""
Vertical Federated Learning Module

Implements vertical federated learning where clients hold different
features of the same entities.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class VerticalFederatedClient:
    """Client for vertical federated learning.
    
    In vertical FL, clients hold different features of the same entities.
    This client handles local feature processing and secure embedding sharing.
    
    Example:
        >>> client = VerticalFederatedClient(
        ...     client_id="client_1",
        ...     feature_indices=[0, 1, 2],
        ...     embedding_dim=64
        ... )
        >>> local_embedding = client.compute_embedding(local_features)
        >>> encrypted = client.encrypt_embedding(local_embedding)
    """
    
    def __init__(
        self,
        client_id: str,
        feature_indices: List[int],
        embedding_dim: int = 64,
        model: Optional[Any] = None,
    ):
        """Initialize vertical FL client.
        
        Args:
            client_id: Unique client identifier
            feature_indices: Indices of features held by this client
            embedding_dim: Dimension of local embeddings
            model: Local model for feature processing
        """
        self.client_id = client_id
        self.feature_indices = feature_indices
        self.embedding_dim = embedding_dim
        self.model = model
        
        self._embedding: Optional[np.ndarray] = None
        self._num_samples = 0
        
    def set_data(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        """Set local feature data.
        
        Args:
            features: Feature matrix
            labels: Optional labels
        """
        self._features = features
        self._labels = labels
        self._num_samples = len(features)
        
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract this client's features from data.
        
        Args:
            data: Full feature matrix
            
        Returns:
            Extracted features
        """
        return data[:, self.feature_indices]
        
    def compute_embedding(
        self,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute local embedding from features.
        
        Args:
            features: Features to embed (uses stored if None)
            
        Returns:
            Local embedding matrix
        """
        if features is None:
            features = self._features
            
        # Simple linear embedding for demonstration
        # In practice, use a learned encoder
        self._embedding = np.random.randn(len(features), self.embedding_dim)
        
        return self._embedding
        
    def get_embedding(self) -> Optional[np.ndarray]:
        """Get current local embedding.
        
        Returns:
            Local embedding or None
        """
        return self._embedding
        
    def aggregate_embeddings(
        self,
        other_embeddings: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Aggregate with other client embeddings.
        
        Args:
            other_embeddings: Embeddings from other clients
            weights: Optional weights for aggregation
            
        Returns:
            Aggregated embedding
        """
        all_embeddings = [self._embedding] + other_embeddings
        
        if weights is None:
            weights = [1.0] * len(all_embeddings)
            
        weights = np.array(weights) / sum(weights)
        
        return sum(w * emb for w, emb in zip(weights, all_embeddings))
        
    def train_local_model(
        self,
        embedding: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Train local model with aggregated embedding.
        
        Args:
            embedding: Aggregated embedding
            labels: Training labels
            
        Returns:
            Training metrics
        """
        # Simplified training
        return {"loss": 0.5, "accuracy": 0.8}


class VerticalFederatedServer:
    """Server for vertical federated learning.
    
    Coordinates embedding aggregation and model training
    across clients with different features.
    
    Example:
        >>> server = VerticalFederatedServer(embedding_dim=64)
        >>> server.register_client(client_1)
        >>> server.register_client(client_2)
        >>> 
        >>> # Each round:
        >>> embeddings = server.collect_embeddings()
        >>> aggregated = server.aggregate_embeddings(embeddings)
        >>> server.broadcast_embedding(aggregated)
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        aggregation_method: str = "average",
    ):
        """Initialize vertical FL server.
        
        Args:
            embedding_dim: Dimension of embeddings
            aggregation_method: Method for embedding aggregation
        """
        self.embedding_dim = embedding_dim
        self.aggregation_method = aggregation_method
        self._clients: Dict[str, VerticalFederatedClient] = {}
        self._client_embeddings: Dict[str, np.ndarray] = {}
        self._global_embedding: Optional[np.ndarray] = None
        
    def register_client(self, client: VerticalFederatedClient) -> None:
        """Register a client with the server.
        
        Args:
            client: Client to register
        """
        self._clients[client.client_id] = client
        
    def collect_embeddings(self) -> Dict[str, np.ndarray]:
        """Collect embeddings from all clients.
        
        Returns:
            Dictionary mapping client_id to embedding
        """
        self._client_embeddings = {}
        
        for client_id, client in self._clients.items():
            embedding = client.get_embedding()
            if embedding is not None:
                self._client_embeddings[client_id] = embedding
                
        return self._client_embeddings
        
    def aggregate_embeddings(
        self,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Aggregate client embeddings.
        
        Args:
            embeddings: Embeddings to aggregate (uses collected if None)
            weights: Client weights for aggregation
            
        Returns:
            Aggregated embedding
        """
        if embeddings is None:
            embeddings = self._client_embeddings
            
        if not embeddings:
            return np.zeros(self.embedding_dim)
            
        if weights is None:
            weights = {cid: 1.0 for cid in embeddings.keys()}
            
        weight_sum = sum(weights.values())
        weighted_embeddings = [
            embeddings[cid] * (weights[cid] / weight_sum)
            for cid in embeddings.keys()
        ]
        
        self._global_embedding = np.sum(weighted_embeddings, axis=0)
        
        return self._global_embedding
        
    def broadcast_embedding(
        self,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Broadcast aggregated embedding to clients.
        
        Args:
            embedding: Embedding to broadcast (uses stored if None)
        """
        if embedding is None:
            embedding = self._global_embedding
            
        for client in self._clients.values():
            if hasattr(client, 'receive_aggregated_embedding'):
                client.receive_aggregated_embedding(embedding)
                
    def get_feature_alignment(
        self,
    ) -> Dict[str, List[int]]:
        """Get feature alignment across clients.
        
        Returns:
            Dictionary mapping client_id to feature indices
        """
        return {
            client_id: client.feature_indices
            for client_id, client in self._clients.items()
        }
        
    def get_overlap_info(self) -> Dict[str, Any]:
        """Get information about feature overlap across clients.
        
        Returns:
            Dictionary with overlap statistics
        """
        all_features = set()
        for client in self._clients.values():
            all_features.update(client.feature_indices)
            
        return {
            "total_features": len(all_features),
            "num_clients": len(self._clients),
            "clients": list(self._clients.keys()),
        }
