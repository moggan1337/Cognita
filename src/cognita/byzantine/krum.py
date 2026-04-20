"""
Krum Aggregator Module

Implements Krum and Multi-Krum Byzantine-resilient aggregation.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cognita.aggregation.base import BaseAggregator
from cognita.byzantine.byzantine_resilient import KrumDefender


class KrumAggregator(BaseAggregator):
    """Krum Byzantine-resilient aggregator.
    
    Krum selects the update that minimizes the sum of squared distances
    to its nearest n-f-2 neighbors, where f is the number of Byzantine
    clients.
    
    Reference: Blanchard et al., "Machine Learning with Adversaries:
    Byzantine Tolerant Gradient Descent" (NeurIPS 2017)
    
    Example:
        >>> aggregator = KrumAggregator(num_byzantine=1)
        >>> aggregated = aggregator.aggregate(client_updates)
    """
    
    def __init__(
        self,
        num_byzantine: int = 1,
        multi_krum: bool = True,
    ):
        """Initialize Krum aggregator.
        
        Args:
            num_byzantine: Maximum number of Byzantine clients
            multi_krum: Use multi-Krum (select multiple updates)
        """
        super().__init__(name="Krum")
        self.num_byzantine = num_byzantine
        self.multi_krum = multi_krum
        self.defender = KrumDefender(
            num_byzantine=num_byzantine,
            multi_krum=multi_krum,
        )
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate using Krum.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated weights
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        
        # Filter with Krum
        filtered = self.defender.filter(updates)
        
        # If no updates remain, fall back to average
        if not filtered:
            return self._compute_weighted_average(updates)
            
        # Aggregate filtered updates
        return self._compute_weighted_average(filtered)
    
    def _compute_scores(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> np.ndarray:
        """Compute Krum scores for all updates.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Array of Krum scores
        """
        n = len(updates)
        weights_only = [u[0] for u in updates]
        
        # Compute distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._compute_distance(weights_only[i], weights_only[j])
                distances[i, j] = dist
                distances[j, i] = dist
                
        # Compute scores
        n_neighbors = n - 2 - self.num_byzantine
        scores = np.zeros(n)
        
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            scores[i] = np.sum(sorted_dists[1:n_neighbors + 1])
            
        return scores
        
    def _compute_distance(
        self,
        w1: Dict[str, np.ndarray],
        w2: Dict[str, np.ndarray],
    ) -> float:
        """Compute distance between two weight vectors.
        
        Args:
            w1: First weights
            w2: Second weights
            
        Returns:
            Euclidean distance
        """
        total = 0.0
        for name in w1.keys():
            if name in w2:
                diff = w1[name].flatten() - w2[name].flatten()
                total += np.dot(diff, diff)
        return np.sqrt(total)


class MultiKrumAggregator(KrumAggregator):
    """Multi-Krum aggregator (selects multiple updates)."""
    
    def __init__(self, num_byzantine: int = 1):
        """Initialize Multi-Krum.
        
        Args:
            num_byzantine: Number of Byzantine clients
        """
        super().__init__(num_byzantine=num_byzantine, multi_krum=True)
        self.name = "MultiKrum"
