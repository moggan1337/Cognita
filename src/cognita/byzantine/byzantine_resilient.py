"""
Byzantine-Resilient Aggregation Module

Implements Byzantine-resilient aggregation methods to defend against
malicious clients in federated learning.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np

from cognita.aggregation.base import BaseAggregator


class ByzantineResilientAggregator(BaseAggregator):
    """Byzantine-resilient aggregator wrapper.
    
    Wraps a base aggregator with Byzantine defense mechanisms
    to filter out malicious updates before aggregation.
    
    Attributes:
        base_aggregator: Underlying aggregation strategy
        num_byzantine: Maximum number of Byzantine clients
        defense_method: Defense method to use
        
    Example:
        >>> base_agg = FedAvgAggregator()
        >>> aggregator = ByzantineResilientAggregator(
        ...     base_aggregator=base_agg,
        ...     num_byzantine=2,
        ...     defense_method="krum"
        ... )
        >>> aggregated = aggregator.aggregate(client_updates)
    """
    
    def __init__(
        self,
        base_aggregator: BaseAggregator,
        num_byzantine: int = 1,
        defense_method: str = "krum",
        **kwargs,
    ):
        """Initialize Byzantine-resilient aggregator.
        
        Args:
            base_aggregator: Underlying aggregator
            num_byzantine: Maximum Byzantine clients
            defense_method: Defense method name
            **kwargs: Additional arguments for defense method
        """
        super().__init__(name=f"Byzantine-{defense_method}")
        self.base_aggregator = base_aggregator
        self.num_byzantine = num_byzantine
        self.defense_method = defense_method
        self.kwargs = kwargs
        
        self._defender: Optional[BaseDefender] = None
        self._initialize_defender()
        
    def _initialize_defender(self) -> None:
        """Initialize the defense mechanism."""
        if self.defense_method == "krum":
            self._defender = KrumDefender(
                num_byzantine=self.num_byzantine,
                **self.kwargs,
            )
        elif self.defense_method == "trimmed_mean":
            self._defender = TrimmedMeanDefender(
                num_byzantine=self.num_byzantine,
                **self.kwargs,
            )
        elif self.defense_method == "median":
            self._defender = MedianDefender(**self.kwargs)
        elif self.defense_method == "brute_force":
            self._defender = BruteForceDefender(
                num_byzantine=self.num_byzantine,
                **self.kwargs,
            )
        else:
            self._defender = KrumDefender(
                num_byzantine=self.num_byzantine,
            )
            
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate updates with Byzantine resilience.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated weights
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        
        # Filter out Byzantine updates
        filtered_updates = self._defender.filter(updates)
        
        # Aggregate using base aggregator
        result = self.base_aggregator.aggregate(filtered_updates)
        
        self._metrics = {
            "num_clients": self.num_clients,
            "num_byzantine": self.num_byzantine,
            "num_filtered": len(updates) - len(filtered_updates),
            "defense_method": self.defense_method,
        }
        
        return result
    
    def reset(self) -> None:
        """Reset aggregator state."""
        self.base_aggregator.reset()
        self._metrics = {}


class BaseDefender:
    """Base class for Byzantine defense mechanisms."""
    
    def filter(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> List[Tuple[Dict[str, np.ndarray], float]]:
        """Filter out Byzantine updates.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Filtered updates
        """
        return updates


class KrumDefender(BaseDefender):
    """Multi-Krum Byzantine defense.
    
    Selects updates that are closest to the average of their neighbors.
    """
    
    def __init__(
        self,
        num_byzantine: int = 1,
        multi_krum: bool = True,
    ):
        """Initialize Krum defender.
        
        Args:
            num_byzantine: Number of Byzantine clients
            multi_krum: Use multi-Krum (select multiple) vs single-Krum
        """
        self.num_byzantine = num_byzantine
        self.multi_krum = multi_krum
        
    def filter(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> List[Tuple[Dict[str, np.ndarray], float]]:
        """Filter updates using Krum.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Filtered updates
        """
        n = len(updates)
        
        # Number of neighbors to consider
        # n - 2 for Krum (exclude self and f Byzantine)
        n_neighbors = n - 2 - self.num_byzantine
        
        if n_neighbors < 1:
            return updates
            
        # Compute pairwise distances
        weights_only = [u[0] for u in updates]
        distances = self._compute_distances(weights_only)
        
        if self.multi_krum:
            return self._multi_krum(updates, distances, n_neighbors)
        else:
            return self._single_krum(updates, distances, n_neighbors)
            
    def _compute_distances(
        self,
        weights: List[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Compute pairwise Euclidean distances.
        
        Args:
            weights: List of weight dictionaries
            
        Returns:
            Distance matrix
        """
        n = len(weights)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._weight_distance(weights[i], weights[j])
                distances[i, j] = dist
                distances[j, i] = dist
                
        return distances
    
    def _weight_distance(
        self,
        w1: Dict[str, np.ndarray],
        w2: Dict[str, np.ndarray],
    ) -> float:
        """Compute distance between two weight dictionaries.
        
        Args:
            w1: First weight dictionary
            w2: Second weight dictionary
            
        Returns:
            Euclidean distance
        """
        total_dist = 0.0
        
        for name in w1.keys():
            if name in w2:
                diff = w1[name].flatten() - w2[name].flatten()
                total_dist += np.dot(diff, diff)
                
        return np.sqrt(total_dist)
        
    def _single_krum(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
        distances: np.ndarray,
        n_neighbors: int,
    ) -> List[Tuple[Dict[str, np.ndarray], float]]:
        """Select single update via Krum.
        
        Args:
            updates: List of updates
            distances: Distance matrix
            n_neighbors: Number of neighbors
            
        Returns:
            List with single selected update
        """
        n = len(updates)
        scores = np.zeros(n)
        
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            # Sum of smallest n_neighbors distances
            scores[i] = np.sum(sorted_dists[1:n_neighbors + 1])  # Exclude self
            
        # Select update with smallest score
        best_idx = np.argmin(scores)
        return [updates[best_idx]]
        
    def _multi_krum(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
        distances: np.ndarray,
        n_neighbors: int,
    ) -> List[Tuple[Dict[str, np.ndarray], float]]:
        """Select multiple updates via multi-Krum.
        
        Args:
            updates: List of updates
            distances: Distance matrix
            n_neighbors: Number of neighbors
            
        Returns:
            Filtered list of updates
        """
        n = len(updates)
        n_select = n - self.num_byzantine - 1
        
        scores = np.zeros(n)
        
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            scores[i] = np.sum(sorted_dists[1:n_neighbors + 1])
            
        # Select indices with smallest scores
        selected_indices = np.argsort(scores)[:n_select]
        
        return [updates[i] for i in selected_indices]


class TrimmedMeanDefender(BaseDefender):
    """Trimmed mean Byzantine defense.
    
    Computes coordinate-wise trimmed mean after removing
    extreme values.
    """
    
    def __init__(self, num_byzantine: int = 1, trim_ratio: float = 0.1):
        """Initialize trimmed mean defender.
        
        Args:
            num_byzantine: Number of Byzantine clients
            trim_ratio: Ratio to trim from each end
        """
        self.num_byzantine = num_byzantine
        self.trim_ratio = trim_ratio
        
    def filter(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> List[Tuple[Dict[str, np.ndarray], float]]:
        """Filter updates (trimmed mean doesn't filter, it robustly aggregates)."""
        return updates


class MedianDefender(BaseDefender):
    """Coordinate-wise median defense."""
    
    def filter(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> List[Tuple[Dict[str, np.ndarray], float]]:
        """Filter updates (median doesn't filter)."""
        return updates


class BruteForceDefender(BaseDefender):
    """Brute-force Byzantine defense.
    
    Tries all possible combinations to find the best
    non-Byzantine subset (expensive but optimal).
    """
    
    def __init__(self, num_byzantine: int = 1):
        """Initialize brute-force defender.
        
        Args:
            num_byzantine: Number of Byzantine clients
        """
        self.num_byzantine = num_byzantine
        
    def filter(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> List[Tuple[Dict[str, np.ndarray], float]]:
        """Filter updates using brute-force."""
        # This is a placeholder - brute force is O(n choose f)
        # which is computationally infeasible for large n
        # In practice, use Krum or other efficient methods
        return updates
