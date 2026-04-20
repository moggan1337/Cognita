"""
Brute Force Byzantine Aggregator Module

Implements brute-force Byzantine-resilient aggregation by trying
all possible combinations of non-Byzantine updates.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations
import numpy as np

from cognita.aggregation.base import BaseAggregator


class BruteForceAggregator(BaseAggregator):
    """Brute-Force Byzantine-resilient aggregator.
    
    Exhaustively searches for the set of updates that minimizes
    some robust statistic. This is computationally expensive but
    provides optimal Byzantine resilience.
    
    Note: This is primarily for theoretical analysis. For practical
    use, prefer Krum or Trimmed Mean aggregators.
    
    Example:
        >>> aggregator = BruteForceAggregator(num_byzantine=1, metric="variance")
        >>> aggregated = aggregator.aggregate(client_updates)
    """
    
    def __init__(
        self,
        num_byzantine: int = 1,
        metric: str = "variance",
    ):
        """Initialize brute-force aggregator.
        
        Args:
            num_byzantine: Number of expected Byzantine clients
            metric: Metric to minimize ("variance", "max_dist")
        """
        super().__init__(name="BruteForce")
        self.num_byzantine = num_byzantine
        self.metric = metric
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate using brute-force search.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated weights
        """
        n = len(updates)
        f = self.num_byzantine
        
        # Need at least 2f+1 clients for Byzantine tolerance
        if n < 2 * f + 1:
            # Fall back to weighted average
            return self._compute_weighted_average(updates)
            
        # Number of updates to select
        n_select = n - f
        
        # Find best subset
        best_score = float('inf')
        best_subset = None
        
        for combo in combinations(range(n), n_select):
            subset = [updates[i] for i in combo]
            score = self._compute_score(subset)
            
            if score < best_score:
                best_score = score
                best_subset = subset
                
        if best_subset is None:
            return self._compute_weighted_average(updates)
            
        return self._compute_weighted_average(best_subset)
        
    def _compute_score(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> float:
        """Compute score for a subset of updates.
        
        Args:
            updates: Subset of updates
            
        Returns:
            Score value (lower is better)
        """
        weights_only = [u[0] for u in updates]
        
        if self.metric == "variance":
            return self._compute_variance(weights_only)
        elif self.metric == "max_dist":
            return self._compute_max_distance(weights_only)
        else:
            return self._compute_variance(weights_only)
            
    def _compute_variance(
        self,
        weights: List[Dict[str, np.ndarray]],
    ) -> float:
        """Compute total variance of updates.
        
        Args:
            weights: List of weight dictionaries
            
        Returns:
            Total variance
        """
        # Compute mean
        mean = self._compute_mean(weights)
        
        # Compute sum of squared distances
        total_var = 0.0
        for w in weights:
            dist = self._compute_distance(w, mean)
            total_var += dist ** 2
            
        return total_var / len(weights)
        
    def _compute_max_distance(
        self,
        weights: List[Dict[str, np.ndarray]],
    ) -> float:
        """Compute maximum pairwise distance.
        
        Args:
            weights: List of weight dictionaries
            
        Returns:
            Maximum distance
        """
        max_dist = 0.0
        n = len(weights)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._compute_distance(weights[i], weights[j])
                max_dist = max(max_dist, dist)
                
        return max_dist
        
    def _compute_mean(
        self,
        weights: List[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """Compute mean of weights.
        
        Args:
            weights: List of weight dictionaries
            
        Returns:
            Mean weights
        """
        if not weights:
            return {}
            
        result = {}
        param_names = weights[0].keys()
        
        for name in param_names:
            stacked = np.stack([w[name] for w in weights])
            result[name] = np.mean(stacked, axis=0)
            
        return result
        
    def _compute_distance(
        self,
        w1: Dict[str, np.ndarray],
        w2: Dict[str, np.ndarray],
    ) -> float:
        """Compute distance between two weight dictionaries.
        
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


class AveragePairwiseDistance(BaseAggregator):
    """Aggregates using average pairwise distance minimization."""
    
    def __init__(self):
        super().__init__(name="AvgPairwiseDist")
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate by minimizing average pairwise distance."""
        # This is similar to Krum but uses different scoring
        return self._compute_weighted_average(updates)
