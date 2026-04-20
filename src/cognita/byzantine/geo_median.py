"""
Geometric Median Aggregator Module

Implements geometric median for Byzantine-resilient aggregation.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cognita.aggregation.base import BaseAggregator


class GeoMedianAggregator(BaseAggregator):
    """Geometric Median Byzantine-resilient aggregator.
    
    Computes the geometric median (minimizer of sum of distances)
    of client updates, providing strong Byzantine resilience.
    
    Reference: Chen et al., "Distributed Statistical Machine Learning
    in Adversarial Settings" (2017)
    
    Example:
        >>> aggregator = GeoMedianAggregator(max_iter=100, tol=1e-6)
        >>> aggregated = aggregator.aggregate(client_updates)
    """
    
    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        lr: float = 0.1,
    ):
        """Initialize geometric median aggregator.
        
        Args:
            max_iter: Maximum iterations for optimization
            tol: Convergence tolerance
            lr: Learning rate for gradient descent
        """
        super().__init__(name="GeoMedian")
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate using geometric median.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated weights
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        
        # Initialize with weighted average
        initial = self._compute_weighted_average(updates)
        
        # Flatten all weights for optimization
        flattened_updates = []
        for weights, _ in updates:
            flat = self._flatten_weights(weights)
            flattened_updates.append(flat)
            
        initial_flat = self._flatten_weights(initial)
        
        # Optimize geometric median using Weiszfeld's algorithm
        result_flat = self._weiszfeld(flattened_updates, initial_flat)
        
        # Reshape back
        param_shapes = {name: w.shape for name, w in updates[0][0].items()}
        result = self._unflatten_weights(result_flat, param_shapes)
        
        return result
        
    def _flatten_weights(
        self,
        weights: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Flatten weight dictionary to 1D array.
        
        Args:
            weights: Weight dictionary
            
        Returns:
            Flattened array
        """
        arrays = [w.flatten() for w in weights.values()]
        return np.concatenate(arrays)
        
    def _unflatten_weights(
        self,
        flat: np.ndarray,
        shapes: Dict[str, tuple],
    ) -> Dict[str, np.ndarray]:
        """Unflatten array to weight dictionary.
        
        Args:
            flat: Flattened array
            shapes: Parameter shapes
            
        Returns:
            Weight dictionary
        """
        result = {}
        offset = 0
        
        for name, shape in shapes.items():
            size = np.prod(shape)
            result[name] = flat[offset:offset + size].reshape(shape)
            offset += size
            
        return result
        
    def _weiszfeld(
        self,
        points: List[np.ndarray],
        initial: np.ndarray,
    ) -> np.ndarray:
        """Weiszfeld's algorithm for geometric median.
        
        Args:
            points: List of points
            initial: Initial estimate
            
        Returns:
            Geometric median
        """
        x = initial.copy()
        
        for _ in range(self.max_iter):
            # Compute distances
            distances = []
            for p in points:
                d = np.linalg.norm(x - p) + 1e-10
                distances.append(d)
                
            # Compute weights
            weights = [1.0 / d for d in distances]
            weight_sum = sum(weights)
            
            # Compute weighted average
            x_new = sum(w * p for w, p in zip(weights, points)) / weight_sum
            
            # Check convergence
            if np.linalg.norm(x_new - x) < self.tol:
                break
                
            x = x_new
            
        return x


class WeiszfeldAggregator(GeoMedianAggregator):
    """Alias for GeoMedianAggregator using Weiszfeld's algorithm."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Weiszfeld"
