"""
Trimmed Mean Aggregator Module

Implements coordinate-wise trimmed mean for Byzantine-resilient aggregation.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cognita.aggregation.base import BaseAggregator


class TrimmedMeanAggregator(BaseAggregator):
    """Trimmed Mean Byzantine-resilient aggregator.
    
    Trimmed mean removes a fraction of the largest and smallest values
    at each coordinate before computing the mean, providing robustness
    against Byzantine attacks.
    
    Reference: Dilmaghani et al., "Byzantine-Robust Distributed
    Learning" (NeurIPS 2019)
    
    Example:
        >>> aggregator = TrimmedMeanAggregator(num_byzantine=1, trim_ratio=0.1)
        >>> aggregated = aggregator.aggregate(client_updates)
    """
    
    def __init__(
        self,
        num_byzantine: int = 1,
        trim_ratio: Optional[float] = None,
    ):
        """Initialize trimmed mean aggregator.
        
        Args:
            num_byzantine: Number of Byzantine clients
            trim_ratio: Fraction to trim from each end (auto-computed if None)
        """
        super().__init__(name="TrimmedMean")
        self.num_byzantine = num_byzantine
        self.trim_ratio = trim_ratio
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate using trimmed mean.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated weights
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        
        # Auto-compute trim ratio if needed
        trim_ratio = self.trim_ratio
        if trim_ratio is None:
            trim_ratio = self.num_byzantine / self.num_clients
            
        # Compute trim count
        n = len(updates)
        n_trim = max(1, int(n * trim_ratio))
        
        weights_only = [u[0] for u in updates]
        
        # Aggregate parameter by parameter
        result = {}
        param_names = weights_only[0].keys()
        
        for name in param_names:
            # Stack all values for this parameter
            param_values = np.stack([w[name] for w in weights_only], axis=0)
            
            # Apply weighted trimmed mean
            trimmed = self._trimmed_mean(
                param_values,
                n_trim,
                [u[1] for u in updates],
            )
            result[name] = trimmed
            
        return result
        
    def _trimmed_mean(
        self,
        values: np.ndarray,
        n_trim: int,
        weights: List[float],
    ) -> np.ndarray:
        """Compute weighted trimmed mean.
        
        Args:
            values: Array of shape (n_clients, *param_shape)
            n_trim: Number of values to trim from each end
            weights: Client weights
            
        Returns:
            Trimmed mean result
        """
        n = values.shape[0]
        
        if n <= 2 * n_trim:
            # Not enough values to trim, use regular mean
            return np.mean(values, axis=0)
            
        # Flatten for sorting
        original_shape = values.shape[1:]
        flat_values = values.reshape(n, -1)
        
        # Sort along client axis
        sorted_indices = np.argsort(flat_values, axis=0)
        
        # Trim from both ends
        trim_start = n_trim
        trim_end = n - n_trim
        
        # Get trimmed values using advanced indexing
        result = np.zeros(flat_values.shape[1])
        
        for j in range(flat_values.shape[1]):
            col_values = flat_values[:, j]
            col_sorted = np.sort(col_values)
            result[j] = np.mean(col_sorted[trim_start:trim_end])
            
        return result.reshape(original_shape)


class CoordinateWiseMedian(BaseAggregator):
    """Coordinate-wise median aggregator."""
    
    def __init__(self):
        """Initialize median aggregator."""
        super().__init__(name="Median")
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate using coordinate-wise median.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated weights
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        
        weights_only = [u[0] for u in updates]
        
        result = {}
        param_names = weights_only[0].keys()
        
        for name in param_names:
            param_values = np.stack([w[name] for w in weights_only], axis=0)
            result[name] = np.median(param_values, axis=0)
            
        return result
