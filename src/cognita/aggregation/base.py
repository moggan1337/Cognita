"""
Base Aggregation Module

Defines the abstract base class for federated learning aggregation strategies.

Author: Cognita Team
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
import numpy as np


class BaseAggregator(ABC):
    """Abstract base class for federated learning aggregators.
    
    All aggregation strategies should inherit from this class and
    implement the aggregate method.
    
    Attributes:
        name: Aggregator name
        num_clients: Number of clients for last aggregation
        
    Example:
        >>> class MyAggregator(BaseAggregator):
        ...     def aggregate(self, updates):
        ...         # Custom aggregation logic
        ...         return aggregated_weights
    """
    
    def __init__(self, name: str = "base"):
        """Initialize the aggregator.
        
        Args:
            name: Aggregator name
        """
        self.name = name
        self.num_clients = 0
        self._metrics: Dict[str, Any] = {}
        
    @abstractmethod
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate client updates.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated weights dictionary
        """
        pass
    
    def reset(self) -> None:
        """Reset aggregator state."""
        self.num_clients = 0
        self._metrics = {}
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregation metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self._metrics.copy()
    
    def _compute_weighted_average(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Compute weighted average of updates.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Weighted average weights
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        total_weight = sum(w for _, w in updates)
        
        result = {}
        param_names = updates[0][0].keys()
        
        for name in param_names:
            weighted_sum = np.zeros_like(updates[0][0][name], dtype=np.float64)
            for weights, weight in updates:
                weighted_sum += weights[name].astype(np.float64) * (weight / total_weight)
            result[name] = weighted_sum.astype(np.float32)
            
        return result
    
    def _compute_median(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Compute element-wise median of updates.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Median weights
        """
        if not updates:
            return {}
            
        weights_only = [u[0] for u in updates]
        param_names = weights_only[0].keys()
        
        result = {}
        for name in param_names:
            stacked = np.stack([w[name] for w in weights_only])
            result[name] = np.median(stacked, axis=0)
            
        return result
    
    def _computeTrimmedMean(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
        trim_ratio: float = 0.1,
    ) -> Dict[str, np.ndarray]:
        """Compute trimmed mean of updates.
        
        Args:
            updates: List of (weights, weight) tuples
            trim_ratio: Ratio of values to trim from each end
            
        Returns:
            Trimmed mean weights
        """
        if not updates:
            return {}
            
        weights_only = [u[0] for u in updates]
        num_trim = max(1, int(len(updates) * trim_ratio))
        param_names = weights_only[0].keys()
        
        result = {}
        for name in param_names:
            stacked = np.stack([w[name] for w in weights_only])
            # Sort along client axis
            sorted_indices = np.argsort(stacked, axis=0)
            # Trim from both ends
            trimmed = np.take_along_axis(
                stacked, sorted_indices[num_trim:-num_trim], axis=0
            ) if num_trim > 0 else stacked
            result[name] = np.mean(trimmed, axis=0)
            
        return result
