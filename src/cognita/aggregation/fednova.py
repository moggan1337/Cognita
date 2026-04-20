"""
FedNova Aggregator Module

Implements FedNova (Federated Normalized Averaging) for accurate
representation of heterogeneous local training epochs.

FedNova: Li et al., "FedNova: Tackling Objective Inconsistency in
Heterogeneous Federated Learning" (2021)

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cognita.aggregation.base import BaseAggregator


class FedNovaAggregator(BaseAggregator):
    """Federated Normalized Averaging (FedNova) aggregator.
    
    FedNova addresses the objective inconsistency problem in heterogeneous
    federated learning by normalizing client updates based on their
    local training工作量 (number of gradient steps).
    
    Algorithm:
        1. Each client performs tau_i local steps
        2. Compute normalized update: delta_i / tau_i
        3. Aggregate: w = sum_i(n_i * delta_i / tau_i) / sum_i(n_i / tau_i)
        
    Attributes:
        tau_eff: Effective number of gradient steps
        
    Example:
        >>> aggregator = FedNovaAggregator()
        >>> # Pass (weights, weight, local_steps) tuples
        >>> aggregated = aggregator.aggregate([
        ...     (weights1, 100, 10),  # (weights, samples, local_steps)
        ...     (weights2, 200, 5),
        ... ])
    """
    
    def __init__(self, normalize: bool = True):
        """Initialize FedNova aggregator.
        
        Args:
            normalize: Normalize by local steps
        """
        super().__init__(name="FedNova")
        self.normalize = normalize
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float, int]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate client updates using FedNova.
        
        Args:
            updates: List of (weights, weight, local_steps) tuples
            
        Returns:
            Aggregated weights
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        
        if self.normalize:
            return self._aggregate_normalized(updates)
        else:
            return self._aggregate_standard(updates)
            
    def _aggregate_normalized(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float, int]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate with normalization by local steps.
        
        Args:
            updates: List of (weights, weight, local_steps)
            
        Returns:
            Normalized aggregated weights
        """
        # Compute normalized weights
        normalized_weights = []
        for weights, num_samples, local_steps in updates:
            # Normalize contribution by local steps
            norm_weight = num_samples / max(local_steps, 1)
            normalized_weights.append((weights, norm_weight))
            
        # Sum of normalized weights
        total_norm_weight = sum(w for _, w in normalized_weights)
        
        if total_norm_weight == 0:
            return {}
            
        result = {}
        param_names = updates[0][0].keys()
        
        for name in param_names:
            weighted_sum = np.zeros_like(updates[0][0][name], dtype=np.float64)
            for (weights, norm_w) in normalized_weights:
                weighted_sum += weights[name].astype(np.float64) * (norm_w / total_norm_weight)
            result[name] = weighted_sum.astype(np.float32)
            
        self._metrics = {
            "num_clients": self.num_clients,
            "total_steps": sum(s for _, _, s in updates),
            "avg_steps": np.mean([s for _, _, s in updates]),
        }
        
        return result
        
    def _aggregate_standard(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float, int]],
    ) -> Dict[str, np.ndarray]:
        """Standard aggregation without normalization.
        
        Args:
            updates: List of (weights, weight, local_steps)
            
        Returns:
            Standard aggregated weights
        """
        # Convert to standard format and use base implementation
        standard_updates = [(w, n) for w, n, _ in updates]
        return self._compute_weighted_average(standard_updates)


class FedNovaWithProximal(FedNovaAggregator):
    """FedNova with proximal term for additional stability."""
    
    def __init__(
        self,
        normalize: bool = True,
        mu: float = 0.01,
    ):
        """Initialize FedNova with proximal term.
        
        Args:
            normalize: Normalize by local steps
            mu: Proximal term coefficient
        """
        super().__init__(normalize=normalize)
        self.name = "FedNova-Prox"
        self.mu = mu
        self._global_weights: Optional[Dict[str, np.ndarray]] = None
        
    def set_global_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set global weights for proximal term.
        
        Args:
            weights: Current global weights
        """
        self._global_weights = weights
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float, int]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate with proximal regularization.
        
        Args:
            updates: List of (weights, weight, local_steps)
            
        Returns:
            Aggregated weights with proximal term
        """
        result = super().aggregate(updates)
        
        # Add proximal correction if global weights available
        if self._global_weights:
            for name in result.keys():
                if name in self._global_weights:
                    # Proximal term correction
                    correction = self.mu * (result[name] - self._global_weights[name])
                    result[name] = result[name] - correction
                    
        return result


class FedNovaWithMomentum(FedNovaAggregator):
    """FedNova with server-side momentum."""
    
    def __init__(
        self,
        normalize: bool = True,
        momentum: float = 0.9,
    ):
        """Initialize FedNova with momentum.
        
        Args:
            normalize: Normalize by local steps
            momentum: Momentum factor
        """
        super().__init__(normalize=normalize)
        self.name = "FedNova-Momentum"
        self.momentum = momentum
        self._velocity: Optional[Dict[str, np.ndarray]] = None
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float, int]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate with momentum.
        
        Args:
            updates: List of (weights, weight, local_steps)
            
        Returns:
            Aggregated weights with momentum
        """
        aggregated = super().aggregate(updates)
        
        # Apply momentum
        if self._velocity is None:
            self._velocity = {k: np.zeros_like(v) for k, v in aggregated.items()}
            
        for name in aggregated.keys():
            self._velocity[name] = (
                self.momentum * self._velocity[name] + aggregated[name]
            )
            
        return self._velocity.copy()
        
    def reset(self) -> None:
        """Reset aggregator state."""
        super().reset()
        self._velocity = None
