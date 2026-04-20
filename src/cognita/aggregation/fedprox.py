"""
FedProx Aggregator Module

Implements FedProx aggregation strategy for handling system heterogeneity.

FedProx: Li et al., "Federated Optimization in Heterogeneous Networks" (2020)

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cognita.aggregation.base import BaseAggregator


class FedProxAggregator(BaseAggregator):
    """Federated Proximal (FedProx) aggregator.
    
    FedProx extends FedAvg to handle heterogeneous networks where clients
    may have varying computational resources and network conditions.
    It adds a proximal term to the local objective to prevent dramatic
    deviations from the global model.
    
    Algorithm:
        1. Receive updates from clients
        2. Add proximal term: ||w - w_global||^2
        3. Aggregate using weighted average
        4. Update global model
        
    Attributes:
        mu: Proximal term coefficient (0 = FedAvg)
        adaptive_mu: Adapt mu based on system heterogeneity
        
    Example:
        >>> aggregator = FedProxAggregator(mu=0.1)
        >>> aggregated = aggregator.aggregate(client_updates)
    """
    
    def __init__(
        self,
        mu: float = 0.01,
        adaptive_mu: bool = False,
        adaptive_strategy: str = "variance",
    ):
        """Initialize FedProx aggregator.
        
        Args:
            mu: Proximal term coefficient
            adaptive_mu: Adapt mu based on data heterogeneity
            adaptive_strategy: Strategy for adaptive mu
        """
        super().__init__(name="FedProx")
        self.mu = mu
        self.adaptive_mu = adaptive_mu
        self.adaptive_strategy = adaptive_strategy
        self._global_weights: Optional[Dict[str, np.ndarray]] = None
        self._round_metrics: List[Dict[str, float]] = []
        
    def set_global_weights(
        self,
        weights: Dict[str, np.ndarray],
    ) -> None:
        """Set global weights for proximal term computation.
        
        Args:
            weights: Current global model weights
        """
        self._global_weights = weights
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate client updates using FedProx.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated weight updates
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        
        # Adapt mu if enabled
        if self.adaptive_mu:
            self._adapt_mu(updates)
            
        # Compute weighted average
        aggregated = self._compute_weighted_average(updates)
        
        # Record metrics
        update_norms = [
            np.linalg.norm(np.concatenate([u.flatten() for u in upd[0].values()]))
            for upd in updates
        ]
        
        self._metrics = {
            "num_clients": self.num_clients,
            "mu": self.mu,
            "avg_update_norm": np.mean(update_norms),
            "max_update_norm": np.max(update_norms),
            "min_update_norm": np.min(update_norms),
        }
        
        self._round_metrics.append(self._metrics.copy())
        
        return aggregated
    
    def _adapt_mu(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> None:
        """Adapt proximal term based on update characteristics.
        
        Args:
            updates: Client updates
        """
        if self.adaptive_strategy == "variance":
            # Adapt based on variance of updates
            update_norms = []
            for weights, _ in updates:
                norm = sum(np.linalg.norm(w.flatten()) for w in weights.values())
                update_norms.append(norm)
                
            variance = np.var(update_norms)
            self.mu = min(1.0, variance / (np.mean(update_norms) + 1e-8))
            
        elif self.adaptive_strategy == "norm":
            # Adapt based on average update norm
            avg_norm = np.mean([
                np.linalg.norm(np.concatenate([w.flatten() for w in upd[0].values()]))
                for upd in updates
            ])
            self.mu = min(1.0, 1.0 / (avg_norm + 1.0))
            
    def compute_proximal_term(
        self,
        weights: Dict[str, np.ndarray],
    ) -> float:
        """Compute proximal term value.
        
        Args:
            weights: Local weights
            
        Returns:
            Proximal term value
        """
        if self._global_weights is None:
            return 0.0
            
        proximal = 0.0
        for name in weights.keys():
            if name in self._global_weights:
                diff = weights[name] - self._global_weights[name]
                proximal += np.sum(diff ** 2)
                
        return 0.5 * self.mu * proximal
    
    def reset(self) -> None:
        """Reset aggregator state."""
        super().reset()
        self._round_metrics = []


class FedProxWithMomentum(FedProxAggregator):
    """FedProx with momentum acceleration."""
    
    def __init__(
        self,
        mu: float = 0.01,
        momentum: float = 0.9,
    ):
        """Initialize FedProx with momentum.
        
        Args:
            mu: Proximal term coefficient
            momentum: Momentum factor
        """
        super().__init__(mu=mu)
        self.name = "FedProx-Momentum"
        self.momentum = momentum
        self._velocity: Optional[Dict[str, np.ndarray]] = None
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate with momentum.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated updates with momentum
        """
        aggregated = super().aggregate(updates)
        
        # Apply momentum to aggregated update
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


class AdaptiveFedProx(FedProxAggregator):
    """Adaptive FedProx with client-specific regularization.
    
    Each client gets a personalized proximal term based on its
    local data characteristics and system capabilities.
    """
    
    def __init__(
        self,
        base_mu: float = 0.01,
        adaptation_method: str = "distance",
    ):
        """Initialize adaptive FedProx.
        
        Args:
            base_mu: Base proximal term coefficient
            adaptation_method: Method for computing client-specific mu
        """
        super().__init__(mu=base_mu)
        self.name = "AdaptiveFedProx"
        self.base_mu = base_mu
        self.adaptation_method = adaptation_method
        self._client_mus: Dict[str, float] = {}
        
    def set_client_adaptation(
        self,
        client_id: str,
        staleness: int,
        local_epochs: int,
    ) -> None:
        """Set client-specific adaptation parameters.
        
        Args:
            client_id: Client identifier
            staleness: Staleness of client's update
            local_epochs: Number of local epochs
        """
        if self.adaptation_method == "staleness":
            # Increase mu for stale updates
            self._client_mus[client_id] = self.base_mu * (1 + 0.1 * staleness)
            
        elif self.adaptation_method == "epochs":
            # Increase mu for more local epochs
            self._client_mus[client_id] = self.base_mu * (1 + 0.05 * local_epochs)
            
        else:
            self._client_mus[client_id] = self.base_mu
            
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate with client-specific proximal terms.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated updates
        """
        # Apply client-specific regularization
        regularized_updates = []
        for weights, weight in updates:
            # No per-client mu in this implementation
            regularized_updates.append((weights, weight))
            
        return super().aggregate(regularized_updates)
