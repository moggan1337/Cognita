"""
FedAvg Aggregator Module

Implements Federated Averaging (FedAvg) aggregation strategy.

FedAvg: McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data" (2017)

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cognita.aggregation.base import BaseAggregator


class FedAvgAggregator(BaseAggregator):
    """Federated Averaging (FedAvg) aggregator.
    
    FedAvg is the most commonly used aggregation strategy in federated
    learning. It computes a weighted average of client model updates,
    where the weights are typically proportional to the number of samples
    each client has.
    
    Algorithm:
        1. Receive model updates from n clients
        2. Compute weights w_i for each client (usually based on data size)
        3. Aggregate: w = sum_i(w_i * delta_i) / sum_i(w_i)
        4. Update global model: w_global += aggregated_delta
        
    Attributes:
        momentum: Momentum for exponential moving average
        server_momentum: Apply momentum at server side
        
    Example:
        >>> aggregator = FedAvgAggregator(momentum=0.9)
        >>> aggregated = aggregator.aggregate([
        ...     (client1_weights, 100),  # (weights, num_samples)
        ...     (client2_weights, 200),
        ...     (client3_weights, 150),
        ... ])
    """
    
    def __init__(
        self,
        momentum: float = 0.0,
        server_momentum: bool = False,
    ):
        """Initialize FedAvg aggregator.
        
        Args:
            momentum: Local gradient momentum
            server_momentum: Apply momentum at server side
        """
        super().__init__(name="FedAvg")
        self.momentum = momentum
        self.server_momentum = server_momentum
        self._momentum_buffer: Optional[Dict[str, np.ndarray]] = None
        self._num_rounds = 0
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate client updates using FedAvg.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated weight updates
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        
        # Compute weighted average
        aggregated = self._compute_weighted_average(updates)
        
        # Apply server-side momentum if enabled
        if self.server_momentum and self.momentum > 0:
            aggregated = self._apply_momentum(aggregated)
            
        self._num_rounds += 1
        
        # Record metrics
        self._metrics = {
            "num_clients": self.num_clients,
            "total_weight": sum(w for _, w in updates),
            "avg_weight": np.mean([w for _, w in updates]),
        }
        
        return aggregated
    
    def _apply_momentum(
        self,
        delta: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Apply momentum to delta updates.
        
        Args:
            delta: Current round delta
            
        Returns:
            Momentum-adjusted delta
        """
        if self._momentum_buffer is None:
            self._momentum_buffer = {
                name: np.zeros_like(arr) for name, arr in delta.items()
            }
            
        result = {}
        for name in delta.keys():
            # Update momentum buffer
            self._momentum_buffer[name] = (
                self.momentum * self._momentum_buffer[name] + delta[name]
            )
            result[name] = self._momentum_buffer[name]
            
        return result
    
    def reset(self) -> None:
        """Reset aggregator state including momentum buffer."""
        super().reset()
        self._momentum_buffer = None
        self._num_rounds = 0


class FedAvgMBAggregator(FedAvgAggregator):
    """FedAvg with Mini-Batch updates.
    
    Variant of FedAvg that handles mini-batch style updates
    where gradients are accumulated over multiple batches.
    """
    
    def __init__(
        self,
        momentum: float = 0.0,
        normalize: bool = True,
    ):
        """Initialize mini-batch FedAvg aggregator.
        
        Args:
            momentum: Momentum factor
            normalize: Normalize by number of local steps
        """
        super().__init__(momentum=momentum)
        self.name = "FedAvg-MB"
        self.normalize = normalize
        self._local_steps: List[int] = []
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate mini-batch updates.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated updates
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        
        # Extract local steps if provided in metadata
        # For now, use equal normalization
        num_updates = len(updates)
        
        result = {}
        param_names = updates[0][0].keys()
        
        for name in param_names:
            total_weight = 0.0
            weighted_sum = np.zeros_like(updates[0][0][name], dtype=np.float64)
            
            for weights, weight in updates:
                weighted_sum += weights[name].astype(np.float64) * weight
                total_weight += weight
                
            if total_weight > 0:
                result[name] = (weighted_sum / total_weight).astype(np.float32)
            else:
                result[name] = weighted_sum.astype(np.float32)
                
        return result


class FedAvgMomentumAggregator(FedAvgAggregator):
    """FedAvg with client-side momentum.
    
    Implements FedAvg where momentum is applied on the client
    side before sending updates to the server.
    """
    
    def __init__(
        self,
        momentum: float = 0.9,
        nesterov: bool = False,
    ):
        """Initialize momentum FedAvg.
        
        Args:
            momentum: Momentum factor
            nesterov: Use Nesterov accelerated gradient
        """
        super().__init__(momentum=momentum)
        self.name = "FedAvg-Momentum"
        self.nesterov = nesterov
