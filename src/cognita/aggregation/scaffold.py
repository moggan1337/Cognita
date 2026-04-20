"""
SCAFFOLD Aggregator Module

Implements SCAFFOLD (Stochastic Controlled Averaging for Federated Learning)
for variance reduction and improved convergence.

SCAFFOLD: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for
Federated Learning" (2020)

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cognita.aggregation.base import BaseAggregator


class SCAFFOLDAggregator(BaseAggregator):
    """SCAFFOLD (Stochastic Controlled Averaging) aggregator.
    
    SCAFFOLD addresses the problem of client drift in federated learning
    by maintaining control variates (server and client controls) that
    reduce variance and accelerate convergence.
    
    Algorithm:
        1. Maintain server control c and client controls c_i
        2. Clients compute: y_i = w_i - c_i (drift correction)
        3. Aggregate: w = sum_i(p_i * y_i) + c
        4. Update server control
        
    Attributes:
        learning_rate: Server learning rate
        control_lr: Control variate learning rate
        
    Example:
        >>> aggregator = SCAFFOLDAggregator(learning_rate=1.0)
        >>> 
        >>> # For each round:
        >>> aggregator.set_global_weights(global_weights)
        >>> 
        >>> # After client training:
        >>> for client in clients:
        ...     client_update = aggregator.process_client_update(
        ...         client, 
        ...         local_weights,
        ...         control_variate
        ...     )
        >>> 
        >>> aggregated = aggregator.aggregate_updates(client_updates)
    """
    
    def __init__(
        self,
        learning_rate: float = 1.0,
        control_lr: float = 1.0,
        momentum: float = 0.0,
    ):
        """Initialize SCAFFOLD aggregator.
        
        Args:
            learning_rate: Server learning rate
            control_lr: Control variate learning rate
            momentum: Momentum for aggregation
        """
        super().__init__(name="SCAFFOLD")
        self.learning_rate = learning_rate
        self.control_lr = control_lr
        self.momentum = momentum
        
        self._server_control: Optional[Dict[str, np.ndarray]] = None
        self._client_controls: Dict[str, Dict[str, np.ndarray]] = {}
        self._prev_weights: Optional[Dict[str, np.ndarray]] = None
        self._num_rounds = 0
        
    def initialize_controls(
        self,
        weight_shapes: Dict[str, Tuple[int, ...]],
    ) -> None:
        """Initialize control variates.
        
        Args:
            weight_shapes: Shapes of model weight tensors
        """
        self._server_control = {
            name: np.zeros(shape) for name, shape in weight_shapes.items()
        }
        
    def register_client(
        self,
        client_id: str,
        weight_shapes: Dict[str, Tuple[int, ...]],
    ) -> Dict[str, np.ndarray]:
        """Register a client and initialize its control.
        
        Args:
            client_id: Client identifier
            weight_shapes: Shapes of model weight tensors
            
        Returns:
            Client's control variate
        """
        self._client_controls[client_id] = {
            name: np.zeros(shape) for name, shape in weight_shapes.items()
        }
        return self._client_controls[client_id]
        
    def set_global_weights(
        self,
        weights: Dict[str, np.ndarray],
    ) -> None:
        """Set global model weights.
        
        Args:
            weights: Current global weights
        """
        self._prev_weights = weights.copy()
        
    def process_client_update(
        self,
        client_id: str,
        local_weights: Dict[str, np.ndarray],
        num_samples: int,
    ) -> Dict[str, Any]:
        """Process a client's update and compute control correction.
        
        Args:
            client_id: Client identifier
            local_weights: Client's local model weights
            num_samples: Number of local training samples
            
        Returns:
            Dictionary with drift-corrected update and control delta
        """
        if self._prev_weights is None or self._server_control is None:
            return {"weights": local_weights, "control_delta": {}}
            
        # Ensure client control exists
        if client_id not in self._client_controls:
            self._client_controls[client_id] = {
                name: np.zeros_like(arr) for name, arr in local_weights.items()
            }
            
        client_control = self._client_controls[client_id]
        server_control = self._server_control
        
        # Compute drift-corrected update
        corrected_weights = {}
        control_delta = {}
        
        for name in local_weights.keys():
            # Compute weight delta
            delta_w = local_weights[name] - self._prev_weights.get(name, np.zeros_like(local_weights[name]))
            
            # Compute control delta
            c_delta = delta_w - np.mean([
                local_weights[name] - self._prev_weights.get(name, np.zeros_like(local_weights[name]))
                for _ in range(1)  # Simplified for single client
            ], axis=0)
            
            # Apply correction: w - c_i + c
            corrected_weights[name] = local_weights[name] - client_control.get(name, np.zeros_like(local_weights[name])) + server_control.get(name, np.zeros_like(local_weights[name]))
            
            control_delta[name] = c_delta
            
        return {
            "weights": corrected_weights,
            "control_delta": control_delta,
            "num_samples": num_samples,
        }
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate client updates using SCAFFOLD.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated weight updates
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        total_weight = sum(w for _, w in updates)
        
        # Weighted average of corrected weights
        aggregated = {}
        param_names = updates[0][0].keys()
        
        for name in param_names:
            weighted_sum = np.zeros_like(updates[0][0][name], dtype=np.float64)
            for weights, weight in updates:
                weighted_sum += weights[name].astype(np.float64) * (weight / total_weight)
            aggregated[name] = weighted_sum.astype(np.float32)
            
        # Update server control
        self._update_server_control(updates, total_weight)
        
        self._num_rounds += 1
        
        self._metrics = {
            "num_clients": self.num_clients,
            "learning_rate": self.learning_rate,
            "num_rounds": self._num_rounds,
        }
        
        return aggregated
    
    def _update_server_control(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
        total_weight: float,
    ) -> None:
        """Update server control variate.
        
        Args:
            updates: Client updates with control deltas
            total_weight: Total weight of updates
        """
        if self._server_control is None:
            return
            
        # Compute average control delta
        for name in self._server_control.keys():
            avg_delta = np.zeros_like(self._server_control[name], dtype=np.float64)
            
            for weights, weight in updates:
                if name in weights:
                    avg_delta += weights[name].astype(np.float64) * (weight / total_weight)
                    
            # Update server control
            self._server_control[name] = (
                self._server_control[name] - self.control_lr * avg_delta
            ).astype(np.float32)
            
    def get_server_control(self) -> Optional[Dict[str, np.ndarray]]:
        """Get current server control variate.
        
        Returns:
            Server control dictionary
        """
        return self._server_control.copy() if self._server_control else None
        
    def get_client_control(self, client_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Get a client's control variate.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Client control dictionary
        """
        return self._client_controls.get(client_id, {}).copy()
        
    def reset(self) -> None:
        """Reset aggregator state."""
        super().reset()
        self._server_control = None
        self._client_controls = {}
        self._prev_weights = None
        self._num_rounds = 0


class SCAFFOLDWithMomentum(SCAFFOLDAggregator):
    """SCAFFOLD with momentum acceleration."""
    
    def __init__(
        self,
        learning_rate: float = 1.0,
        momentum: float = 0.9,
    ):
        """Initialize SCAFFOLD with momentum.
        
        Args:
            learning_rate: Server learning rate
            momentum: Momentum factor
        """
        super().__init__(learning_rate=learning_rate, momentum=momentum)
        self.name = "SCAFFOLD-Momentum"
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


class VRLSCAFFOLD(SCAFFOLDAggregator):
    """SCAFFOLD with Variance Reduction.
    
    Enhanced version of SCAFFOLD with additional variance reduction
    techniques for more stable convergence.
    """
    
    def __init__(
        self,
        learning_rate: float = 1.0,
        vr_decay: float = 0.9,
        control_lr: float = 1.0,
    ):
        """Initialize VRL-SCAFFOLD.
        
        Args:
            learning_rate: Server learning rate
            vr_decay: Variance reduction decay factor
            control_lr: Control learning rate
        """
        super().__init__(learning_rate=learning_rate, control_lr=control_lr)
        self.name = "VRL-SCAFFOLD"
        self.vr_decay = vr_decay
        self._vr_buffer: Optional[Dict[str, np.ndarray]] = None
        
    def _apply_variance_reduction(
        self,
        aggregated: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Apply variance reduction to aggregated update.
        
        Args:
            aggregated: Raw aggregated update
            
        Returns:
            Variance-reduced update
        """
        if self._vr_buffer is None:
            self._vr_buffer = {k: np.zeros_like(v) for k, v in aggregated.items()}
            
        result = {}
        for name in aggregated.keys():
            # Exponential moving average of updates
            self._vr_buffer[name] = (
                self.vr_decay * self._vr_buffer[name] +
                (1 - self.vr_decay) * aggregated[name]
            )
            result[name] = self._vr_buffer[name]
            
        return result
    
    def reset(self) -> None:
        """Reset aggregator state."""
        super().reset()
        self._vr_buffer = None
