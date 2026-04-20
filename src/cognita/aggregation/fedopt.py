"""
FedOpt Aggregators Module

Implements FedOpt (Federated Optimization) adaptive federated learning
methods including FedAdam, FedAdagrad, and FedYogi.

FedOpt: Reddi et al., "Adaptive Federated Optimization" (2021)

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cognita.aggregation.base import BaseAggregator


class FedAdamAggregator(BaseAggregator):
    """FedAdam (Adaptive Federated Momentums) aggregator.
    
    Adapts the Adam optimizer to the federated learning setting,
    maintaining server-side momentum and second-moment estimates.
    
    Algorithm:
        1. Compute weighted average of client updates
        2. Update server momentum: m_t = beta1 * m_{t-1} + (1-beta1) * delta
        3. Update server second moment: v_t = beta2 * v_{t-1} + (1-beta2) * delta^2
        4. Apply: w = w - lr * m_t / (sqrt(v_t) + epsilon)
        
    Attributes:
        lr: Server learning rate
        beta1: First moment decay
        beta2: Second moment decay
        epsilon: Numerical stability constant
        
    Example:
        >>> aggregator = FedAdamAggregator(lr=0.01, beta1=0.9, beta2=0.99)
        >>> aggregated = aggregator.aggregate(client_updates)
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.99,
        epsilon: float = 1e-8,
        tau: float = 1e-7,
    ):
        """Initialize FedAdam aggregator.
        
        Args:
            lr: Server learning rate
            beta1: First moment decay rate
            beta2: Second moment decay rate
            epsilon: Numerical stability parameter
            tau: Server momentum correction term
        """
        super().__init__(name="FedAdam")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.tau = tau
        
        self._momentum: Optional[Dict[str, np.ndarray]] = None
        self._second_moment: Optional[Dict[str, np.ndarray]] = None
        self._round = 0
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate client updates using FedAdam.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated weight updates
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        self._round += 1
        
        # Compute weighted average of updates
        aggregated = self._compute_weighted_average(updates)
        
        # Initialize momentum buffers if needed
        if self._momentum is None:
            self._momentum = {k: np.zeros_like(v) for k, v in aggregated.items()}
            self._second_moment = {k: np.zeros_like(v) for k, v in aggregated.items()}
            
        # Bias correction
        t = self._round
        beta1_correction = 1 - self.beta1 ** t
        beta2_correction = 1 - self.beta2 ** t
        
        result = {}
        for name in aggregated.keys():
            delta = aggregated[name].astype(np.float64)
            
            # Update first moment
            self._momentum[name] = (
                self.beta1 * self._momentum[name].astype(np.float64) +
                (1 - self.beta1) * delta
            )
            
            # Update second moment
            self._second_moment[name] = (
                self.beta2 * self._second_moment[name].astype(np.float64) +
                (1 - self.beta2) * (delta ** 2)
            )
            
            # Compute corrected estimates
            m_hat = self._momentum[name] / beta1_correction
            v_hat = self._second_moment[name] / beta2_correction
            
            # Apply update with server momentum correction
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon + self.tau)
            result[name] = update.astype(np.float32)
            
        self._metrics = {
            "num_clients": self.num_clients,
            "round": self._round,
            "lr": self.lr,
        }
        
        return result
    
    def reset(self) -> None:
        """Reset aggregator state."""
        super().reset()
        self._momentum = None
        self._second_moment = None
        self._round = 0


class FedAdagradAggregator(BaseAggregator):
    """FedAdagrad (Adaptive Federated Adagrad) aggregator.
    
    Applies Adagrad-style adaptive learning rates to federated learning,
    scaling updates inversely proportional to the square root of
    accumulated gradients.
    
    Algorithm:
        1. Compute weighted average of client updates
        2. Accumulate squared gradients: v_t = v_{t-1} + delta^2
        3. Apply: w = w - lr * delta / (sqrt(v_t) + epsilon)
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        epsilon: float = 1e-8,
        initial_accumulator: float = 0.1,
    ):
        """Initialize FedAdagrad aggregator.
        
        Args:
            lr: Server learning rate
            epsilon: Numerical stability parameter
            initial_accumulator: Initial value for accumulators
        """
        super().__init__(name="FedAdagrad")
        self.lr = lr
        self.epsilon = epsilon
        self.initial_accumulator = initial_accumulator
        
        self._accumulator: Optional[Dict[str, np.ndarray]] = None
        self._round = 0
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate client updates using FedAdagrad.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated weight updates
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        self._round += 1
        
        # Compute weighted average
        aggregated = self._compute_weighted_average(updates)
        
        # Initialize accumulator if needed
        if self._accumulator is None:
            self._accumulator = {
                k: np.full_like(v, self.initial_accumulator)
                for k, v in aggregated.items()
            }
            
        result = {}
        for name in aggregated.keys():
            delta = aggregated[name].astype(np.float64)
            
            # Accumulate squared gradients
            self._accumulator[name] = (
                self._accumulator[name].astype(np.float64) + (delta ** 2)
            )
            
            # Compute adaptive learning rate
            accum_sqrt = np.sqrt(self._accumulator[name])
            
            # Apply update
            update = self.lr * delta / (accum_sqrt + self.epsilon)
            result[name] = update.astype(np.float32)
            
        self._metrics = {
            "num_clients": self.num_clients,
            "round": self._round,
        }
        
        return result
    
    def reset(self) -> None:
        """Reset aggregator state."""
        super().reset()
        self._accumulator = None
        self._round = 0


class FedYogiAggregator(BaseAggregator):
    """FedYogi (Adaptive Federated Yogi) aggregator.
    
    Uses a more aggressive second-moment estimation than FedAdam,
    with sign-based updates for improved convergence.
    
    Algorithm:
        1. Compute weighted average of client updates
        2. Update: v_t = v_{t-1} - (1-beta2) * sign(v_{t-1} - delta^2) * delta^2
        3. Apply Adam-style update
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """Initialize FedYogi aggregator.
        
        Args:
            lr: Server learning rate
            beta1: First moment decay
            beta2: Second moment decay
            epsilon: Numerical stability parameter
        """
        super().__init__(name="FedYogi")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self._momentum: Optional[Dict[str, np.ndarray]] = None
        self._second_moment: Optional[Dict[str, np.ndarray]] = None
        self._round = 0
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate client updates using FedYogi.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated weight updates
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        self._round += 1
        
        # Compute weighted average
        aggregated = self._compute_weighted_average(updates)
        
        # Initialize buffers
        if self._momentum is None:
            self._momentum = {k: np.zeros_like(v) for k, v in aggregated.items()}
            self._second_moment = {k: np.zeros_like(v) + 1e-3 for k, v in aggregated.items()}
            
        # Bias correction
        t = self._round
        beta1_correction = 1 - self.beta1 ** t
        beta2_correction = 1 - self.beta2 ** t
        
        result = {}
        for name in aggregated.keys():
            delta = aggregated[name].astype(np.float64)
            
            # Update first moment (like Adam)
            self._momentum[name] = (
                self.beta1 * self._momentum[name] + (1 - self.beta1) * delta
            )
            
            # Update second moment with Yogi sign-based update
            v_prev = self._second_moment[name]
            sign = np.sign(v_prev - (1 - self.beta2) * (delta ** 2))
            self._second_moment[name] = (
                self.beta2 * v_prev + (1 - self.beta2) * sign * (delta ** 2)
            )
            
            # Bias-corrected estimates
            m_hat = self._momentum[name] / beta1_correction
            v_hat = np.abs(self._second_moment[name]) / beta2_correction
            
            # Apply update
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            result[name] = update.astype(np.float32)
            
        self._metrics = {
            "num_clients": self.num_clients,
            "round": self._round,
        }
        
        return result
    
    def reset(self) -> None:
        """Reset aggregator state."""
        super().reset()
        self._momentum = None
        self._second_moment = None
        self._round = 0
