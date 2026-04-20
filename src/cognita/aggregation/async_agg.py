"""
Asynchronous Aggregation Module

Implements asynchronous federated learning aggregation strategies
for handling stragglers and varying client response times.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from cognita.aggregation.base import BaseAggregator


@dataclass
class AsyncUpdate:
    """Container for asynchronous client update."""
    client_id: str
    round_num: int
    weights: Dict[str, np.ndarray]
    weight: float
    timestamp: float
    staleness: int = 0


class AsyncAggregator(BaseAggregator):
    """Asynchronous aggregation strategy.
    
    Handles client updates that arrive at different times,
    with staleness-aware weighting to prevent degradation
    from stale updates.
    
    Attributes:
        staleness_weight: Decay factor for stale updates
        max_staleness: Maximum allowed staleness
        staleness_threshold: Threshold for staleness compensation
        
    Example:
        >>> aggregator = AsyncAggregator(staleness_weight=0.99, max_staleness=10)
        >>> aggregator.set_current_round(5)
        >>> 
        >>> # As updates arrive:
        >>> aggregator.add_update(client_id, weights, round_num=3)
        >>> aggregator.add_update(client_id2, weights2, round_num=5)
        >>> 
        >>> # Get aggregated result:
        >>> aggregated = aggregator.get_aggregation()
    """
    
    def __init__(
        self,
        staleness_weight: float = 0.99,
        max_staleness: int = 10,
        staleness_threshold: int = 5,
        compensation_method: str = "decay",
    ):
        """Initialize async aggregator.
        
        Args:
            staleness_weight: Exponential decay for stale updates
            max_staleness: Maximum staleness before dropping updates
            staleness_threshold: Apply compensation above this staleness
            compensation_method: Method for staleness compensation
        """
        super().__init__(name="Async")
        self.staleness_weight = staleness_weight
        self.max_staleness = max_staleness
        self.staleness_threshold = staleness_threshold
        self.compensation_method = compensation_method
        
        self._current_round = 0
        self._pending_updates: List[AsyncUpdate] = []
        self._processed_rounds: Set[int] = set()
        self._accumulated_weights: Optional[Dict[str, np.ndarray]] = None
        self._accumulated_weight = 0.0
        
    def set_current_round(self, round_num: int) -> None:
        """Set the current round number.
        
        Args:
            round_num: Current round
        """
        self._current_round = round_num
        
    def add_update(
        self,
        client_id: str,
        weights: Dict[str, np.ndarray],
        weight: float,
        round_num: int,
        timestamp: float,
    ) -> None:
        """Add a client update to the pending queue.
        
        Args:
            client_id: Client identifier
            weights: Client model weights
            weight: Client weight (e.g., number of samples)
            round_num: Round the update was computed for
            timestamp: Time update was received
        """
        staleness = self._current_round - round_num
        
        # Drop if too stale
        if staleness > self.max_staleness:
            return
            
        update = AsyncUpdate(
            client_id=client_id,
            round_num=round_num,
            weights=weights,
            weight=weight,
            timestamp=timestamp,
            staleness=staleness,
        )
        
        self._pending_updates.append(update)
        
    def get_pending_updates(self) -> List[AsyncUpdate]:
        """Get all pending updates.
        
        Returns:
            List of pending updates
        """
        return self._pending_updates.copy()
        
    def aggregate(
        self,
        updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate updates asynchronously.
        
        Args:
            updates: List of (weights, weight) tuples
            
        Returns:
            Aggregated weights
        """
        if not updates:
            return {}
            
        self.num_clients = len(updates)
        
        # Compute staleness-aware weights
        staleness_weights = []
        for _, weight in updates:
            # Staleness decay
            decay = self.staleness_weight ** self._current_round
            adjusted_weight = weight * decay
            staleness_weights.append(adjusted_weight)
            
        # Normalize weights
        total_weight = sum(staleness_weights)
        normalized_weights = [w / total_weight for w in staleness_weights]
        
        # Aggregate
        result = {}
        param_names = updates[0][0].keys()
        
        for name in param_names:
            weighted_sum = np.zeros_like(updates[0][0][name], dtype=np.float64)
            for (weights, _), norm_w in zip(updates, normalized_weights):
                weighted_sum += weights[name].astype(np.float64) * norm_w
            result[name] = weighted_sum.astype(np.float32)
            
        self._metrics = {
            "num_clients": self.num_clients,
            "current_round": self._current_round,
            "pending_updates": len(self._pending_updates),
        }
        
        return result
    
    def get_staleness_compensated_weight(
        self,
        weight: float,
        staleness: int,
    ) -> float:
        """Compute staleness-compensated weight.
        
        Args:
            weight: Original weight
            staleness: Update staleness
            
        Returns:
            Compensated weight
        """
        if self.compensation_method == "decay":
            return weight * (self.staleness_weight ** staleness)
        
        elif self.compensation_method == "linear":
            decay = max(0, 1 - staleness / self.max_staleness)
            return weight * decay
            
        elif self.compensation_method == "quadratic":
            decay = max(0, 1 - (staleness / self.max_staleness) ** 2)
            return weight * decay
            
        elif self.compensation_method == "threshold":
            if staleness <= self.staleness_threshold:
                return weight
            excess = staleness - self.staleness_threshold
            decay = self.staleness_weight ** excess
            return weight * decay
            
        return weight
        
    def clear_processed(self) -> None:
        """Clear processed updates."""
        self._pending_updates = [
            u for u in self._pending_updates
            if u.staleness <= self.max_staleness
        ]
        
    def reset(self) -> None:
        """Reset aggregator state."""
        super().reset()
        self._pending_updates = []
        self._processed_rounds = set()
        self._accumulated_weights = None
        self._accumulated_weight = 0.0


class PartialAsynchronous(AsyncAggregator):
    """Partial asynchronous aggregation.
    
    Combines synchronous and asynchronous approaches by waiting
    for a minimum number of updates before aggregating, then
    processing remaining updates asynchronously.
    """
    
    def __init__(
        self,
        min_updates: int = 3,
        max_wait_time: float = 60.0,
        **kwargs,
    ):
        """Initialize partial async aggregator.
        
        Args:
            min_updates: Minimum updates before starting aggregation
            max_wait_time: Maximum time to wait for updates
        """
        super().__init__(**kwargs)
        self.name = "PartialAsync"
        self.min_updates = min_updates
        self.max_wait_time = max_wait_time
        self._waiting_updates: List[AsyncUpdate] = []
        self._aggregation_start: Optional[float] = None
        
    def start_aggregation_round(self) -> None:
        """Start a new aggregation round."""
        self._waiting_updates = []
        self._aggregation_start = None
        
    def add_update(
        self,
        client_id: str,
        weights: Dict[str, np.ndarray],
        weight: float,
        round_num: int,
        timestamp: float,
    ) -> None:
        """Add update to waiting pool.
        
        Args:
            client_id: Client identifier
            weights: Client weights
            weight: Client weight
            round_num: Round number
            timestamp: Update timestamp
        """
        staleness = self._current_round - round_num
        
        if staleness > self.max_staleness:
            return
            
        update = AsyncUpdate(
            client_id=client_id,
            round_num=round_num,
            weights=weights,
            weight=weight,
            timestamp=timestamp,
            staleness=staleness,
        )
        
        self._waiting_updates.append(update)
        
        # Start timer on first update
        if self._aggregation_start is None:
            self._aggregation_start = timestamp
            
    def should_aggregate(self) -> bool:
        """Check if should perform aggregation.
        
        Returns:
            True if should aggregate
        """
        if len(self._waiting_updates) >= self.min_updates:
            return True
            
        if self._aggregation_start:
            import time
            elapsed = time.time() - self._aggregation_start
            if elapsed >= self.max_wait_time:
                return True
                
        return False
        
    def aggregate(self) -> Optional[Dict[str, np.ndarray]]:
        """Aggregate waiting updates.
        
        Returns:
            Aggregated weights or None
        """
        if not self.should_aggregate():
            return None
            
        updates = [
            (u.weights, self.get_staleness_compensated_weight(u.weight, u.staleness))
            for u in self._waiting_updates
        ]
        
        result = super().aggregate(updates)
        self._waiting_updates = []
        self._aggregation_start = None
        
        return result


from typing import Set
