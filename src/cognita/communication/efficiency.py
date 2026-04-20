"""
Communication Efficiency Module

Implements techniques for improving communication efficiency
in federated learning.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class CommunicationEfficiency:
    """Communication efficiency techniques for federated learning.
    
    Provides methods for reducing communication overhead while
    maintaining model quality.
    
    Example:
        >>> efficiency = CommunicationEfficiency()
        >>> 
        >>> # Check if update is significant enough
        >>> if efficiency.should_communicate(update, prev_update, threshold=0.01):
        ...     send_update(update)
        ... else:
        ...     skip_communication()
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        compression_ratio: float = 0.1,
    ):
        """Initialize communication efficiency.
        
        Args:
            similarity_threshold: Threshold for skip communication
            compression_ratio: Compression ratio for updates
        """
        self.similarity_threshold = similarity_threshold
        self.compression_ratio = compression_ratio
        self._prev_update: Optional[Dict[str, np.ndarray]] = None
        self._skipped_rounds = 0
        self._total_rounds = 0
        
    def should_communicate(
        self,
        current: Dict[str, np.ndarray],
        previous: Optional[Dict[str, np.ndarray]] = None,
        threshold: Optional[float] = None,
    ) -> bool:
        """Check if update is significantly different from previous.
        
        Args:
            current: Current model weights
            previous: Previous model weights
            threshold: Custom similarity threshold
            
        Returns:
            True if should communicate
        """
        threshold = threshold or self.similarity_threshold
        self._total_rounds += 1
        
        if previous is None:
            self._prev_update = current.copy()
            return True
            
        # Compute similarity
        similarity = self._compute_similarity(current, previous)
        
        if similarity >= threshold:
            self._skipped_rounds += 1
            return False
            
        self._prev_update = current.copy()
        return True
        
    def _compute_similarity(
        self,
        current: Dict[str, np.ndarray],
        previous: Dict[str, np.ndarray],
    ) -> float:
        """Compute similarity between updates.
        
        Args:
            current: Current weights
            previous: Previous weights
            
        Returns:
            Similarity score (0-1)
        """
        total_norm = 0.0
        current_norm = 0.0
        
        for name in current.keys():
            if name in previous:
                delta = current[name] - previous[name]
                total_norm += np.sum(delta ** 2)
                current_norm += np.sum(current[name] ** 2)
                
        if current_norm == 0:
            return 1.0
            
        # Cosine similarity
        # For same direction, similarity is high
        return 1.0 - min(1.0, np.sqrt(total_norm) / (np.sqrt(current_norm) + 1e-10))
        
    def get_skipped_ratio(self) -> float:
        """Get ratio of skipped communications.
        
        Returns:
            Ratio of skipped rounds
        """
        if self._total_rounds == 0:
            return 0.0
        return self._skipped_rounds / self._total_rounds
        
    def reset_stats(self) -> None:
        """Reset communication statistics."""
        self._skipped_rounds = 0
        self._total_rounds = 0
        self._prev_update = None


class GradientEstimation:
    """Gradient estimation for reducing communication.
    
    Estimates gradients locally to reduce frequency of
    communication with server.
    """
    
    def __init__(
        self,
        momentum: float = 0.9,
        decay: float = 0.99,
    ):
        """Initialize gradient estimator.
        
        Args:
            momentum: Momentum for gradient estimation
            decay: Decay for running average
        """
        self.momentum = momentum
        self.decay = decay
        self._estimated_gradients: Optional[Dict[str, np.ndarray]] = None
        self._running_avg: Optional[Dict[str, np.ndarray]] = None
        
    def estimate(
        self,
        local_gradients: Dict[str, np.ndarray],
        use_momentum: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Estimate gradients for transmission.
        
        Args:
            local_gradients: Actual local gradients
            use_momentum: Apply momentum to estimate
            
        Returns:
            Estimated gradients
        """
        if use_momentum and self._estimated_gradients is not None:
            return {
                name: self.momentum * self._estimated_gradients.get(name, np.zeros_like(g))
                + (1 - self.momentum) * g
                for name, g in local_gradients.items()
            }
        return local_gradients.copy()
        
    def update_running_avg(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> None:
        """Update running average of gradients.
        
        Args:
            gradients: New gradients
        """
        if self._running_avg is None:
            self._running_avg = {name: g.copy() for name, g in gradients.items()}
        else:
            self._running_avg = {
                name: self.decay * self._running_avg.get(name, np.zeros_like(g))
                + (1 - self.decay) * g
                for name, g in gradients.items()
            }
        self._estimated_gradients = self._running_avg.copy()
        
    def get_estimated(self) -> Optional[Dict[str, np.ndarray]]:
        """Get current estimated gradients.
        
        Returns:
            Estimated gradients or None
        """
        return self._estimated_gradients.copy() if self._estimated_gradients else None
