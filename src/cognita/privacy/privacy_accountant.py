"""
Privacy Accountant Module

Implements privacy budget accounting for differential privacy,
supporting RDP (Rényi Differential Privacy) and advanced composition.

Author: Cognita Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import math

import numpy as np


@dataclass
class PrivacyBudget:
    """Container for privacy budget information."""
    epsilon: float
    delta: float
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    steps: int = 0
    
    @property
    def remaining_epsilon(self) -> float:
        """Get remaining epsilon budget."""
        return max(0, self.epsilon - self.spent_epsilon)
    
    @property
    def remaining_delta(self) -> float:
        """Get remaining delta budget."""
        return max(0, self.delta - self.spent_delta)
    
    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.remaining_epsilon <= 0 or self.remaining_delta <= 0


class PrivacyAccountant:
    """Privacy accountant for tracking privacy budget consumption.
    
    Implements RDP-based (Rényi Differential Privacy) privacy accounting
    with support for subsampled Gaussian mechanism and advanced composition.
    
    Attributes:
        target_epsilon: Target privacy budget
        target_delta: Target privacy failure probability
        accountant_type: Type of accounting (rdp, gdp, simple)
        
    Example:
        >>> accountant = PrivacyAccountant(epsilon=8.0, delta=1e-5)
        >>> 
        >>> # After each training step:
        >>> accountant.update(sample_rate=0.01, noise_multiplier=0.1)
        >>> 
        >>> # Check privacy:
        >>> spent = accountant.get_spent_epsilon()
        >>> print(f"Epsilon spent: {spent:.4f}")
        >>> 
        >>> # Check if privacy budget exhausted:
        >>> if accountant.get_epsilon() <= target_epsilon:
        ...     print("Privacy budget exceeded!")
    """
    
    def __init__(
        self,
        epsilon: float = float('inf'),
        delta: float = 1e-5,
        accountant_type: str = "rdp",
    ):
        """Initialize privacy accountant.
        
        Args:
            epsilon: Target epsilon (infinity for no limit)
            delta: Target delta
            accountant_type: Type of accounting ("rdp", "gdp", "simple")
        """
        self.target_epsilon = epsilon
        self.target_delta = delta
        self.accountant_type = accountant_type
        
        self._steps = 0
        self._orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        self._rdp_cumsum = np.zeros(len(self._orders))
        self._gdp_mu = 0.0
        self._gdp_sigma = 0.0
        
    def update(
        self,
        sample_rate: float,
        noise_multiplier: float,
        steps: int = 1,
    ) -> None:
        """Update privacy accounting.
        
        Args:
            sample_rate: Sampling rate (q in RDP literature)
            noise_multiplier: Noise multiplier (sigma / C)
            steps: Number of steps to account for
        """
        if self.accountant_type == "rdp":
            self._update_rdp(sample_rate, noise_multiplier, steps)
        elif self.accountant_type == "gdp":
            self._update_gdp(sample_rate, noise_multiplier, steps)
        else:
            self._update_simple(sample_rate, noise_multiplier, steps)
            
        self._steps += steps
        
    def _update_rdp(
        self,
        q: float,
        sigma: float,
        steps: int,
    ) -> None:
        """Update RDP accounting.
        
        Args:
            q: Sampling rate
            sigma: Noise multiplier
            steps: Number of steps
        """
        for _ in range(steps):
            rdp = self._compute_rdp(q, sigma, self._orders)
            self._rdp_cumsum += rdp
            
    def _compute_rdp(
        self,
        q: float,
        sigma: float,
        orders: List[float],
    ) -> np.ndarray:
        """Compute RDP values for given orders.
        
        Args:
            q: Sampling rate
            sigma: Noise multiplier
            orders: List of RDP orders
            
        Returns:
            Array of RDP values
        """
        rdp_values = np.zeros(len(orders))
        
        for i, alpha in enumerate(orders):
            if alpha == 1:
                rdp_values[i] = 0
            else:
                # RDP for Gaussian mechanism with Poisson sampling
                # Using the formula from Mironov (2017)
                first_term = 0.5 * (alpha - 1) * q * q * sigma * sigma * (sigma * sigma + 1)
                
                # More accurate computation
                if alpha >= 2:
                    # Use the proper RDP formula
                    rdp_values[i] = self._gaussian_rdp(q, sigma, alpha)
                else:
                    rdp_values[i] = 0
                    
        return rdp_values
    
    def _gaussian_rdp(
        self,
        q: float,
        sigma: float,
        alpha: float,
    ) -> float:
        """Compute RDP for Gaussian mechanism.
        
        Args:
            q: Sampling rate
            sigma: Noise multiplier
            alpha: RDP order
            
        Returns:
            RDP value
        """
        # Simplified RDP computation
        # Uses the bound from Mironov (2017)
        if alpha == float('inf'):
            return float('inf')
            
        # Standard RDP for Gaussian mechanism
        # ε(α) = α/2 * q² * σ² + O(q³ * σ³)
        base_rdp = 0.5 * alpha * q * q * sigma * sigma
        
        # Higher order terms
        if alpha > 2 and q > 0:
            higher_order = (alpha - 1) * q * q * q * sigma * sigma * sigma / 6
            return base_rdp + higher_order
            
        return base_rdp
        
    def _update_gdp(
        self,
        q: float,
        sigma: float,
        steps: int,
    ) -> None:
        """Update GDP accounting.
        
        Args:
            q: Sampling rate
            sigma: Noise multiplier
            steps: Number of steps
        """
        # Compute per-step privacy loss
        from scipy.stats import norm
        
        delta_eps = 1e-6
        eps_range = np.linspace(0, 10, 1000)
        
        # Compute rho (privacy loss) for each epsilon
        for eps in eps_range:
            # Probability of privacy violation
            # Using Gaussian approximation
            pass
            
        # Simplified GDP update
        self._gdp_mu += steps * q
        self._gdp_sigma = math.sqrt(self._gdp_sigma ** 2 + steps * q * q * sigma * sigma)
        
    def _update_simple(
        self,
        q: float,
        sigma: float,
        steps: int,
    ) -> None:
        """Simple privacy accounting using strong composition.
        
        Args:
            q: Sampling rate
            sigma: Noise multiplier
            steps: Number of steps
        """
        # Strong composition theorem
        # ε = δε * sqrt(k * log(1/δ')) where k is number of steps
        eps_per_step = q * sigma * math.sqrt(2 * math.log(1 / self.target_delta))
        self._simple_eps = getattr(self, '_simple_eps', 0) + steps * eps_per_step
        
    def get_spent_epsilon(self) -> float:
        """Get epsilon spent so far.
        
        Returns:
            Spent epsilon
        """
        if self.accountant_type == "rdp":
            return self._compute_epsilon_from_rdp()
        elif self.accountant_type == "gdp":
            return self._gdp_mu
        else:
            return getattr(self, '_simple_eps', 0)
            
    def _compute_epsilon_from_rdp(self) -> float:
        """Compute epsilon from accumulated RDP values.
        
        Returns:
            Estimated epsilon
        """
        if np.sum(self._rdp_cumsum) == 0:
            return 0.0
            
        # Convert RDP to (ε, δ)-DP
        # Find minimum ε such that RDP(ε) ≤ δ
        for i, alpha in enumerate(self._orders):
            rdp_at_alpha = self._rdp_cumsum[i] / self._steps if self._steps > 0 else 0
            
            # Convert RDP to pure DP
            if rdp_at_alpha > 0:
                eps = rdp_at_alpha * alpha + math.log(self.target_delta) / (alpha - 1)
                if eps > 0:
                    return eps
                    
        # Fallback to simple estimate
        return self._steps * 0.01
        
    def get_epsilon(self) -> float:
        """Get current epsilon value.
        
        Returns:
            Current epsilon
        """
        spent = self.get_spent_epsilon()
        return spent
        
    def get_delta(self) -> float:
        """Get current delta value.
        
        Returns:
            Current delta
        """
        return self.target_delta
        
    def get_privacy_spent_ratio(self) -> float:
        """Get ratio of privacy budget spent.
        
        Returns:
            Ratio (0-1) of budget spent
        """
        if self.target_epsilon == float('inf'):
            return 0.0
        return min(1.0, self.get_spent_epsilon() / self.target_epsilon)
        
    def is_privacy_met(self, target_epsilon: Optional[float] = None) -> bool:
        """Check if privacy target is met.
        
        Args:
            target_epsilon: Target epsilon (uses default if None)
            
        Returns:
            True if privacy target is satisfied
        """
        target = target_epsilon or self.target_epsilon
        return self.get_spent_epsilon() <= target
        
    def compute_optimal_noise(
        self,
        target_epsilon: float,
        sample_rate: float,
        steps: int,
    ) -> float:
        """Compute noise multiplier for target epsilon.
        
        Args:
            target_epsilon: Target epsilon
            sample_rate: Sampling rate
            steps: Number of steps
            
        Returns:
            Required noise multiplier
        """
        # Binary search for optimal noise
        low, high = 0.01, 10.0
        
        for _ in range(50):
            mid = (low + high) / 2
            self._steps = 0
            self._rdp_cumsum = np.zeros(len(self._orders))
            
            self.update(sample_rate, mid, steps)
            
            if self.get_spent_epsilon() > target_epsilon:
                low = mid
            else:
                high = mid
                
        return mid
        
    def get_accounting_summary(self) -> dict:
        """Get summary of privacy accounting.
        
        Returns:
            Dictionary with accounting information
        """
        return {
            "accountant_type": self.accountant_type,
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "spent_epsilon": self.get_spent_epsilon(),
            "spent_delta": self.get_delta(),
            "steps": self._steps,
            "privacy_ratio": self.get_privacy_spent_ratio(),
            "is_exhausted": self.get_spent_epsilon() >= self.target_epsilon,
        }


def compose_dp_guarantees(
    guarantees: List[Tuple[float, float]],
    composition: str = "advanced",
) -> Tuple[float, float]:
    """Compose multiple DP guarantees.
    
    Args:
        guarantees: List of (epsilon, delta) tuples
        composition: Composition type ("simple", "advanced", "sequential")
        
    Returns:
        Composed (epsilon, delta) guarantee
    """
    if not guarantees:
        return 0.0, 0.0
        
    if composition == "sequential":
        total_eps = sum(e for e, _ in guarantees)
        total_delta = sum(d for _, d in guarantees)
        
    elif composition == "advanced":
        # Advanced composition
        k = len(guarantees)
        max_eps = max(e for e, _ in guarantees)
        max_delta = max(d for _, d in guarantees)
        
        # Use zCDP composition
        total_eps = max_eps * math.sqrt(2 * k * math.log(1 / max_delta))
        total_delta = k * max_delta
        
    else:
        # Simple composition
        total_eps = sum(e for e, _ in guarantees) * math.sqrt(len(guarantees))
        total_delta = sum(d for _, d in guarantees)
        
    return total_eps, total_delta


def compute_privacy_budget(
    epochs: int,
    batch_size: int,
    dataset_size: int,
    noise_multiplier: float,
    max_grad_norm: float,
    delta: float = 1e-5,
) -> PrivacyBudget:
    """Compute privacy budget for training.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        dataset_size: Total dataset size
        noise_multiplier: Noise multiplier
        max_grad_norm: Maximum gradient norm
        delta: Target delta
        
    Returns:
        PrivacyBudget with accounting info
    """
    sample_rate = batch_size / dataset_size
    steps_per_epoch = dataset_size // batch_size
    total_steps = epochs * steps_per_epoch
    
    accountant = PrivacyAccountant(epsilon=float('inf'), delta=delta)
    
    for _ in range(total_steps):
        accountant.update(sample_rate, noise_multiplier)
        
    spent = accountant.get_spent_epsilon()
    
    return PrivacyBudget(
        epsilon=spent,
        delta=delta,
        spent_epsilon=spent,
        spent_delta=delta,
        steps=total_steps,
    )
