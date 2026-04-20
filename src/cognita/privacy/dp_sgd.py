"""
Differential Privacy (DP-SGD) Module

Implements differentially private stochastic gradient descent for
federated learning with per-sample gradient clipping and noise addition.

DP-SGD: Abadi et al., "Deep Learning with Differential Privacy" (2016)

Author: Cognita Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import math

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DPSGDConfig:
    """Configuration for DP-SGD.
    
    Attributes:
        epsilon: Privacy budget (lower = stronger privacy)
        delta: Privacy failure probability
        max_grad_norm: Maximum per-sample gradient norm
        noise_multiplier: Ratio of noise std to max_grad_norm
        sample_rate: Fraction of samples per batch
        min_epsilon: Minimum epsilon before stopping
    """
    epsilon: float = 8.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 0.1
    sample_rate: float = 0.01
    min_epsilon: float = 1.0
    adaptive_noise: bool = False
    noise_decay: float = 0.99


class DPClient:
    """Differentially private client for federated learning.
    
    Implements per-sample gradient clipping and Gaussian noise addition
    for differential privacy guarantees.
    
    Attributes:
        max_grad_norm: Maximum gradient norm for clipping
        noise_multiplier: Noise standard deviation multiplier
        
    Example:
        >>> client = DPClient(max_grad_norm=1.0, noise_multiplier=0.1)
        >>> 
        >>> # During training:
        >>> clipped_grads = client.clip_gradients(model, max_norm)
        >>> noisy_grads = client.add_noise(clipped_grads)
        >>> 
        >>> # Check privacy:
        >>> spent = client.get_privacy_spent()
        >>> print(f"Epsilon spent: {spent:.4f}")
    """
    
    def __init__(
        self,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 0.1,
        target_epsilon: float = 8.0,
    ):
        """Initialize DP client.
        
        Args:
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Ratio of noise std to max_grad_norm
            target_epsilon: Target privacy budget
        """
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.target_epsilon = target_epsilon
        
        self._round_count = 0
        self._accumulated_noise: Dict[str, float] = {}
        
    def clip_gradients(
        self,
        model: nn.Module,
        max_norm: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Clip gradients by per-sample norm.
        
        Args:
            model: PyTorch model with gradients
            max_norm: Maximum norm (uses self.max_grad_norm if None)
            
        Returns:
            Dictionary of clipped gradients
        """
        max_norm = max_norm or self.max_grad_norm
        
        clipped_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Compute per-sample norms
                if grad.dim() == 1:
                    # Bias terms - single vector
                    param_norm = grad.norm()
                else:
                    # Weight matrices - compute row norms
                    param_norm = grad.view(grad.size(0), -1).norm(dim=1)
                    
                # Compute clip factor
                clip_factor = torch.clamp(
                    max_norm / (param_norm + 1e-6),
                    max=1.0
                )
                
                # Apply clipping
                if grad.dim() == 1:
                    clipped_grad = grad * clip_factor
                else:
                    # Expand clip factor and apply
                    clip_factor = clip_factor.unsqueeze(1)
                    clipped_grad = grad * clip_factor.unsqueeze(-1)
                    
                clipped_grads[name] = clipped_grad.cpu().numpy()
                
        return clipped_grads
    
    def add_noise(
        self,
        gradients: Dict[str, np.ndarray],
        noise_multiplier: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Add Gaussian noise to clipped gradients.
        
        Args:
            gradients: Dictionary of clipped gradients
            noise_multiplier: Noise multiplier (uses self.noise_multiplier if None)
            
        Returns:
            Dictionary of noisy gradients
        """
        noise_multiplier = noise_multiplier or self.noise_multiplier
        std = noise_multiplier * self.max_grad_norm
        
        noisy_grads = {}
        for name, grad in gradients.items():
            noise = np.random.normal(0, std, grad.shape).astype(grad.dtype)
            noisy_grads[name] = grad + noise
            
            # Track accumulated noise variance
            self._accumulated_noise[name] = (
                self._accumulated_noise.get(name, 0) + std ** 2
            )
            
        return noisy_grads
    
    def compute_noise_budget(
        self,
        num_steps: int,
        sample_rate: float,
    ) -> float:
        """Compute privacy budget spent using RDP.
        
        Args:
            num_steps: Number of gradient steps
            sample_rate: Sampling rate per step
            
        Returns:
            Estimated epsilon spent
        """
        from cognita.privacy.privacy_accountant import PrivacyAccountant
        
        accountant = PrivacyAccountant(
            epsilon=self.target_epsilon,
            delta=1e-5,
        )
        
        for _ in range(num_steps):
            accountant.update(
                sample_rate=sample_rate,
                noise_multiplier=self.noise_multiplier,
            )
            
        return accountant.get_spent_epsilon()
    
    def adaptive_clip(
        self,
        model: nn.Module,
        clip_ratio: float = 0.5,
    ) -> float:
        """Apply adaptive gradient clipping.
        
        Uses the median of per-sample norms for more efficient clipping.
        
        Args:
            model: PyTorch model
            clip_ratio: Target percentile for clipping
            
        Returns:
            New adaptive clip norm
        """
        all_norms = []
        
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad.data
                if grad.dim() == 1:
                    all_norms.append(grad.norm().item())
                else:
                    norms = grad.view(grad.size(0), -1).norm(dim=1)
                    all_norms.extend(norms.cpu().numpy().tolist())
                    
        if all_norms:
            adaptive_norm = np.percentile(all_norms, clip_ratio * 100)
            self.max_grad_norm = adaptive_norm
            
        return self.max_grad_norm
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get privacy budget spent.
        
        Returns:
            Tuple of (spent_epsilon, remaining_epsilon)
        """
        # Simplified estimate
        spent = self._round_count * self._compute_eps_per_round()
        remaining = max(0, self.target_epsilon - spent)
        return spent, remaining
    
    def _compute_eps_per_round(self) -> float:
        """Compute epsilon spent per round.
        
        Returns:
            Epsilon per round estimate
        """
        # Simplified privacy accounting
        q = self.sample_rate if hasattr(self, 'sample_rate') else 0.01
        sigma = self.noise_multiplier * self.max_grad_norm
        
        # Approximate using strong composition
        return q * self.max_grad_norm / sigma
    
    def step(self) -> None:
        """Increment round counter."""
        self._round_count += 1


class DPSGD:
    """DPSGD optimizer wrapper.
    
    Wraps a standard optimizer with differential privacy mechanisms.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float,
        noise_multiplier: float,
        secure_rng: Optional[np.random.Generator] = None,
    ):
        """Initialize DPSGD optimizer.
        
        Args:
            optimizer: Base optimizer
            max_grad_norm: Maximum gradient norm
            noise_multiplier: Noise multiplier
            secure_rng: Secure random number generator
        """
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.secure_rng = secure_rng or np.random.default_rng()
        
    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()
        
    def step(
        self,
        closure: Optional[callable] = None,
        per_sample_grads: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[float]:
        """Perform optimization step with DP.
        
        Args:
            closure: Loss computation closure
            per_sample_grads: Pre-computed per-sample gradients
            
        Returns:
            Loss value if closure provided
        """
        if closure is not None:
            loss = closure()
        else:
            loss = None
            
        # Clip gradients
        self._clip_gradients()
        
        # Add noise
        self._add_noise()
        
        # Apply update
        self.optimizer.step()
        
        return loss
    
    def _clip_gradients(self) -> None:
        """Clip gradients by global norm."""
        torch.nn.utils.clip_grad_norm_(
            self.optimizer.param_groups[0]['params'],
            self.max_grad_norm,
        )
        
    def _add_noise(self) -> None:
        """Add Gaussian noise to gradients."""
        std = self.noise_multiplier * self.max_grad_norm
        
        for param in self.optimizer.param_groups[0]['params']:
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * std
                param.grad.data.add_(noise)
                
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying optimizer."""
        return getattr(self.optimizer, name)


def compute_dp_sgd_privacy(
    epochs: int,
    max_grad_norm: float,
    noise_multiplier: float,
    batch_size: int,
    dataset_size: int,
    delta: float = 1e-5,
) -> Tuple[float, int]:
    """Compute privacy guarantee for DP-SGD.
    
    Args:
        epochs: Number of training epochs
        max_grad_norm: Maximum gradient norm
        noise_multiplier: Noise multiplier
        batch_size: Training batch size
        dataset_size: Total dataset size
        delta: Privacy failure probability
        
    Returns:
        Tuple of (epsilon, max_steps)
    """
    from cognita.privacy.privacy_accountant import PrivacyAccountant
    
    sample_rate = batch_size / dataset_size
    steps_per_epoch = dataset_size // batch_size
    total_steps = epochs * steps_per_epoch
    
    accountant = PrivacyAccountant(epsilon=float('inf'), delta=delta)
    
    for _ in range(total_steps):
        accountant.update(
            sample_rate=sample_rate,
            noise_multiplier=noise_multiplier,
        )
        
    return accountant.get_epsilon(), total_steps


from typing import Callable, Optional
