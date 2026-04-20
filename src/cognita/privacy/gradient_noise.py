"""
Gradient Noise Module

Provides utilities for adding noise to gradients for privacy preservation.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Dict, Optional, Union
import numpy as np


def add_gaussian_noise(
    gradients: Dict[str, np.ndarray],
    std: float,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Add Gaussian noise to gradients.
    
    Args:
        gradients: Dictionary of gradient arrays
        std: Standard deviation of noise
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of noisy gradients
    """
    rng = np.random.default_rng(seed)
    
    noisy_grads = {}
    for name, grad in gradients.items():
        noise = rng.normal(0, std, grad.shape).astype(grad.dtype)
        noisy_grads[name] = grad + noise
        
    return noisy_grads


def add_laplace_noise(
    gradients: Dict[str, np.ndarray],
    scale: float,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Add Laplace noise to gradients.
    
    Args:
        gradients: Dictionary of gradient arrays
        scale: Scale parameter (b) for Laplace distribution
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of noisy gradients
    """
    rng = np.random.default_rng(seed)
    
    noisy_grads = {}
    for name, grad in gradients.items():
        noise = rng.laplace(0, scale, grad.shape).astype(grad.dtype)
        noisy_grads[name] = grad + noise
        
    return noisy_grads


def calibrate_noise_for_dp(
    epsilon: float,
    delta: float,
    sensitivity: float,
    mechanism: str = "gaussian",
) -> float:
    """Calibrate noise standard deviation for differential privacy.
    
    Args:
        epsilon: Privacy budget
        delta: Privacy failure probability
        sensitivity: Sensitivity of the function
        mechanism: Noise mechanism ("gaussian" or "laplace")
        
    Returns:
        Calibrated noise parameter
    """
    if mechanism == "gaussian":
        # For Gaussian mechanism: σ ≥ √(2 ln(1.25/δ)) * Δf / ε
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        return sigma
        
    elif mechanism == "laplace":
        # For Laplace mechanism: b = Δf / ε
        b = sensitivity / epsilon
        return b
        
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


def compute_adaptive_noise(
    gradients: Dict[str, np.ndarray],
    base_noise: float,
    gradient_norms: Optional[Dict[str, float]] = None,
    adaptation_factor: float = 0.1,
) -> float:
    """Compute adaptive noise based on gradient magnitudes.
    
    Args:
        gradients: Dictionary of gradients
        base_noise: Base noise level
        gradient_norms: Pre-computed gradient norms
        adaptation_factor: How much to adapt noise
        
    Returns:
        Adapted noise level
    """
    if gradient_norms is None:
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += np.sum(grad ** 2)
        gradient_norms = {"total": np.sqrt(total_norm)}
        
    avg_norm = np.mean(list(gradient_norms.values()))
    
    # Increase noise for high-norm gradients
    adaptive_noise = base_noise * (1 + adaptation_factor * avg_norm)
    
    return adaptive_noise
