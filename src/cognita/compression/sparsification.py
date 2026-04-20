"""
Sparsification Module

Implements gradient sparsification methods for compression.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np


class TopKCompressor:
    """Top-K gradient sparsification.
    
    Keeps only the k largest (by magnitude) elements of each gradient.
    
    Example:
        >>> compressor = TopKCompressor(ratio=0.1)
        >>> sparse_grads, info = compressor.compress(gradients)
        >>> restored = compressor.decompress(sparse_grads, info)
    """
    
    def __init__(
        self,
        ratio: float = 0.1,
        per_layer: bool = True,
    ):
        """Initialize Top-K compressor.
        
        Args:
            ratio: Fraction of elements to keep
            per_layer: Apply per-layer vs globally
        """
        self.ratio = ratio
        self.per_layer = per_layer
        
    def compress(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compress gradients using Top-K.
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            Tuple of (sparse_grads, info_dict)
        """
        sparse = {}
        info = {}
        
        if self.per_layer:
            for name, grad in gradients.items():
                sparse_grad, grad_info = self._compress_layer(grad, name)
                sparse.update(sparse_grad)
                info.update(grad_info)
        else:
            # Global Top-K
            all_values = np.concatenate([g.flatten() for g in gradients.values()])
            k = max(1, int(len(all_values) * self.ratio))
            
            threshold = np.sort(np.abs(all_values))[-k] if len(all_values) > k else 0
            
            for name, grad in gradients.items():
                mask = np.abs(grad) >= threshold
                sparse[name] = grad * mask
                info[f"{name}_mask"] = mask
                
        return sparse, info
        
    def _compress_layer(
        self,
        grad: np.ndarray,
        name: str,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compress a single layer gradient."""
        flat = grad.flatten()
        k = max(1, int(len(flat) * self.ratio))
        
        indices = np.argpartition(np.abs(flat), -k)[-k:]
        values = flat[indices]
        
        return (
            {name: values},
            {
                f"{name}_indices": indices,
                f"{name}_shape": np.array(grad.shape),
            },
        )
        
    def decompress(
        self,
        sparse: Dict[str, np.ndarray],
        info: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Decompress Top-K compressed gradients."""
        restored = {}
        
        for name, values in sparse.items():
            if name in info:
                continue
                
            indices = info.get(f"{name}_indices")
            shape = info.get(f"{name}_shape")
            
            if indices is not None and shape is not None:
                full = np.zeros(np.prod(shape).astype(int))
                full[indices.astype(int)] = values
                restored[name] = full.reshape(shape.astype(int))
            else:
                restored[name] = values
                
        return restored


class RandomKCompressor:
    """Random-K gradient sparsification.
    
    Keeps k randomly selected elements of each gradient.
    
    Example:
        >>> compressor = RandomKCompressor(ratio=0.1, seed=42)
        >>> sparse_grads, info = compressor.compress(gradients)
    """
    
    def __init__(
        self,
        ratio: float = 0.1,
        seed: Optional[int] = None,
        rescale: bool = True,
    ):
        """Initialize Random-K compressor.
        
        Args:
            ratio: Fraction of elements to keep
            seed: Random seed
            rescale: Rescale values to maintain expectation
        """
        self.ratio = ratio
        self.seed = seed
        self.rescale = rescale
        self.rng = np.random.default_rng(seed)
        
    def compress(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compress gradients using Random-K.
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            Tuple of (sparse_grads, info_dict)
        """
        sparse = {}
        info = {}
        
        for name, grad in gradients.items():
            flat = grad.flatten()
            k = max(1, int(len(flat) * self.ratio))
            
            indices = self.rng.choice(len(flat), k, replace=False)
            values = flat[indices]
            
            # Compute scaling factor for unbiased compression
            scale = len(flat) / k if self.rescale else 1.0
            
            sparse[name] = values
            info[f"{name}_indices"] = indices
            info[f"{name}_shape"] = np.array(grad.shape)
            info[f"{name}_scale"] = np.array([scale])
            
        return sparse, info
        
    def decompress(
        self,
        sparse: Dict[str, np.ndarray],
        info: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Decompress Random-K compressed gradients."""
        restored = {}
        
        for name, values in sparse.items():
            indices = info.get(f"{name}_indices")
            shape = info.get(f"{name}_shape")
            scale = info.get(f"{name}_scale", np.array([1.0]))[0]
            
            if indices is not None and shape is not None:
                full = np.zeros(np.prod(shape).astype(int))
                full[indices.astype(int)] = values * scale
                restored[name] = full.reshape(shape.astype(int))
            else:
                restored[name] = values
                
        return restored


class ThresholdCompressor:
    """Threshold-based gradient sparsification.
    
    Keeps elements above a certain magnitude threshold.
    """
    
    def __init__(
        self,
        threshold: float = 0.01,
        absolute: bool = True,
    ):
        """Initialize threshold compressor.
        
        Args:
            threshold: Magnitude threshold
            absolute: Use absolute thresholding
        """
        self.threshold = threshold
        self.absolute = absolute
        
    def compress(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compress gradients using threshold."""
        sparse = {}
        info = {}
        
        for name, grad in gradients.items():
            if self.absolute:
                mask = np.abs(grad) > self.threshold
            else:
                mask = grad > self.threshold
                
            sparse[name] = grad * mask
            info[f"{name}_mask"] = mask
            
        return sparse, info
        
    def decompress(
        self,
        sparse: Dict[str, np.ndarray],
        info: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Decompress threshold-compressed gradients."""
        restored = {}
        
        for name, values in sparse.items():
            restored[name] = values
            
        return restored
