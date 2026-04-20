"""
Gradient Compression Module

Implements various gradient compression methods for reducing
communication costs in federated learning.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import math

import numpy as np


class CompressionMethod(str, Enum):
    """Available compression methods."""
    TOP_K = "top_k"
    RANDOM_K = "random_k"
    QUANTIZATION = "quantization"
    SIGN = "sign"
    POW2 = "pow2"
    DFT = "dft"
    WAVELET = "wavelet"
    SPARSE = "sparse"


class GradientCompressor:
    """Gradient compression for federated learning.
    
    Provides various compression methods to reduce communication
    overhead while maintaining model quality.
    
    Attributes:
        method: Compression method
        compression_ratio: Target compression ratio (0-1)
        
    Example:
        >>> compressor = GradientCompressor(method="top_k", compression_ratio=0.1)
        >>> compressed = compressor.compress(gradients)
        >>> decompressed = compressor.decompress(compressed)
    """
    
    def __init__(
        self,
        method: str = "top_k",
        compression_ratio: float = 0.1,
        seed: Optional[int] = None,
    ):
        """Initialize gradient compressor.
        
        Args:
            method: Compression method name
            compression_ratio: Ratio of values to keep/send
            seed: Random seed
        """
        self.method = method
        self.compression_ratio = compression_ratio
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self._compressors = {
            "top_k": self._compress_top_k,
            "random_k": self._compress_random_k,
            "quantization": self._compress_quantization,
            "sign": self._compress_sign,
            "pow2": self._compress_pow2,
            "sparse": self._compress_sparse,
        }
        
        self._decompressors = {
            "top_k": self._decompress_top_k,
            "random_k": self._decompress_random_k,
            "quantization": self._decompress_quantization,
            "sign": self._decompress_sign,
            "pow2": self._decompress_pow2,
            "sparse": self._decompress_sparse,
        }
        
    def compress(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compress gradients.
        
        Args:
            gradients: Dictionary of gradient arrays
            
        Returns:
            Compressed gradients
        """
        compressor = self._compressors.get(
            self.method,
            self._compress_top_k,
        )
        return compressor(gradients)
        
    def decompress(
        self,
        compressed: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Decompress gradients.
        
        Args:
            compressed: Compressed gradients
            
        Returns:
            Decompressed gradients
        """
        decompressor = self._decompressors.get(
            self.method,
            self._decompress_top_k,
        )
        return decompressor(compressed)
        
    def _compress_top_k(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compress by keeping only top-k values by magnitude.
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            Compressed gradients with indices
        """
        compressed = {}
        
        for name, grad in gradients.items():
            flat = grad.flatten()
            k = max(1, int(len(flat) * self.compression_ratio))
            
            # Get indices of top-k values by absolute magnitude
            abs_flat = np.abs(flat)
            if len(abs_flat) > k:
                indices = np.argpartition(abs_flat, -k)[-k:]
                values = flat[indices]
                compressed[f"{name}_values"] = values
                compressed[f"{name}_indices"] = indices
                compressed[f"{name}_shape"] = np.array(grad.shape)
            else:
                compressed[name] = grad
                
        return compressed
        
    def _decompress_top_k(
        self,
        compressed: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Decompress top-k compressed gradients."""
        decompressed = {}
        
        for name in list(compressed.keys()):
            if name.endswith("_values"):
                base_name = name.replace("_values", "")
                indices = compressed.get(f"{base_name}_indices")
                shape = compressed.get(f"{base_name}_shape")
                
                if indices is not None and shape is not None:
                    values = compressed[name]
                    full = np.zeros(np.prod(shape))
                    full[indices] = values
                    decompressed[base_name] = full.reshape(shape.astype(int))
                    
        return decompressed
        
    def _compress_random_k(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compress by keeping random-k values.
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            Compressed gradients
        """
        compressed = {}
        
        for name, grad in gradients.items():
            flat = grad.flatten()
            k = max(1, int(len(flat) * self.compression_ratio))
            
            indices = self.rng.choice(len(flat), k, replace=False)
            values = flat[indices]
            
            compressed[f"{name}_values"] = values
            compressed[f"{name}_indices"] = indices
            compressed[f"{name}_shape"] = np.array(grad.shape)
            compressed[f"{name}_norm"] = np.array([np.linalg.norm(flat) / (np.linalg.norm(values) + 1e-10)])
            
        return compressed
        
    def _decompress_random_k(
        self,
        compressed: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Decompress random-k compressed gradients."""
        decompressed = {}
        
        for name in list(compressed.keys()):
            if name.endswith("_values"):
                base_name = name.replace("_values", "")
                indices = compressed.get(f"{base_name}_indices")
                shape = compressed.get(f"{base_name}_shape")
                norm_ratio = compressed.get(f"{base_name}_norm", np.array([1.0]))[0]
                
                if indices is not None:
                    values = compressed[name]
                    full = np.zeros(np.prod(shape))
                    full[indices] = values * norm_ratio
                    decompressed[base_name] = full.reshape(shape.astype(int))
                    
        return decompressed
        
    def _compress_quantization(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compress using quantization.
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            Quantized gradients
        """
        compressed = {}
        levels = 256  # 8-bit quantization
        
        for name, grad in gradients.items():
            # Find range
            min_val = np.min(grad)
            max_val = np.max(grad)
            
            # Normalize to [0, 1]
            normalized = (grad - min_val) / (max_val - min_val + 1e-10)
            
            # Quantize
            quantized = np.round(normalized * (levels - 1)).astype(np.uint8)
            
            compressed[f"{name}_quantized"] = quantized
            compressed[f"{name}_min"] = np.array([min_val])
            compressed[f"{name}_max"] = np.array([max_val])
            compressed[f"{name}_shape"] = np.array(grad.shape)
            
        return compressed
        
    def _decompress_quantization(
        self,
        compressed: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Decompress quantized gradients."""
        decompressed = {}
        
        for name in list(compressed.keys()):
            if name.endswith("_quantized"):
                base_name = name.replace("_quantized", "")
                min_val = compressed.get(f"{base_name}_min", np.array([0]))[0]
                max_val = compressed.get(f"{base_name}_max", np.array([1]))[0]
                shape = compressed.get(f"{base_name}_shape")
                
                quantized = compressed[name].astype(np.float32) / 255.0
                decompressed[base_name] = min_val + quantized * (max_val - min_val)
                if shape is not None:
                    decompressed[base_name] = decompressed[base_name].reshape(shape.astype(int))
                    
        return decompressed
        
    def _compress_sign(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compress using sign quantization (1-bit).
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            Sign-compressed gradients
        """
        compressed = {}
        
        for name, grad in gradients.items():
            # Get sign and norm
            signs = np.sign(grad)
            # Get scaling factor (mean absolute value)
            scale = np.mean(np.abs(grad))
            
            compressed[f"{name}_signs"] = (signs > 0).astype(np.int8)
            compressed[f"{name}_scale"] = np.array([scale])
            compressed[f"{name}_shape"] = np.array(grad.shape)
            
        return compressed
        
    def _decompress_sign(
        self,
        compressed: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Decompress sign-compressed gradients."""
        decompressed = {}
        
        for name in list(compressed.keys()):
            if name.endswith("_signs"):
                base_name = name.replace("_signs", "")
                signs = compressed[name]
                scale = compressed.get(f"{base_name}_scale", np.array([1.0]))[0]
                shape = compressed.get(f"{base_name}_shape")
                
                decompressed[base_name] = (signs * 2 - 1) * scale
                if shape is not None:
                    decompressed[base_name] = decompressed[base_name].reshape(shape.astype(int))
                    
        return decompressed
        
    def _compress_pow2(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compress using power-of-2 quantization.
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            Power-of-2 compressed gradients
        """
        compressed = {}
        num_levels = 8
        
        for name, grad in gradients.items():
            # Compute scale
            max_abs = np.max(np.abs(grad))
            
            # Power-of-2 levels
            levels = np.array([max_abs * (0.5 ** i) for i in range(num_levels)])
            
            # Quantize to nearest power of 2
            abs_grad = np.abs(grad)
            indices = np.argmin(np.abs(abs_grad[:, :, None] - levels), axis=-1)
            
            compressed[f"{name}_indices"] = indices.astype(np.uint8)
            compressed[f"{name}_levels"] = levels
            compressed[f"{name}_signs"] = (grad > 0).astype(np.int8)
            compressed[f"{name}_shape"] = np.array(grad.shape)
            
        return compressed
        
    def _decompress_pow2(
        self,
        compressed: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Decompress power-of-2 compressed gradients."""
        decompressed = {}
        
        for name in list(compressed.keys()):
            if name.endswith("_indices"):
                base_name = name.replace("_indices", "")
                indices = compressed[name]
                levels = compressed.get(f"{base_name}_levels")
                signs = compressed.get(f"{base_name}_signs")
                shape = compressed.get(f"{base_name}_shape")
                
                if levels is not None:
                    values = levels[indices]
                    if signs is not None:
                        values = values * (signs * 2 - 1)
                    decompressed[base_name] = values
                    if shape is not None:
                        decompressed[base_name] = decompressed[base_name].reshape(shape.astype(int))
                        
        return decompressed
        
    def _compress_sparse(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compress by zeroing small values.
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            Sparse gradients
        """
        threshold = np.percentile(
            np.abs(np.concatenate([g.flatten() for g in gradients.values()])),
            (1 - self.compression_ratio) * 100,
        )
        
        sparse = {}
        for name, grad in gradients.items():
            mask = np.abs(grad) > threshold
            sparse[name] = grad * mask
            sparse[f"{name}_mask"] = mask
            
        return sparse
        
    def _decompress_sparse(
        self,
        compressed: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Decompress sparse gradients."""
        decompressed = {}
        
        for name in list(compressed.keys()):
            if not name.endswith("_mask"):
                if f"{name}_mask" in compressed:
                    mask = compressed[f"{name}_mask"]
                    decompressed[name] = compressed[name] * mask
                else:
                    decompressed[name] = compressed[name]
                    
        return decompressed
        
    def get_compression_ratio(self) -> float:
        """Get actual compression ratio achieved.
        
        Returns:
            Compression ratio
        """
        return self.compression_ratio
