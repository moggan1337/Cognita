"""
Quantization Module

Implements quantization for gradient compression.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np


class Quantizer:
    """Quantizer for gradient compression.
    
    Provides various quantization schemes for compressing
    model updates before transmission.
    
    Example:
        >>> quantizer = Quantizer(levels=256)
        >>> indices, scale = quantizer.quantize(gradients)
        >>> restored = quantizer.dequantize(indices, scale)
    """
    
    def __init__(
        self,
        levels: int = 256,
        method: str = "uniform",
    ):
        """Initialize quantizer.
        
        Args:
            levels: Number of quantization levels
            method: Quantization method ("uniform", "log", "power")
        """
        self.levels = levels
        self.method = method
        
    def quantize(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Quantize data.
        
        Args:
            data: Data to quantize
            
        Returns:
            Tuple of (indices, scale_info)
        """
        if self.method == "uniform":
            return self._uniform_quantize(data)
        elif self.method == "log":
            return self._log_quantize(data)
        elif self.method == "power":
            return self._power_quantize(data)
        else:
            return self._uniform_quantize(data)
            
    def dequantize(
        self,
        indices: np.ndarray,
        scale_info: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Dequantize data.
        
        Args:
            indices: Quantized indices
            scale_info: Scaling information
            
        Returns:
            Dequantized data
        """
        if self.method == "uniform":
            return self._uniform_dequantize(indices, scale_info)
        elif self.method == "log":
            return self._log_dequantize(indices, scale_info)
        elif self.method == "power":
            return self._power_dequantize(indices, scale_info)
        else:
            return self._uniform_dequantize(indices, scale_info)
            
    def _uniform_quantize(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Uniform quantization."""
        min_val = np.min(data)
        max_val = np.max(data)
        
        normalized = (data - min_val) / (max_val - min_val + 1e-10)
        indices = np.round(normalized * (self.levels - 1)).astype(np.uint32)
        
        scale_info = {
            "min": np.array([min_val]),
            "max": np.array([max_val]),
            "levels": np.array([self.levels]),
        }
        
        return indices, scale_info
        
    def _uniform_dequantize(
        self,
        indices: np.ndarray,
        scale_info: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Uniform dequantization."""
        min_val = scale_info["min"][0]
        max_val = scale_info["max"][0]
        levels = scale_info["levels"][0]
        
        normalized = indices.astype(np.float32) / (levels - 1)
        return min_val + normalized * (max_val - min_val)
        
    def _log_quantize(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Logarithmic quantization."""
        abs_data = np.abs(data)
        sign = np.sign(data)
        
        log_data = np.log2(abs_data + 1e-10)
        min_log = np.min(log_data)
        max_log = np.max(log_data)
        
        normalized = (log_data - min_log) / (max_log - min_log + 1e-10)
        indices = np.round(normalized * (self.levels - 1)).astype(np.uint32)
        
        scale_info = {
            "min_log": np.array([min_log]),
            "max_log": np.array([max_log]),
            "levels": np.array([self.levels]),
            "sign": sign,
        }
        
        return indices, scale_info
        
    def _log_dequantize(
        self,
        indices: np.ndarray,
        scale_info: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Logarithmic dequantization."""
        min_log = scale_info["min_log"][0]
        max_log = scale_info["max_log"][0]
        levels = scale_info["levels"][0]
        sign = scale_info["sign"]
        
        normalized = indices.astype(np.float32) / (levels - 1)
        log_val = min_log + normalized * (max_log - min_log)
        
        return 2 ** log_val * sign
        
    def _power_quantize(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Power-of-2 quantization."""
        abs_data = np.abs(data)
        max_abs = np.max(abs_data)
        
        levels = np.array([max_abs * (0.5 ** i) for i in range(self.levels)])
        indices = np.argmin(np.abs(abs_data[:, :, None] - levels), axis=-1)
        
        scale_info = {
            "max_abs": np.array([max_abs]),
            "levels": levels,
        }
        
        return indices, scale_info
        
    def _power_dequantize(
        self,
        indices: np.ndarray,
        scale_info: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Power-of-2 dequantization."""
        levels = scale_info["levels"]
        return levels[indices]
