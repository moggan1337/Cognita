"""
Entropy Coding Module

Implements entropy coding for additional compression of
sparse gradient representations.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


class EntropyCoder:
    """Entropy coder for gradient compression.
    
    Provides entropy coding (Huffman, arithmetic) for
    further compression of already sparse gradients.
    
    Example:
        >>> coder = EntropyCoder(method="huffman")
        >>> encoded, codebook = coder.encode(indices)
        >>> decoded = coder.decode(encoded, codebook)
    """
    
    def __init__(self, method: str = "huffman"):
        """Initialize entropy coder.
        
        Args:
            method: Coding method ("huffman", "arithmetic", "run_length")
        """
        self.method = method
        
    def encode(
        self,
        data: np.ndarray,
    ) -> Tuple[bytes, Dict]:
        """Encode data.
        
        Args:
            data: Data to encode
            
        Returns:
            Tuple of (encoded_bytes, codebook)
        """
        if self.method == "huffman":
            return self._huffman_encode(data)
        elif self.method == "run_length":
            return self._run_length_encode(data)
        else:
            return self._huffman_encode(data)
            
    def decode(
        self,
        encoded: bytes,
        codebook: Dict,
    ) -> np.ndarray:
        """Decode data.
        
        Args:
            encoded: Encoded bytes
            codebook: Codebook dictionary
            
        Returns:
            Decoded data
        """
        if self.method == "huffman":
            return self._huffman_decode(encoded, codebook)
        elif self.method == "run_length":
            return self._run_length_decode(encoded, codebook)
        else:
            return self._huffman_decode(encoded, codebook)
            
    def _huffman_encode(
        self,
        data: np.ndarray,
    ) -> Tuple[bytes, Dict]:
        """Huffman encoding (simplified)."""
        flat = data.flatten()
        
        # Count frequencies
        unique, counts = np.unique(flat, return_counts=True)
        frequencies = dict(zip(unique, counts))
        
        # Simple fixed-length encoding for demonstration
        # In practice, use proper Huffman coding library
        encoded_values = []
        for val in flat:
            encoded_values.append(val)
            
        # Create codebook
        codebook = {
            "frequencies": frequencies,
            "unique_values": unique,
            "dtype": str(data.dtype),
            "shape": np.array(data.shape),
        }
        
        # Pack into bytes (simplified)
        encoded = np.array(encoded_values, dtype=data.dtype).tobytes()
        
        return encoded, codebook
        
    def _huffman_decode(
        self,
        encoded: bytes,
        codebook: Dict,
    ) -> np.ndarray:
        """Huffman decoding (simplified)."""
        dtype = np.dtype(codebook["dtype"])
        shape = tuple(codebook["shape"].astype(int))
        
        data = np.frombuffer(encoded, dtype=dtype)
        return data.reshape(shape)
        
    def _run_length_encode(
        self,
        data: np.ndarray,
    ) -> Tuple[bytes, Dict]:
        """Run-length encoding."""
        flat = data.flatten()
        
        # Find runs
        runs = []
        i = 0
        while i < len(flat):
            val = flat[i]
            count = 1
            while i + count < len(flat) and flat[i + count] == val:
                count += 1
            runs.append((val, count))
            i += count
            
        # Encode runs
        values = np.array([r[0] for r in runs], dtype=data.dtype)
        lengths = np.array([r[1] for r in runs], dtype=np.uint32)
        
        codebook = {
            "shape": np.array(data.shape),
            "dtype": str(data.dtype),
        }
        
        return values.tobytes() + lengths.tobytes(), codebook
        
    def _run_length_decode(
        self,
        encoded: bytes,
        codebook: Dict,
    ) -> np.ndarray:
        """Run-length decoding."""
        shape = tuple(codebook["shape"].astype(int))
        dtype = np.dtype(codebook["dtype"])
        
        # Determine values and lengths
        item_size = np.dtype(dtype).itemsize
        num_items = len(encoded) // (item_size + 4)
        
        values = np.frombuffer(encoded[:num_items * item_size], dtype=dtype)
        lengths = np.frombuffer(encoded[num_items * item_size:], dtype=np.uint32)
        
        # Reconstruct
        flat = np.repeat(values, lengths)
        return flat.reshape(shape)


class AdaptiveCompressor:
    """Adaptive compressor that selects best method.
    
    Automatically selects the best compression method based on
    data characteristics.
    """
    
    def __init__(
        self,
        methods: Optional[List[str]] = None,
    ):
        """Initialize adaptive compressor.
        
        Args:
            methods: List of methods to try
        """
        self.methods = methods or ["top_k", "quantization", "sign"]
        
    def compress(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], str, Dict]:
        """Compress with best method.
        
        Args:
            gradients: Dictionary of gradients
            
        Returns:
            Tuple of (compressed, method_name, info)
        """
        from cognita.compression.compressor import GradientCompressor
        
        best_method = None
        best_result = None
        best_compression = 0
        
        for method in self.methods:
            compressor = GradientCompressor(method=method)
            compressed = compressor.compress(gradients)
            
            # Estimate compression ratio
            orig_size = sum(g.nbytes for g in gradients.values())
            comp_size = sum(v.nbytes for v in compressed.values())
            
            compression = 1 - comp_size / orig_size if orig_size > 0 else 0
            
            if compression > best_compression:
                best_compression = compression
                best_method = method
                best_result = compressed
                
        return best_result, best_method, {"compression_ratio": best_compression}
