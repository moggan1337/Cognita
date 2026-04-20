"""
Serialization Module

Provides utilities for serializing and deserializing model weights.

Author: Cognita Team
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


def serialize_weights(
    weights: Dict[str, np.ndarray],
    format: str = "numpy",
) -> bytes:
    """Serialize model weights.
    
    Args:
        weights: Dictionary of weights
        format: Serialization format ("numpy", "pickle", "json")
        
    Returns:
        Serialized bytes
    """
    if format == "pickle":
        return pickle.dumps(weights)
    elif format == "json":
        # Convert to serializable format
        serializable = {
            k: v.tolist() for k, v in weights.items()
        }
        return json.dumps(serializable).encode()
    else:
        # Numpy format
        arrays = [weights[k] for k in sorted(weights.keys())]
        return b"".join(a.tobytes() for a in arrays)
        
        
def deserialize_weights(
    data: bytes,
    format: str = "numpy",
    shapes: Optional[Dict[str, tuple]] = None,
) -> Dict[str, np.ndarray]:
    """Deserialize model weights.
    
    Args:
        data: Serialized bytes
        format: Serialization format
        shapes: Required for numpy format
        
    Returns:
        Dictionary of weights
    """
    if format == "pickle":
        return pickle.loads(data)
    elif format == "json":
        loaded = json.loads(data.decode())
        return {k: np.array(v) for k, v in loaded.items()}
    else:
        # Numpy format
        if shapes is None:
            raise ValueError("shapes required for numpy format")
            
        weights = {}
        offset = 0
        for name in sorted(shapes.keys()):
            shape = shapes[name]
            size = np.prod(shape) * 4  # float32
            weights[name] = np.frombuffer(data[offset:offset+size], dtype=np.float32)
            weights[name] = weights[name].reshape(shape)
            offset += size
            
        return weights
        
        
def save_weights(
    weights: Dict[str, np.ndarray],
    path: Union[str, Path],
) -> None:
    """Save weights to file.
    
    Args:
        weights: Dictionary of weights
        path: File path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save weights
    np.savez(path, **weights)
    
    
def load_weights(
    path: Union[str, Path],
) -> Dict[str, np.ndarray]:
    """Load weights from file.
    
    Args:
        path: File path
        
    Returns:
        Dictionary of weights
    """
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
