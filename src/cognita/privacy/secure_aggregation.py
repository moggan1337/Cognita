"""
Secure Aggregation Module

Implements secure aggregation protocols for privacy-preserving
federated learning, including secret sharing and secure summation.

Author: Cognita Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import secrets
from dataclasses import dataclass

import numpy as np


@dataclass
class SecureAggregationConfig:
    """Configuration for secure aggregation."""
    num_clients: int
    threshold: int  # Minimum clients needed for reconstruction
    prime: int = 2**61 - 1  # Large prime for finite field arithmetic
    use_dh_key_exchange: bool = True
    compression_enabled: bool = True


class SecretSharer:
    """Secret sharing using Shamir's Secret Sharing scheme.
    
    Splits a secret into multiple shares such that any threshold
    number of shares can reconstruct the secret.
    """
    
    def __init__(self, threshold: int, num_shares: int, prime: int = 2**61 - 1):
        """Initialize secret sharer.
        
        Args:
            threshold: Minimum shares needed for reconstruction
            num_shares: Total number of shares to generate
            prime: Prime modulus for finite field
        """
        self.threshold = threshold
        self.num_shares = num_shares
        self.prime = prime
        
    def share(
        self,
        secret: Union[int, np.ndarray],
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Split secret into shares.
        
        Args:
            secret: Secret to split
            seed: Random seed for reproducibility
            
        Returns:
            List of (x, y) coordinate shares
        """
        rng = np.random.default_rng(seed)
        
        if isinstance(secret, np.ndarray):
            return self._share_array(secret, rng)
            
        # Generate random coefficients for polynomial
        coefficients = [secret % self.prime]
        for _ in range(self.threshold - 1):
            coefficients.append(rng.integers(0, self.prime))
            
        # Evaluate polynomial at each share point
        shares = []
        for x in range(1, self.num_shares + 1):
            y = sum(c * (x ** i) for i, c in enumerate(coefficients)) % self.prime
            shares.append((x, y))
            
        return shares
    
    def _share_array(
        self,
        secret: np.ndarray,
        rng: np.random.Generator,
    ) -> List[List[Tuple[int, int]]]:
        """Share a numpy array element-wise.
        
        Args:
            secret: Array to share
            rng: Random number generator
            
        Returns:
            List of shares for each element
        """
        shares = []
        secret_flat = secret.flatten()
        
        for value in secret_flat:
            value_int = int(value.view(np.uint64)) % self.prime
            coeff = [value_int]
            for _ in range(self.threshold - 1):
                coeff.append(rng.integers(0, self.prime))
                
            element_shares = []
            for x in range(1, self.num_shares + 1):
                y = sum(c * (x ** i) for i, c in enumerate(coeff)) % self.prime
                element_shares.append((x, y))
            shares.append(element_shares)
            
        return shares
    
    def reconstruct(
        self,
        shares: List[Tuple[int, int]],
    ) -> Union[int, np.ndarray]:
        """Reconstruct secret from shares.
        
        Args:
            shares: List of (x, y) shares
            
        Returns:
            Reconstructed secret
        """
        if len(shares) < self.threshold:
            raise ValueError(
                f"Need at least {self.threshold} shares, got {len(shares)}"
            )
            
        x_s = np.array([s[0] for s in shares[:self.threshold]])
        y_s = np.array([s[1] for s in shares[:self.threshold]])
        
        # Lagrange interpolation at x=0
        result = 0
        for i in range(self.threshold):
            # Lagrange basis polynomial
            numerator = 1
            denominator = 1
            for j in range(self.threshold):
                if i != j:
                    numerator = (numerator * (0 - x_s[j])) % self.prime
                    denominator = (denominator * (x_s[i] - x_s[j])) % self.prime
                    
            lagrange_coeff = (numerator * pow(denominator, -1, self.prime)) % self.prime
            result = (result + y_s[i] * lagrange_coeff) % self.prime
            
        return result


class SecureAggregator:
    """Secure aggregation for federated learning.
    
    Implements protocols for securely aggregating model updates
    from multiple clients without exposing individual updates.
    
    Attributes:
        config: Secure aggregation configuration
        
    Example:
        >>> config = SecureAggregationConfig(num_clients=10, threshold=7)
        >>> aggregator = SecureAggregator(config)
        >>> 
        >>> # Each client:
        >>> shares = aggregator.create_shares(client_weights, client_id)
        >>> 
        >>> # Server:
        >>> aggregated = aggregator.aggregate(shares)
    """
    
    def __init__(self, config: SecureAggregationConfig):
        """Initialize secure aggregator.
        
        Args:
            config: Secure aggregation configuration
        """
        self.config = config
        self.sharer = SecretSharer(
            threshold=config.threshold,
            num_shares=config.num_clients,
            prime=config.prime,
        )
        
        self._client_masks: Dict[int, Dict[str, np.ndarray]] = {}
        self._client_seeds: Dict[int, int] = {}
        self._collected_shares: Dict[int, List[Any]] = {}
        
    def generate_mask(
        self,
        client_id: int,
        shape: Tuple[int, ...],
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate random mask for a client.
        
        Args:
            client_id: Client identifier
            shape: Shape of weights to mask
            seed: Random seed
            
        Returns:
            Random mask array
        """
        if seed is None:
            seed = secrets.randbelow(self.config.prime)
            
        rng = np.random.default_rng(seed)
        mask = rng.integers(
            0, self.config.prime, size=shape, dtype=np.uint64
        ).astype(np.int64) % self.config.prime
        
        self._client_seeds[client_id] = seed
        return mask
        
    def create_shares(
        self,
        data: Union[np.ndarray, Dict[str, np.ndarray]],
        client_id: int,
    ) -> Dict[str, Any]:
        """Create shares of data for secure aggregation.
        
        Args:
            data: Data to share (array or dict of arrays)
            client_id: Client identifier
            
        Returns:
            Dictionary of shares for transmission
        """
        shares_data = {}
        
        if isinstance(data, dict):
            for name, array in data.items():
                shares = self._share_array(array)
                shares_data[name] = shares
        else:
            shares_data["_default"] = self._share_array(data)
            
        return {
            "client_id": client_id,
            "shares": shares_data,
        }
        
    def _share_array(self, array: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Share a numpy array.
        
        Args:
            array: Array to share
            
        Returns:
            List of shares (one per client)
        """
        # Flatten and convert to integers
        flat = array.flatten()
        flat_int = (flat * 1e6).astype(np.int64) % self.config.prime
        
        # Create shares for each element
        shares = [[] for _ in range(self.config.num_clients)]
        
        for value in flat_int:
            element_shares = self.sharer.share(value)
            for i, share in enumerate(element_shares):
                shares[i].append(share)
                
        return shares
        
    def aggregate(
        self,
        shares_list: List[Dict[str, Any]],
        target_shape: Optional[Tuple[int, ...]] = None,
    ) -> Optional[np.ndarray]:
        """Aggregate shares from clients.
        
        Args:
            shares_list: List of share dictionaries from clients
            target_shape: Expected output shape
            
        Returns:
            Aggregated result or None if insufficient shares
        """
        if len(shares_list) < self.config.threshold:
            return None
            
        # Reconstruct each share set
        reconstructed = []
        
        for shares_data in shares_list:
            shares = shares_data["shares"]
            
            # Get first array's shares
            if "_default" in shares:
                share_arrays = shares["_default"]
            else:
                # Take first key
                first_key = next(iter(shares))
                share_arrays = shares[first_key]
                
            # Reconstruct element by element
            num_elements = len(share_arrays)
            reconstructed_values = []
            
            for element_shares in share_arrays[:num_elements]:
                value = self.sharer.reconstruct(element_shares)
                reconstructed_values.append(value)
                
            reconstructed.append(np.array(reconstructed_values))
            
        # Average the reconstructed values
        result = np.mean(reconstructed, axis=0)
        
        if target_shape:
            result = result.reshape(target_shape)
            
        return result
        
    def compute_pairwise_masks(
        self,
        client_ids: List[int],
        shape: Tuple[int, ...],
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Compute pairwise masks for secure sum protocol.
        
        Args:
            client_ids: Participating client IDs
            shape: Shape of weights
            
        Returns:
            Dictionary of pairwise masks
        """
        masks = {}
        rng = np.random.default_rng()
        
        for i, client_i in enumerate(client_ids):
            for j, client_j in enumerate(client_ids):
                if i < j:
                    seed = rng.integers(0, 2**31)
                    mask = rng.integers(0, self.config.prime, size=shape)
                    masks[(client_i, client_j)] = mask
                    masks[(client_j, client_i)] = -mask % self.config.prime
                    
        return masks
        
    def secure_sum(
        self,
        values: Dict[int, np.ndarray],
        client_ids: List[int],
    ) -> np.ndarray:
        """Compute secure sum of values from multiple clients.
        
        Args:
            values: Dictionary mapping client_id to value
            client_ids: Participating client IDs
            
        Returns:
            Sum of all values
        """
        shape = next(iter(values.values())).shape
        
        # Generate pairwise masks
        masks = self.compute_pairwise_masks(client_ids, shape)
        
        # Each client computes masked value
        masked_values = {}
        for client_id in client_ids:
            masked = values[client_id].copy()
            
            # Add and subtract masks
            for other_id in client_ids:
                if client_id < other_id:
                    masked = (masked + masks.get((client_id, other_id), 0)) % self.config.prime
                elif client_id > other_id:
                    masked = (masked + masks.get((client_id, other_id), 0)) % self.config.prime
                    
            masked_values[client_id] = masked
            
        # Server sums all masked values (masks cancel out)
        total = sum(masked_values.values()) % self.config.prime
        
        return total.astype(np.float32) / 1e6  # Convert back to float
        
    def hash_commitment(
        self,
        data: np.ndarray,
    ) -> str:
        """Create hash commitment for data.
        
        Args:
            data: Data to commit to
            
        Returns:
            Hash commitment string
        """
        data_bytes = data.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()
        
    def verify_commitment(
        self,
        data: np.ndarray,
        commitment: str,
    ) -> bool:
        """Verify hash commitment.
        
        Args:
            data: Data to verify
            commitment: Expected commitment
            
        Returns:
            True if commitment matches
        """
        return self.hash_commitment(data) == commitment


def create_secure_aggregation(
    num_clients: int,
    threshold: Optional[int] = None,
) -> SecureAggregator:
    """Create a secure aggregator.
    
    Args:
        num_clients: Number of participating clients
        threshold: Threshold for reconstruction (default: 2/3 of clients)
        
    Returns:
        Configured SecureAggregator
    """
    if threshold is None:
        threshold = (2 * num_clients) // 3
        
    config = SecureAggregationConfig(
        num_clients=num_clients,
        threshold=threshold,
    )
    
    return SecureAggregator(config)
