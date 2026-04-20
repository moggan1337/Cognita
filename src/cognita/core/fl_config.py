"""
Federated Learning Configuration Module

This module provides configuration classes for federated learning experiments,
including client settings, server settings, and privacy parameters.

Author: Cognita Team
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np


class AggregationStrategy(str, Enum):
    """Available aggregation strategies."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    FEDMED = "fedmed"
    FEDNOVA = "fednova"
    ASYNC = "async"
    FEDADAM = "fedadam"
    FEDADagrad = "fedadagrad"


class FLType(str, Enum):
    """Federated learning type."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class ClientMode(str, Enum):
    """Client operating mode."""
    TRAINING = "training"
    EVALUATION = "evaluation"
    STANDBY = "standby"


@dataclass
class PrivacyConfig:
    """Differential privacy configuration.
    
    Attributes:
        epsilon: Privacy budget (lower = stronger privacy)
        delta: Privacy failure probability
        max_grad_norm: Maximum gradient norm for clipping
        noise_multiplier: Ratio of noise std to max_grad_norm
        secure_aggregation: Enable secure aggregation
        communication_compression: Enable compression
        compression_ratio: Target compression ratio (0-1)
    """
    epsilon: float = 8.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 0.1
    secure_aggregation: bool = False
    communication_compression: bool = False
    compression_ratio: float = 0.5
    adaptive_noise: bool = True
    target_epsilon: float = 8.0
    min_epsilon: float = 1.0
    
    def __post_init__(self):
        """Validate privacy configuration."""
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0 < self.delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be non-negative")


@dataclass
class ClientConfig:
    """Configuration for a federated learning client.
    
    Attributes:
        client_id: Unique client identifier
        enabled: Whether client is active
        local_epochs: Number of local training epochs
        batch_size: Training batch size
        learning_rate: Local learning rate
        optimizer: Optimizer name
        momentum: Momentum for optimizer
        weight: Client weight for aggregation
        data_partition: Indices of local data partition
        features: Feature indices for vertical FL
        labels: Label indices for vertical FL
        mode: Client operating mode
        privacy: Privacy configuration
    """
    client_id: str
    enabled: bool = True
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight: float = 1.0
    data_partition: Optional[np.ndarray] = None
    features: Optional[List[int]] = None
    labels: Optional[List[int]] = None
    mode: ClientMode = ClientMode.TRAINING
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    
    def __hash__(self):
        return hash(self.client_id)
    
    def __eq__(self, other):
        if not isinstance(other, ClientConfig):
            return False
        return self.client_id == other.client_id


@dataclass
class ServerConfig:
    """Configuration for the federated learning server.
    
    Attributes:
        num_rounds: Total number of communication rounds
        clients_per_round: Number of clients per round
        min_clients: Minimum clients required to proceed
        aggregation_strategy: Strategy for model aggregation
        model_save_path: Path to save global model
        checkpoint_interval: Rounds between checkpoints
        early_stopping_rounds: Rounds for early stopping
        early_stopping_delta: Minimum improvement threshold
        timeout: Client response timeout in seconds
        sample_strategy: Client sampling strategy
        fl_type: Horizontal or vertical FL
    """
    num_rounds: int = 100
    clients_per_round: int = 10
    min_clients: int = 3
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDAVG
    model_save_path: str = "./checkpoints"
    checkpoint_interval: int = 10
    early_stopping_rounds: int = 20
    early_stopping_delta: float = 0.001
    timeout: float = 300.0
    sample_strategy: str = "uniform"  # uniform, stratified, velocity
    fl_type: FLType = FLType.HORIZONTAL
    evaluation_interval: int = 5
    test_interval: int = 10
    
    def __post_init__(self):
        """Validate server configuration."""
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be positive")
        if self.clients_per_round < self.min_clients:
            raise ValueError("clients_per_round must be >= min_clients")


@dataclass
class ByzantineConfig:
    """Byzantine resilience configuration.
    
    Attributes:
        enabled: Enable Byzantine resilience
        attack_type: Type of attack to defend against
        num_byzantine: Maximum number of Byzantine clients
        defense_method: Defense method (krum, trimmed_mean, etc.)
        krum_multiplier: Multiplier for Krum neighbor selection
        trim_ratio: Ratio to trim for trimmed mean
    """
    enabled: bool = False
    attack_type: str = "label_flipping"  # label_flipping, gaussian, zero_grad
    num_byzantine: int = 0
    defense_method: str = "krum"
    krum_multiplier: float = 2.0
    trim_ratio: float = 0.1
    fgsm_epsilon: float = 0.1


@dataclass 
class CompressionConfig:
    """Model compression configuration.
    
    Attributes:
        enabled: Enable compression
        method: Compression method (top_k, random_k, quantization, etc.)
        compression_ratio: Ratio of parameters to keep
        quantize_levels: Number of quantization levels
        warmup_rounds: Rounds before compression starts
        adaptive: Adapt compression based on bandwidth
    """
    enabled: bool = False
    method: str = "top_k"
    compression_ratio: float = 0.1
    quantize_levels: int = 8
    warmup_rounds: int = 5
    adaptive: bool = False
    sparsity_pattern: str = "uniform"


@dataclass
class CommunicationConfig:
    """Communication efficiency configuration.
    
    Attributes:
        enable_compression: Enable gradient compression
        enable_skip_comm: Skip communication for similar updates
        similarity_threshold: Threshold for skip communication
        enable_structured_updates: Use structured updates
        gradient_estimation: Use gradient estimation
    """
    enable_compression: bool = False
    enable_skip_comm: bool = False
    similarity_threshold: float = 0.95
    enable_structured_updates: bool = False
    gradient_estimation: bool = False
    staleness_weight: float = 0.5
    max_staleness: int = 5


class FLConfig:
    """Main federated learning configuration.
    
    This class aggregates all configuration options for a federated
    learning experiment, including client, server, privacy, and
    Byzantine resilience settings.
    
    Example:
        >>> config = FLConfig(
        ...     server=ServerConfig(num_rounds=50, clients_per_round=5),
        ...     client=ClientConfig(local_epochs=3),
        ...     privacy=PrivacyConfig(epsilon=8.0, delta=1e-5),
        ...     byzantine=ByzantineConfig(enabled=True, num_byzantine=1)
        ... )
        >>> print(config.to_dict())
    """
    
    def __init__(
        self,
        server: Optional[ServerConfig] = None,
        client: Optional[ClientConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
        byzantine: Optional[ByzantineConfig] = None,
        compression: Optional[CompressionConfig] = None,
        communication: Optional[CommunicationConfig] = None,
    ):
        self.server = server or ServerConfig()
        self.client = client or ClientConfig(client_id="default")
        self.privacy = privacy or PrivacyConfig()
        self.byzantine = byzantine or ByzantineConfig()
        self.compression = compression or CompressionConfig()
        self.communication = communication or CommunicationConfig()
        self._client_registry: Dict[str, ClientConfig] = {}
        
    def add_client(self, client_config: ClientConfig) -> None:
        """Register a client with the configuration.
        
        Args:
            client_config: Client configuration to register
        """
        self._client_registry[client_config.client_id] = client_config
        
    def remove_client(self, client_id: str) -> bool:
        """Remove a client from the registry.
        
        Args:
            client_id: ID of client to remove
            
        Returns:
            True if client was removed, False if not found
        """
        return self._client_registry.pop(client_id, None) is not None
        
    def get_client(self, client_id: str) -> Optional[ClientConfig]:
        """Get client configuration by ID.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Client configuration or None
        """
        return self._client_registry.get(client_id)
    
    def get_active_clients(self) -> List[ClientConfig]:
        """Get list of active (enabled) clients.
        
        Returns:
            List of active client configurations
        """
        return [c for c in self._client_registry.values() if c.enabled]
    
    def get_client_weights(self) -> Dict[str, float]:
        """Get normalized client weights for aggregation.
        
        Returns:
            Dictionary mapping client_id to normalized weight
        """
        active = self.get_active_clients()
        total_weight = sum(c.weight for c in active)
        if total_weight == 0:
            return {c.client_id: 1.0 / len(active) for c in active}
        return {c.client_id: c.weight / total_weight for c in active}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "server": {
                "num_rounds": self.server.num_rounds,
                "clients_per_round": self.server.clients_per_round,
                "min_clients": self.server.min_clients,
                "aggregation_strategy": self.server.aggregation_strategy.value,
                "fl_type": self.server.fl_type.value,
            },
            "privacy": {
                "epsilon": self.privacy.epsilon,
                "delta": self.privacy.delta,
                "max_grad_norm": self.privacy.max_grad_norm,
                "noise_multiplier": self.privacy.noise_multiplier,
                "secure_aggregation": self.privacy.secure_aggregation,
            },
            "byzantine": {
                "enabled": self.byzantine.enabled,
                "defense_method": self.byzantine.defense_method,
                "num_byzantine": self.byzantine.num_byzantine,
            },
            "compression": {
                "enabled": self.compression.enabled,
                "method": self.compression.method,
                "compression_ratio": self.compression.compression_ratio,
            },
            "num_registered_clients": len(self._client_registry),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FLConfig":
        """Create configuration from dictionary.
        
        Args:
            data: Dictionary with configuration data
            
        Returns:
            FLConfig instance
        """
        server = ServerConfig(**data.get("server", {}))
        client = ClientConfig(client_id="default", **data.get("client", {}))
        privacy = PrivacyConfig(**data.get("privacy", {}))
        byzantine = ByzantineConfig(**data.get("byzantine", {}))
        compression = CompressionConfig(**data.get("compression", {}))
        
        return cls(
            server=server,
            client=client,
            privacy=privacy,
            byzantine=byzantine,
            compression=compression,
        )
    
    def copy(self) -> "FLConfig":
        """Create a deep copy of the configuration.
        
        Returns:
            Deep copy of this configuration
        """
        return copy.deepcopy(self)
    
    def update_privacy_budget(self, spent: float) -> None:
        """Update remaining privacy budget.
        
        Args:
            spent: Privacy budget spent in current round
        """
        self.privacy.epsilon = max(
            self.privacy.min_epsilon, 
            self.privacy.epsilon - spent
        )
    
    def get_privacy_spent_ratio(self) -> float:
        """Get ratio of privacy budget spent.
        
        Returns:
            Ratio of budget spent (0-1)
        """
        return 1 - (self.privacy.epsilon / self.privacy.target_epsilon)


class ExperimentTracker:
    """Track federated learning experiment metrics.
    
    This class provides utilities for tracking and logging experiment
    metrics, including round-by-round performance and client statistics.
    """
    
    def __init__(self):
        self.rounds: List[Dict[str, Any]] = []
        self.client_history: Dict[str, List[Dict[str, Any]]] = {}
        
    def log_round(
        self,
        round_num: int,
        metrics: Dict[str, float],
        num_clients: int,
        duration: float,
    ) -> None:
        """Log metrics for a round.
        
        Args:
            round_num: Current round number
            metrics: Dictionary of metric names to values
            num_clients: Number of clients participated
            duration: Round duration in seconds
        """
        self.rounds.append({
            "round": round_num,
            "metrics": metrics,
            "num_clients": num_clients,
            "duration": duration,
        })
        
    def log_client(
        self,
        client_id: str,
        round_num: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log client-specific metrics.
        
        Args:
            client_id: Client identifier
            round_num: Current round number
            metrics: Client metrics dictionary
        """
        if client_id not in self.client_history:
            self.client_history[client_id] = []
        self.client_history[client_id].append({
            "round": round_num,
            **metrics,
        })
        
    def get_round_metrics(self, round_num: int) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific round.
        
        Args:
            round_num: Round number to retrieve
            
        Returns:
            Round metrics or None if not found
        """
        for r in self.rounds:
            if r["round"] == round_num:
                return r
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.rounds:
            return {}
            
        return {
            "total_rounds": len(self.rounds),
            "total_clients": len(self.client_history),
            "avg_round_duration": np.mean([r["duration"] for r in self.rounds]),
            "final_metrics": self.rounds[-1]["metrics"] if self.rounds else {},
            "best_metrics": {
                k: max(r["metrics"].get(k, 0) for r in self.rounds)
                for k in self.rounds[0]["metrics"].keys()
            } if self.rounds else {},
        }
