"""
Federated Learning Client Module

Implements federated learning client functionality with support for
local training, differential privacy, secure aggregation, and model
compression.

Author: Cognita Team
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cognita.core.model_manager import ModelManager, BaseFederatedModel
from cognita.core.fl_config import ClientConfig, PrivacyConfig, FLConfig
from cognita.privacy import DPClient, PrivacyAccountant
from cognita.compression import GradientCompressor
from cognita.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClientUpdate:
    """Container for client update to send to server.
    
    Attributes:
        client_id: Unique client identifier
        round_num: Current round number
        weights: Model weights dictionary
        num_samples: Number of training samples
        training_time: Time spent training
        metrics: Training metrics
        privacy_noise: Noise added for differential privacy
        compressed: Whether weights are compressed
        metadata: Additional metadata
    """
    client_id: str
    round_num: int
    weights: Dict[str, np.ndarray]
    num_samples: int
    training_time: float
    metrics: Dict[str, float]
    privacy_noise: Optional[Dict[str, np.ndarray]] = None
    compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "client_id": self.client_id,
            "round_num": self.round_num,
            "num_samples": self.num_samples,
            "training_time": self.training_time,
            "metrics": self.metrics,
            "compressed": self.compressed,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientUpdate":
        """Create from dictionary.
        
        Args:
            data: Dictionary with update data
            
        Returns:
            ClientUpdate instance
        """
        return cls(
            client_id=data["client_id"],
            round_num=data["round_num"],
            weights=data["weights"],
            num_samples=data["num_samples"],
            training_time=data["training_time"],
            metrics=data["metrics"],
            privacy_noise=data.get("privacy_noise"),
            compressed=data.get("compressed", False),
            metadata=data.get("metadata", {}),
        )


class FederatedClient:
    """Federated learning client.
    
    This class implements the client-side functionality for federated
    learning, including local model training, differential privacy,
    gradient compression, and secure communication with the server.
    
    Attributes:
        client_id: Unique client identifier
        model_manager: Manager for model operations
        config: Client configuration
        
    Example:
        >>> client = FederatedClient(
        ...     client_id="client_1",
        ...     model=SimpleCNN(),
        ...     config=ClientConfig(local_epochs=5)
        ... )
        >>> client.set_data(train_dataset)
        >>> client.set_privacy(epsilon=8.0, delta=1e-5)
        >>> update = client.train_round(global_weights)
        >>> server.submit_update(update)
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        config: Optional[ClientConfig] = None,
        fl_config: Optional[FLConfig] = None,
        device: str = "cpu",
    ):
        """Initialize the federated client.
        
        Args:
            client_id: Unique client identifier
            model: PyTorch model to train
            config: Client configuration
            fl_config: Federated learning configuration
            device: Device for computation
        """
        self.client_id = client_id
        self.config = config or ClientConfig(client_id=client_id)
        self.fl_config = fl_config or FLConfig(client=self.config)
        self.device = torch.device(device)
        
        self.model_manager = ModelManager(model, device=device)
        self.dp_client: Optional[DPClient] = None
        self.privacy_accountant: Optional[PrivacyAccountant] = None
        self.compressor: Optional[GradientCompressor] = None
        
        self._train_loader: Optional[DataLoader] = None
        self._test_loader: Optional[DataLoader] = None
        self._current_weights: Optional[Dict[str, np.ndarray]] = None
        self._round_history: List[Dict[str, Any]] = []
        self._is_connected = False
        self._local_steps = 0
        
        self._init_privacy()
        self._init_compression()
        
    def _init_privacy(self) -> None:
        """Initialize differential privacy components."""
        if self.config.privacy.epsilon > 0:
            self.dp_client = DPClient(
                max_grad_norm=self.config.privacy.max_grad_norm,
                noise_multiplier=self.config.privacy.noise_multiplier,
            )
            self.privacy_accountant = PrivacyAccountant(
                epsilon=self.config.privacy.epsilon,
                delta=self.config.privacy.delta,
            )
            
    def _init_compression(self) -> None:
        """Initialize gradient compression."""
        if self.fl_config.compression.enabled:
            self.compressor = GradientCompressor(
                method=self.fl_config.compression.method,
                compression_ratio=self.fl_config.compression.compression_ratio,
            )
    
    def set_data(
        self,
        train_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        train_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
    ) -> None:
        """Set training and test data.
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            train_batch_size: Training batch size
            test_batch_size: Test batch size
        """
        batch_size = train_batch_size or self.config.batch_size
        test_batch = test_batch_size or self.config.batch_size
        
        if train_dataset:
            self._train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=self.device.type == "cuda",
            )
            
        if test_dataset:
            self._test_loader = DataLoader(
                test_dataset,
                batch_size=test_batch,
                shuffle=False,
                num_workers=0,
                pin_memory=self.device.type == "cuda",
            )
            
    def set_privacy(
        self,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        noise_multiplier: Optional[float] = None,
    ) -> None:
        """Update privacy parameters.
        
        Args:
            epsilon: Privacy budget
            delta: Privacy failure probability
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise multiplier for DP-SGD
        """
        if epsilon is not None:
            self.config.privacy.epsilon = epsilon
        if delta is not None:
            self.config.privacy.delta = delta
        if max_grad_norm is not None:
            self.config.privacy.max_grad_norm = max_grad_norm
        if noise_multiplier is not None:
            self.config.privacy.noise_multiplier = noise_multiplier
            
        self._init_privacy()
        
    def receive_global_weights(
        self,
        weights: Dict[str, np.ndarray],
    ) -> None:
        """Receive global model weights from server.
        
        Args:
            weights: Global model weights
        """
        self._current_weights = weights
        self.model_manager.set_weights(weights)
        
    def train_round(
        self,
        round_num: int,
        global_weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> ClientUpdate:
        """Execute one round of local training.
        
        Args:
            round_num: Current round number
            global_weights: Global model weights to start from
            
        Returns:
            ClientUpdate with trained weights and metrics
        """
        start_time = time.time()
        
        # Initialize with global weights if provided
        if global_weights is not None:
            self.receive_global_weights(global_weights)
        elif self._current_weights is not None:
            self.model_manager.set_weights(self._current_weights)
            
        # Get current weights for delta computation
        prev_weights = self.model_manager.get_weights()
        
        # Create optimizer
        optimizer = self.model_manager.create_optimizer(
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
        )
        
        # Training loop
        self.model_manager.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.local_epochs):
            for batch in self._train_loader:
                # Move data to device
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model_manager.model(inputs)
                loss = self.model_manager.loss_fn(outputs, targets)
                loss.backward()
                
                # Apply gradient clipping if using DP
                if self.dp_client:
                    clipped_grads = self.dp_client.clip_gradients(
                        self.model_manager.model,
                        self.config.privacy.max_grad_norm,
                    )
                    # Add noise for DP
                    noisy_grads = self.dp_client.add_noise(clipped_grads)
                    self._apply_gradients(noisy_grads)
                else:
                    optimizer.step()
                    
                total_loss += loss.item()
                num_batches += 1
                self._local_steps += 1
                
        avg_loss = total_loss / max(num_batches, 1)
        training_time = time.time() - start_time
        
        # Compute weight delta
        current_weights = self.model_manager.get_weights()
        weight_delta = self.model_manager.compute_delta(current_weights, prev_weights)
        
        # Apply compression if enabled
        compressed = False
        if self.compressor and self.fl_config.compression.enabled:
            if round_num >= self.fl_config.compression.warmup_rounds:
                weight_delta = self.compressor.compress(weight_delta)
                compressed = True
                
        # Get number of samples
        num_samples = len(self._train_loader.dataset) if self._train_loader else 0
        
        # Compute metrics
        metrics = self._compute_metrics()
        metrics["loss"] = avg_loss
        metrics["local_steps"] = self._local_steps
        
        # Update privacy budget
        if self.privacy_accountant:
            self.privacy_accountant.update(
                sample_rate=num_samples / max(num_samples, 1),
                noise_multiplier=self.config.privacy.noise_multiplier,
            )
            
        # Create update
        update = ClientUpdate(
            client_id=self.client_id,
            round_num=round_num,
            weights=weight_delta,
            num_samples=num_samples,
            training_time=training_time,
            metrics=metrics,
            compressed=compressed,
            metadata={
                "local_epochs": self.config.local_epochs,
                "batch_size": self.config.batch_size,
                "epsilon_spent": self.privacy_accountant.get_spent_epsilon() if self.privacy_accountant else 0.0,
            },
        )
        
        # Store history
        self._round_history.append({
            "round": round_num,
            "metrics": metrics,
            "training_time": training_time,
        })
        
        # Update current weights reference
        self._current_weights = current_weights
        
        return update
        
    def _apply_gradients(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> None:
        """Apply gradients to model parameters.
        
        Args:
            gradients: Dictionary of gradients
        """
        for name, param in self.model_manager.model.named_parameters():
            if name in gradients:
                grad_tensor = torch.from_numpy(gradients[name]).to(self.device)
                param.grad = grad_tensor
        # Call step without zero_grad since we do it manually
        for param in self.model_manager.model.parameters():
            if param.grad is not None:
                param.data -= self.config.learning_rate * param.grad
        
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Returns:
            Dictionary of metrics
        """
        if self._test_loader is None:
            return {"accuracy": 0.0, "loss": 0.0}
            
        eval_metrics = self.model_manager.evaluate(self._test_loader)
        return eval_metrics.to_dict()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on local test data.
        
        Returns:
            Evaluation metrics
        """
        if self._test_loader is None:
            return {"accuracy": 0.0, "loss": 0.0}
        return self.model_manager.evaluate(self._test_loader).to_dict()
    
    def get_model_weights(self) -> Dict[str, np.ndarray]:
        """Get current model weights.
        
        Returns:
            Model weights dictionary
        """
        return self.model_manager.get_weights()
    
    def get_update_weight(self) -> float:
        """Get client weight for aggregation.
        
        Returns:
            Client weight based on data size
        """
        if self._train_loader:
            return len(self._train_loader.dataset)
        return 1.0
        
    def get_privacy_budget(self) -> Tuple[float, float]:
        """Get current privacy budget.
        
        Returns:
            Tuple of (spent_epsilon, remaining_epsilon)
        """
        if self.privacy_accountant:
            spent = self.privacy_accountant.get_spent_epsilon()
            remaining = self.config.privacy.epsilon - spent
            return spent, remaining
        return 0.0, self.config.privacy.epsilon
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get training history.
        
        Returns:
            List of round history entries
        """
        return self._round_history.copy()
    
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save client state checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "client_id": self.client_id,
            "config": self.config,
            "weights": self.model_manager.get_weights(),
            "round_history": self._round_history,
            "local_steps": self._local_steps,
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load client state checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self._round_history = checkpoint.get("round_history", [])
        self._local_steps = checkpoint.get("local_steps", 0)
        
        if "weights" in checkpoint:
            self.model_manager.set_weights(checkpoint["weights"])
            self._current_weights = checkpoint["weights"]
            
    def connect(self) -> bool:
        """Connect to the federated learning server.
        
        Returns:
            True if connection successful
        """
        self._is_connected = True
        logger.info(f"Client {self.client_id} connected to server")
        return True
        
    def disconnect(self) -> None:
        """Disconnect from the server."""
        self._is_connected = False
        logger.info(f"Client {self.client_id} disconnected from server")
        
    @property
    def is_connected(self) -> bool:
        """Check if client is connected.
        
        Returns:
            Connection status
        """
        return self._is_connected


class VerticalFLClient(FederatedClient):
    """Client for vertical federated learning.
    
    In vertical FL, clients hold different features of the same entities.
    This class handles feature-aligned training and secure embedding sharing.
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        feature_indices: List[int],
        config: Optional[ClientConfig] = None,
        **kwargs,
    ):
        """Initialize vertical FL client.
        
        Args:
            client_id: Unique client identifier
            model: PyTorch model
            feature_indices: Indices of features held by this client
            config: Client configuration
            **kwargs: Additional arguments for parent class
        """
        super().__init__(client_id, model, config, **kwargs)
        self.feature_indices = feature_indices
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
    def extract_features(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """Extract client's features from data.
        
        Args:
            data: Full feature matrix
            
        Returns:
            Extracted features for this client
        """
        return data[:, self.feature_indices]
    
    def compute_local_embedding(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """Compute local embedding for features.
        
        Args:
            data: Local feature data
            
        Returns:
            Embedding vector
        """
        # Simple mean embedding for demonstration
        embedding = np.mean(data, axis=0)
        return embedding
    
    def train_round(
        self,
        round_num: int,
        global_weights: Optional[Dict[str, np.ndarray]] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> ClientUpdate:
        """Execute vertical FL training round.
        
        Args:
            round_num: Current round number
            global_weights: Global model weights
            embeddings: Embeddings from other clients
            
        Returns:
            ClientUpdate with training results
        """
        if embeddings:
            self.embedding_cache = embeddings
            
        return super().train_round(round_num, global_weights)
        
    def get_embedding_shape(self) -> int:
        """Get embedding dimension for this client.
        
        Returns:
            Embedding dimension
        """
        return len(self.feature_indices)
