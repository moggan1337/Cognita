"""
Model Manager Module

Handles neural network model operations for federated learning,
including model creation, weight management, serialization, and
gradient computation.

Author: Cognita Team
"""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    loss: float
    accuracy: float
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }


class BaseFederatedModel(nn.Module):
    """Base class for federated learning models.
    
    This class extends PyTorch nn.Module with federated learning
    specific functionality like gradient tracking and model updates.
    """
    
    def __init__(self):
        super().__init__()
        self._gradient_history = []
        
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get model weights as a dictionary.
        
        Returns:
            Dictionary mapping parameter names to tensors
        """
        return {name: param.clone() for name, param in self.named_parameters()}
    
    def set_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Set model weights from a dictionary.
        
        Args:
            weights: Dictionary mapping parameter names to tensors
        """
        for name, param in self.named_parameters():
            if name in weights:
                param.data = weights[name].clone().to(param.device)
                
    def add_gradient(self, gradient: Dict[str, torch.Tensor]) -> None:
        """Add gradient to history for tracking.
        
        Args:
            gradient: Dictionary of gradients to track
        """
        self._gradient_history.append({
            name: grad.clone() for name, grad in gradient.items()
        })
        
    def get_gradient_norm(self) -> float:
        """Compute the L2 norm of gradients.
        
        Returns:
            L2 norm of all gradients
        """
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def clip_gradients(self, max_norm: float) -> float:
        """Clip gradients by global norm.
        
        Args:
            max_norm: Maximum allowed gradient norm
            
        Returns:
            Total gradient norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
    
    def federalize(self) -> "BaseFederatedModel":
        """Create a federated copy of the model.
        
        Returns:
            New model instance with copied weights
        """
        model = copy.deepcopy(self)
        return model


class ModelManager:
    """Manager for federated learning model operations.
    
    This class provides a high-level interface for managing neural network
    models in federated learning scenarios, including weight aggregation,
    model serialization, and evaluation utilities.
    
    Attributes:
        model: The managed PyTorch model
        device: Device for computation (cuda/cpu)
        optimizer_class: Optimizer class for training
        loss_fn: Loss function for training
        
    Example:
        >>> manager = ModelManager(model=MyModel(), device="cuda")
        >>> manager.save_checkpoint("checkpoint.pt", round=5)
        >>> manager.load_checkpoint("checkpoint.pt")
        >>> metrics = manager.evaluate(test_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        optimizer_class: type = optim.SGD,
        loss_fn: Optional[nn.Module] = None,
    ):
        """Initialize the model manager.
        
        Args:
            model: PyTorch model to manage
            device: Device for computation
            optimizer_class: Optimizer class for training
            loss_fn: Loss function (default: CrossEntropyLoss)
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self._checkpoints: Dict[int, Dict[str, Any]] = {}
        
    @property
    def parameters(self) -> List[torch.Tensor]:
        """Get model parameters as list.
        
        Returns:
            List of model parameters
        """
        return list(self.model.parameters())
    
    @property
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get model state dictionary.
        
        Returns:
            Model state dictionary
        """
        return self.model.state_dict()
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get model weights as numpy arrays.
        
        Returns:
            Dictionary mapping parameter names to numpy arrays
        """
        state = self.model.state_dict()
        return {k: v.cpu().numpy() for k, v in state.items()}
    
    def set_weights(self, weights: Dict[str, Union[np.ndarray, torch.Tensor]]) -> None:
        """Set model weights from dictionary.
        
        Args:
            weights: Dictionary mapping parameter names to weights
        """
        state = {}
        for k, v in weights.items():
            if isinstance(v, np.ndarray):
                state[k] = torch.from_numpy(v)
            else:
                state[k] = v
            state[k] = state[k].to(self.device)
        self.model.load_state_dict(state)
    
    def average_weights(
        self,
        weight_updates: List[Tuple[Dict[str, np.ndarray], float]],
    ) -> Dict[str, np.ndarray]:
        """Compute weighted average of model weights.
        
        Args:
            weight_updates: List of (weights, weight) tuples
            
        Returns:
            Averaged weights dictionary
        """
        if not weight_updates:
            return {}
            
        result = {}
        total_weight = sum(w for _, w in weight_updates)
        
        for name in weight_updates[0][0].keys():
            weighted_sum = np.zeros_like(weight_updates[0][0][name], dtype=np.float64)
            for weights, client_weight in weight_updates:
                weighted_sum += weights[name].astype(np.float64) * (client_weight / total_weight)
            result[name] = weighted_sum.astype(np.float32)
            
        return result
    
    def compute_delta(
        self,
        current_weights: Dict[str, np.ndarray],
        prev_weights: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compute weight delta (difference).
        
        Args:
            current_weights: Current model weights
            prev_weights: Previous model weights
            
        Returns:
            Weight delta dictionary
        """
        return {
            name: current_weights[name] - prev_weights[name]
            for name in current_weights.keys()
        }
    
    def apply_delta(
        self,
        base_weights: Dict[str, np.ndarray],
        delta: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Apply weight delta to base weights.
        
        Args:
            base_weights: Base weights to update
            delta: Weight delta to apply
            
        Returns:
            Updated weights
        """
        return {
            name: base_weights[name] + delta[name]
            for name in base_weights.keys()
        }
    
    def create_optimizer(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> torch.optim.Optimizer:
        """Create optimizer for model training.
        
        Args:
            lr: Learning rate
            momentum: Momentum factor
            weight_decay: L2 regularization
            **kwargs: Additional optimizer arguments
            
        Returns:
            Configured optimizer
        """
        if self.optimizer_class == optim.SGD:
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                **kwargs,
            )
        elif self.optimizer_class == optim.Adam:
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **kwargs,
            )
        elif self.optimizer_class == optim.AdamW:
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **kwargs,
            )
        return self.optimizer_class(self.model.parameters(), lr=lr, **kwargs)
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Execute single training step.
        
        Args:
            batch: (inputs, targets) tuple
            optimizer: Optimizer for gradient descent
            
        Returns:
            Loss value for the step
        """
        self.model.train()
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        
        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(
        self,
        data_loader: DataLoader,
        metrics: Optional[List[str]] = None,
    ) -> ModelMetrics:
        """Evaluate model on data.
        
        Args:
            data_loader: DataLoader for evaluation data
            metrics: List of metrics to compute
            
        Returns:
            ModelMetrics with evaluation results
        """
        self.model.eval()
        metrics = metrics or ["loss", "accuracy"]
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        # Compute additional metrics if sklearn is available
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        except ImportError:
            precision, recall, f1 = 0.0, 0.0, 0.0
            
        return ModelMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
        )
    
    def get_gradient(
        self,
        data_loader: DataLoader,
    ) -> Dict[str, np.ndarray]:
        """Compute gradients on data.
        
        Args:
            data_loader: DataLoader for gradient computation
            
        Returns:
            Dictionary of gradients
        """
        self.model.train()
        self.model.zero_grad()
        
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            break
            
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.cpu().numpy()
                
        return gradients
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        round_num: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            round_num: Current round number
            metadata: Additional metadata to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "round": round_num,
            "metadata": metadata or {},
        }
        
        if round_num is not None:
            self._checkpoints[round_num] = checkpoint
            
        torch.save(checkpoint, path)
        
    def load_checkpoint(
        self,
        path: Union[str, Path],
        load_optimizer: bool = False,
    ) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
        return {
            "round": checkpoint.get("round"),
            "metadata": checkpoint.get("metadata", {}),
        }
    
    def save_weights(self, path: Union[str, Path]) -> None:
        """Save model weights as numpy files.
        
        Args:
            path: Directory to save weights
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        weights = self.get_weights()
        for name, tensor in weights.items():
            np.save(path / f"{name.replace('.', '_')}.npy", tensor)
            
    def load_weights(self, path: Union[str, Path]) -> None:
        """Load model weights from numpy files.
        
        Args:
            path: Directory containing weight files
        """
        path = Path(path)
        weights = {}
        
        for file in path.glob("*.npy"):
            name = file.stem.replace('_', '.')
            weights[name] = np.load(file)
            
        self.set_weights(weights)


class SimpleCNN(BaseFederatedModel):
    """Simple CNN for federated learning experiments.
    
    A lightweight convolutional neural network suitable for
    MNIST, CIFAR-10, and similar image classification tasks.
    
    Architecture:
        - Conv layers with batch normalization
        - Max pooling
        - Fully connected layers
        - Dropout for regularization
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        """Initialize the CNN.
        
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class SimpleMLP(BaseFederatedModel):
    """Simple MLP for federated learning experiments.
    
    A multilayer perceptron suitable for tabular data and
    simple classification tasks.
    
    Architecture:
        - Input layer
        - Hidden layers with batch normalization and dropout
        - Output layer
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        """Initialize the MLP.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        hidden_dims = hidden_dims or [512, 256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        x = x.view(x.size(0), -1)
        return self.network(x)
