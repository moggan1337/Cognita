"""
Cognita - Federated Learning Platform
=====================================

A comprehensive federated learning platform with privacy-preserving mechanisms,
supporting horizontal and vertical federated learning, differential privacy,
secure aggregation, and Byzantine-resilient protocols.

Example:
    >>> from cognita import FederatedClient, FederatedServer, FedAvgAggregator
    >>> from cognita.privacy import DPClient, PrivacyAccountant
    >>> from cognita.aggregation import SCAFFOLDAggregator
    
    >>> # Setup server
    >>> server = FederatedServer(model, aggregator=FedAvgAggregator())
    >>> server.start()
    
    >>> # Add clients
    >>> client = FederatedClient(model, client_id="client_1")
    >>> client.set_data(train_data, train_labels)
    >>> client.set_privacy(epsilon=8.0, delta=1e-5)
"""

__version__ = "1.0.0"
__author__ = "Cognita Team"

from cognita.core.client import FederatedClient
from cognita.core.server import FederatedServer
from cognita.core.model_manager import ModelManager
from cognita.core.fl_config import FLConfig, ClientConfig
from cognita.aggregation import (
    FedAvgAggregator,
    FedProxAggregator,
    SCAFFOLDAggregator,
    AsyncAggregator,
)
from cognita.privacy import DPClient, PrivacyAccountant, PrivacyBudget
from cognita.byzantine import ByzantineResilientAggregator, KrumAggregator, TrimmedMeanAggregator

__all__ = [
    # Core
    "FederatedClient",
    "FederatedServer",
    "ModelManager",
    "FLConfig",
    "ClientConfig",
    # Aggregation
    "FedAvgAggregator",
    "FedProxAggregator",
    "SCAFFOLDAggregator",
    "AsyncAggregator",
    # Privacy
    "DPClient",
    "PrivacyAccountant",
    "PrivacyBudget",
    # Byzantine
    "ByzantineResilientAggregator",
    "KrumAggregator",
    "TrimmedMeanAggregator",
]
