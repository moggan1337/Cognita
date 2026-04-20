"""Core module for Cognita federated learning platform."""

from cognita.core.fl_config import FLConfig, ClientConfig
from cognita.core.model_manager import ModelManager
from cognita.core.client import FederatedClient
from cognita.core.server import FederatedServer
from cognita.core.coordinator import FederatedCoordinator

__all__ = [
    "FLConfig",
    "ClientConfig",
    "ModelManager",
    "FederatedClient",
    "FederatedServer",
    "FederatedCoordinator",
]
