"""Aggregation strategies for federated learning."""

from cognita.aggregation.base import BaseAggregator
from cognita.aggregation.fedavg import FedAvgAggregator
from cognita.aggregation.fedprox import FedProxAggregator
from cognita.aggregation.scaffold import SCAFFOLDAggregator
from cognita.aggregation.async_agg import AsyncAggregator
from cognita.aggregation.fednova import FedNovaAggregator
from cognita.aggregation.fedopt import FedAdamAggregator, FedAdagradAggregator

__all__ = [
    "BaseAggregator",
    "FedAvgAggregator",
    "FedProxAggregator",
    "SCAFFOLDAggregator",
    "AsyncAggregator",
    "FedNovaAggregator",
    "FedAdamAggregator",
    "FedAdagradAggregator",
]
