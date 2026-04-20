"""Unit tests for aggregation strategies."""

import pytest
import numpy as np
from cognita.aggregation import (
    FedAvgAggregator,
    FedProxAggregator,
    SCAFFOLDAggregator,
)
from cognita.aggregation import FedAdamAggregator, FedAdagradAggregator


class TestFedAvg:
    """Tests for FedAvg aggregator."""
    
    def test_simple_average(self):
        """Test simple weighted average."""
        aggregator = FedAvgAggregator()
        
        # Create mock updates
        weights1 = {"layer1": np.array([1.0, 2.0, 3.0])}
        weights2 = {"layer1": np.array([4.0, 5.0, 6.0])}
        
        updates = [
            (weights1, 1.0),
            (weights2, 1.0),
        ]
        
        result = aggregator.aggregate(updates)
        
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_allclose(result["layer1"], expected, rtol=1e-5)
        
    def test_weighted_average(self):
        """Test weighted average with different weights."""
        aggregator = FedAvgAggregator()
        
        weights1 = {"layer1": np.array([0.0, 0.0])}
        weights2 = {"layer1": np.array([10.0, 10.0])}
        
        updates = [
            (weights1, 1.0),
            (weights2, 3.0),
        ]
        
        result = aggregator.aggregate(updates)
        
        # Expected: (0 + 30) / 4 = 7.5
        expected = np.array([7.5, 7.5])
        np.testing.assert_allclose(result["layer1"], expected, rtol=1e-5)
        
    def test_empty_updates(self):
        """Test with empty updates."""
        aggregator = FedAvgAggregator()
        result = aggregator.aggregate([])
        assert result == {}


class TestFedProx:
    """Tests for FedProx aggregator."""
    
    def test_proximal_term(self):
        """Test proximal term computation."""
        aggregator = FedProxAggregator(mu=0.1)
        
        global_weights = {"layer1": np.array([1.0, 1.0])}
        aggregator.set_global_weights(global_weights)
        
        local_weights = {"layer1": np.array([2.0, 2.0])}
        
        proximal = aggregator.compute_proximal_term(local_weights)
        
        # ||w - w_global||^2 = (1 + 1) = 2
        # 0.5 * mu * 2 = 0.1
        assert abs(proximal - 0.1) < 1e-5


class TestSCAFFOLD:
    """Tests for SCAFFOLD aggregator."""
    
    def test_control_initialization(self):
        """Test control variate initialization."""
        aggregator = SCAFFOLDAggregator()
        
        shapes = {"layer1": (2, 3), "layer2": (3,)}
        aggregator.initialize_controls(shapes)
        
        assert aggregator._server_control is not None
        assert "layer1" in aggregator._server_control
        assert "layer2" in aggregator._server_control


class TestFedAdam:
    """Tests for FedAdam aggregator."""
    
    def test_aggregation(self):
        """Test FedAdam aggregation."""
        aggregator = FedAdamAggregator(lr=0.1)
        
        weights1 = {"layer1": np.array([1.0, 2.0])}
        weights2 = {"layer1": np.array([3.0, 4.0])}
        
        updates = [
            (weights1, 1.0),
            (weights2, 1.0),
        ]
        
        result = aggregator.aggregate(updates)
        
        # Should have some momentum accumulated
        assert "layer1" in result
        assert result["layer1"].shape == (2,)
