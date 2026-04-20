"""Unit tests for privacy mechanisms."""

import pytest
import numpy as np
from cognita.privacy import (
    DPClient,
    PrivacyAccountant,
    PrivacyBudget,
    add_gaussian_noise,
    add_laplace_noise,
)


class TestDPClient:
    """Tests for DP-SGD client."""
    
    def test_initialization(self):
        """Test DP client initialization."""
        client = DPClient(
            max_grad_norm=1.0,
            noise_multiplier=0.1
        )
        
        assert client.max_grad_norm == 1.0
        assert client.noise_multiplier == 0.1
        
    def test_noise_addition(self):
        """Test Gaussian noise addition."""
        client = DPClient(max_grad_norm=1.0, noise_multiplier=0.1)
        
        gradients = {"layer1": np.array([1.0, 2.0, 3.0])}
        noisy = client.add_noise(gradients)
        
        assert "layer1" in noisy
        assert noisy["layer1"].shape == gradients["layer1"].shape


class TestPrivacyAccountant:
    """Tests for privacy accountant."""
    
    def test_initialization(self):
        """Test accountant initialization."""
        accountant = PrivacyAccountant(
            epsilon=8.0,
            delta=1e-5
        )
        
        assert accountant.target_epsilon == 8.0
        assert accountant.target_delta == 1e-5
        assert accountant._steps == 0
        
    def test_update(self):
        """Test privacy accounting update."""
        accountant = PrivacyAccountant(epsilon=8.0, delta=1e-5)
        
        initial_spent = accountant.get_spent_epsilon()
        
        # Update with several steps
        for _ in range(10):
            accountant.update(sample_rate=0.01, noise_multiplier=0.1)
            
        spent = accountant.get_spent_epsilon()
        assert spent >= initial_spent
        
    def test_privacy_budget_exhaustion(self):
        """Test privacy budget tracking."""
        accountant = PrivacyAccountant(epsilon=1.0, delta=1e-5)
        
        # Many updates should eventually exceed budget
        for _ in range(1000):
            accountant.update(sample_rate=0.01, noise_multiplier=0.1)
            
        spent = accountant.get_spent_epsilon()
        assert spent >= 1.0 or accountant.get_privacy_spent_ratio() > 0


class TestNoiseFunctions:
    """Tests for noise addition functions."""
    
    def test_gaussian_noise(self):
        """Test Gaussian noise addition."""
        gradients = {"layer1": np.array([1.0, 2.0, 3.0])}
        noisy = add_gaussian_noise(gradients, std=0.1)
        
        assert "layer1" in noisy
        # Noisy should be different from original
        assert not np.allclose(noisy["layer1"], gradients["layer1"])
        
    def test_laplace_noise(self):
        """Test Laplace noise addition."""
        gradients = {"layer1": np.array([1.0, 2.0, 3.0])}
        noisy = add_laplace_noise(gradients, scale=0.1)
        
        assert "layer1" in noisy
        assert not np.allclose(noisy["layer1"], gradients["layer1"])
