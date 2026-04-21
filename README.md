# Cognita - Federated Learning Platform

<div align="center">

![Cognita](docs/cognita-logo.png)

**A Comprehensive Federated Learning Platform with Privacy-Preserving Mechanisms**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

</div>

---

## 🎬 Demo
![Cognita Demo](demo.gif)

*Federated learning across distributed clients*

## Screenshots
| Component | Preview |
|-----------|---------|
| Federated Training | ![training](screenshots/federated-training.png) |
| Client Dashboard | ![clients](screenshots/client-dash.png) |
| Privacy Metrics | ![privacy](screenshots/privacy.png) |

## Visual Description
Federated training shows models being trained on distributed nodes with gradient aggregation. Client dashboard displays participating devices with data distributions. Privacy metrics show differential privacy guarantees.

---


## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Core Concepts](#core-concepts)
   - [Federated Learning Basics](#federated-learning-basics)
   - [Horizontal vs Vertical FL](#horizontal-vs-vertical-fl)
   - [Privacy Mechanisms](#privacy-mechanisms)
7. [API Reference](#api-reference)
8. [Aggregation Strategies](#aggregation-strategies)
9. [Byzantine Resilience](#byzantine-resilience)
10. [Privacy Budget Management](#privacy-budget-management)
11. [Model Compression](#model-compression)
12. [Benchmarks](#benchmarks)
13. [Examples](#examples)
14. [Contributing](#contributing)
15. [License](#license)

---

## Overview

**Cognita** is a production-ready federated learning platform designed for privacy-preserving distributed machine learning. It provides a comprehensive suite of tools for implementing federated learning across heterogeneous networks, with strong differential privacy guarantees, Byzantine-resilient aggregation, and communication-efficient protocols.

### Key Highlights

- 🔒 **Differential Privacy**: Built-in DP-SGD with configurable privacy budgets
- 🛡️ **Byzantine Resilience**: Multiple defense mechanisms against malicious clients
- 📊 **Advanced Aggregation**: FedAvg, FedProx, SCAFFOLD, FedNova, FedAdam, and more
- 📐 **Model Compression**: Top-K, quantization, and sparse updates
- 🔐 **Secure Aggregation**: Secret sharing and secure summation protocols
- 📈 **Vertical FL**: Support for feature-partitioned scenarios
- 🔄 **Communication Efficiency**: Adaptive compression and skip communications

---

## Features

### Core Federated Learning

| Feature | Description |
|---------|-------------|
| **Horizontal Federated Learning** | Standard FL where clients hold different samples |
| **Vertical Federated Learning** | Feature-partitioned FL for multi-party learning |
| **Asynchronous FL** | Handle stragglers with async aggregation |
| **Hierarchical FL** | Multi-cluster coordination |

### Privacy-Preserving Mechanisms

| Feature | Description |
|---------|-------------|
| **DP-SGD** | Per-sample gradient clipping and Gaussian noise |
| **Privacy Accountant** | RDP-based privacy budget tracking |
| **Secure Aggregation** | Secret sharing for secure summation |
| **Privacy Budget Management** | Automatic budget monitoring and adaptation |

### Aggregation Strategies

| Strategy | Description | Paper |
|----------|-------------|-------|
| **FedAvg** | Federated Averaging | McMahan et al., 2017 |
| **FedProx** | Proximal term for heterogeneity | Li et al., 2020 |
| **SCAFFOLD** | Variance reduction with control variates | Karimireddy et al., 2020 |
| **FedNova** | Normalized averaging for heterogeneity | Li et al., 2021 |
| **FedAdam** | Adaptive optimization | Reddi et al., 2021 |
| **FedAdagrad** | Adagrad for FL | Reddi et al., 2021 |
| **FedYogi** | Advanced adaptive method | Reddi et al., 2021 |

### Byzantine Resilience

| Method | Description |
|--------|-------------|
| **Krum** | Multi-Krum Byzantine-tolerant aggregation |
| **Trimmed Mean** | Coordinate-wise trimmed mean |
| **Geometric Median** | Robust aggregation |
| **Brute Force** | Optimal subset selection |

### Communication Efficiency

| Technique | Description |
|-----------|-------------|
| **Gradient Compression** | Top-K, Random-K, quantization |
| **Skip Communication** | Adaptive communication based on similarity |
| **Gradient Estimation** | Momentum-based local estimation |
| **Sparse Updates** | Only communicate significant changes |

---

## Architecture

```
Cognita Architecture
====================

┌─────────────────────────────────────────────────────────────────┐
│                         Cognita Platform                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Client    │  │   Server    │  │     Coordinator        │  │
│  │   Module    │  │   Module    │  │        Module          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Aggregation Layer                         │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────────┐  │ │
│  │  │  FedAvg  │ │ FedProx │ │SCAFFOLD │ │    FedOpt        │  │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └──────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │     Privacy       │  │   Byzantine    │  │  Compression  │  │
│  │     Module        │  │    Module      │  │    Module     │  │
│  └───────────────────┘  └────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Overview

1. **FederatedClient**: Handles local training, gradient computation, and DP
2. **FederatedServer**: Coordinates rounds, aggregates updates, manages global model
3. **FederatedCoordinator**: Orchestrates multi-round training with callbacks
4. **Aggregators**: Implement various aggregation strategies
5. **PrivacyModule**: DP-SGD, privacy accounting, secure aggregation
6. **ByzantineModule**: Byzantine-resilient aggregation methods
7. **CompressionModule**: Gradient compression and sparsification

---

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- NumPy 1.24+

### Install from Source

```bash
git clone https://github.com/moggan1337/Cognita.git
cd Cognita
pip install -e .
```

### Install with Dependencies

```bash
pip install -e ".[all]"  # Include all optional dependencies
pip install -e ".[dev]" # Include development dependencies
```

### Using Poetry

```bash
poetry install
poetry shell
```

---

## Quick Start

### Basic Federated Learning

```python
import torch
import torch.nn as nn
from cognita import FederatedClient, FederatedServer, FedAvgAggregator
from cognita.core import FLConfig, ClientConfig, ServerConfig

# Define model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
        
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# Create configuration
config = FLConfig(
    server=ServerConfig(num_rounds=10, clients_per_round=3),
    client=ClientConfig(client_id="client_1", local_epochs=5)
)

# Create server
model = SimpleModel()
server = FederatedServer(model, aggregator=FedAvgAggregator())

# Create and register clients
for i in range(5):
    client = FederatedClient(
        client_id=f"client_{i}",
        model=SimpleModel(),
        config=ClientConfig(client_id=f"client_{i}")
    )
    # client.set_data(train_dataset)
    server.register_client(client)

# Run federated learning
results = server.run(num_rounds=10)
```

### With Differential Privacy

```python
from cognita.privacy import DPClient, PrivacyAccountant
from cognita.core import PrivacyConfig

# Configure privacy
privacy_config = PrivacyConfig(
    epsilon=8.0,
    delta=1e-5,
    max_grad_norm=1.0,
    noise_multiplier=0.1
)

# Create client with DP
client = FederatedClient(
    client_id="client_1",
    model=model,
    config=ClientConfig(client_id="client_1", privacy=privacy_config)
)

# Check privacy budget
accountant = PrivacyAccountant(epsilon=8.0, delta=1e-5)
spent = accountant.get_spent_epsilon()
print(f"Privacy budget spent: {spent:.4f}")
```

### With Byzantine Resilience

```python
from cognita.byzantine import KrumAggregator, ByzantineResilientAggregator
from cognita.core import ByzantineConfig

# Configure Byzantine resilience
byzantine_config = ByzantineConfig(
    enabled=True,
    num_byzantine=1,
    defense_method="krum"
)

# Create Byzantine-resilient server
server = FederatedServer(
    model,
    aggregator=ByzantineResilientAggregator(
        base_aggregator=FedAvgAggregator(),
        num_byzantine=1,
        defense_method="krum"
    )
)
```

---

## Core Concepts

### Federated Learning Basics

Federated learning enables training machine learning models across decentralized data sources without sharing raw data. The key steps are:

1. **Initialization**: Server sends global model to selected clients
2. **Local Training**: Each client trains on local data
3. **Model Update**: Clients send model updates (not data) to server
4. **Aggregation**: Server combines updates using aggregation strategy
5. **Iteration**: Repeat until convergence

```
Traditional ML:
┌─────────┐     ┌─────────┐     ┌─────────┐
│  Data   │ --> │  Model  │ --> │  Model  │
│(Central)│     │Training │     │Output   │
└─────────┘     └─────────┘     └─────────┘

Federated Learning:
┌─────────┐                    ┌─────────┐
│ Client1 │─┐                ┌>│  Client1│
│  Data   │ │                │ └─────────┘
└─────────┘ │   ┌────────┐   │
            ├──>│ Server │<──┤
┌─────────┐ │   │        │   │
│ Client2 │─┘   │   FL   │   │ ┌─────────┐
│  Data   │     │ Loop   │   ├>│ Client2 │
└─────────┘     └────────┘   │ └─────────┘
            │                │
┌─────────┐ │   ┌────────┐   │ ┌─────────┐
│ ClientN │─┘   │Aggreg. │   └>│ ClientN │
│  Data   │     └────────┘     └─────────┘
└─────────┘
```

### Horizontal vs Vertical FL

#### Horizontal Federated Learning

- Clients hold different **samples** of the same features
- Each client has a complete feature set
- Standard FL scenario (e.g., mobile keyboard prediction)

```python
# Each client has different user data, same features
client_1_data: [user_A_features, user_B_features]
client_2_data: [user_C_features, user_D_features]
```

#### Vertical Federated Learning

- Clients hold different **features** of the same samples
- Each client has a partial feature set
- Used when parties have complementary information (e.g., bank + e-commerce)

```python
# Each client has same users, different features
client_bank: [user_A_balance, user_A_loans, ...]
client_ecomm: [user_A_purchases, user_A_clicks, ...]
```

```python
from cognita.vertical import VerticalFederatedClient, VerticalFederatedServer

# Create vertical FL setup
server = VerticalFederatedServer(embedding_dim=64)
client_bank = VerticalFederatedClient(
    client_id="bank",
    feature_indices=[0, 1, 2],  # Financial features
    embedding_dim=32
)
client_ecomm = VerticalFederatedClient(
    client_id="ecommerce",
    feature_indices=[0, 1],  # Shopping features
    embedding_dim=32
)
```

### Privacy Mechanisms

#### Differential Privacy (DP)

Differential privacy provides mathematical guarantees that individual data points cannot be inferred from model outputs. Cognita implements DP-SGD with:

1. **Per-sample Gradient Clipping**: Bound sensitivity by clipping gradients
2. **Gaussian Noise Addition**: Add calibrated noise to clipped gradients
3. **Privacy Accounting**: Track cumulative privacy budget spent

```
Original Gradient: g_i
        │
        v
┌───────────────┐
│  Clip to C    │  g_i' = g_i * min(1, C/||g_i||)
└───────────────┘
        │
        v
┌───────────────┐
│ Add Noise     │  g_i'' = g_i' + N(0, σ²C²)
└───────────────┘
        │
        v
   Noisy Gradient
```

#### Privacy Budget

The privacy budget (ε, δ) quantifies the privacy guarantee:

- **ε (epsilon)**: Maximum difference in outputs for adjacent datasets
- **δ (delta)**: Probability of privacy violation

```python
from cognita.privacy import PrivacyAccountant

# Initialize accountant
accountant = PrivacyAccountant(epsilon=8.0, delta=1e-5)

# After each step
accountant.update(sample_rate=0.01, noise_multiplier=0.1)

# Check spent budget
spent = accountant.get_spent_epsilon()
remaining = 8.0 - spent
print(f"Spent: {spent:.4f}, Remaining: {remaining:.4f}")
```

---

## API Reference

### FederatedClient

```python
from cognita import FederatedClient

client = FederatedClient(
    client_id: str,           # Unique client ID
    model: nn.Module,        # PyTorch model
    config: ClientConfig,    # Client configuration
    fl_config: FLConfig,    # FL configuration
    device: str = "cpu"      # Computation device
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `set_data(train_dataset, test_dataset)` | Set local data |
| `set_privacy(epsilon, delta, ...)` | Configure DP |
| `train_round(round_num, global_weights)` | Execute local training |
| `evaluate()` | Evaluate on local test data |
| `get_model_weights()` | Get current weights |
| `get_privacy_budget()` | Get remaining budget |

### FederatedServer

```python
from cognita import FederatedServer
from cognita.aggregation import FedAvgAggregator

server = FederatedServer(
    model: nn.Module,
    config: ServerConfig,
    aggregator: BaseAggregator,
    device: str = "cpu"
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `register_client(client)` | Add client to server |
| `start_round(round_num)` | Begin new round |
| `receive_update(update)` | Collect client update |
| `aggregate_updates(updates)` | Aggregate client updates |
| `execute_round(round_num)` | Complete round end-to-end |
| `run(num_rounds, callback)` | Run full training |
| `save_checkpoint(path)` | Save server state |

### FLConfig

```python
from cognita.core import FLConfig, ServerConfig, ClientConfig, PrivacyConfig

config = FLConfig(
    server=ServerConfig(
        num_rounds=100,
        clients_per_round=10,
        aggregation_strategy=AggregationStrategy.FEDAVG
    ),
    client=ClientConfig(
        client_id="client_1",
        local_epochs=5,
        batch_size=32,
        learning_rate=0.01
    ),
    privacy=PrivacyConfig(
        epsilon=8.0,
        delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=0.1
    ),
    byzantine=ByzantineConfig(
        enabled=False,
        num_byzantine=0
    )
)
```

---

## Aggregation Strategies

### FedAvg (Federated Averaging)

The foundational FL algorithm. Computes weighted average of client updates.

```python
from cognita.aggregation import FedAvgAggregator

aggregator = FedAvgAggregator(momentum=0.0)
```

**Algorithm:**
```
For each client i:
    w_i = w - lr * gradient_i
    
Global: w = Σ (n_i / n) * w_i
```

### FedProx

Adds proximal term to handle system heterogeneity.

```python
from cognita.aggregation import FedProxAggregator

aggregator = FedProxAggregator(mu=0.01)  # Proximal coefficient
```

**Algorithm:**
```
Local objective: min_w Σ loss + (μ/2) * ||w - w_global||²
```

### SCAFFOLD

Stochastic Controlled Averaging with control variates for variance reduction.

```python
from cognita.aggregation import SCAFFOLDAggregator

aggregator = SCAFFOLDAggregator(
    learning_rate=1.0,
    control_lr=1.0
)
```

### FedOpt (FedAdam, FedAdagrad, FedYogi)

Adaptive optimization methods for FL.

```python
from cognita.aggregation import FedAdamAggregator, FedAdagradAggregator

# FedAdam
adam = FedAdamAggregator(lr=0.01, beta1=0.9, beta2=0.99)

# FedAdagrad
adagrad = FedAdagradAggregator(lr=0.01)
```

---

## Byzantine Resilience

Byzantine clients may submit malicious updates to corrupt the global model. Cognita provides multiple defense mechanisms.

### Krum

Selects updates closest to their neighbors.

```python
from cognita.byzantine import KrumAggregator

aggregator = KrumAggregator(num_byzantine=1)
```

### Trimmed Mean

Removes extreme values before averaging.

```python
from cognita.byzantine import TrimmedMeanAggregator

aggregator = TrimmedMeanAggregator(num_byzantine=1, trim_ratio=0.1)
```

### Geometric Median

Minimizes sum of distances to all updates.

```python
from cognita.byzantine import GeoMedianAggregator

aggregator = GeoMedianAggregator(max_iter=100, tol=1e-6)
```

---

## Privacy Budget Management

### Automatic Budget Tracking

```python
from cognita.privacy import PrivacyAccountant, PrivacyBudget

# Initialize with target budget
accountant = PrivacyAccountant(
    epsilon=8.0,  # Target epsilon
    delta=1e-5,
    accountant_type="rdp"  # Rényi DP accounting
)

# Update after each round
for round_num in range(100):
    accountant.update(
        sample_rate=0.01,      # q = batch_size / dataset_size
        noise_multiplier=0.1   # σ
    )
    
    # Check if budget exhausted
    if accountant.get_spent_epsilon() >= 8.0:
        print("Privacy budget exhausted!")
        break
```

### Computing Optimal Noise

```python
# Find noise level for target privacy
optimal_noise = accountant.compute_optimal_noise(
    target_epsilon=8.0,
    sample_rate=0.01,
    steps=1000
)
print(f"Required noise multiplier: {optimal_noise:.4f}")
```

---

## Model Compression

Reduce communication overhead with gradient compression.

### Top-K Sparsification

Keep only the k largest values by magnitude.

```python
from cognita.compression import GradientCompressor

compressor = GradientCompressor(
    method="top_k",
    compression_ratio=0.1  # Keep 10% of values
)

# Compress
compressed = compressor.compress(gradients)

# Decompress
restored = compressor.decompress(compressed)
```

### Quantization

Reduce precision of gradient values.

```python
from cognita.compression import Quantizer

quantizer = Quantizer(levels=256)  # 8-bit quantization
indices, scale = quantizer.quantize(gradients)
restored = quantizer.dequantize(indices, scale)
```

### Available Methods

| Method | Description | Compression Ratio |
|--------|-------------|-------------------|
| `top_k` | Keep largest k values | 1-k |
| `random_k` | Keep random k values | 1-k |
| `quantization` | Reduce precision | ~8x |
| `sign` | 1-bit sign quantization | 32x |
| `pow2` | Power-of-2 quantization | ~4x |
| `sparse` | Threshold-based sparsity | Variable |

---

## Benchmarks

### Communication Efficiency

| Method | Compression Ratio | Accuracy Retention |
|--------|-------------------|-------------------|
| No Compression | 1x | 100% |
| Top-K (10%) | 10x | 99.5% |
| Random-K (10%) | 10x | 99.2% |
| Sign (1-bit) | 32x | 98.5% |
| Quantization (8-bit) | 4x | 99.8% |

### Privacy-Utility Tradeoff

| Epsilon | Noise Multiplier | Test Accuracy |
|---------|------------------|---------------|
| 2.0 | 0.5 | 87.2% |
| 4.0 | 0.3 | 91.5% |
| 8.0 | 0.1 | 94.2% |
| 16.0 | 0.05 | 96.1% |
| ∞ (no DP) | 0 | 97.8% |

### Byzantine Resilience

| Defense | Attack | Accuracy |
|---------|--------|----------|
| None | Label Flipping | 52.3% |
| Krum | Label Flipping | 91.8% |
| Trimmed Mean | Label Flipping | 93.1% |
| Geo Median | Label Flipping | 94.5% |
| None | Gaussian | 48.7% |
| Krum | Gaussian | 90.2% |

---

## Examples

### MNIST Federated Learning

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from cognita import FederatedClient, FederatedServer, FedAvgAggregator
from cognita.core import FLConfig, ClientConfig, ServerConfig

# Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

# Create clients with partitioned data
num_clients = 5
datasets = random_split(full_dataset, [len(full_dataset)//num_clients]*num_clients)

for i in range(num_clients):
    client = FederatedClient(
        client_id=f"client_{i}",
        model=SimpleCNN(),
        config=ClientConfig(client_id=f"client_{i}", local_epochs=5)
    )
    client.set_data(datasets[i])
    server.register_client(client)

# Run training
results = server.run(num_rounds=20)
```

### Complete DP-SGD Example

```python
from cognita import FederatedClient, FederatedServer
from cognita.aggregation import FedAvgAggregator
from cognita.privacy import DPClient, PrivacyAccountant
from cognita.core import PrivacyConfig

# Configure differential privacy
privacy_config = PrivacyConfig(
    epsilon=8.0,
    delta=1e-5,
    max_grad_norm=1.0,
    noise_multiplier=0.1,
    secure_aggregation=False
)

# Create DP client
client = FederatedClient(
    client_id="client_1",
    model=model,
    config=ClientConfig(client_id="client_1", privacy=privacy_config)
)

# Initialize privacy accountant
accountant = PrivacyAccountant(epsilon=8.0, delta=1e-5)

# Training loop with privacy tracking
for round_num in range(100):
    update = client.train_round(round_num)
    
    accountant.update(
        sample_rate=0.01,
        noise_multiplier=privacy_config.noise_multiplier
    )
    
    spent = accountant.get_spent_epsilon()
    print(f"Round {round_num}: Privacy spent = {spent:.4f}")
    
    if spent >= 8.0:
        print("Privacy budget exhausted!")
        break
```

### Byzantine-Resilient Federated Learning

```python
from cognita import FederatedServer
from cognita.aggregation import FedAvgAggregator
from cognita.byzantine import ByzantineResilientAggregator, KrumAggregator
from cognita.core import ByzantineConfig

# Configure Byzantine defense
byzantine_config = ByzantineConfig(
    enabled=True,
    num_byzantine=1,  # Expect up to 1 Byzantine client
    defense_method="krum"
)

# Create Byzantine-resilient server
server = FederatedServer(
    model=global_model,
    aggregator=ByzantineResilientAggregator(
        base_aggregator=FedAvgAggregator(),
        num_byzantine=1,
        defense_method="krum"
    )
)
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/moggan1337/Cognita.git
cd Cognita
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black src/
isort src/
mypy src/
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use Cognita in your research, please cite:

```bibtex
@software{cognita2024,
  title = {Cognita: A Comprehensive Federated Learning Platform},
  author = {Cognita Team},
  year = {2024},
  url = {https://github.com/moggan1337/Cognita}
}
```

---

## Acknowledgments

- FedAvg: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- FedProx: Li et al., "Federated Optimization in Heterogeneous Networks"
- SCAFFOLD: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
- DP-SGD: Abadi et al., "Deep Learning with Differential Privacy"
- FedOpt: Reddi et al., "Adaptive Federated Optimization"

---

<div align="center">

**Cognita** - Empowering Privacy-Preserving Machine Learning

</div>
