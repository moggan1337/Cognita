"""
Simple Federated Learning Example

This script demonstrates basic federated learning with Cognita.
"""

import torch
import torch.nn as nn
from cognita import FederatedClient, FederatedServer, FedAvgAggregator
from cognita.core import FLConfig, ClientConfig, ServerConfig


class SimpleModel(nn.Module):
    """Simple linear model for demonstration."""
    
    def __init__(self, input_dim=10, output_dim=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)


def main():
    # Create model
    model = SimpleModel()
    
    # Configure server
    server_config = ServerConfig(
        num_rounds=5,
        clients_per_round=2,
        checkpoint_interval=10
    )
    
    # Create federated server
    server = FederatedServer(
        model=model,
        config=server_config,
        aggregator=FedAvgAggregator()
    )
    
    # Create and register clients
    num_clients = 3
    clients = []
    
    for i in range(num_clients):
        client_config = ClientConfig(
            client_id=f"client_{i}",
            local_epochs=2,
            batch_size=16,
            learning_rate=0.01
        )
        
        client = FederatedClient(
            client_id=f"client_{i}",
            model=SimpleModel(),
            config=client_config
        )
        
        # In a real scenario, you would set data here:
        # client.set_data(train_dataset)
        
        server.register_client(client)
        clients.append(client)
        
    print(f"Registered {len(clients)} clients")
    print(f"Server configured for {server_config.num_rounds} rounds")
    
    # Simulate a few rounds
    for round_num in range(3):
        # Start round
        sampled_clients = server.start_round(round_num)
        print(f"Round {round_num}: Sampled clients: {sampled_clients}")
        
        # In a real scenario, clients would train here:
        # for client_id in sampled_clients:
        #     update = clients[client_id].train_round(round_num, global_weights)
        #     server.receive_update(update)
        
        # Aggregate
        updates = server.get_pending_updates()
        if len(updates) >= server.config.min_clients:
            aggregated = server.aggregate_updates(updates)
            print(f"Round {round_num}: Aggregated {len(updates)} updates")
    
    print("Federated learning completed!")


if __name__ == "__main__":
    main()
