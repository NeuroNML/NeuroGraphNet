import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, input_length=3000, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_length, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: [num_nodes, time_steps]  
        returns: [num_nodes, hidden_dim]  
        """
        print("MLP input mean:", x.mean().item(), "std:", x.std().item())
        return self.encoder(x)
