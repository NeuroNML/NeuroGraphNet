import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class EEGGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, heads=1, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Binary classification
        )

    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        print(f"After GAT 1")
        print(f"mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        x = F.elu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        print(f"After GAT 2")
        print(f"mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")

        x = F.elu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        print(f"After pooling")
        print(f"mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
      
        return self.classifier(x)