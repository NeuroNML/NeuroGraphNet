import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class EEGGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 64], output_dim=1, dropout=0.2):
        super().__init__()
        self.gcn1 = EEGGCNLayer(input_dim, hidden_dims[0], dropout)
        self.gcn2 = EEGGCNLayer(hidden_dims[0], hidden_dims[1], dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

    def forward(self, x, edge_index, batch):
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return out