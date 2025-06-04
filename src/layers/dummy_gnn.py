from torch_geometric.nn import GCNConv, global_mean_pool
from torch import nn
import torch



class DummyGNN(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=64, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        #x = self.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        #x = self.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.linear(x)
