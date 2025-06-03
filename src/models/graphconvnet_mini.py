import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, BatchNorm

class EEGGraphConvNetMini(nn.Module):
    def __init__(self, in_channels=6, hidden1=16, hidden2=32, mlp_dims=(32, 16, 2)):
        super(EEGGraphConvNetMini, self).__init__()

        # GCN layers
        self.conv1 = GCNConv(in_channels, hidden1)
        self.bn1 = BatchNorm(hidden1)
        
        self.conv2 = GCNConv(hidden1, hidden2)
        self.bn2 = BatchNorm(hidden2)

        # MLP layers after pooling
        self.fc1 = nn.Linear(hidden2, mlp_dims[0])
        self.fc2 = nn.Linear(mlp_dims[0], mlp_dims[1])
        self.fc3 = nn.Linear(mlp_dims[1], mlp_dims[2])

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, edge_index, batch):
        # Graph Conv Block 1
        x = self.conv1(x, edge_index)
        x = self.leaky_relu(x)
        x = self.bn1(x)

        # Graph Conv Block 2
        x = self.conv2(x, edge_index)
        x = self.leaky_relu(x)
        x = self.bn2(x)

        # Global Pooling
        x = global_add_pool(x, batch)

        # MLP
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        return x
