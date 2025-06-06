import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, BatchNorm, global_mean_pool, global_max_pool

class EEGGraphConvNetMini(nn.Module):
    def __init__(self, in_channels=60, dropout=0.5,  hidden1=64, hidden2=32, mlp_dims=(16, 1)):
        super(EEGGraphConvNetMini, self).__init__()

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # GCN layers
        self.conv1 = GCNConv(in_channels, hidden1)
        self.bn1 = BatchNorm(hidden1)
        
        self.conv2 = GCNConv(hidden1, hidden2)
        self.bn2 = BatchNorm(hidden2)

        # MLP layers after pooling
        self.mlp = nn.Sequential(nn.Linear(hidden2, mlp_dims[0]),
                                self.leaky_relu,
                                nn.Dropout(dropout),
                                nn.Linear(mlp_dims[0], mlp_dims[1]))
                               # self.leaky_relu(),
                               # nn.Dropout(dropout),
                               # nn.Linear(mlp_dims[1], mlp_dims[2]))

        

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
        x = self.mlp(x)

        return x
