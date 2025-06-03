import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

# Custom imports
from src.utils.models_funcs import pooling


class ChebEEGNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, pooling_type, K=1):
        super(ChebEEGNet, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        self.pooling_type = pooling_type

    def forward(self, x, edge_index, batch):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # batch: [num_nodes] indicating graph membership
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = pooling(x, batch, pooling_type=self.pooling_type) # aggregate node embeddings into a graph embedding
        x = self.linear(x)
        return x
