import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear, BatchNorm1d, Dropout


class EEGGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        """
        Simple GCN model for EEG classification
        
        Args:
            in_channels: Input feature dimensions (number of time points)
            hidden_channels: Number of hidden units
            out_channels: Output feature dimensions before classification
            num_classes: Number of classes for classification
        """
        super(EEGGCN, self).__init__()
        
        # First GCN layer
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        
        # Second GCN layer
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        
        # Third GCN layer
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn3 = BatchNorm1d(out_channels)
        
        # Output layer
        self.linear = Linear(out_channels, num_classes)
        
        # Dropout for regularization
        self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index, batch):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            torch.Tensor: Class logits
        """
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third GCN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling: combine all node features for each graph
        x = global_mean_pool(x, batch)
        
        # Final classification
        x = self.linear(x)
        
        return x