import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear, BatchNorm1d, Dropout


class EEGGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, num_conv_layers=3, dropout=0.5):
        """
        GCN model for EEG classification with configurable number of conv layers
        
        Args:
            in_channels: Input feature dimensions (number of time points)
            hidden_channels: Number of hidden units
            out_channels: Output feature dimensions before classification
            num_classes: Number of classes for classification
            num_conv_layers: Number of GCN convolutional layers (default: 3)
            dropout: Dropout probability (default: 0.5)
        """
        super(EEGGCN, self).__init__()
        
        self.num_conv_layers = num_conv_layers
        self.dropout_p = dropout
        self.dropout = torch.nn.Dropout(dropout)
        
        # Create a ModuleList to hold our convolutional layers
        self.conv_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        
        # First layer: in_channels -> hidden_channels
        self.conv_layers.append(GCNConv(in_channels, hidden_channels))
        self.bn_layers.append(BatchNorm1d(hidden_channels))
        
        # Middle layers: hidden_channels -> hidden_channels
        for i in range(1, num_conv_layers - 1):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
            self.bn_layers.append(BatchNorm1d(hidden_channels))
        
        # Last layer: hidden_channels -> out_channels
        if num_conv_layers > 1:
            self.conv_layers.append(GCNConv(hidden_channels, out_channels))
            self.bn_layers.append(BatchNorm1d(out_channels))
        else:
            # If only one layer, go directly from in_channels to out_channels
            self.conv_layers[-1] = GCNConv(in_channels, out_channels)
            self.bn_layers[-1] = BatchNorm1d(out_channels)
        
        # Output layer
        self.linear = Linear(out_channels, num_classes)

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
        # Apply all GCN layers except the last one
        for i in range(self.num_conv_layers - 1):
            x = self.conv_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Apply the last GCN layer (without dropout)
        if self.num_conv_layers > 0:
            x = self.conv_layers[-1](x, edge_index)
            x = self.bn_layers[-1](x)
            x = F.relu(x)
        
        # Global pooling: combine all node features for each graph
        x = global_mean_pool(x, batch)
        
        # Final classification
        x = self.linear(x)
        
        return x