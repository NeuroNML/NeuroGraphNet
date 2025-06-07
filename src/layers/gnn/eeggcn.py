import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear, BatchNorm1d, Dropout, ModuleList


class EEGGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, num_conv_layers=3, dropout=0.5):
        super(EEGGCN, self).__init__()
        
        # Ensure num_conv_layers is not negative, handle 0 if it means direct to linear
        if num_conv_layers < 0:
            raise ValueError("num_conv_layers cannot be negative.")

        self.num_conv_layers = num_conv_layers
        self.dropout = Dropout(dropout)
        
        self.conv_layers = ModuleList()
        self.bn_layers = ModuleList()
        
        current_dim = in_channels

        if num_conv_layers > 0:
            # First layer
            self.conv_layers.append(GCNConv(current_dim, hidden_channels if num_conv_layers > 1 else out_channels))
            self.bn_layers.append(BatchNorm1d(hidden_channels if num_conv_layers > 1 else out_channels))
            current_dim = hidden_channels if num_conv_layers > 1 else out_channels
            
            # Middle layers
            for _ in range(1, num_conv_layers - 1):
                self.conv_layers.append(GCNConv(current_dim, hidden_channels))
                self.bn_layers.append(BatchNorm1d(hidden_channels))
                # current_dim remains hidden_channels
            
            # Last layer (if num_conv_layers > 1 and it's different from the first)
            if num_conv_layers > 1:
                self.conv_layers.append(GCNConv(current_dim, out_channels))
                self.bn_layers.append(BatchNorm1d(out_channels))
            
            # The input to the linear layer will be out_channels if GCN layers are used
            linear_in_features = out_channels
        else:
            # If no GCN layers, the input to linear is the original in_channels
            linear_in_features = in_channels 

        # Output layer
        self.linear = Linear(linear_in_features, num_classes)

    def forward(self, x, edge_index, batch):
        if self.num_conv_layers > 0:
            for i in range(self.num_conv_layers): # Iterate through all GCN layers
                x = self.conv_layers[i](x, edge_index)
                x = self.bn_layers[i](x)
                x = F.relu(x)
                # Apply dropout to all GCN layers except the very last one before pooling
                if i < self.num_conv_layers - 1:
                    x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Final classification
        x = self.linear(x)
        
        return x