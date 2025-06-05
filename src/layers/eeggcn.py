import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear, BatchNorm1d, Dropout


class EEGGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, num_conv_layers=3, dropout=0.5):
        super(EEGGCN, self).__init__()
        
        # Ensure num_conv_layers is not negative, handle 0 if it means direct to linear
        if num_conv_layers < 0:
            raise ValueError("num_conv_layers cannot be negative.")

        self.num_conv_layers = num_conv_layers
        self.dropout = torch.nn.Dropout(dropout)
        
        self.conv_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        
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
            # (or you might want a projection if in_channels is too large)
            # This example assumes direct pass-through of initial features after pooling.
            linear_in_features = in_channels 
            # Or: self.projection = Linear(in_channels, out_channels)
            # linear_in_features = out_channels

        # Output layer
        self.linear = Linear(linear_in_features, num_classes)

    def forward(self, x, edge_index, batch):
        if self.num_conv_layers > 0:
            for i in range(self.num_conv_layers): # Iterate through all GCN layers
                x = self.conv_layers[i](x, edge_index)
                x = self.bn_layers[i](x)
                x = F.relu(x)
                # Apply dropout to all GCN layers except the very last one before pooling,
                # or apply to all based on preference.
                # Your original code applied it to all but the last activation.
                if i < self.num_conv_layers - 1: # Common to apply dropout to hidden layers
                    x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # If num_conv_layers == 0 and you added a projection:
        # if self.num_conv_layers == 0 and hasattr(self, 'projection'):
        #     x = self.projection(x)
        #     x = F.relu(x) # Optional activation
            
        # Final classification
        x = self.linear(x)
        
        return x