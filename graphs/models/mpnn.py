import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear, BatchNorm1d, Dropout, BatchNorm1d, Sequential, ReLU

class EEGMPNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, 
                 num_mp_layers=3, aggr="add", dropout=0.5):
        """
        MPNN model for EEG classification with configurable number of message passing layers
        
        Args:
            in_channels: Input feature dimensions (number of time points)
            hidden_channels: Number of hidden units
            out_channels: Output feature dimensions before classification
            num_classes: Number of classes for classification
            num_mp_layers: Number of message passing layers (default: 3)
            aggr: Aggregation method for messages ("add", "mean", or "max") (default: "add")
            dropout: Dropout probability (default: 0.5)
        """
        super(EEGMPNN, self).__init__()
        
        self.num_mp_layers = num_mp_layers
        self.dropout_p = dropout
        self.dropout = torch.nn.Dropout(dropout)
        
        # Create ModuleLists to hold message passing layers and batch norm layers
        self.mp_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        
        # First layer: in_channels -> hidden_channels
        self.mp_layers.append(MessagePassing(in_channels, hidden_channels, aggr=aggr))
        self.bn_layers.append(BatchNorm1d(hidden_channels))
        
        # Middle layers: hidden_channels -> hidden_channels
        for i in range(1, num_mp_layers - 1):
            self.mp_layers.append(MessagePassing(hidden_channels, hidden_channels, aggr=aggr))
            self.bn_layers.append(BatchNorm1d(hidden_channels))
        
        # Last layer: hidden_channels -> out_channels
        if num_mp_layers > 1:
            self.mp_layers.append(MessagePassing(hidden_channels, out_channels, aggr=aggr))
            self.bn_layers.append(BatchNorm1d(out_channels))
        else:
            # If only one layer, go directly from in_channels to out_channels
            self.mp_layers[-1] = MessagePassing(in_channels, out_channels, aggr=aggr)
            self.bn_layers[-1] = BatchNorm1d(out_channels)
        
        # Output layer for classification
        self.linear = Linear(out_channels, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            edge_attr: Edge features [num_edges, edge_feature_dim] (optional)
            
        Returns:
            torch.Tensor: Class logits
        """
        # Apply all message passing layers except the last one
        for i in range(self.num_mp_layers - 1):
            x = self.mp_layers[i](x, edge_index, edge_attr)
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Apply the last message passing layer (without dropout)
        if self.num_mp_layers > 0:
            x = self.mp_layers[-1](x, edge_index, edge_attr)
            x = self.bn_layers[-1](x)
            x = F.relu(x)
        
        # Global pooling: combine all node features for each graph
        x = global_mean_pool(x, batch)
        
        # Final classification
        x = self.linear(x)
        
        return x


class MessagePassing(torch.nn.Module):
    """
    Custom Message Passing layer with configurable message and update functions
    """
    def __init__(self, in_channels, out_channels, aggr="add"):
        """
        Initialize a Message Passing layer
        
        Args:
            in_channels: Input feature dimensions
            out_channels: Output feature dimensions
            aggr: Aggregation method for messages ("add", "mean", or "max")
        """
        super(MessagePassing, self).__init__()
        
        self.aggr = aggr
        
        # Message function (edge-level): transforms source node features
        self.message_mlp = Sequential(
            Linear(in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels)
        )
        
        # Update function (node-level): combines node features with aggregated messages
        self.update_mlp = Sequential(
            Linear(in_channels + out_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels)
        )
        
        # Optional edge feature integration
        self.edge_encoder = Linear(1, out_channels, bias=False)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass for message passing layer
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim] (optional)
            
        Returns:
            torch.Tensor: Updated node features [num_nodes, out_channels]
        """
        source, target = edge_index
        
        # Compute messages from source nodes
        message = self.message_mlp(x[source])
        
        # If edge attributes are provided, incorporate them into messages
        if edge_attr is not None:
            # Ensure edge_attr is 2D
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)
            edge_embedding = self.edge_encoder(edge_attr)
            message = message * edge_embedding
        
        # Aggregate messages for each target node
        if self.aggr == "add":
            aggregated = torch.scatter_add(message, target, dim=0, dim_size=x.size(0))
        elif self.aggr == "mean":
            aggregated = torch.scatter_mean(message, target, dim=0, dim_size=x.size(0))
        elif self.aggr == "max":
            aggregated = torch.scatter_max(message, target, dim=0, dim_size=x.size(0))[0]
        else:
            raise ValueError(f"Unsupported aggregation type: {self.aggr}")
        
        # Update node features by combining with aggregated messages
        updated = self.update_mlp(torch.cat([x, aggregated], dim=1))
        
        return updated