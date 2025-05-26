import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout, Sequential, ReLU

# Custom imports
from src.utils.models_funcs import pooling

class EEGMPNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_mp_layers=3, pooling_message='mean', pooling_type="mean", dropout_prob=0.5):
        """
        MPNN model for EEG classification with configurable number of message passing layers
        
        Args:
            in_channels: Input feature dimensions 
            hidden_channels: Number of hidden units
            out_channels: Output feature dimensions before classification
            num_classes: Number of classes for classification
            num_mp_layers: Number of message passing layers (default: 3)
            pooling_message: Pooling method for message passing (default: 'max')
            global_pooling_type: Pooling method for global pooling (default: 'mean')
            dropout: Dropout probability (default: 0.5)
        """
        super(EEGMPNN, self).__init__()
        
        self.num_mp_layers = num_mp_layers
        self.dropout_p = dropout_prob
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.final_pooling = pooling_type
        
        # Create ModuleLists to hold message passing layers and batch norm layers
        self.mp_layers = torch.nn.ModuleList()
        #self.bn_layers = torch.nn.ModuleList()
        if num_mp_layers == 1:
             self.mp_layers.append(MessagePassing(in_channels, out_channels, aggr=pooling_message))
        else:
    
            # First layer: in_channels -> hidden_channels
            self.mp_layers.append(MessagePassing(in_channels, hidden_channels, aggr=pooling_message))
            #self.bn_layers.append(BatchNorm1d(hidden_channels))
            
            # Middle layers: hidden_channels -> hidden_channels
            for _ in range(1, num_mp_layers - 2):
                self.mp_layers.append(MessagePassing(hidden_channels, hidden_channels, aggr=pooling_message))
                #self.bn_layers.append(BatchNorm1d(hidden_channels))
            
            # Last layer: hidden_channels -> out_channels
            self.mp_layers.append(MessagePassing(hidden_channels, out_channels, aggr=pooling_message))
            #self.bn_layers.append(BatchNorm1d(out_channels))
          
            
        #self.bn_layers = torch.nn.ModuleList([BatchNorm1d(hidden_channels) for _ in range(num_mp_layers - 1)])
        # Output layer for binary classification
        self.linear = Linear(out_channels, 1)

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
        for i in range(len(self.mp_layers) - 1):
            x = self.mp_layers[i](x, edge_index, edge_attr)
            x = F.relu(x)
            #x = self.dropout(x)
        
   
        # Apply the last message passing layer (no ReLU or dropout)
        x = self.mp_layers[-1](x, edge_index, edge_attr)
        
      
        # Aggregate node features into graph-level representation
        x = pooling(x, batch, pooling_type=self.final_pooling)
        
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