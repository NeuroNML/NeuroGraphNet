# General imports
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, global_mean_pool

# Custom imports
from src.utils.models_funcs import pooling


class EEGGCN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        out_channels,
        num_conv_layers=3,
        pooling_type="mean",
        dropout_prob=0.5,
    ):
        """
        Baseline GCN model for EEG classification.

        Args:
            hidden_channels: Number of hidden units in intermediate layers
            out_channels: Feature dimension before final classifier
            num_conv_layers: Number of GCN layers
        """
        super(EEGGCN, self).__init__()

        # --------------------- Convolutional layers --------------------- #
        self.num_conv_layers = num_conv_layers
        self.conv_layers = torch.nn.ModuleList()

        # Number of input channels
        in_channels = 250 * 12  # 3000 - 250Hz sampling rate, 12 s
        if num_conv_layers == 1:
            self.conv_layers.append(GCNConv(in_channels, out_channels))
        else:
            self.conv_layers.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_conv_layers - 2):
                self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
            self.conv_layers.append(GCNConv(hidden_channels, out_channels))

        # --------------------- Classifier --------------------- #
        self.linear = Linear(out_channels, 1)
        # Define the pooling type
        self.pooling_type = pooling_type

        # --------------------- Dropout --------------------- #
        self.dropout = Dropout(p=dropout_prob)  # Enables dropout only during training

    def forward(self, x, edge_index, batch):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes_batch, in_channels] # all nodes features from all graphs concatenated
            edge_index: Edge list [2, num_edges] # all edges from all graphs concatenated
            batch: Graph batch vector [num_nodes_batch]
                # Tells the model which graph each node belongs to
                # e.g. [0, 0, 0, 1, 1, 1] means first three nodes belong to graph 0 and last three to graph 1

        Returns:
            Class logits [num_graphs, num_classes]
        """
        # Input x: [total_nodes, in_channels]
        for i in range(self.num_conv_layers - 1):
            x = self.conv_layers[i](
                x, edge_index
            )  # Update each node’s embedding with neighbor information
            x = F.relu(x)  # Output: [total_nodes, hidden_dim]
            x = self.dropout(x)

        # Last layer - no dropout
        x = self.conv_layers[-1](x, edge_index)
        x = F.relu(x)

        # Aggregate node features into graph-level representation
        x = pooling(x, batch, pooling_type=self.pooling_type)

        return self.linear(x)  #  graph-level embedding -> logit value
