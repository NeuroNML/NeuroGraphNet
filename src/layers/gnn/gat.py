import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, BatchNorm1d
from torch_geometric.nn import GATConv

from src.utils.model import pooling


class EEGGAT(torch.nn.Module):
    """
    Graph Attention Network for EEG signal processing.
    Features:
    - Optional CNN preprocessing for time series reduction
    - Multiple GAT layers with configurable dimensions and attention heads
    - Batch normalization
    - Dropout for regularization
    - Configurable pooling strategy
    - Optional MLP head for classification
    """
    def __init__(
        self,
        in_channels=3000,
        hidden_channels=64,
        out_channels=64,
        num_conv_layers=3,
        heads=1,
        pooling_type="mean",
        dropout_prob=0.5,
        use_cnn_preprocessing=False,
        use_batch_norm=True,
        mlp_dims=None,
    ):
        """
        Args:
            in_channels: Input feature dimension (default: 3000 for EEG)
            hidden_channels: Number of hidden units in intermediate layers
            out_channels: Feature dimension before final classifier
            num_conv_layers: Number of GAT layers
            heads: Number of attention heads
            pooling_type: Type of graph pooling ("mean", "max", or "sum")
            dropout_prob: Dropout probability
            use_cnn_preprocessing: Whether to use CNN for time series reduction
            use_batch_norm: Whether to use batch normalization
            mlp_dims: List of dimensions for MLP head (if None, uses single linear layer)
        """
        super(EEGGAT, self).__init__()

        # CNN preprocessing for time series reduction
        if use_cnn_preprocessing:
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=15, stride=2, padding=7),
                nn.LeakyReLU(),
                nn.Conv1d(8, 16, kernel_size=15, stride=2, padding=7),
                nn.LeakyReLU(),
                nn.Conv1d(16, 1, kernel_size=3, stride=1, padding=1),
            )
            self.project_to = nn.Linear(750, in_channels)
        else:
            self.cnn = None
            self.project_to = None

        # GAT layers
        self.num_conv_layers = num_conv_layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.heads = heads

        # First layer
        if num_conv_layers == 1:
            self.conv_layers.append(GATConv(in_channels, out_channels, heads=heads, dropout=dropout_prob))
            if use_batch_norm and self.batch_norms is not None:
                self.batch_norms.append(BatchNorm1d(out_channels * heads))
        else:
            self.conv_layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_prob))
            if use_batch_norm and self.batch_norms is not None:
                self.batch_norms.append(BatchNorm1d(hidden_channels * heads))
            
            # Intermediate layers
            for i in range(num_conv_layers - 2):
                # For intermediate layers, use concat=True (default) except for the last layer
                self.conv_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout_prob))
                if use_batch_norm and self.batch_norms is not None:
                    self.batch_norms.append(BatchNorm1d(hidden_channels * heads))
            
            # Final layer - use concat=False to get the desired output dimension
            self.conv_layers.append(GATConv(hidden_channels * heads, out_channels, heads=heads, concat=False, dropout=dropout_prob))
            if use_batch_norm and self.batch_norms is not None:
                self.batch_norms.append(BatchNorm1d(out_channels))

        # Pooling and dropout
        self.pooling_type = pooling_type
        self.dropout = Dropout(p=dropout_prob)

        # MLP head - if mlp_dims is None, don't create any classifier
        self.use_classifier = mlp_dims is not None
        if self.use_classifier:
            mlp_layers = []
            prev_dim = out_channels
            assert mlp_dims is not None, "mlp_dims must be provided for classifier"
            for dim in mlp_dims:
                mlp_layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout_prob)
                ])
                prev_dim = dim
            mlp_layers.append(nn.Linear(prev_dim, 1))
            self.mlp = nn.Sequential(*mlp_layers)
        else:
            self.mlp = None

    def forward(self, x, edge_index, batch):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes_batch, in_channels]
            edge_index: Edge list [2, num_edges]
            batch: Graph batch vector [num_nodes_batch]

        Returns:
            Class logits [num_graphs, 1] or features [num_graphs, out_channels]
        """
        # CNN preprocessing
        if self.cnn is not None:
            x = x.unsqueeze(1)  # [N, 1, T]
            x = self.cnn(x).squeeze(1)  # [N, T/4]

        # GAT layers
        for i in range(self.num_conv_layers - 1):
            x = self.conv_layers[i](x, edge_index)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = F.elu(x)
            x = self.dropout(x)

        # Final layer
        x = self.conv_layers[-1](x, edge_index)
        if self.batch_norms is not None:
            x = self.batch_norms[-1](x)
        x = F.elu(x)

        # Perform pooling
        x = pooling(x, batch, self.pooling_type)

        # Return features or logits based on whether classifier is used
        if self.use_classifier and self.mlp is not None:
            return self.mlp(x)
        else:
            return x