# General imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, BatchNorm1d
from torch_geometric.nn import GCNConv

from src.utils.model import pooling

class EEGGCN(torch.nn.Module):
    """
    Graph Convolutional Network for EEG signal processing.
    Features:
    - Optional CNN preprocessing for time series reduction
    - Multiple GCN layers with configurable dimensions
    - Batch normalization
    - Dropout for regularization
    - Configurable pooling strategy
    - Optional MLP head for classification
    """
    def __init__(
        self,
        in_channels=3000,
        hidden_channels=640,
        out_channels=64,
        num_conv_layers=3,
        pooling_type="max",
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
            num_conv_layers: Number of GCN layers
            pooling_type: Type of graph pooling ("mean" or "sum")
            dropout_prob: Dropout probability
            use_cnn_preprocessing: Whether to use CNN for time series reduction
            use_batch_norm: Whether to use batch normalization
            mlp_dims: List of dimensions for MLP head (if None, uses single linear layer)
        """
        super(EEGGCN, self).__init__()

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

        # GCN layers
        self.num_conv_layers = num_conv_layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        # First layer
        if num_conv_layers == 1:
            self.conv_layers.append(GCNConv(in_channels, out_channels))
            if use_batch_norm and self.batch_norms is not None:
                self.batch_norms.append(BatchNorm1d(out_channels))
        else:
            self.conv_layers.append(GCNConv(in_channels, hidden_channels))
            if use_batch_norm and self.batch_norms is not None:
                self.batch_norms.append(BatchNorm1d(hidden_channels))
            
            # Intermediate layers
            for _ in range(num_conv_layers - 2):
                self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
                if use_batch_norm and self.batch_norms is not None:
                    self.batch_norms.append(BatchNorm1d(hidden_channels))
            
            # Final layer
            self.conv_layers.append(GCNConv(hidden_channels, out_channels))
            if use_batch_norm and self.batch_norms is not None:
                self.batch_norms.append(BatchNorm1d(out_channels))

        # Pooling and dropout
        self.pooling_type = pooling_type
        self.dropout = Dropout(p=dropout_prob)

        # MLP head
        if mlp_dims is not None:
            mlp_layers = []
            prev_dim = out_channels
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
            self.mlp = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, batch):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes_batch, in_channels]
            edge_index: Edge list [2, num_edges]
            batch: Graph batch vector [num_nodes_batch]

        Returns:
            Class logits [num_graphs, 1]
        """
        # CNN preprocessing
        if self.cnn is not None:
            x = x.unsqueeze(1)  # [N, 1, T]
            x = self.cnn(x).squeeze(1)  # [N, T/4]
            # x = self.project_to(x)  # [N, in_channels]

        # GCN layers
        for i in range(self.num_conv_layers - 1):
            x = self.conv_layers[i](x, edge_index)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = F.leaky_relu(x)
            x = self.dropout(x)

        # Final layer
        x = self.conv_layers[-1](x, edge_index)
        if self.batch_norms is not None:
            x = self.batch_norms[-1](x)
        x = F.leaky_relu(x)

        # Perform pooling
        x = pooling(x, batch, self.pooling_type)

        # MLP head
        return self.mlp(x)
