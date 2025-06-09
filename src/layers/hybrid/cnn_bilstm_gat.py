from typing import Optional
import torch
import torch.nn as nn
import logging

from src.layers.gnn.gat import EEGGAT
from src.layers.encoders.cnnbilstm_encoder import EEGCNNBiLSTMEncoder

logger = logging.getLogger(__name__)


class EEGCNNBiLSTMGAT(nn.Module):
    """
    Hybrid model combining a CNN-BiLSTM encoder for temporal feature extraction
    per EEG channel and a Graph Attention Network (GAT) for spatial feature integration.

    This architecture addresses the project's objective of exploring graph-based methods
    by leveraging the inherent spatial relationships between EEG electrodes with attention mechanisms.
    The temporal encoder processes each 12-second EEG window (3000 time points)
    for each of the 19 electrodes, while the GAT integrates information across electrodes
    with learnable attention weights.
    """

    def __init__(
        self,
        # Node feature dimensions
        node_input_dim: int = 3000,  # Time steps per channel
        # Temporal encoder parameters (CNN-BiLSTM)
        cnn_dropout: float = 0.25,
        lstm_hidden_dim: int = 64,
        lstm_out_dim: int = 64,  # This becomes node feature dim for GAT
        lstm_dropout: float = 0.25,
        lstm_num_layers: int = 1,
        encoder_use_batch_norm: bool = True,
        encoder_use_layer_norm: bool = False,
        # GAT parameters
        hidden_dim: int = 64,  # GAT hidden dimensions
        out_channels: int = 32,  # GAT output dimensions
        num_conv_layers: int = 3,
        gat_heads: int = 4,  # Number of attention heads
        pooling_type: str = "mean",
        gat_dropout: float = 0.5,
        gat_use_batch_norm: bool = True,
        # Graph features
        graph_feature_dim: int = 0,
        use_graph_features: bool = True,
        # Classification
        num_classes: int = 1,
        classifier_dropout: float = 0.5,
        # General parameters
        num_channels: int = 19,  # Number of EEG channels
    ):
        """
        Initializes the EEGCNNBiLSTMGAT model.

        Args:
            node_input_dim (int): Input feature dimension per node (EEG channel).
            cnn_dropout (float): Dropout probability for the CNN part of the temporal encoder.
            lstm_hidden_dim (int): Number of hidden units in the LSTM part of the temporal encoder.
            lstm_out_dim (int): Output feature dimension from the LSTM part of the temporal encoder.
                               This also serves as the input dimension for the GAT.
            lstm_dropout (float): Dropout probability for the LSTM part of the temporal encoder.
            lstm_num_layers (int): Number of layers in the BiLSTM.
            encoder_use_batch_norm (bool): Whether to use batch normalization in encoder layers.
            encoder_use_layer_norm (bool): Whether to use layer normalization in encoder layers.
            hidden_dim (int): Number of hidden units in the GAT layers.
            out_channels (int): Output feature dimensions from the last GAT layer.
            num_conv_layers (int): Number of GAT convolutional layers.
            gat_heads (int): Number of attention heads in GAT layers.
            pooling_type (str): Type of graph pooling ("mean", "max", or "sum").
            gat_dropout (float): Dropout probability for GAT layers.
            gat_use_batch_norm (bool): Whether to use batch normalization in GAT layers.
            graph_feature_dim (int): Dimension of additional graph-level features.
            use_graph_features (bool): Whether to incorporate graph-level features.
            num_classes (int): Number of output classes (e.g., 1 for binary seizure detection).
            classifier_dropout (float): Dropout probability for the final classifier.
            num_channels (int): The fixed number of EEG channels (e.g., 19).
        """
        super().__init__()

        # Store configuration
        self.num_channels = num_channels
        self.time_encoder_output_dim = lstm_out_dim
        self.use_graph_features = use_graph_features
        self.graph_feature_dim = graph_feature_dim

        # Initialize the temporal encoder (CNN-BiLSTM)
        self.channel_encoder = EEGCNNBiLSTMEncoder(
            in_channels=node_input_dim,
            cnn_dropout=cnn_dropout,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_out_dim=lstm_out_dim,
            lstm_dropout=lstm_dropout,
            num_layers=lstm_num_layers,
            use_batch_norm=encoder_use_batch_norm,
            use_layer_norm=encoder_use_layer_norm,
            add_classifier=False,  # We'll add our own classifier
        )

        # Initialize the Graph Attention Network without built-in classifier
        self.gat = EEGGAT(
            in_channels=self.time_encoder_output_dim,
            hidden_channels=hidden_dim,
            out_channels=out_channels,
            num_conv_layers=num_conv_layers,
            heads=gat_heads,
            pooling_type=pooling_type,
            dropout_prob=gat_dropout,
            use_batch_norm=gat_use_batch_norm,
            use_cnn_preprocessing=False,
            mlp_dims=None,  # No built-in classifier
        )

        # Calculate final feature dimension for classifier
        final_feature_dim = out_channels
        if self.use_graph_features:
            final_feature_dim += self.graph_feature_dim

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(final_feature_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, num_classes)
        )

        logger.info(f"Initialized EEGCNNBiLSTMGAT with {num_conv_layers} GAT layers, "
                   f"{gat_heads} attention heads, and {out_channels} output channels")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_labels: torch.Tensor,
        graph_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the CNN-BiLSTM-GAT model.

        Args:
            x (torch.Tensor): Input EEG signals. Expected shape:
                            [num_graphs_in_batch, num_channels, time_steps]
                            For example, [batch_size, 19, 3000].
            edge_index (torch.Tensor): Graph connectivity in COO format. Shape: [2, num_edges_in_batch].
                                     Node indices are assumed to be global across the batch
                                     as handled by PyTorch Geometric's DataLoader.
            batch_labels (torch.Tensor): Batch vector mapping each node to its respective graph.
                                       Shape: [total_num_nodes_in_batch]. Used for global pooling.
            graph_features (torch.Tensor, optional): Additional graph-level features.
                                                    Shape: [num_graphs_in_batch, graph_feature_dim].

        Returns:
            torch.Tensor: Class logits for each graph in the batch. Shape: [num_graphs_in_batch, num_classes].
        """
        # Step 1: Encode each channel's time series into a fixed-size embedding
        # The temporal encoder processes all graphs and channels in parallel
        # Input shape: [num_graphs_in_batch, num_channels, time_steps]
        # Output shape: [num_graphs_in_batch * num_channels, lstm_out_dim]
        node_features = self.channel_encoder(x)

        # Step 2: Pass node features through GAT to get graph-level representations
        # The GAT will handle the spatial relationships between EEG channels
        # Input: [num_nodes_total, feature_dim], edge_index, batch_labels
        # Output: [num_graphs_in_batch, out_channels]
        graph_embeddings = self.gat(node_features, edge_index, batch_labels)

        # Step 3: Optionally concatenate additional graph features
        if self.use_graph_features and graph_features is not None:
            graph_embeddings = torch.cat([graph_embeddings, graph_features], dim=1)

        # Step 4: Final classification
        logits = self.classifier(graph_embeddings)

        return logits

    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor, batch_labels: torch.Tensor):
        """
        Extract attention weights from GAT layers for visualization and analysis.
        
        Args:
            x (torch.Tensor): Input EEG signals.
            edge_index (torch.Tensor): Graph connectivity.
            batch_labels (torch.Tensor): Batch vector.
            
        Returns:
            List of attention weight tensors from each GAT layer.
        """
        # Encode temporal features
        node_features = self.channel_encoder(x)
        
        # Extract attention weights from GAT layers
        attention_weights = []
        current_features = node_features
        
        for i, conv_layer in enumerate(self.gat.conv_layers):
            # Get attention weights if available
            if hasattr(conv_layer, 'attention'):
                current_features, (edge_index_out, alpha) = conv_layer(current_features, edge_index, return_attention_weights=True)
                attention_weights.append(alpha)
            else:
                current_features = conv_layer(current_features, edge_index)
            
            # Apply batch norm and activation if not final layer
            if i < len(self.gat.conv_layers) - 1:
                if self.gat.batch_norms is not None:
                    current_features = self.gat.batch_norms[i](current_features)
                current_features = torch.nn.functional.elu(current_features)
                current_features = self.gat.dropout(current_features)
        
        return attention_weights
