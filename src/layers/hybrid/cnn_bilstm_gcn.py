from typing import Optional
import torch
import torch.nn as nn
import logging

from src.layers.gnn.gcn import EEGGCN
from src.layers.encoders.cnnbilstm_encoder import EEGCNNBiLSTMEncoder

logger = logging.getLogger(__name__)


class EEGCNNBiLSTMGCN(nn.Module):
    """
    Hybrid model combining a CNN-BiLSTM encoder for temporal feature extraction
    per EEG channel and a Graph Convolutional Network (GCN) for spatial feature integration.

    This architecture addresses the project's objective of exploring graph-based methods
    by leveraging the inherent spatial relationships between EEG electrodes.
    The temporal encoder processes each 12-second EEG window (3000 time points)
    for each of the 19 electrodes, while the GCN integrates information across electrodes.
    """

    def __init__(
        self,
        # Node feature dimensions
        node_input_dim: int = 3000,  # Time steps per channel
        # Temporal encoder parameters (CNN-BiLSTM)
        cnn_dropout: float = 0.25,
        lstm_hidden_dim: int = 64,
        lstm_out_dim: int = 64,  # This becomes node feature dim for GCN
        lstm_dropout: float = 0.25,
        lstm_num_layers: int = 1,
        encoder_use_batch_norm: bool = True,
        encoder_use_layer_norm: bool = False,
        # GCN parameters
        hidden_dim: int = 64,  # GCN hidden dimensions
        out_channels: int = 32,  # GCN output dimensions
        num_conv_layers: int = 3,
        pooling_type: str = "mean",
        gcn_dropout: float = 0.5,
        gcn_use_batch_norm: bool = True,
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
        Initializes the EEGCNNBiLSTMGCN.

        Args:
            node_input_dim (int): Input dimension for each node (time steps per channel).
            cnn_dropout (float): Dropout probability for the CNN part of the temporal encoder.
            lstm_hidden_dim (int): Number of hidden units in the LSTM part of the temporal encoder.
            lstm_out_dim (int): Output feature dimension from the LSTM part of the temporal encoder.
                                This also serves as the input dimension for the GCN.
            lstm_dropout (float): Dropout probability for the LSTM part of the temporal encoder.
            lstm_num_layers (int): Number of layers in the CNN_BiLSTM_Encoder.
            hidden_dim (int): Number of hidden units in the GCN layers.
            out_channels (int): Output feature dimensions from the last GCN layer before pooling.
            num_conv_layers (int): Number of GCN convolutional layers.
            gcn_dropout (float): Dropout probability for GCN layers.
            graph_feature_dim (int): Dimension of graph-level features.
            use_graph_features (bool): Whether to use graph-level features.
            num_classes (int): Number of output classes.
            classifier_dropout (float): Dropout for classifier layers.
            num_channels (int): The fixed number of EEG channels (e.g., 19).
        """
        super().__init__()

        self.num_channels = num_channels
        self.use_graph_features = use_graph_features and graph_feature_dim > 0

        # Compute the classifier input dimension
        # After GCN pooling, we get out_channels features
        classifier_input_dim = out_channels
        if self.use_graph_features:
            classifier_input_dim += graph_feature_dim

        # Store output dimension for reference
        self.node_feature_dim = lstm_out_dim

        # Initialize the temporal encoder for each EEG channel.
        self.channel_encoder = EEGCNNBiLSTMEncoder(
            cnn_dropout=cnn_dropout,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_out_dim=lstm_out_dim,
            lstm_dropout=lstm_dropout,
            num_layers=lstm_num_layers,
            use_batch_norm=encoder_use_batch_norm,
            use_layer_norm=encoder_use_layer_norm,
            add_classifier=False,
        )

        # Initialize the Graph Convolutional Network.
        # Input dimension is the output of the LSTM encoder (lstm_out_dim).
        # The GCN will process the node features extracted by the LSTM encoder.
        self.gcn = EEGGCN(
            in_channels=lstm_out_dim,
            hidden_channels=hidden_dim,    # Use hidden_dim as hidden_channels
            out_channels=out_channels,     # Use out_channels as final output before pooling
            num_conv_layers=num_conv_layers,
            pooling_type=pooling_type,
            dropout_prob=gcn_dropout,
            use_batch_norm=gcn_use_batch_norm,
            use_cnn_preprocessing=False, # no feature reduction since we use directly the LSTM output
            mlp_dims=None, # Disable built-in classifier, as we need a more complex one
        )

        # Classifier layers to map GCN output to class logits.
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Log model configuration
        logger.info(f"EEGCNNBiLSTMGCN initialized:")
        logger.info(f"  - Node input dim: {node_input_dim}")
        logger.info(f"  - Node feature dim (LSTM output): {lstm_out_dim}")
        logger.info(f"  - GCN hidden dim: {hidden_dim}")
        logger.info(f"  - Graph feature dim: {graph_feature_dim}")
        logger.info(f"  - Use graph features: {self.use_graph_features}")
        logger.info(f"  - Classifier input dim: {classifier_input_dim}")
        logger.info(f"  - Num classes: {num_classes}")
        logger.info(f"  - Num channels: {num_channels}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, graph_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the CNN-BiLSTM-GCN model.

        Args:
            x (torch.Tensor): Input EEG signals. Expected shape:
                              [num_graphs_in_batch * num_channels, time_steps]
                              For example, [batch_size * 19, 3000].
            edge_index (torch.Tensor): Graph connectivity in COO format. Shape: [2, num_edges_in_batch].
                                       Node indices are assumed to be global across the batch
                                       as handled by PyTorch Geometric's DataLoader.
            batch (torch.Tensor): Batch vector mapping each node to its respective graph.
                                  Shape: [total_num_nodes_in_batch]. Used for global pooling.
            graph_features (torch.Tensor, optional): Graph-level features. Shape: [num_graphs_in_batch, graph_feature_dim].

        Returns:
            torch.Tensor: Class logits for each graph in the batch. Shape: [num_graphs_in_batch, num_classes].
        """
        node_features = self.channel_encoder(x)
        gcn_output = self.gcn(node_features, edge_index, batch) # [num_graphs_in_batch, out_channels]
        
        # Combine with graph-level features if available
        if self.use_graph_features:
            # check if graph features have been loaded
            if graph_features is None:
                raise ValueError(
                    "EEGCNNBiLSTMGCN is configured to use graph features (self.use_graph_features is True), "
                    "but graph_features were not provided (is None) in the forward pass. "
                    f"GCN output has {gcn_output.shape[-1]} features, classifier expects "
                    f"{self.classifier[0].in_features} features."
                )
            # Ensure that the graph features are of the expected shape
            if graph_features.shape[0] != gcn_output.shape[0]:
                raise ValueError(
                    f"Graph features shape {graph_features.shape} does not match GCN output shape {gcn_output.shape}. "
                    "Ensure that graph_features has the same number of rows as the number of graphs in the batch."
                )

            # Concatenate node-level aggregated features with graph-level features
            print("Graph features shape:", graph_features.shape)
            print(graph_features)
            combined_features = torch.cat([gcn_output, graph_features], dim=1)
        else:
            combined_features = gcn_output
        
        # Final classification
        logits = self.classifier(combined_features)

        return logits