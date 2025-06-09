import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.encoders.cnnbilstmattention_encoder import EEGCNNBiLSTMAttentionEncoder
from src.layers.gnn.gat import EEGGAT


class EEGCNNBiLSTMAttentionGAT(nn.Module):
    """
    Hybrid model combining a CNN-BiLSTM-Attention encoder for temporal feature extraction
    per EEG channel and a Graph Attention Network (GAT) for spatial feature integration.

    This architecture extends the CNN-BiLSTM-GAT model by incorporating attention mechanisms
    both in the BiLSTM component (temporal attention) and in the GAT component (spatial attention),
    allowing the model to focus on the most relevant temporal and spatial patterns in the EEG signals.
    """

    def __init__(
        self,
        # Parameters for the CNN encoder
        cnn_dropout_prob: float = 0.25,
        # Parameters for the BiLSTM-Attention encoder
        lstm_hidden_dim: int = 64,
        lstm_out_dim: int = 64,  # This will be the time_encoder_output_dim for the GAT
        encoder_use_batch_norm: bool = True,
        encoder_use_layer_norm: bool = False,
        lstm_num_layers: int = 1,
        lstm_dropout_prob: float = 0.25,
        # Parameters for the EEGGAT (graph attention network)
        gat_hidden_channels: int = 64,
        gat_out_channels: int = 32,
        gat_heads: int = 4,  # Number of attention heads
        gat_pooling_type: str = "mean",
        gat_use_batch_norm: bool = True,
        gat_num_layers: int = 3,
        gat_dropout_prob: float = 0.5,
        # General parameters
        num_channels: int = 19,  # Number of EEG channels
    ):
        """
        Initializes the EEGCNNBiLSTMAttentionGAT.

        Args:
            cnn_dropout_prob (float): Dropout probability for the CNN part of the temporal encoder.
            lstm_hidden_dim (int): Number of hidden units in the LSTM part of the temporal encoder.
            lstm_out_dim (int): Output feature dimension from the LSTM part of the temporal encoder.
                               This also serves as the input dimension for the GAT.
            encoder_use_batch_norm (bool): Whether to use batch normalization in encoder layers.
            encoder_use_layer_norm (bool): Whether to use layer normalization in encoder layers.
            lstm_num_layers (int): Number of layers in the BiLSTM.
            lstm_dropout_prob (float): Dropout probability for the LSTM part of the temporal encoder.
            gat_hidden_channels (int): Number of hidden units in the GAT layers.
            gat_out_channels (int): Output feature dimensions from the last GAT layer before final classification.
            gat_heads (int): Number of attention heads in GAT layers.
            gat_pooling_type (str): Type of graph pooling ("mean", "max", or "sum").
            gat_use_batch_norm (bool): Whether to use batch normalization in GAT layers.
            gat_num_layers (int): Number of GAT convolutional layers.
            gat_dropout_prob (float): Dropout probability for GAT layers.
            num_channels (int): The fixed number of EEG channels (e.g., 19).
        """
        super().__init__()

        self.num_channels = num_channels
        self.time_encoder_output_dim = lstm_out_dim

        self.channel_encoder = EEGCNNBiLSTMAttentionEncoder(
            cnn_dropout=cnn_dropout_prob,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_out_dim=lstm_out_dim,
            lstm_dropout=lstm_dropout_prob,
            num_layers=lstm_num_layers,
            use_batch_norm=encoder_use_batch_norm,
            use_layer_norm=encoder_use_layer_norm,
            add_classifier=False,
        )

        # Initialize the Graph Attention Network
        self.gat = EEGGAT(
            in_channels=self.time_encoder_output_dim,
            hidden_channels=gat_hidden_channels,
            num_conv_layers=gat_num_layers,
            heads=gat_heads,
            pooling_type=gat_pooling_type,
            out_channels=gat_out_channels,
            dropout_prob=gat_dropout_prob,
            use_batch_norm=gat_use_batch_norm,
            use_cnn_preprocessing=False,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN-BiLSTM-Attention-GAT model.

        Args:
            x (torch.Tensor): Input EEG signals. Expected shape:
                            [num_graphs_in_batch, num_channels, time_steps]
                            For example, [batch_size, 19, 3000].
            edge_index (torch.Tensor): Graph connectivity in COO format. Shape: [2, num_edges_in_batch].
                                     Node indices are assumed to be global across the batch
                                     as handled by PyTorch Geometric's DataLoader.
            batch_labels (torch.Tensor): Batch vector mapping each node to its respective graph.
                                       Shape: [total_num_nodes_in_batch]. Used for global pooling.

        Returns:
            torch.Tensor: Class logits for each graph in the batch. Shape: [num_graphs_in_batch, num_classes].
        """
        # Encode each channel's time series into a fixed-size embedding.
        # The output `node_features` will have shape:
        # [num_graphs_in_batch * num_channels, lstm_out_dim]
        node_features = self.channel_encoder(x)
        
        # Pass through GAT
        logits = self.gat(node_features, edge_index, batch_labels)

        return logits
