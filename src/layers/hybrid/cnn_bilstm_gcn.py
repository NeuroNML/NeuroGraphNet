import torch
import torch.nn as nn

from src.layers.gnn.gcn import EEGGCN
from src.layers.encoders.cnnbilstm_encoder import EEGCNNBiLSTMEncoder


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
        # Parameters for the CNN_BiLSTM_Encoder (temporal encoder)
        cnn_dropout_prob: float = 0.25,
        lstm_hidden_dim: int = 64,
        lstm_out_dim: int = 64,  # This will be the time_encoder_output_dim for the GCN
        encoder_use_batch_norm: bool = True,
        encoder_use_layer_norm: bool = False,
        lstm_num_layers: int = 1,
        lstm_dropout_prob: float = 0.25,
        # Parameters for the EEGGCN (graph neural network)
        gcn_hidden_channels: int = 64,
        gcn_out_channels: int = 32,
        gcn_pooling_type: str = "mean",
        gcn_use_batch_norm: bool = True,
        gcn_num_layers: int = 3,
        gcn_dropout_prob: float = 0.5,
        # General parameters
        num_channels: int = 19,  # Number of EEG channels
    ):
        """
        Initializes the EEGCNNBiLSTMGCN.

        Args:
            cnn_dropout (float): Dropout probability for the CNN part of the temporal encoder.
            lstm_hidden_dim (int): Number of hidden units in the LSTM part of the temporal encoder.
            lstm_out_dim (int): Output feature dimension from the LSTM part of the temporal encoder.
                                This also serves as the input dimension for the GCN.
            lstm_dropout (float): Dropout probability for the LSTM part of the temporal encoder.
            num_layers (int): Number of layers in the CNN_BiLSTM_Encoder.
            gcn_hidden_channels (int): Number of hidden units in the GCN layers.
            gcn_out_channels (int): Output feature dimensions from the last GCN layer before final classification.
            num_gcn_layers (int): Number of GCN convolutional layers.
            gcn_dropout (float): Dropout probability for GCN layers.
            num_channels (int): The fixed number of EEG channels (e.g., 19).
        """
        super().__init__()

        self.num_channels = num_channels
        # The output dimension of the channel encoder is now directly lstm_out_dim
        self.time_encoder_output_dim = lstm_out_dim

        # Initialize the temporal encoder for each EEG channel.
        self.channel_encoder = EEGCNNBiLSTMEncoder(
            cnn_dropout=cnn_dropout_prob,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_out_dim=lstm_out_dim,
            lstm_dropout=lstm_dropout_prob,
            num_layers=lstm_num_layers,
            use_batch_norm=encoder_use_batch_norm,
            use_layer_norm=encoder_use_layer_norm,
            add_classifier=False,
        )

        # Initialize the Graph Convolutional Network.
        # Its input dimension (in_channels) will be the output dimension
        # produced by the channel_encoder (lstm_out_dim).
        self.gcn = EEGGCN(
            in_channels=self.time_encoder_output_dim,
            hidden_channels=gcn_hidden_channels,
            num_conv_layers=gcn_num_layers,
            pooling_type=gcn_pooling_type,
            out_channels=gcn_out_channels,
            dropout_prob=gcn_dropout_prob,
            use_batch_norm=gcn_use_batch_norm,
            use_cnn_preprocessing=False,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM-GNN model.

        Args:
            x (torch.Tensor): Input EEG signals. Expected shape:
                              [num_graphs_in_batch, num_channels, time_steps]
                              For example, [batch_size, 19, 3000].
            edge_index (torch.Tensor): Graph connectivity in COO format. Shape: [2, num_edges_in_batch].
                                       Node indices are assumed to be global across the batch
                                       as handled by PyTorch Geometric's DataLoader.
            batch (torch.Tensor): Batch vector mapping each node to its respective graph.
                                  Shape: [total_num_nodes_in_batch]. Used for global pooling.

        Returns:
            torch.Tensor: Class logits for each graph in the batch. Shape: [num_graphs_in_batch, 1].
        """
        # The input 'x' represents a batch of EEG recordings,
        # where each recording has multiple channels (nodes) and time steps.
        # x shape: [num_graphs_in_batch * num_channels, time_steps]

        # Encode each channel's time series into a fixed-size embedding.
        # The output `node_features` will have shape:
        # [num_graphs_in_batch * num_channels, lstm_out_dim]
        node_features = self.channel_encoder(x)

        # Pass the extracted node features (embeddings) and the graph structure
        # (edge_index and batch tensors) to the GCN.
        # The `batch` tensor correctly maps the flattened `node_features` back
        # to their respective graphs for aggregation within the GCN.
        logits = self.gcn(node_features, edge_index, batch_labels)

        # return predictions logits
        return logits