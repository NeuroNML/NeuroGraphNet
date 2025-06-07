import torch
import torch.nn as nn

from src.layers.cnn_bilstm import CNN_BiLSTM_Encoder
from src.layers.gnn.gcn import EEGGCN
from src.layers.hybrid.cnn_bilstm import EEGCNNBiLSTMEncoder


class EEGCNNBiLSTMGNN(nn.Module):
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
        cnn_dropout: float = 0.25,
        lstm_hidden_dim: int = 64,
        lstm_out_dim: int = 64,  # This will be the time_encoder_output_dim for the GCN
        lstm_dropout: float = 0.25,
        num_layers: int = 1,
        # Parameters for the EEGGCN (graph neural network)
        gcn_hidden_channels: int = 64,
        gcn_out_channels: int = 32,
        num_gcn_layers: int = 3,
        gcn_dropout: float = 0.5,
        num_classes: int = 1,  # For binary classification (seizure/non-seizure)
        num_channels: int = 19,  # Number of EEG channels
    ):
        """
        Initializes the EEGCNNBiLSTMGNN.

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
            num_classes (int): Number of output classes (e.g., 1 for binary seizure detection).
            num_channels (int): The fixed number of EEG channels (e.g., 19).
        """
        super().__init__()

        self.num_channels = num_channels
        # The output dimension of the channel encoder is now directly lstm_out_dim
        self.time_encoder_output_dim = lstm_out_dim

        # Initialize the temporal encoder for each EEG channel.
        # Parameters for CNN_BiLSTM_Encoder are passed directly.
        self.channel_encoder = CNN_BiLSTM_Encoder(
            cnn_dropout=cnn_dropout,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_out_dim=lstm_out_dim,
            lstm_dropout=lstm_dropout,
            num_layers=num_layers,
        )

        # Initialize the Graph Convolutional Network.
        # Its input dimension (in_channels) will be the output dimension
        # produced by the channel_encoder (lstm_out_dim).
        self.gcn = EEGGCN(
            in_channels=self.time_encoder_output_dim,  # This is lstm_out_dim
            hidden_channels=gcn_hidden_channels,
            out_channels=gcn_out_channels,
            num_classes=num_classes,  # This is used for the final linear layer within EEGGCN
            num_conv_layers=num_gcn_layers,
            dropout=gcn_dropout,
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
            torch.Tensor: Class logits for each graph in the batch. Shape: [num_graphs_in_batch, num_classes].
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