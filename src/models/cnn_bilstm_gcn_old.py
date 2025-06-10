import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear, BatchNorm1d

class EEGBiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder for EEG signal processing.
    Features:
    - Configurable number of layers
    - Dropout between layers
    - Layer normalization
    - Projection layer for output dimension control
    """
    def __init__(self, input_size=128, hidden_dim=64, out_dim=64, 
                 dropout=0.25, num_layers=1, use_layer_norm=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.project = nn.Sequential(
            nn.Linear(2 * hidden_dim, out_dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(2 * hidden_dim) if use_layer_norm else nn.Identity()

    def forward(self, x):
        # x: [B, T, input_size]
        x, _ = self.lstm(x)  # [B, T, 2*hidden_dim]
        x = x[:, -1, :]  # Take last time step
        x = self.layer_norm(x)
        x = self.project(x)  # [B, out_dim]
        x = self.dropout(x)
        return x

class EEGGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, num_conv_layers=3, dropout=0.5):
        super(EEGGCN, self).__init__()
        
        # Ensure num_conv_layers is not negative, handle 0 if it means direct to linear
        if num_conv_layers < 0:
            raise ValueError("num_conv_layers cannot be negative.")

        self.num_conv_layers = num_conv_layers
        self.dropout = torch.nn.Dropout(dropout)
        
        self.conv_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        
        current_dim = in_channels

        if num_conv_layers > 0:
            # First layer
            self.conv_layers.append(GCNConv(current_dim, hidden_channels if num_conv_layers > 1 else out_channels))
            self.bn_layers.append(BatchNorm1d(hidden_channels if num_conv_layers > 1 else out_channels))
            current_dim = hidden_channels if num_conv_layers > 1 else out_channels
            
            # Middle layers
            for _ in range(1, num_conv_layers - 1):
                self.conv_layers.append(GCNConv(current_dim, hidden_channels))
                self.bn_layers.append(BatchNorm1d(hidden_channels))
                # current_dim remains hidden_channels
            
            # Last layer (if num_conv_layers > 1 and it's different from the first)
            if num_conv_layers > 1:
                self.conv_layers.append(GCNConv(current_dim, out_channels))
                self.bn_layers.append(BatchNorm1d(out_channels))
            
            # The input to the linear layer will be out_channels if GCN layers are used
            linear_in_features = out_channels
        else:
            # If no GCN layers, the input to linear is the original in_channels
            # (or you might want a projection if in_channels is too large)
            # This example assumes direct pass-through of initial features after pooling.
            linear_in_features = in_channels 
            # Or: self.projection = Linear(in_channels, out_channels)
            # linear_in_features = out_channels

        # Output layer
        self.linear = Linear(linear_in_features, num_classes)

    def forward(self, x, edge_index, batch):
        if self.num_conv_layers > 0:
            for i in range(self.num_conv_layers): # Iterate through all GCN layers
                x = self.conv_layers[i](x, edge_index)
                x = self.bn_layers[i](x)
                x = F.relu(x)
                # Apply dropout to all GCN layers except the very last one before pooling,
                # or apply to all based on preference.
                # Your original code applied it to all but the last activation.
                if i < self.num_conv_layers - 1: # Common to apply dropout to hidden layers
                    x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # If num_conv_layers == 0 and you added a projection:
        # if self.num_conv_layers == 0 and hasattr(self, 'projection'):
        #     x = self.projection(x)
        #     x = F.relu(x) # Optional activation
            
        # Final classification
        x = self.linear(x)
        
        return x

class EEGCNNEncoder(nn.Module):
    """
    CNN Encoder for EEG signal processing.
    Features:
    - Multiple convolutional layers with increasing channels
    - Batch normalization
    - Dropout for regularization
    - Max pooling for dimensionality reduction
    """
    def __init__(self, in_channels=1, dropout=0.25, use_batch_norm=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(64) if use_batch_norm else nn.Identity()

    def forward(self, x):
        # x: [B, 1, T]
        x = self.relu(self.conv1(x))  # [B, 32, T]
        x = self.pool(x)  # [B, 32, T/2]
        x = self.dropout(x)

        x = self.relu(self.conv2(x))  # [B, 64, T/2]
        x = self.batch_norm(x)
        x = self.pool(x)  # [B, 64, T/4]
        x = self.dropout(x)

        x = self.relu(self.conv3(x))  # [B, 128, T/4]
        x = self.pool(x)  # [B, 128, T/8]
        x = self.dropout(x)

        return x

class EEGBiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder for EEG signal processing.
    Features:
    - Configurable number of layers
    - Dropout between layers
    - Layer normalization
    - Projection layer for output dimension control
    """
    def __init__(self, input_size=128, hidden_dim=64, out_dim=64, 
                 dropout=0.25, num_layers=1, use_layer_norm=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.project = nn.Sequential(
            nn.Linear(2 * hidden_dim, out_dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(2 * hidden_dim) if use_layer_norm else nn.Identity()

    def forward(self, x):
        # x: [B, T, input_size]
        x, _ = self.lstm(x)  # [B, T, 2*hidden_dim]
        x = x[:, -1, :]  # Take last time step
        x = self.layer_norm(x)
        x = self.project(x)  # [B, out_dim]
        x = self.dropout(x)
        return x

class EEGConvBiLSTM(nn.Module):
# -------------------------- MODEL -----------------------#

    def __init__(self, output_dim_lstm =128, in_channels=1):
        super().__init__()
        # Result CNN: compress the time series while increasing the number of channels.
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), # Input: 1D time signal for each channel [batch_size, 19, time_samples=fs*12] -> 32 conv. filters (-5+4+1->perseve temporal length)
            nn.ReLU(),
            nn.MaxPool1d(2), # Time sample dim reduced by 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Output: [batch_size, 64, time_samples/4=fs*3]
        # self.rnn = nn.LSTM(input_size=64, hidden_size=output_dim_lstm, batch_first=True)
        # NOTE: now using a bidirectional LSTM in order to capture both forward and backward temporal information
        self.rnn = nn.LSTM(input_size=64, hidden_size=output_dim_lstm, batch_first=True, bidirectional=True)
        # Output: final hidden state -> [1, batch_size, 128]
        #self.fc = nn.Linear(128, 1)

    def forward(self, x):
        print('Before CNN')
        print(f'mean:{x.mean()},std:{ x.std()}')
        x = x.unsqueeze(1)            # [B, 3000] -> [B, 1, 3000]
        x = self.cnn(x)               # [B, 1, 3000] -> [B, 64, 3000/4= 750]
        print('After CNN')
        print(f'mean:{x.mean()},std:{ x.std()}')
        x = x.permute(0, 2, 1)        # [B, 64, T/4] -> [B, T/4, 64]
        _, (hn, _) = self.rnn(x)      # hn: [1, B, 128]
        out = hn.squeeze(0)           # [B, 128]
        print('After LSTM')
        print(f'mean:{out.mean()},std:{out.std()}')
        return out # No sigmoid to get prob. ->[batch_size, probs.]; already incorporated in loss
    

class EEGCNN(nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, 1, 3000]
        # Describe: [1,3000]
        x = self.relu(self.conv1(x))  # [32, 3000]
        x = self.pool(x)  # [32, 1500]
        x = self.dropout(x)

        x = self.relu(self.conv2(x))  # [64, 1500]
        x = self.pool(x)  # [64, 750]
        x = self.dropout(x)

        x = self.relu(self.conv3(x))  # [128, 750]
        x = self.pool(x)  # [128, 375]
        x = self.dropout(x)

        return x


# Bi-LSTM
class EEGBiLSTM(nn.Module):
    def __init__(self, hidden_dim=64, out_dim=64, dropout=0.25, input_size=128): # Inpot size fixed to ouput of CNN
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.project = nn.Sequential(
            nn.Linear(2 * hidden_dim, out_dim), nn.ReLU()  # e.g., 128 → 64
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x, _ = self.lstm(x)  # [B, T, 2H] -> Each 'time step' gets two hidden vectors
        x = x[:, -1, :]  # Single vector per node -> summary representation
        x = self.project(x)  # Reduce by half and apply Relu -> [B, H]
        x = self.dropout(x)
        return x


# Combined model
class CNN_BiLSTM_Encoder(nn.Module):
    def __init__(
        self,
        cnn_dropout=0.25,
        lstm_hidden_dim=64,
        lstm_out_dim=64,
        lstm_dropout=0.25,
    ):
        super().__init__()
        self.cnn_path = EEGCNN(dropout=cnn_dropout)
        self.lstm_path = EEGBiLSTM(
            hidden_dim=lstm_hidden_dim,
            out_dim=lstm_out_dim,
            dropout=lstm_dropout,
        )


    def forward(self, x):
        # x: [B, T]
        x = x.unsqueeze(1)  # [B, 1, 3000]
        lstm_input = self.cnn_path(x).permute(
            0, 2, 1
        )  # Permute output -> [B, 375, 128]
        embedding = self.lstm_path(lstm_input)

        return embedding

class LSTM_GCN_Model(nn.Module):
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
        # Parameters for the EEGGCN (graph neural network)
        gcn_hidden_channels: int = 64,
        gcn_out_channels: int = 32,
        num_gcn_layers: int = 3,
        gcn_dropout: float = 0.5,
        num_classes: int = 1,  # For binary classification (seizure/non-seizure)
        num_channels: int = 19,  # Number of EEG channels
    ):
        """
        Initializes the LSTM_GNN_Model.

        Args:
            cnn_dropout (float): Dropout probability for the CNN part of the temporal encoder.
            lstm_hidden_dim (int): Number of hidden units in the LSTM part of the temporal encoder.
            lstm_out_dim (int): Output feature dimension from the LSTM part of the temporal encoder.
                                This also serves as the input dimension for the GCN.
            lstm_dropout (float): Dropout probability for the LSTM part of the temporal encoder.
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