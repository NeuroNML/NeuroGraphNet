import torch.nn as nn

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

class EEGCNNBiLSTM(nn.Module):
    """
    Combined CNN-BiLSTM model for EEG signal processing.
    Features:
    - CNN for spatial feature extraction
    - BiLSTM for temporal processing
    - Configurable architecture parameters
    - Optional classification head
    """
    def __init__(self, in_channels=1, cnn_dropout=0.25, lstm_hidden_dim=64,
                 lstm_out_dim=64, lstm_dropout=0.25, num_layers=1,
                 use_batch_norm=True, use_layer_norm=True, add_classifier=False):
        super().__init__()
        self.cnn = EEGCNNEncoder(in_channels=in_channels, dropout=cnn_dropout,
                         use_batch_norm=use_batch_norm)
        self.lstm = EEGBiLSTMEncoder(
            input_size=128,  # CNN output channels
            hidden_dim=lstm_hidden_dim,
            out_dim=lstm_out_dim,
            dropout=lstm_dropout,
            num_layers=num_layers,
            use_layer_norm=use_layer_norm
        )
        self.classifier = nn.Linear(lstm_out_dim, 1) if add_classifier else nn.Identity()

    def forward(self, x):
        # x: [B, T]
        x = x.unsqueeze(1)  # [B, 1, T]
        x = self.cnn(x)  # [B, 128, T/8]
        x = x.permute(0, 2, 1)  # [B, T/8, 128]
        x = self.lstm(x)  # [B, lstm_out_dim]
        return self.classifier(x)  # [B, 1] if classifier, else [B, lstm_out_dim]

class EEGCNNBiLSTMEncoder(nn.Module):
    """
    CNN-BiLSTM Encoder for EEG signal processing.
    """
    def __init__(self, in_channels=1, cnn_dropout=0.25, lstm_hidden_dim=64,
                 lstm_out_dim=64, lstm_dropout=0.25, num_layers=1,
                 use_batch_norm=True, use_layer_norm=True, add_classifier=False):
        super().__init__()
        self.cnn = EEGCNNEncoder(in_channels=in_channels, dropout=cnn_dropout,
                         use_batch_norm=use_batch_norm)
        self.lstm = EEGBiLSTMEncoder(
            input_size=128,  # CNN output channels
            hidden_dim=lstm_hidden_dim,
            out_dim=lstm_out_dim,
            dropout=lstm_dropout,
            num_layers=num_layers,
            use_layer_norm=use_layer_norm
        )
        self.classifier = nn.Linear(lstm_out_dim, 1) if add_classifier else nn.Identity()

    def forward(self, x):
        # x: [B, T]
        x = x.unsqueeze(1)  # [B, 1, T]
        x = self.cnn(x)  # [B, 128, T/8]
        x = x.permute(0, 2, 1)  # [B, T/8, 128]
        x = self.lstm(x)  # [B, lstm_out_dim]
        return self.classifier(x)  # [B, 1] if classifier, else [B, lstm_out_dim]

# ... existing code for CNN_BiLSTM_Encoder class ...
