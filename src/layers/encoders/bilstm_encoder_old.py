import torch.nn as nn

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