import torch.nn as nn

class EEGConvBiLSTM(nn.Module):
    """
    Small CNN + Bi-LSTM + sigmoid logits
    """
    def __init__(self, in_ch=19, hidden=128, num_layers=2, p_drop=0.3):
        super().__init__()
        # Convolutional layers to extract features
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Dropout1d(p_drop) # dropout after conv layers
        )
        # Bi-LSTM layers to capture temporal dependencies
        # Note: hidden size is doubled because of bidirectional
        self.lstm = nn.LSTM(
            64, hidden, num_layers,
            batch_first=True, bidirectional=True,
            dropout=p_drop # dropout between LSTM layers
        )
        # Layer normalization to stabilize training
        # Note: we apply layer norm to the last hidden state
        self.norm = nn.LayerNorm(hidden * 2)
        # Fully connected layer to output logits
        self.fc   = nn.Linear(hidden * 2, 1)

    def forward(self, x):                # x: (B, T, C)
        x = x.permute(0, 2, 1)           # (B, C, T)
        x = self.cnn(x)                  # (B, 64, T/2)
        x = x.permute(0, 2, 1)           # (B, T/2, 64)
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1])
        return self.fc(out)              # logits
