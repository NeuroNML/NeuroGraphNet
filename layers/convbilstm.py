import torch.nn as nn

class EEGConvBiLSTM(nn.Module):
    """
    Small CNN + Bi-LSTM + sigmoid logits
    """
    def __init__(self, in_ch=19, hidden=128, num_layers=2, p_drop=0.3):
        super().__init__()
        # temporal CNN front‑end
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Dropout2d(p_drop)                         # locked dropout
        )
        # Bi‑LSTM
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=p_drop
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.fc   = nn.Linear(hidden * 2, 1)

    def forward(self, x):             # x: (B, T, C)
        x = x.permute(0, 2, 1)        # (B, C, T)
        x = self.cnn(x)               # (B, 64, T/2)
        x = x.permute(0, 2, 1)        # (B, T/2, 64)
        out, _ = self.lstm(x)         # (B, T/2, 2*hidden)
        out = self.norm(out[:, -1])   # last time‑step
        return self.fc(out)           # logits
