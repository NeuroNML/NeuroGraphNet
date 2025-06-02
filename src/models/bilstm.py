import torch
import torch.nn as nn

class EEGBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=1):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False  # BiLSTM
        )
        # Output will be [B, hidden_size]
        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """
        x: [batch_size, time_steps]
        """
        x = x.unsqueeze(-1)  # [B, T] → [B, T, 1]
        _, (hn, _) = self.bilstm(x) 
        hn = hn.squeeze(0)  
        return self.output_layer(hn)  # [B, hidden_size]
