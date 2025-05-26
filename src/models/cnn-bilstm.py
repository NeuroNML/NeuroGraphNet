import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN
class EEGCNN(nn.Module):
    def __init__(self, in_channels=1, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Output shape: [B, 64, 1]
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64, 32)
        self.bn = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, 1, T]
        x = self.relu(self.conv1(x))       # [B, 32, T]
        x = self.pool(x)                   # [B, 32, T/2]
        x = self.relu(self.conv2(x))       # [B, 64, T/2]
        x = self.global_pool(x)            # [B, 64, 1]
        x = self.flatten(x)                # [B, 64]
        x = self.relu(self.dense(x))       # [B, 32]
        x = self.bn(x)
        x = self.dropout(x)
        return x


# Bi-LSTM 
class EEGBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_dims=[64, 32], dropout=0.25):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_dims[0],
                             batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=2*hidden_dims[0], hidden_size=hidden_dims[1],
                             batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(2 * hidden_dims[1], 10)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(10)

    def forward(self, x):
        # x: [B, T] → reshape to [B, T, 1]
        x = x.unsqueeze(-1)
        x, _ = self.lstm1(x)               # [B, T, 2*64]
        x, _ = self.lstm2(x)               # [B, T, 2*32]
        x = x[:, -1, :]                    # Take last timestep → [B, 64]
        x = self.relu(self.dense(x))       # [B, 10]
        x = self.bn(x)
        return x


# Combined model
class CNN_BiLSTM_Encoder(nn.Module):
    def __init__(self, time_steps=3000):
        super().__init__()
        self.cnn_path = EEGCNN(in_channels=1, time_steps=time_steps)
        self.lstm_path = EEGBiLSTM(input_size=1)
        self.fusion = nn.Sequential(
            nn.Linear(32 + 10, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Binary classification
        )

    def forward(self, x):
        # x: [B, T]
        x_cnn = self.cnn_path(x.unsqueeze(1))  # [B, 32]
        x_lstm = self.lstm_path(x)             # [B, 10]
        x_fused = torch.cat([x_cnn, x_lstm], dim=1)  # [B, 42]
        out = self.fusion(x_fused)             # [B, 1]
        return out
