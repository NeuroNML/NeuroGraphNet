import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN
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
        x = x.unsqueeze(1)  # [B, 1, 3000
        lstm_input = self.cnn_path(x).permute(
            0, 2, 1
        )  # Permute output -> [B, 375, 128]
        embedding = self.lstm_path(lstm_input)

        return embedding
