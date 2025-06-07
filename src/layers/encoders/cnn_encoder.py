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