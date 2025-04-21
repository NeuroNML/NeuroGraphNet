import torch
import torch.nn as nn
import torch.optim as optim

class EEGCNN(nn.Module):
    def __init__(self, input_channels=19, num_classes=1, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            # conv block 1
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # conv block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # conv block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # global pooling → fixed 128‐dim output per example
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # expect x: (batch, seq_len, channels)  
        # conv1d wants (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        feat = self.features(x)
        return self.classifier(feat)
