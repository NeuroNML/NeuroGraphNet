import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMEmb(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=64, bidirectional=False, dropout=0.3):
        super().__init__()

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # CNN Encoder applied to each channel
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.project = nn.Linear(lstm_output_dim, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):
        """
        x: [B, 19, 3000]
        Returns:
            logits: [B, 1]
            embeddings: [B, 19, 128]
        """
        B, C, T = x.shape  # [batch, 19, 3000]
        x = x.view(B * C, 1, T)  # [B*19, 1, 3000]

        x = self.conv(x)  # [B*19, 64, T]
        x = F.adaptive_avg_pool1d(x, 256)   # [B*19, 64, T']
        x = x.permute(0, 2, 1)  # [B*19, T', 64] for LSTM

        lstm_out, _ = self.lstm(x)  # [B*19, T', hidden_dim]
        lstm_out = self.dropout(lstm_out)
        pooled = lstm_out.mean(dim=1)  # [B*19, hidden_dim]

        embeddings = self.project(pooled)  # [B*19, 128]
        embeddings = embeddings.view(B, C, -1)  # [B, 19, 128]

        graph_embeddings = embeddings.mean(dim=1)  # [B, 128]
        logits = self.classifier(graph_embeddings)  # [B, 1]

        return logits, embeddings
