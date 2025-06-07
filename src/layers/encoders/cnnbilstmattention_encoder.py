import torch.nn as nn

from src.layers.encoders.cnn_encoder import EEGCNNEncoder
from src.layers.encoders.bilstm_attention_encoder import EEGBiLSTMEncoder

class EEGCNNBiLSTMAttentionEncoder(nn.Module):
    """
    CNN-BiLSTM Encoder for EEG signal processing.
    Features:
    - CNN for spatial feature extraction
    - BiLSTM + Attention for temporal processing
    - Configurable architecture parameters
    - Optional classification head
    """
    def __init__(self, in_channels=1, cnn_dropout=0.25, lstm_hidden_dim=64,
                 lstm_out_dim=64, lstm_dropout=0.25, num_layers=1,
                 use_batch_norm=True, use_layer_norm=True, add_classifier=False):
        super().__init__()
        self.cnn = EEGCNNEncoder(
            in_channels=in_channels,
            dropout=cnn_dropout,
            use_batch_norm=use_batch_norm
        )

        self.lstm = EEGBiLSTMEncoder(
            input_dim=128,  # CNN output channels
            hidden_dim=lstm_hidden_dim,
            dropout=lstm_dropout,
            num_layers=num_layers,
            use_layer_norm=use_layer_norm,
        )
        self.classifier = nn.Linear(lstm_out_dim, 1) if add_classifier else nn.Identity()

    def forward(self, x):
        # x: [B, T]
        x = x.unsqueeze(1)  # [B, 1, T]
        x = self.cnn(x)  # [B, 128, T/8]
        x = x.permute(0, 2, 1)  # [B, T/8, 128]
        x = self.lstm(x)  # [B, lstm_out_dim]
        return self.classifier(x)  # [B, 1] if classifier, else [B, lstm_out_dim]