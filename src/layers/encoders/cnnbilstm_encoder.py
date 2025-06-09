import torch.nn as nn

from src.layers.encoders.cnn_encoder import EEGCNNEncoder
from src.layers.encoders.bilstm_encoder import EEGBiLSTMEncoder
class EEGCNNBiLSTMEncoder(nn.Module):
    """
    CNN-BiLSTM Encoder for EEG signal processing.
    Features:
    - CNN for spatial feature extraction
    - BiLSTM for temporal processing
    - Configurable architecture parameters
    - Optional classification head
    """
    def __init__(self, in_channels=1, cnn_out_channels=128, cnn_dropout=0.25, use_cnn_batch_norm=True,
                 lstm_hidden_dim=64, lstm_out_dim=64, lstm_dropout=0.25, num_lstm_layers=1,
                 use_lstm_layer_norm=True, add_classifier=False):
        super().__init__()

        # --- CNN Encoder (Feature Extractor) ---
        # The CNN will output feature maps (sequences) for the BiLSTM, so global_pool_output=False
        self.cnn = EEGCNNEncoder(
            in_channels=in_channels,
            out_dim=cnn_out_channels,        # The number of channels in the CNN's output feature maps
            dropout_rate=cnn_dropout,
            use_batch_norm=use_cnn_batch_norm,
            global_pool_output=False         # Crucial: CNN outputs sequence for LSTM
        )

        # --- BiLSTM Encoder for temporal processing ---
        # input_size for LSTM should be the out_dim (number of channels) from the CNN encoder
        self.lstm = EEGBiLSTMEncoder(
            input_size=cnn_out_channels, # CNN output channels become LSTM input features
            hidden_dim=lstm_hidden_dim,
            out_dim=lstm_out_dim,
            dropout=lstm_dropout,
            num_layers=num_lstm_layers,
            use_layer_norm=use_lstm_layer_norm
        )
        self.classifier = nn.Linear(lstm_out_dim, 1) if add_classifier else nn.Identity()

    def forward(self, x):
        """
        Forward pass for the CNN-BiLSTM encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps) for single channel,
                              or (batch_size, channels, time_steps) for multi-channel.
        Returns:
            torch.Tensor: Output of the classifier if add_classifier=True (shape [B, 1]),
                          else the output of the BiLSTM (shape [B, lstm_out_dim]).
        """
        # Ensure input is (batch_size, channels, time_steps) for the CNN encoder
        # This handles cases where input is (B, T) and in_channels=1 for CNN.
        if x.dim() == 2:
            x = x.unsqueeze(1) # Becomes (B, 1, T)

        # Pass through CNN to get feature maps (B, cnn_out_channels, time_steps_out)
        x = self.cnn(x)

        # Permute dimensions for BiLSTM: (B, sequence_length, input_size)
        # where sequence_length is time_steps_out from CNN, and input_size is cnn_out_channels
        x = x.permute(0, 2, 1) # [B, time_steps_out, cnn_out_channels]

        # Pass through BiLSTM
        x = self.lstm(x) # [B, lstm_out_dim] (assuming BiLSTM outputs a single vector per batch item)

        # Apply classifier if configured
        return self.classifier(x)