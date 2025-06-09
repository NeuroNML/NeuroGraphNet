import torch.nn as nn
import torch.nn.functional as F

from src.layers.encoders.cnnbilstm_encoder import EEGCNNBiLSTMEncoder
from src.layers.mlp.mlp import EEGMLPClassifier
from src.layers.encoders.cnn_encoder import EEGCNNEncoder

class EEGCNNLSTMClassifier(nn.Module):
    """
    Combines a CNN-BiLSTM Encoder for feature extraction with an MLP for classification.
    """
    def __init__(
        self,
        input_channels=1,           # For EEGCNNBiLSTMEncoder: E.g., 1 for single-channel, or num_electrodes
        # EEGCNNEncoder parameters (passed through EEGCNNBiLSTMEncoder)
        cnn_out_dim=128,        # Output channels of CNN in EEGCNNBiLSTMEncoder
        cnn_dropout_prob=0.25,
        cnn_use_batch_norm=True,
        # EEGBiLSTMEncoder parameters (passed through EEGCNNBiLSTMEncoder)
        lstm_hidden_dim=64,
        lstm_out_dim=64,             # Output dimension of the BiLSTM encoder
        lstm_dropout_prob=0.25,
        num_lstm_layers=1,
        use_lstm_layer_norm=True,
        # EEGMLPClassifier specific parameters
        mlp_hidden_dims=[256, 128],  # Hidden layers for the MLP classifier head
        output_dim=1,                # Final output dimension (e.g., 1 for binary classification)
        mlp_dropout_prob=0.3,
        use_batch_norm_mlp=True,     # Whether to use batch norm in MLP
        activation_mlp="leaky_relu"  # Activation for MLP layers
    ):
        super().__init__()

        # --- CNN-BiLSTM Encoder (Feature Extractor) ---
        # The add_classifier=False ensures that EEGCNNBiLSTMEncoder returns raw features
        # (lstm_out_dim) rather than a classification score.
        self.cnn_lstm_encoder = EEGCNNBiLSTMEncoder(
            in_channels=input_channels,
            cnn_out_channels=cnn_out_dim,
            cnn_dropout=cnn_dropout_prob,
            use_cnn_batch_norm=cnn_use_batch_norm,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_out_dim=lstm_out_dim,
            lstm_dropout=lstm_dropout_prob, # Use the renamed parameter
            num_lstm_layers=num_lstm_layers,
            use_lstm_layer_norm=use_lstm_layer_norm,
            add_classifier=False,          # Crucial: Encoder outputs features, not classification
        )

        # --- MLP Classifier (Classification Head) ---
        # The input_dim for the MLP is simply the output dimension of the CNN-BiLSTM encoder.
        mlp_input_dim = lstm_out_dim
        self.mlp_classifier_head = EEGMLPClassifier(
            input_dim=mlp_input_dim, # 1D input for MLP, which is the output of the CNN-BiLSTM encoder
            hidden_dims=mlp_hidden_dims,
            output_dim=output_dim,
            dropout_prob=mlp_dropout_prob,
            use_batch_norm=use_batch_norm_mlp,
            activation=activation_mlp,
        )

    def forward(self, x):
        """
        Forward pass for the combined CNN-LSTM classifier.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps)
                              or (batch_size, time_steps) if input_channels=1.
        Returns:
            torch.Tensor: Logits for classification of shape (batch_size, output_dim).
        """
        # Pass data through the CNN-BiLSTM encoder
        # The EEGCNNBiLSTMEncoder's forward method handles the unsqueeze(1) for (B, T) inputs.
        features = self.cnn_lstm_encoder(x) # Shape: (batch_size, lstm_out_dim)

        # Pass extracted features through the MLP classification head
        logits = self.mlp_classifier_head(features) # Shape: (batch_size, output_dim)
        return logits

