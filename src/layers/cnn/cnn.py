import torch.nn as nn
import torch.nn.functional as F

from src.layers.mlp.mlp import EEGMLPClassifier

class EEGCNNInternalEncoder(nn.Module):
    """
    CNN Encoder for EEG signal processing, extracting features from raw time-series data.
    """
    def __init__(self, in_channels, out_dim=128): # in_channels is now a mandatory arg
        """
        Args:
            in_channels (int): Number of input channels (e.g., 1 for single-channel EEG, or num_electrodes).
            out_dim (int): The dimension of the final feature vector extracted by the CNN.
        """
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        # Convolutional Block 1
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)

        # Convolutional Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        # Convolutional Block 3
        self.conv3 = nn.Conv1d(64, out_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_dim)

        # Global pooling layer to reduce time dimension to 1
        self.pool = nn.AdaptiveAvgPool1d(1)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass for the CNN encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).
        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, out_dim).
        """
        # EEGCNNEncoder now *assumes* (B, C, T) input.
        # The `unsqueeze(1)` logic for (B, T) input is handled by EEGMLPClassifier
        # before calling this encoder.

        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))

        x = self.pool(x).squeeze(-1)
        return x

class EEGCNNClassifier(nn.Module):
    """
    Combines a CNN Encoder for feature extraction with an MLP for classification.
    This class now uses the EEGMLPClassifier as its classification head.
    """
    def __init__(
        self,
        input_channels=1, # E.g., 1 for single-channel, or 19 for multi-channel
        cnn_out_dim=128,  # Output dimension of the CNN encoder
        mlp_hidden_dims=[256, 128], # Hidden layers for the MLP classifier head
        output_dim=1,     # Final output dimension (e.g., 1 for binary classification)
        dropout_prob=0.3,
        use_batch_norm_mlp=True, # Whether to use batch norm in MLP
        activation_mlp="leaky_relu" # Activation for MLP layers
    ):
        super().__init__()

        # --- CNN Encoder (Feature Extractor) ---
        # The in_channels for the encoder should match the actual input channels of the EEG data
        self.cnn_encoder = EEGCNNInternalEncoder(in_channels=input_channels, out_dim=cnn_out_dim)

        # --- MLP Classifier (Classification Head) ---
        # Instantiate EEGMLPClassifier as the classification head.
        # It will receive the flattened features from the CNN encoder.
        # Pass the cnn_out_dim as input_dim to the MLP.
        # Ensure argument names match EEGMLPClassifier's __init__ method.
        self.mlp_classifier_head = EEGMLPClassifier(
            input_dim=cnn_out_dim,       # Input to MLP is the output of the CNN encoder
            hidden_dims=mlp_hidden_dims, 
            output_dim=output_dim,
            dropout_prob=dropout_prob,
            use_batch_norm=use_batch_norm_mlp,
            activation=activation_mlp,
            # For the MLP head, we don't need 2D input specific parameters or CNN encoder
            # as its input is already a 1D feature vector from the EEGCNNEncoder.
            input_channels=None, # Explicitly set to None for MLP's internal config
            input_time_steps=None, # Explicitly set to None for MLP's internal config
        )

    def forward(self, x):
        """
        Forward pass for the combined CNN-MLP classifier.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps)
                              or (batch_size, time_steps) if input_channels=1 is used
                              for the EEGCNNEncoder (handled by EEGMLPClassifier's `unsqueeze`
                              if it were directly calling EEGCNNEncoder, but here we explicitly
                              handle it before passing to self.cnn_encoder).
        Returns:
            torch.Tensor: Logits for classification of shape (batch_size, output_dim).
        """
        # Ensure input is (batch_size, channels, time_steps) for the CNN encoder
        # If the input_channels of EEGCNNClassifier is 1, and input x is (B, T), unsqueeze it here.
        if x.dim() == 2 and self.cnn_encoder.conv1.in_channels == 1:
            x = x.unsqueeze(1) # Becomes (B, 1, T)

        # Pass data through the CNN encoder
        cnn_features = self.cnn_encoder(x) # Shape: (batch_size, cnn_out_dim)

        # Pass extracted features through the MLP classification head
        # The MLPClassifier expects a 1D input, which cnn_features already is.
        logits = self.mlp_classifier_head(cnn_features) # Shape: (batch_size, output_dim)
        return logits

