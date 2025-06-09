import torch.nn as nn
import torch.nn.functional as F

from src.layers.mlp.mlp import EEGMLPClassifier
from src.layers.encoders.cnn_encoder import EEGCNNEncoder

class EEGCNNClassifier(nn.Module):
    """
    Combines a CNN Encoder for feature extraction with an MLP for classification.
    This class now uses the EEGMLPClassifier as its classification head and a configurable EEGCNNEncoder.
    """
    def __init__(
        self,
        input_channels=1,           # For EEGCNNEncoder: E.g., 1 for single-channel, or num_electrodes
        # EEGCNNEncoder specific parameters
        cnn_out_channels_conv1=32,
        cnn_out_channels_conv2=64,
        cnn_out_dim=128,            # Output dimension of the CNN encoder (also out_channels_conv3)
        cnn_kernel_sizes=(7, 5, 3),
        cnn_paddings=(3, 2, 1),
        activation_cnn='leaky_relu',
        cnn_dropout_prob=0.25,
        cnn_use_batch_norm=True,
        # EEGMLPClassifier specific parameters
        mlp_hidden_dims=[256, 128], # Hidden layers for the MLP classifier head
        output_dim=1,               # Final output dimension (e.g., 1 for binary classification)
        mlp_dropout_prob=0.3,       # Renamed from dropout_prob for clarity
        use_batch_norm_mlp=True,    # Whether to use batch norm in MLP
        activation_mlp="leaky_relu" # Activation for MLP layers
    ):
        super().__init__()

        # --- CNN Encoder (Feature Extractor) ---
        self.cnn_encoder = EEGCNNEncoder(
            in_channels=input_channels,
            out_channels_conv1=cnn_out_channels_conv1,
            out_channels_conv2=cnn_out_channels_conv2,
            out_dim=cnn_out_dim,
            kernel_sizes=cnn_kernel_sizes,
            paddings=cnn_paddings,
            activation_type=activation_cnn,
            dropout_rate=cnn_dropout_prob,
            use_batch_norm=cnn_use_batch_norm,
            global_pool_output=True # use_global_pool=True for final pooling
        )

        # --- MLP Classifier (Classification Head) ---
        self.mlp_classifier_head = EEGMLPClassifier(
            input_dim=cnn_out_dim, # The input_dim is the output dimension of the CNN encoder
            hidden_dims=mlp_hidden_dims,
            output_dim=output_dim,
            dropout_prob=mlp_dropout_prob, # Use the renamed parameter
            use_batch_norm=use_batch_norm_mlp,
            activation=activation_mlp,
            input_channels=None, # Explicitly set to None for MLP's internal config
            input_time_steps=None, # Explicitly set to None for MLP's internal config
        )

    def forward(self, x):
        """
        Forward pass for the combined CNN-MLP classifier.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps)
                              or (batch_size, time_steps) if input_channels=1 is used
                              for the EEGCNNEncoder.
        Returns:
            torch.Tensor: Logits for classification of shape (batch_size, output_dim).
        """
        # Ensure input is (batch_size, channels, time_steps) for the CNN encoder
        # Check against the in_channels parameter of the instantiated cnn_encoder
        if x.dim() == 2 and self.cnn_encoder.conv1.in_channels == 1:
            x = x.unsqueeze(1) # Becomes (B, 1, T)
        elif x.dim() == 2 and self.cnn_encoder.conv1.in_channels != 1:
            raise ValueError(
                f"Input tensor has 2 dimensions (B, T), but cnn_encoder expects "
                f"{self.cnn_encoder.conv1.in_channels} input channels. Unsqueeze manually if appropriate."
            )
        elif x.dim() == 3 and x.shape[1] != self.cnn_encoder.conv1.in_channels:
             raise ValueError(
                f"Input tensor has {x.shape[1]} channels, but cnn_encoder expects "
                f"{self.cnn_encoder.conv1.in_channels} input channels."
            )

        # Pass data through the CNN encoder
        cnn_features = self.cnn_encoder(x) # Shape: (B, cnn_out_dim) or (B, cnn_out_dim, cnn_final_pool_output_size)

        # If the output of CNN is not flat (i.e., cnn_final_pool_output_size > 1), flatten it.
        if cnn_features.dim() > 2:
            cnn_features = cnn_features.view(cnn_features.size(0), -1)

        # Pass extracted features through the MLP classification head
        logits = self.mlp_classifier_head(cnn_features) # Shape: (batch_size, output_dim)
        return logits

