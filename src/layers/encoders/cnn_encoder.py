import torch.nn as nn

class EEGCNNEncoder(nn.Module):
    """
    CNN Encoder for EEG signal processing, designed to extract features from raw time-series data.
    This version is a fusion of previous encoders, offering enhanced configurability.

    Features:
    - Configurable 3-block CNN architecture (channel sizes, kernel sizes, paddings).
    - Choice of activation function (ReLU or LeakyReLU).
    - Batch normalization after each convolutional layer (configurable).
    - Dropout for regularization (configurable rate).
    - Adaptive average pooling for fixed-size output vector.
    - Kaiming weight initialization.
    - Produces a flat feature vector suitable for downstream classification/regression tasks.
    """
    def __init__(self,
                 in_channels=1,
                 out_channels_conv1=32,
                 out_channels_conv2=64,
                 out_dim=128,  # Output dimension, also #channels for conv3
                 kernel_sizes=(7, 5, 3), # Kernel sizes for conv1, conv2, conv3
                 paddings=(3, 2, 1),     # Paddings for conv1, conv2, conv3
                 activation_type='leaky_relu', # 'relu' or 'leaky_relu'
                 dropout_rate=0.25,
                 use_batch_norm=True,
                 global_pool_output=False,
                ):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 1 for single-channel EEG).
            out_channels_conv1 (int): Number of output channels for the first convolutional layer.
            out_channels_conv2 (int): Number of output channels for the second convolutional layer.
            out_dim (int): The dimension of the final feature vector and output channels of the third conv layer.
            kernel_sizes (tuple of int): Kernel sizes for the three convolutional layers.
            paddings (tuple of int): Paddings for the three convolutional layers.
            activation_type (str): Type of activation function ('relu' or 'leaky_relu').
            dropout_rate (float): Dropout rate (0 to 1). If 0, dropout is disabled.
            use_batch_norm (bool): Whether to use batch normalization.
            final_pool_output_size (int): The target output size of the time dimension for adaptive pooling.
        """
        super().__init__()

        # Activation Function
        if activation_type == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation_type == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation_type: {activation_type}. Choose 'relu' or 'leaky_relu'.")

        self.use_batch_norm = use_batch_norm
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Convolutional Block 1
        self.conv1 = nn.Conv1d(in_channels, out_channels_conv1, kernel_size=kernel_sizes[0], padding=paddings[0])
        self.bn1 = nn.BatchNorm1d(out_channels_conv1) if self.use_batch_norm else nn.Identity()

        # Convolutional Block 2
        self.conv2 = nn.Conv1d(out_channels_conv1, out_channels_conv2, kernel_size=kernel_sizes[1], padding=paddings[1])
        self.bn2 = nn.BatchNorm1d(out_channels_conv2) if self.use_batch_norm else nn.Identity()

        # Convolutional Block 3
        self.conv3 = nn.Conv1d(out_channels_conv2, out_dim, kernel_size=kernel_sizes[2], padding=paddings[2])
        self.bn3 = nn.BatchNorm1d(out_dim) if self.use_batch_norm else nn.Identity()

        # Global Pooling Layer
        self.pool = nn.AdaptiveAvgPool1d(1) if global_pool_output else nn.Identity()


        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights of the network.
        - Conv1d layers: Kaiming normal initialization.
        - BatchNorm1d layers: Weights to 1, biases to 0.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nonlinearity = 'leaky_relu' if isinstance(self.activation, nn.LeakyReLU) else 'relu'
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass for the CNN encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, time_steps).
        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, out_dim) if final_pool_output_size is 1,
                          otherwise (batch_size, out_dim, final_pool_output_size).
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Apply global pooling only if configured
        if self.global_pool_output:
            x = self.pool(x).squeeze(-1) # shape: (B, out_dim)
        return x

        if self.pool.output_size == 1:
            x = x.squeeze(-1)  # Shape: [B, out_dim]
        return x