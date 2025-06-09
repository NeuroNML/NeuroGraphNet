import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGMLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron for EEG signal processing.
    Features:
    - Configurable number of layers and dimensions
    - Optional batch normalization
    - Dropout for regularization
    - Optional residual connections
    - Configurable activation functions
    """
    def __init__(
        self,
        input_dim=228,
        hidden_dims=[1024, 512, 256],
        output_dim=1,
        dropout_prob=0.25,
        use_batch_norm=True,
        use_residual=False,
        activation="relu",
        input_channels=None,
        input_time_steps=None
    ):
        """
        Args:
            input_dim: Input feature dimension (default: 228 for EEG features) 
                      OR when using 2D signal input, this represents the flattened dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (default: 1 for binary classification)
            dropout_prob: Dropout probability
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
            activation: Activation function ("relu", "leaky_relu", or "elu")
            input_channels: Number of input channels for 2D signal input (e.g., 19 for EEG)
            input_time_steps: Number of time steps for 2D signal input (e.g., 3000)
        """
        super().__init__()
        
        # Store 2D input configuration
        self.input_channels = input_channels
        self.input_time_steps = input_time_steps
        self.is_2d_input = input_channels is not None and input_time_steps is not None
        
        # Calculate actual input dimension for the MLP layers
        if self.is_2d_input:
            if input_channels is None or input_time_steps is None:
                raise ValueError("input_channels and input_time_steps must be provided for 2D input mode")
            actual_input_dim = input_channels * input_time_steps
        else:
            actual_input_dim = input_dim
        
        # Set activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build layers
        layers = []
        prev_dim = actual_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout_prob)
            ])
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        self.use_residual = use_residual and (actual_input_dim == output_dim)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor 
               - For 1D features: [batch_size, input_dim]
               - For 2D signal features: [batch_size, channels, time_steps]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        # Handle 2D signal input if configured
        if self.is_2d_input:
            if x.dim() == 3:  # [batch_size, channels, time_steps]
                # Flatten to [batch_size, channels * time_steps]
                x = x.view(x.size(0), -1)
            elif x.dim() == 2:
                # Input is already flattened, which is fine for flattened mode
                pass
            else:
                raise ValueError(f"Expected 3D input [batch_size, channels, time_steps] for 2D signal mode, got shape {x.shape}")
        
        # Validate input dimensions for residual connections
        if self.use_residual and x.shape[-1] != self.layers[0].in_features:
            raise ValueError("Input dimension must match the first layer's input dimension for residual connections.")

        # Apply the MLP layers
        if self.use_residual:
            return x + self.layers(x)
        return self.layers(x)
