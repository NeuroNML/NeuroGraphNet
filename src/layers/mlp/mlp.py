import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGMLP(nn.Module):
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
        input_dim=3000,
        hidden_dims=[1024, 512, 256],
        output_dim=1,
        dropout_prob=0.25,
        use_batch_norm=True,
        use_residual=False,
        activation="relu"
    ):
        """
        Args:
            input_dim: Input feature dimension (default: 3000 for EEG)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (default: 1 for binary classification)
            dropout_prob: Dropout probability
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
            activation: Activation function ("relu", "leaky_relu", or "elu")
        """
        super().__init__()
        
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
        prev_dim = input_dim
        
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
        self.use_residual = use_residual and (input_dim == output_dim)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        if self.use_residual:
            return x + self.layers(x)
        return self.layers(x)

class EEGMLPEncoder(EEGMLP):
    """
    MLP Encoder for EEG signal processing.
    This is a specialized version of EEGMLP that focuses on encoding
    node features without a final classification layer.
    """
    def __init__(
        self,
        input_dim=3000,
        hidden_dims=[1024, 512],
        output_dim=128,
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            **kwargs
        )

    def forward(self, x):
        """
        x: [num_nodes, time_steps]  
        returns: [num_nodes, hidden_dim]  
        """
        print("MLP input mean:", x.mean().item(), "std:", x.std().item())
        return self.layers(x)
