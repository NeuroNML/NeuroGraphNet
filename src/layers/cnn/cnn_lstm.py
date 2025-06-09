import torch.nn as nn
import torch.nn.functional as F

from src.layers.encoders.cnnbilstm_encoder import EEGCNNBiLSTMEncoder
from src.layers.mlp.mlp import EEGMLPClassifier
from src.layers.encoders.cnn_encoder import EEGCNNEncoder
from src.layers.encoders.bilstm_encoder import EEGBiLSTMEncoder

class EEGCNNBiLSTMClassifier(nn.Module):
    """
    Specialized CNN-BiLSTM Classifier optimized for EEG seizure detection.
    
    This architecture has been specifically tuned to detect epileptiform patterns by:
    1. Using optimized CNN kernels for different EEG frequency bands relevant to seizures
    2. Implementing a deep BiLSTM to capture complex temporal dependencies in seizure evolution
    3. Employing a robust MLP classifier head with appropriate regularization
    """
    def __init__(
        self,
        input_channels=19,
        # CNN parameters 
        cnn_out_channels_conv1=32,
        cnn_out_channels_conv2=64,
        cnn_out_dim=128,
        cnn_kernel_sizes=(9, 7, 5),
        cnn_paddings=(4, 3, 2),
        cnn_dropout_prob=0.3,
        cnn_use_batch_norm=True,
        # BiLSTM parameters
        lstm_hidden_dim=128,
        lstm_out_dim=128,
        lstm_dropout_prob=0.3,
        num_lstm_layers=2,
        use_lstm_layer_norm=True,
        # MLP classifier parameters
        mlp_hidden_dims=[256, 128],
        output_dim=1,
        mlp_dropout_prob=0.3,
        use_batch_norm_mlp=True,
        activation_mlp="leaky_relu"
    ):
        super().__init__()

        # --- Custom CNN Encoder optimized for seizure detection ---
        self.cnn = EEGCNNEncoder(
            in_channels=input_channels,
            out_channels_conv1=cnn_out_channels_conv1,
            out_channels_conv2=cnn_out_channels_conv2,
            out_dim=cnn_out_dim,
            kernel_sizes=cnn_kernel_sizes,
            paddings=cnn_paddings,
            activation_type='leaky_relu',
            dropout_rate=cnn_dropout_prob,
            use_batch_norm=cnn_use_batch_norm,
            global_pool_output=False       # We need sequences for the LSTM
        )
        
        # --- BiLSTM Encoder for temporal seizure pattern analysis ---
        self.lstm = EEGBiLSTMEncoder(
            input_size=cnn_out_dim,      # CNN output channels become LSTM input features
            hidden_dim=lstm_hidden_dim,
            out_dim=lstm_out_dim,
            dropout=lstm_dropout_prob,
            num_layers=num_lstm_layers,
            use_layer_norm=use_lstm_layer_norm
        )
        
        # --- MLP Classifier (Classification Head) ---
        self.mlp_classifier_head = EEGMLPClassifier(
            input_dim=lstm_out_dim,      # Input dimension from LSTM output
            hidden_dims=mlp_hidden_dims,
            output_dim=output_dim,
            dropout_prob=mlp_dropout_prob,
            use_batch_norm=use_batch_norm_mlp,
            activation=activation_mlp,
        )

    def forward(self, x):
        """
        Forward pass for seizure detection.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps)
                              For EEG seizure detection, typically shape is (batch_size, 19, 3000)
                              representing 19 EEG channels with 3000 time points (e.g., 12 seconds at 250Hz).
        
        Returns:
            torch.Tensor: Logits for seizure classification of shape (batch_size, output_dim).
                         Values > 0 indicate higher probability of seizure after sigmoid activation.
        """
        # Ensure input is (batch_size, channels, time_steps) for the CNN encoder
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension if missing
            
        # Step 1: Extract frequency-domain and spatial features with CNN
        x = self.cnn(x)  # Shape: (batch_size, cnn_out_dim, time_steps_out)
        
        # Step 2: Process the CNN features with BiLSTM to capture temporal dynamics of seizures
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, time_steps_out, cnn_out_dim) for LSTM
        x = self.lstm(x)        # Shape: (batch_size, lstm_out_dim)
        
        # Step 3: Classify the extracted spatio-temporal features using the MLP head
        logits = self.mlp_classifier_head(x)  # Shape: (batch_size, output_dim)
        return logits
