import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class EEGLSTMClassifier(nn.Module):
    """
    LSTM model for EEG signal processing.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_classes=1, input_type='feature', bidirectional=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_val = dropout
        self.bidirectional = bidirectional
        if input_type not in ['feature', 'signal', 'embedding']:
            raise ValueError("Type must be one of ['feature', 'signal', 'embedding']")
        self.input_type = input_type

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, # Ensure this is True if your input is (batch, seq, feature)
            dropout=self.dropout_val if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Adjust the FC layer input size based on bidirectional
        fc_input_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout_fc = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_type == 'feature':
            # x expected shape: (batch_size, features)
            if x.ndim == 2:  # Ensure this operation happens if input is 2D
                x = x.unsqueeze(1)  # Add sequence dimension: (batch_size, 1, features)
            lstm_out, (hn, cn) = self.lstm(x)
            
            if self.bidirectional:
                # For bidirectional LSTM, concatenate forward and backward hidden states
                # hn shape: (num_layers * 2, batch_size, hidden_dim)
                # Get the last layer's forward and backward hidden states
                forward_hidden = hn[-2]  # Forward direction of last layer
                backward_hidden = hn[-1]  # Backward direction of last layer
                last_hidden_state = torch.cat([forward_hidden, backward_hidden], dim=1)
            else:
                last_hidden_state = hn[-1] 
                
            last_hidden_state_dropped = self.dropout_fc(last_hidden_state)
            output = self.fc(last_hidden_state_dropped)
            return output
        elif self.input_type == 'signal':
            # x shape: (batch_size, sensors, time_steps) -> (512, 19, 3000)
            # Reshape to (batch_size * sensors, time_steps, 1) for LSTM processing
            batch_size, n_sensors, time_steps = x.shape
            x = x.view(batch_size * n_sensors, time_steps, 1)
            
            lstm_out, (hn, cn) = self.lstm(x)
            
            if self.bidirectional:
                # For bidirectional LSTM, concatenate forward and backward hidden states
                forward_hidden = hn[-2]  # Forward direction of last layer
                backward_hidden = hn[-1]  # Backward direction of last layer
                last_hidden_state = torch.cat([forward_hidden, backward_hidden], dim=1)
                # last_hidden_state shape: (batch_size * n_sensors, hidden_dim * 2)
                effective_hidden_dim = self.hidden_dim * 2
            else:
                last_hidden_state = hn[-1]  # (batch_size * n_sensors, hidden_dim)
                effective_hidden_dim = self.hidden_dim
            
            # Reshape back to (batch_size, n_sensors, effective_hidden_dim)
            last_hidden_state = last_hidden_state.view(batch_size, n_sensors, effective_hidden_dim)
            
            # Aggregate across sensors (mean pooling)
            aggregated = torch.mean(last_hidden_state, dim=1)  # (batch_size, effective_hidden_dim)
            
            aggregated_dropped = self.dropout_fc(aggregated)
            output = self.fc(aggregated_dropped)
            return output
        elif self.input_type == 'embedding':
            raise NotImplementedError("Embedding type is not implemented yet.")
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")
