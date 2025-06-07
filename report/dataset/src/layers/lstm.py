import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class LSTM(nn.Module):
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

class LSTMAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.2,
                 attention_dim: Optional[int] = None, bidirectional: bool = True, input_type: str = 'feature'):
        super().__init__()

        if input_type not in ['feature', 'signal', 'embedding']:
            raise ValueError("input_type must be one of ['feature', 'signal', 'embedding']")
        self.input_type = input_type
        
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # The input_dim here is the feature size for each step fed into the LSTM
        # For 'feature' type, it's total_features (e.g., 228)
        # For 'signal' type (processed per sensor), it should be 1
        self.actual_lstm_input_dim = input_dim

        self.lstm = nn.LSTM(
            input_size=self.actual_lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        if attention_dim is None:
            attention_dim = hidden_dim # Default attention internal dim

        self.attention_w = nn.Linear(lstm_output_dim, attention_dim) 
        self.attention_u = nn.Linear(attention_dim, 1, bias=False) 
        
        self.dropout_fc = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, 1) # Output one class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.input_type == 'feature':
            # x original shape: (batch_size, features) e.g., [512, 228]
            # Here, self.actual_lstm_input_dim should be == features (228)
            if x.ndim == 2:
                # Reshape to (batch_size, 1, features) for LSTM
                x = x.unsqueeze(1)
            # x shape is now (batch_size, 1, self.actual_lstm_input_dim)
            
            # lstm_out shape: (batch_size, 1, lstm_output_dim)
            lstm_out, (h_n, c_n) = self.lstm(x)
            
            # Attention over the single time step
            # u_it shape: (batch_size, 1, attention_dim)
            u_it = torch.tanh(self.attention_w(lstm_out))
            # alpha_it_unnormalized shape: (batch_size, 1, 1)
            alpha_it_unnormalized = self.attention_u(u_it)
            # attention_weights shape: (batch_size, 1, 1) (softmax over dim=1 is trivial here)
            attention_weights = F.softmax(alpha_it_unnormalized, dim=1)
            
            # context_vector shape: (batch_size, lstm_output_dim)
            context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        elif self.input_type == 'signal':
            # x original shape: (batch_size, num_sensors, time_steps) e.g., [512, 19, 3000]
            # For this mode, self.actual_lstm_input_dim should have been initialized to 1
            if self.actual_lstm_input_dim != 1:
                raise ValueError(f"For 'signal' input_type, LSTMAttention should be initialized with input_dim=1, "
                                 f"but got {self.actual_lstm_input_dim}")

            batch_size, num_sensors, time_steps = x.shape
            
            # Reshape to (batch_size * num_sensors, time_steps, 1)
            x_reshaped = x.contiguous().view(batch_size * num_sensors, time_steps, 1)
            
            # lstm_out shape: (batch_size * num_sensors, time_steps, lstm_output_dim)
            lstm_out, (h_n, c_n) = self.lstm(x_reshaped)
            
            # Attention mechanism applied to each sensor's time series output
            # u_it shape: (batch_size * num_sensors, time_steps, attention_dim)
            u_it = torch.tanh(self.attention_w(lstm_out))
            # alpha_it_unnormalized shape: (batch_size * num_sensors, time_steps, 1)
            alpha_it_unnormalized = self.attention_u(u_it)
            # attention_weights shape: (batch_size * num_sensors, time_steps, 1)
            attention_weights = F.softmax(alpha_it_unnormalized, dim=1) # Softmax over time_steps
            
            # context_vector_per_sensor shape: (batch_size * num_sensors, lstm_output_dim)
            context_vector_per_sensor = torch.sum(attention_weights * lstm_out, dim=1)
            
            # Reshape back to (batch_size, num_sensors, lstm_output_dim)
            lstm_output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
            context_vector_batched = context_vector_per_sensor.view(batch_size, num_sensors, lstm_output_dim)
            
            # Aggregate across sensors (e.g., mean pooling)
            # context_vector shape: (batch_size, lstm_output_dim)
            context_vector = torch.mean(context_vector_batched, dim=1)

        elif self.input_type == 'embedding':
            raise NotImplementedError("Input type 'embedding' is not implemented yet for LSTMAttention.")
        
        else:
            raise ValueError(f"Unknown input_type: {self.input_type} in LSTMAttention forward pass.")
            
        context_vector_dropped = self.dropout_fc(context_vector)
        logits = self.fc(context_vector_dropped)  # [batch_size, 1]
        
        return logits