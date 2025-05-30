import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, # Corrected to input_size for consistency with nn.LSTM docs
            hidden_size=hidden_dim, # Corrected to hidden_size
            num_layers=num_layers,
            batch_first=True,
            # Dropout in nn.LSTM is applied between layers if num_layers > 1.
            # For num_layers=1, this dropout argument has no effect.
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout_fc = nn.Dropout(dropout) # Dropout before the final fully connected layer
        self.fc = nn.Linear(hidden_dim, 1)  # Output for binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, input_dim]
        """
        # out shape: [batch_size, seq_len, hidden_dim]
        # h_n shape: [num_layers * num_directions, batch_size, hidden_dim]
        # c_n shape: [num_layers * num_directions, batch_size, hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the output of the last time step for classification
        last_timestep_output = lstm_out[:, -1, :]  # Shape: [batch_size, hidden_dim]
        
        # Apply dropout before the final classification layer
        last_timestep_output_dropped = self.dropout_fc(last_timestep_output)
        
        logits = self.fc(last_timestep_output_dropped)  # Shape: [batch_size, 1]
        return logits

class BiLSTM(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        # Input to fc is hidden_dim * 2 due to concatenation of forward and backward passes
        self.dropout_fc = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, input_dim]
        """
        # lstm_out shape: [batch_size, seq_len, hidden_dim * 2]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the output of the last time step (concatenation of last fwd and last bwd hidden states)
        last_timestep_output = lstm_out[:, -1, :]  # Shape: [batch_size, hidden_dim * 2]
        
        last_timestep_output_dropped = self.dropout_fc(last_timestep_output)
        
        logits = self.fc(last_timestep_output_dropped)  # Shape: [batch_size, 1]
        return logits

class LSTMAttention(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, num_layers=1, dropout=0.2, attention_dim:int = None, bidirectional:bool = True): # type hint
        super().__init__()
        if attention_dim is None:
            attention_dim = hidden_dim 

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism layers
        self.attention_w = nn.Linear(hidden_dim, attention_dim) 
        self.attention_u = nn.Linear(attention_dim, 1, bias=False) 
        
        self.dropout_fc = nn.Dropout(dropout)
        # FC layer operates on the context vector which has 'hidden_dim' features
        self.fc = nn.Linear(hidden_dim, 1) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, input_dim]
        """
        lstm_out, (h_n, c_n) = self.lstm(x) # lstm_out shape: [batch_size, seq_len, hidden_dim]
        
        # Attention mechanism
        u_it = torch.tanh(self.attention_w(lstm_out)) # [batch_size, seq_len, attention_dim]
        alpha_it_unnormalized = self.attention_u(u_it)    # [batch_size, seq_len, 1]
        attention_weights = F.softmax(alpha_it_unnormalized, dim=1) # [batch_size, seq_len, 1]
        
        # Compute context vector (weighted sum of lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1) # [batch_size, hidden_dim]
        
        context_vector_dropped = self.dropout_fc(context_vector)
        logits = self.fc(context_vector_dropped)  # [batch_size, 1]
        # For debugging or analysis, you might want to return attention_weights as well:
        # return logits, attention_weights 
        return logits