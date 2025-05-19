import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model
class LSTM(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)  # Output for binary classification
        self.dropout_fc = nn.Dropout(dropout) # Adding dropout before the final FC layer

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, input_dim]
        """
        out, (h_n, c_n) = self.lstm(x)  # out shape: [batch_size, seq_len, hidden_dim]
        last_timestep_output = out[:, -1, :]  # [batch_size, hidden_dim]
        last_timestep_output_dropped = self.dropout_fc(last_timestep_output) # Apply dropout before the final classification layer
        logits = self.fc(last_timestep_output_dropped)  # [batch_size, 1]
        return logits

class BiLSTM(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        # For BiLSTM, hidden_dim is the size for one direction.
        # The LSTM output for each direction will be concatenated.
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0, # Dropout in LSTM is applied between layers
            bidirectional=True
        )
        # The input to the fully connected layer will be hidden_dim * 2
        # because we concatenate the outputs of the forward and backward LSTMs.
        self.fc = nn.Linear(hidden_dim * 2, 1) # Output for binary classification
        self.dropout_fc = nn.Dropout(dropout) # Adding dropout before the final FC layer

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, input_dim]
        """
        # out shape: [batch_size, seq_len, hidden_dim * 2] (due to bidirectional)
        # h_n shape: [num_layers * 2, batch_size, hidden_dim] (2 for num_directions)
        # c_n shape: [num_layers * 2, batch_size, hidden_dim]
        out, (h_n, c_n) = self.lstm(x)
        
        # We use the output of the last time step for classification.
        # `out[:, -1, :]` conveniently gives the concatenation of the last forward
        # hidden state and the last backward hidden state (which is the first actual time step's backward output).
        last_timestep_output = out[:, -1, :]  # Shape: [batch_size, hidden_dim * 2]
        
        # Apply dropout
        last_timestep_output_dropped = self.dropout_fc(last_timestep_output)
        
        logits = self.fc(last_timestep_output_dropped)  # Shape: [batch_size, 1]
        return logits

class LSTMAttention(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, num_layers=1, dropout=0.2, attention_dim=None):
        super().__init__()
        if attention_dim is None:
            attention_dim = hidden_dim # A common choice for simplicity

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False # Using unidirectional LSTM for this attention example, can be extended to BiLSTM
        )
        
        # Attention mechanism layers
        # Takes LSTM outputs and computes attention scores
        self.attention_w = nn.Linear(hidden_dim, attention_dim)
        # Computes the unnormalized attention scores (before softmax)
        # This context vector u_w is learned
        self.attention_u = nn.Linear(attention_dim, 1, bias=False) 
        
        self.fc = nn.Linear(hidden_dim, 1) # Output for binary classification from context vector
        self.dropout_fc = nn.Dropout(dropout)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, input_dim]
        """
        # lstm_out shape: [batch_size, seq_len, hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention mechanism
        # Pass LSTM outputs through a linear layer and a non-linearity (tanh)
        # u_it shape: [batch_size, seq_len, attention_dim]
        u_it = torch.tanh(self.attention_w(lstm_out))
        
        # Compute scores for each time step
        # alpha_it_unnormalized shape: [batch_size, seq_len, 1]
        alpha_it_unnormalized = self.attention_u(u_it)
        
        # Apply softmax to get attention weights
        # alpha_it shape: [batch_size, seq_len, 1]
        attention_weights = F.softmax(alpha_it_unnormalized, dim=1)
        
        # Compute the context vector by taking a weighted sum of LSTM outputs
        # lstm_out shape: [batch_size, seq_len, hidden_dim]
        # attention_weights shape: [batch_size, seq_len, 1]
        # We want to multiply them element-wise after broadcasting attention_weights
        # and then sum over the time dimension (dim=1).
        # context_vector shape: [batch_size, hidden_dim]
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Apply dropout to the context vector
        context_vector_dropped = self.dropout_fc(context_vector)
        
        logits = self.fc(context_vector_dropped)  # Shape: [batch_size, 1]
        return logits #, attention_weights # Optionally return attention_weights for visualization/interpretation