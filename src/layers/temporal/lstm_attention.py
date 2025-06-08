import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class EEGLSTMAttention(nn.Module):
    """
    LSTM model with attention for sequence processing.
    Assumes input x is of shape (batch_size, seq_len, input_dim)
    or (batch_size, input_dim) which is treated as seq_len=1.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.2,
                 attention_dim: Optional[int] = None, bidirectional: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # The input_dim here is the feature size for each step fed into the LSTM
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
        # x original shape: (batch_size, features) or (batch_size, seq_len, features)
        # self.actual_lstm_input_dim should be == features
        
        if x.ndim == 2:
            # Reshape (batch_size, features) to (batch_size, 1, features) for LSTM
            x = x.unsqueeze(1)
        # x shape is now (batch_size, seq_len, self.actual_lstm_input_dim)
        
        # lstm_out shape: (batch_size, seq_len, lstm_output_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention mechanism
        # u_it shape: (batch_size, seq_len, attention_dim)
        u_it = torch.tanh(self.attention_w(lstm_out))
        # alpha_it_unnormalized shape: (batch_size, seq_len, 1)
        alpha_it_unnormalized = self.attention_u(u_it)
        # attention_weights shape: (batch_size, seq_len, 1) (softmax over seq_len dimension)
        attention_weights = F.softmax(alpha_it_unnormalized, dim=1)
        
        # context_vector shape: (batch_size, lstm_output_dim)
        # Weighted sum of lstm_out based on attention_weights
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
            
        context_vector_dropped = self.dropout_fc(context_vector)
        logits = self.fc(context_vector_dropped)  # [batch_size, 1]
        
        return logits