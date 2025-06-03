import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_classes=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_val = dropout

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, # Ensure this is True if your input is (batch, seq, feature)
            dropout=self.dropout_val if num_layers > 1 else 0
        )

        self.dropout_fc = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: (batch_size, seq_len, input_dim) when batch_first=True
        
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        # hn shape: (num_layers * num_directions, batch_size, hidden_dim)
        # cn shape: (num_layers * num_directions, batch_size, hidden_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # To get the features from the last time step of the last layer:
        # hn[-1] gives the hidden state of the last layer for all time steps.
        # For a unidirectional LSTM, num_directions is 1.
        # We want the hidden state from the last layer, which is hn[-1]
        # This will have shape: [batch_size, hidden_dim]
        last_hidden_state = hn[-1] 
        
        # Apply dropout before the final classification layer
        last_hidden_state_dropped = self.dropout_fc(last_hidden_state)
        
        output = self.fc(last_hidden_state_dropped)
        return output

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