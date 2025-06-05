import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self,  hidden_size=64,  dropout=0.3, input_size=95, bidirectional=True, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1) 
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim) = (N, 3000, 19) FOR TIME SIGNAL(processing_sessions)
        x = x.permute(0,2,1)# (B,C,T)->(B,T,C)
        lstm_out, _ = self.lstm(x)         # (N, 3000, 128)
        pooled = lstm_out.mean(dim=1)      # (N, 128)
        pooled = self.dropout(pooled)
        logits = self.mlp(pooled)          # (N, 1)
        return logits, 1
    
      


    '''
    def forward(self, x):
        # If x time series: [B,C,T]
        x = x.permute(0, 2, 1) # Need:[B,T,C]-> (B, 3000, 19) -> After LSTM: (B,3000, emb_dim)
      
        _, (hn, _) = self.bilstm(x)  # hn: [num_layers * num_directions, B, H]

        if self.bidirectional:
            hn = torch.cat((hn[-2], hn[-1]), dim=1)  # [B, 2H]
        else:
            hn = hn[-1]  # [B, H]

        return self.mlp(hn), 1 # [B, 1]
    '''