import torch
import torch.nn as nn


class EEGBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=1):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False  # BiLSTM
        )
        # Output will be [B, hidden_size]
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        #self.output_layer = nn.Linear(1, hidden_size)

        self.apply(self.init_weights)

    def init_weights(self, m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    '''
    
    def forward(self, x):
        x = x.unsqueeze(-1)  # [B, T, 1]
        return torch.relu(self.output_layer(torch.mean(x, dim=1)))  # [B, 1] → up to [B, H]
    '''

    
    def forward(self, x):
        """
        x: [batch_size, time_steps]
        """
        print(f"Input to EEGBiLSTM.forward - Shape: {x.shape}, Min={x.min().item():.4f}, Max={x.max().item():.4f}, Mean={x.mean().item():.4f}, Std={x.std().item():.4f}")
        x = x.unsqueeze(-1)  # [B, T] → [B, T, 1]
       
        '''
        output, _ = self.bilstm(x)  # output: [B, T, H]
        print(f"LSTM output: shape={output.shape}, min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}, std={output.std().item():.4f}")
        # Apply mean or max pooling over time
        pooled = output.mean(dim=1)  # or torch.max(output, dim=1).values
        return self.output_layer(pooled)  # [B, hidden_size]
    
        '''
        _, (hn, _) = self.bilstm(x) 
        hn = hn.squeeze(0)  
        print("hn:", hn.min(), hn.max(), hn.mean(), hn.std())
        return self.output_layer(hn)  # [B, hidden_size]
    
