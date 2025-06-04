import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, BatchNorm

class EEGGraphConvLSTMNet(nn.Module):
    def __init__(self, in_channels=3000, lstm_hidden=128,
                 gcn_dims=[320, 180, 90, 50], mlp_dims=[32, 16]):
        super(EEGGraphConvLSTMNet, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,           # EEG: (batch, time, features), here features=1
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        # Project BiLSTM output (2 * hidden) to GCN input dimension
        self.project = nn.Linear(2 * lstm_hidden, gcn_dims[0])

        # GCN stack
        self.gcn_layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        prev_dim = gcn_dims[0]
        for dim in gcn_dims:
            self.gcn_layers.append(GCNConv(prev_dim, dim))
            self.bns.append(BatchNorm(dim))
            prev_dim = dim

        # MLP classifier head
        mlp_layers = []
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(prev_dim, dim))
            mlp_layers.append(nn.LeakyReLU())
            prev_dim = dim
        mlp_layers.append(nn.Linear(prev_dim, 1))  # Output a single logit
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x, edge_index, batch):
        """
        x: shape (num_nodes * batch_size, 3000)
        edge_index: graph edges
        batch: graph batch assignment
        """
        B = x.size(0)
        x = x.unsqueeze(-1)  # → (B, 3000, 1)

        lstm_out, _ = self.lstm(x)  # → (B, 3000, 2 * lstm_hidden)
        x = lstm_out[:, -1, :]      # Take last timestep → (B, 2 * lstm_hidden)
        x = self.project(x)         # → (B, gcn_input_dim)

        for gcn, bn in zip(self.gcn_layers, self.bns):
            x = gcn(x, edge_index)
            x = F.leaky_relu(x)
            x = bn(x)

        x = global_add_pool(x, batch)  # → (batch_size, final_gcn_dim)
        x = self.mlp(x)                # → (batch_size, 1)
        return x  # Single logit, use sigmoid outside
