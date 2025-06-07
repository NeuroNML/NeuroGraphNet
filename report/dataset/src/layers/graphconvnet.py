import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, BatchNorm

class EEGGraphConvNet(nn.Module):
    def __init__(self, in_channels=3000, reduced_channels=1280,
                 gcn_hidden_dims=[640, 512, 256, 256], mlp_dims=[128, 64]):
        super(EEGGraphConvNet, self).__init__()

        # 1D CNN to reduce time-series from 3000 → 1280
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=15, stride=2, padding=7),  # 3000 → ~1500
            nn.LeakyReLU(),
            nn.Conv1d(8, 16, kernel_size=15, stride=2, padding=7), # ~1500 → ~750
            nn.LeakyReLU(),
            nn.Conv1d(16, 1, kernel_size=3, stride=1, padding=1),  # → shape (1, 750)
        )

        self.project_to = nn.Linear(750, reduced_channels)  # → 1280

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        prev_dim = reduced_channels
        for dim in gcn_hidden_dims:
            self.gcn_layers.append(GCNConv(prev_dim, dim))
            self.bns.append(BatchNorm(dim))
            prev_dim = dim

        # MLP head (single output)
        mlp_layers = []
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(prev_dim, dim))
            mlp_layers.append(nn.LeakyReLU())
            prev_dim = dim
        mlp_layers.append(nn.Linear(prev_dim, 1))  # Single logit
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x, edge_index, batch):
        

        # x shape: (num_nodes * batch_size, 3000)
        x = x.unsqueeze(1)              # (N, 1, 3000)
        x = self.cnn(x).squeeze(1)      # (N, 750)
        x = self.project_to(x)          # (N, 1280)

        # GCN stack
        for gcn, bn in zip(self.gcn_layers, self.bns):
            x = gcn(x, edge_index)
            x = F.leaky_relu(x)
            x = bn(x)

        # Pool all node features per graph
        x = global_add_pool(x, batch)   # (batch_size, 256)

        # Final MLP for binary classification
        x = self.mlp(x)                 # (batch_size, 1)
        return x        