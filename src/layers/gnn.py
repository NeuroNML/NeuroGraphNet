import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool



class EEGGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 64], output_dim=1, dropout=0.2):
        super().__init__()
        self.gcn1 = EEGGCNLayer(input_dim, hidden_dims[0], dropout)
        self.gcn2 = EEGGCNLayer(hidden_dims[0], hidden_dims[1], dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

    def forward(self, x, edge_index, batch):
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        x = global_mean_pool(x, batch)  # Pool per graph
        out = self.classifier(x)
        return out
'''

class SimpleGNN(nn.Module):
    def __init__(self, in_channels=12, hidden_channels=16, out_channels=1):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.relu = nn.ReLU()
            self.post_mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels),
            )

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.post_mlp(x)



    
    def __init__(self, in_channels=12, hidden_channels=16, out_channels=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv1 = nn.Linear(in_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.b1 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index, batch):
        print(f"edge_index shape: {edge_index.shape}")
        print(f"unique nodes in edge_index: {edge_index.unique().numel()} / {x.size(0)}")
        print(f"Input x stats: mean={x.mean():.4f}, std={x.std():.4f}, shape={x.shape}")
        x = F.relu(self.conv1(x))
        print("conv1 weights:", self.conv1.weight)

        print(f"After conv1: mean={x.mean():.4f}, std={x.std():.4f}, shape={x.shape}")
        x = global_mean_pool(x, batch)
        print(f"After pooling: mean={x.mean():.4f}, std={x.std():.4f}, shape={x.shape}")
        out = self.linear(x)
        print(f"Final logits: mean={out.mean():.4f}, std={out.std():.4f}, shape={out.shape}")
        return out
    
    def __init__(self, in_channels=12, hidden_channels=16, out_channels=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Ignore edge_index entirely
        x = self.mlp(x)                      # [num_nodes_batch, hidden]
        x = global_mean_pool(x, batch)       # [num_graphs, hidden]
        return self.linear(x)                # [num_graphs, 1]    
    '''
