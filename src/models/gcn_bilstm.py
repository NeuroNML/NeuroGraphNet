import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN_LSTM(nn.Module):
    def __init__(self, dropout=0.4, input_dim=3000, bidirectional=True, hidden_gcn=256, gcn_out_dim=16, lstm_hidden=16):
        super().__init__()

        # GCN stack
        self.conv1 = GCNConv(input_dim, hidden_gcn)
        self.conv2 = GCNConv(hidden_gcn, hidden_gcn)
        self.conv3 = GCNConv(hidden_gcn, hidden_gcn)
        self.conv4 = GCNConv(hidden_gcn, hidden_gcn)
        self.conv5 = GCNConv(hidden_gcn, gcn_out_dim)
        self.relu = nn.ReLU()

        # LSTM
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=gcn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional= bidirectional
        )

        # Final classifier
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden * self.num_directions, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1) 
        )

    def forward(self, x, edge_index, batch):
        """
        x: Node features [num_nodes, input_dim]
        edge_index: [2, num_edges]
        batch: [num_nodes] mapping each node to a graph in the batch
        """
        x = self.relu(self.conv1(x, edge_index))  # [N, 256]
        x = self.relu(self.conv2(x, edge_index))  # [N, 256]
        x = self.relu(self.conv3(x, edge_index))  # [N, 256]
        x = self.relu(self.conv4(x, edge_index))  # [N, 256]
        x = self.relu(self.conv5(x, edge_index))  # [N, 16]

        # Convert to batch-wise structure for LSTM
        #  
        batch_size = batch.max().item() + 1
        num_nodes_per_graph = x.size(0) // batch_size # [B*N,F] -> N (all graphs contain sma enum of nodes)
        x = x.view(batch_size, num_nodes_per_graph, -1)  # [B, T, F]

        lstm_out, (hn, _) = self.lstm(x)  # hn: [num_layers, B, H]; if bidirectional: 2 dirs
        #pooled = lstm_out.mean(dim=1) # Or could take last layers (2 strategirs to try)
        hn = hn.transpose(0, 1)         # → (B, 2, H)
        hn = hn.reshape(hn.size(0), -1) # → (B, 2*H)
        final_lstm = hn
        out = self.mlp(final_lstm)  # [B, 1]
        return out