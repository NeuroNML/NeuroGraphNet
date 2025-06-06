import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN_LSTM(nn.Module):

    import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN_LSTM(nn.Module):
    def __init__(self, bidirectional=True, dropout=0.4, cnn_out_channels=8, kernel_size=15, stride=5,
                 gcn_hidden=64, gcn_out=32, lstm_hidden=64):
        super().__init__()



        ''' 0.61 (TOO SLOW)
        self.n_channels = 19

        self.gcn1 = GCNConv(1, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_out)
        self.relu = nn.ReLU()

        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=gcn_out * self.n_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.num_directions * lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, batch):
        """
        x: [B * N, T] (3000 time steps per node)
        edge_index: [2, E]
        batch: [B * N] — node-to-graph assignment
        """
        B = batch.max().item() + 1
        T = x.size(1)
        gcn_embeddings = []

        for t in range(T):
            x_t = x[:, t].unsqueeze(1)              # [B*N, 1] — features at time t
            h = self.relu(self.gcn1(x_t, edge_index))
            h = self.relu(self.gcn2(h, edge_index))  # [B*N, gcn_out]

            # Reshape to [B, N * gcn_out] using batch
            h = h.view(B, self.n_channels * h.size(-1))  # [B, N * gcn_out]
            gcn_embeddings.append(h)

        # Stack over time
        x_lstm = torch.stack(gcn_embeddings, dim=1)  # [B, T, N * gcn_out]

        # Temporal modeling
        lstm_out, (hn, _) = self.lstm(x_lstm)
        hn = hn.transpose(0, 1).reshape(B, -1)       # [B, H * num_directions]

        return self.mlp(hn)                          # [B, 1]


    '''



    ''' F1: 0.58
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
    
        '''