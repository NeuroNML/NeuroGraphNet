import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout, Sequential, ReLU
from torch_geometric.nn import global_mean_pool  


class GNN_Encoder(nn.Module):
    """
    EEGNet model that combines a time encoder (CNN or LSTM) with a GNN for EEG data.
    This model first encodes the time series data for each node using the specified time encoder,
    and then applies a GNN to the resulting node embeddings.
    Args:
        encoder: A time series encoder (e.g., CNN or LSTM) that processes the time steps for each node.
        gnn: A GNN model that takes the node embeddings from the time encoder and performs message passing.
    """

    def __init__(self, encoder, gnn):
        super().__init__()
        self.encoder = encoder  
        self.gnn = gnn  

    def forward(self, x, edge_index, batch):
        # x: [num_nodes_batch, time_steps] 
        print(x)
        # Apply the encoder to each node independently
        node_embeddings = self.encoder(x)  # shape: [B, embedding_dim]
        print("Encoder output:", node_embeddings.shape, node_embeddings.mean().item(), node_embeddings.std().item())

        print("edge_index shape:", edge_index.shape)
        print("num edges:", edge_index.size(1))
        print("edge_index[:,:5]:", edge_index[:, :5])

        # Pass embeddings to the GNN
        return  self.gnn(node_embeddings, edge_index, batch)



'''
class GNN_Encoder(nn.Module):
    def __init__(self, encoder, gnn):
        super().__init__()
        self.encoder = encoder  # processes each node independently
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, batch):
        # x: [num_nodes_total, input_dim]
        # batch: [num_nodes_total] -> tells which node belongs to which graph

        node_embeddings = self.encoder(x)  # [num_nodes_total, embedding_dim]
        graph_embeddings = global_mean_pool(node_embeddings, batch)  

        out = self.mlp(graph_embeddings)  # [num_graphs, 1]
        return out
'''