import torch
import torch.nn as nn

class EEGGNNEncoder(nn.Module):
    '''
    EEGNet model that combines a time encoder (CNN or LSTM) with a GNN for EEG data.
    This model first encodes the time series data for each node using the specified time encoder,
    and then applies a GNN to the resulting node embeddings.
    Args:
        encoder: A time series encoder (e.g., CNN or LSTM) that processes the time steps for each node.
        gnn: A GNN model that takes the node embeddings from the time encoder and performs message passing.
    '''

    def __init__(self, encoder, gnn):
        super().__init__()
        self.encoder = encoder  
        self.gnn = gnn  

    def forward(self, x, edge_index, batch):
        # x: [num_nodes_batch, time_steps] 
        # Apply the encoder to each node independently
        node_embeddings = self.encoder(x)  # shape: [num_nodes, embedding_dim]

        # Pass embeddings to the GNN
        return self.gnn(node_embeddings, edge_index, batch)