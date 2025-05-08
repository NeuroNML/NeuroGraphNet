import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm1d, global_mean_pool
from torch_geometric.nn import Linear
from layers.eeggcn import EEGGCN

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, num_layers=1, dropout=0.2, bidirectional=False):
        """
        LSTM-based encoder for EEG temporal signals
        
        Args:
            input_dim: Number of input features (typically number of channels in EEG)
            hidden_dim: Size of hidden state
            num_layers: Number of recurrent layers
            dropout: Dropout probability (only applied between layers if num_layers > 1)
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: Last hidden state [batch_size, hidden_dim * (2 if bidirectional else 1)]
        """
        # Process through LSTM
        out, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            forward = h_n[2*self.num_layers-2]
            backward = h_n[2*self.num_layers-1]
            last_hidden = torch.cat((forward, backward), dim=1)
        else:
            # Get the hidden state from the last layer
            last_hidden = h_n[self.num_layers-1]
            
        return last_hidden


class HybridLSTMGCN(torch.nn.Module):
    def __init__(self, seq_length, num_channels, lstm_hidden_dim, gcn_hidden_dim, gcn_out_dim, 
                 num_classes, num_lstm_layers=1, num_gcn_layers=3, 
                 lstm_dropout=0.2, gcn_dropout=0.5, bidirectional=False):
        """
        Hybrid LSTM-GCN model for EEG classification
        
        Args:
            seq_length: Length of the EEG sequence for each channel
            num_channels: Number of EEG channels/electrodes
            lstm_hidden_dim: Size of LSTM hidden state
            gcn_hidden_dim: Size of GCN hidden layers
            gcn_out_dim: Output feature dimensions before classification
            num_classes: Number of classes for classification
            num_lstm_layers: Number of LSTM layers
            num_gcn_layers: Number of GCN convolutional layers
            lstm_dropout: Dropout probability for LSTM
            gcn_dropout: Dropout probability for GCN
            bidirectional: Whether to use bidirectional LSTM
        """
        super(HybridLSTMGCN, self).__init__()
        
        # LSTM for temporal encoding
        self.lstm_encoder = LSTMEncoder(
            input_dim=seq_length,  # Each node's sequence becomes the input
            hidden_dim=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim
        
        # GCN for spatial processing (using your existing EEGGCN class)
        self.gcn = EEGGCN(
            in_channels=lstm_output_dim,  # LSTM output becomes GCN input
            hidden_channels=gcn_hidden_dim,
            out_channels=gcn_out_dim,
            num_classes=num_classes,
            num_conv_layers=num_gcn_layers,
            dropout=gcn_dropout
        )

    def forward(self, node_sequences, edge_index, batch):
        """
        Forward pass
        
        Args:
            node_sequences: Node features as sequences [num_nodes, seq_length, num_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            torch.Tensor: Class logits
        """
        num_nodes = node_sequences.size(0)
        
        # Process each node's sequence through LSTM to get temporal embeddings
        node_embeddings = []
        for i in range(num_nodes):
            # Extract sequence for current node
            node_seq = node_sequences[i]  # [seq_length, num_channels]
            # Process through LSTM encoder
            node_emb = self.lstm_encoder(node_seq.unsqueeze(0))  # Add batch dim for LSTM
            node_embeddings.append(node_emb.squeeze(0))  # Remove batch dim
        
        # Stack embeddings to create node feature matrix
        x = torch.stack(node_embeddings)  # [num_nodes, lstm_output_dim]
        
        # Pass node embeddings through GCN
        logits = self.gcn(x, edge_index, batch)
        
        return logits


class BatchedHybridLSTMGCN(torch.nn.Module):
    """
    More efficient version of HybridLSTMGCN that processes all node sequences in parallel
    """
    def __init__(self, seq_length, num_channels, lstm_hidden_dim, gcn_hidden_dim, gcn_out_dim, 
                 num_classes, num_lstm_layers=1, num_gcn_layers=3, 
                 lstm_dropout=0.2, gcn_dropout=0.5, bidirectional=False):
        super(BatchedHybridLSTMGCN, self).__init__()
        
        # LSTM for temporal encoding
        self.lstm_encoder = LSTMEncoder(
            input_dim=seq_length,
            hidden_dim=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim
        
        # GCN for spatial processing
        self.gcn = EEGGCN(
            in_channels=lstm_output_dim,
            hidden_channels=gcn_hidden_dim,
            out_channels=gcn_out_dim,
            num_classes=num_classes,
            num_conv_layers=num_gcn_layers,
            dropout=gcn_dropout
        )
    
    def forward(self, node_sequences, edge_index, batch):
        """
        Forward pass with batch processing
        
        Args:
            node_sequences: Node features as sequences [num_nodes, seq_length, num_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            torch.Tensor: Class logits
        """
        # Reshape for batch processing: [num_nodes, seq_length, num_channels] -> [num_nodes, num_channels, seq_length]
        reshaped_seqs = node_sequences.transpose(1, 2)
        
        # Process all node sequences in parallel through LSTM
        batch_size, num_channels, seq_length = reshaped_seqs.size()
        
        # Flatten batch and channels to process all sequences at once
        flattened = reshaped_seqs.reshape(-1, seq_length).unsqueeze(1)  # [batch_size*num_channels, 1, seq_length]
        
        # Process through LSTM
        all_embeddings = self.lstm_encoder(flattened)  # [batch_size*num_channels, lstm_output_dim]
        
        # Reshape back to [batch_size, num_channels*lstm_output_dim]
        lstm_output_dim = all_embeddings.size(1)
        node_embeddings = all_embeddings.reshape(batch_size, num_channels * lstm_output_dim)
        
        # Pass through GCN
        logits = self.gcn(node_embeddings, edge_index, batch)
        
        return logits