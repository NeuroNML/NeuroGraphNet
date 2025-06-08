"""
Demonstration script for using graph features in EEG graph datasets.

This script shows how to:
1. Initialize a dataset with graph feature extraction
2. Create a custom data loader that handles graph features
3. Use graph features in a model
"""

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from pathlib import Path
import pandas as pd
import logging

from src.data.dataset_graph import GraphEEGDataset
from src.utils.graph_features import batch_graph_features

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphFeatureAwareGCN(nn.Module):
    """
    GCN model that can utilize both node features and graph-level features.
    """
    
    def __init__(
        self, 
        node_input_dim: int,
        graph_feature_dim: int = 0,
        hidden_dim: int = 64,
        num_classes: int = 1,
        num_conv_layers: int = 2,
        dropout: float = 0.5,
        use_graph_features: bool = True
    ):
        super().__init__()
        
        self.use_graph_features = use_graph_features and graph_feature_dim > 0
        
        # GCN layers for node-level processing
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(node_input_dim, hidden_dim))
        
        for _ in range(num_conv_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Combine node-level and graph-level features
        classifier_input_dim = hidden_dim
        if self.use_graph_features:
            classifier_input_dim += graph_feature_dim
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"GraphFeatureAwareGCN initialized:")
        logger.info(f"  - Node input dim: {node_input_dim}")
        logger.info(f"  - Graph feature dim: {graph_feature_dim}")
        logger.info(f"  - Use graph features: {self.use_graph_features}")
        logger.info(f"  - Classifier input dim: {classifier_input_dim}")
    
    def forward(self, x, edge_index, batch, graph_features=None):
        # Node-level processing with GCN
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = torch.relu(x)
            if i < len(self.conv_layers) - 1:  # Don't apply dropout to last layer
                x = self.dropout(x)
        
        # Global pooling to get graph-level representations
        x = global_mean_pool(x, batch)  # Shape: [batch_size, hidden_dim]
        
        # Combine with graph-level features if available
        if self.use_graph_features and graph_features is not None:
            x = torch.cat([x, graph_features], dim=1)
        
        # Final classification
        out = self.classifier(x)
        return out


def create_example_dataset_with_graph_features():
    """
    Example function showing how to create a dataset with graph features enabled.
    """
    
    # Example parameters - adjust these for your actual data
    root = Path("data/graph_dataset_with_features")
    
    # Create dummy clips DataFrame for demonstration
    clips = pd.DataFrame({
        'id': range(100),
        'patient': [f'patient_{i//10}' for i in range(100)],
        'session': [f'session_{i//20}' for i in range(100)],
        'start_time': [i * 12.0 for i in range(100)],
        'end_time': [(i + 1) * 12.0 for i in range(100)],
        'label': [i % 2 for i in range(100)],  # Binary labels
        'signals_path': ['dummy_path.parquet' for _ in range(100)]
    })
    
    # Initialize dataset with graph features enabled
    dataset = GraphEEGDataset(
        root=root,
        clips=clips,
        signal_folder=Path("data/train/signals"),
        embeddings_dir=Path("data/embeddings"),
        use_embeddings=False,
        extracted_features_dir=Path("data/extracted_features"),
        use_selected_features=False,
        edge_strategy="correlation",
        correlation_threshold=0.5,
        top_k=5,
        force_reprocess=True,
        extract_graph_features=True,  # Enable graph features
        graph_feature_types=['degree', 'clustering', 'centrality']  # Specify which features to extract
    )
    
    return dataset


def train_with_graph_features(dataset, num_epochs: int = 10):
    """
    Example training loop that utilizes graph features.
    """
    
    # Get dimensions for model initialization
    sample = dataset[0]
    node_input_dim = sample.x.shape[1]
    
    # Count graph features
    graph_feature_names = [attr for attr in dir(sample) if attr.startswith('graph_')]
    graph_feature_dim = len(graph_feature_names)
    
    logger.info(f"Node input dimension: {node_input_dim}")
    logger.info(f"Graph feature dimension: {graph_feature_dim}")
    logger.info(f"Graph features: {graph_feature_names}")
    
    # Initialize model
    model = GraphFeatureAwareGCN(
        node_input_dim=node_input_dim,
        graph_feature_dim=graph_feature_dim,
        hidden_dim=64,
        use_graph_features=graph_feature_dim > 0
    )
    
    # Create data loader
    dataloader = GeoDataLoader(dataset, batch_size=16, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Get graph features if available
            graph_features = getattr(batch, 'graph_features', None)
            
            # Forward pass
            out = model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                graph_features=graph_features
            )
            
            # Compute loss
            loss = criterion(out.squeeze(), batch.y.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")


def analyze_graph_features(dataset):
    """
    Analyze the extracted graph features to understand their distributions.
    """
    logger.info("Analyzing graph features...")
    
    if len(dataset) == 0:
        logger.warning("Dataset is empty, cannot analyze features")
        return
    
    # Get first sample to identify features
    sample = dataset[0]
    graph_feature_names = [attr for attr in dir(sample) if attr.startswith('graph_')]
    
    if not graph_feature_names:
        logger.info("No graph features found in dataset")
        return
    
    # Collect features from all samples
    feature_stats = {name: [] for name in graph_feature_names}
    
    for i in range(min(len(dataset), 1000)):  # Analyze first 1000 samples
        data = dataset[i]
        for feature_name in graph_feature_names:
            if hasattr(data, feature_name):
                value = getattr(data, feature_name)
                if isinstance(value, torch.Tensor):
                    feature_stats[feature_name].append(value.item())
                else:
                    feature_stats[feature_name].append(value)
    
    # Print statistics
    for feature_name, values in feature_stats.items():
        if values:
            values_tensor = torch.tensor(values)
            logger.info(f"{feature_name}:")
            logger.info(f"  Mean: {values_tensor.mean():.4f}")
            logger.info(f"  Std: {values_tensor.std():.4f}")
            logger.info(f"  Min: {values_tensor.min():.4f}")
            logger.info(f"  Max: {values_tensor.max():.4f}")


if __name__ == "__main__":
    # Example usage
    logger.info("Creating dataset with graph features...")
    
    # This is a demonstration - you would use your actual data
    try:
        dataset = create_example_dataset_with_graph_features()
        logger.info(f"Dataset created with {len(dataset)} samples")
        
        if len(dataset) > 0:
            # Analyze graph features
            analyze_graph_features(dataset)
            
            # Demonstrate training with graph features
            logger.info("Starting training demonstration...")
            train_with_graph_features(dataset, num_epochs=3)
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        logger.info("This is expected if you don't have the actual data files")
        logger.info("The code shows how to use graph features when you have real data")
