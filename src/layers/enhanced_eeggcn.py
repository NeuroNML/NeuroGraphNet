"""
Enhanced EEG GCN layer that incorporates graph-level features alongside node-level features.

This demonstrates how to modify existing models to utilize the new graph features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear, BatchNorm1d, Dropout
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EnhancedEEGGCN(torch.nn.Module):
    """
    Enhanced version of EEGGCN that can utilize graph-level features in addition to node features.
    
    This model:
    1. Processes node features through GCN layers
    2. Aggregates node features into graph-level representations
    3. Combines with pre-computed graph features (clustering, centrality, etc.)
    4. Makes final predictions using the combined features
    """
    
    def __init__(
        self, 
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_classes: int = 1,
        num_conv_layers: int = 3,
        dropout: float = 0.5,
        graph_feature_dim: int = 0,
        use_graph_features: bool = True,
        graph_feature_fusion: str = "concat"  # "concat", "add", or "gated"
    ):
        super().__init__()
        
        self.num_conv_layers = num_conv_layers
        self.use_graph_features = use_graph_features and graph_feature_dim > 0
        self.graph_feature_fusion = graph_feature_fusion
        
        # GCN layers for node feature processing
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        # First layer: in_channels -> hidden_channels
        self.conv_layers.append(GCNConv(in_channels, hidden_channels))
        self.bn_layers.append(BatchNorm1d(hidden_channels))
        
        # Middle layers: hidden_channels -> hidden_channels
        for i in range(1, num_conv_layers - 1):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
            self.bn_layers.append(BatchNorm1d(hidden_channels))
        
        # Last layer: hidden_channels -> out_channels
        if num_conv_layers > 1:
            self.conv_layers.append(GCNConv(hidden_channels, out_channels))
            self.bn_layers.append(BatchNorm1d(out_channels))
        else:
            # If only one layer, go directly from in_channels to out_channels
            self.conv_layers[-1] = GCNConv(in_channels, out_channels)
            self.bn_layers[-1] = BatchNorm1d(out_channels)
        
        # Determine final feature dimension based on fusion strategy
        final_feature_dim = out_channels
        
        if self.use_graph_features:
            if self.graph_feature_fusion == "concat":
                final_feature_dim = out_channels + graph_feature_dim
            elif self.graph_feature_fusion == "add":
                # For addition, we need to project graph features to match node features
                self.graph_feature_projection = Linear(graph_feature_dim, out_channels)
                final_feature_dim = out_channels
            elif self.graph_feature_fusion == "gated":
                # Gated fusion learns how to combine features
                self.gate = nn.Sequential(
                    Linear(out_channels + graph_feature_dim, out_channels),
                    nn.Sigmoid()
                )
                self.graph_feature_projection = Linear(graph_feature_dim, out_channels)
                final_feature_dim = out_channels
            else:
                raise ValueError(f"Unknown fusion strategy: {self.graph_feature_fusion}")
        
        # Output layer
        self.linear = Linear(final_feature_dim, num_classes)
        self.dropout = Dropout(dropout)
        
        logger.info(f"EnhancedEEGGCN initialized:")
        logger.info(f"  - Node input channels: {in_channels}")
        logger.info(f"  - Hidden channels: {hidden_channels}")
        logger.info(f"  - Output channels: {out_channels}")
        logger.info(f"  - Graph feature dim: {graph_feature_dim}")
        logger.info(f"  - Use graph features: {self.use_graph_features}")
        logger.info(f"  - Fusion strategy: {self.graph_feature_fusion}")
        logger.info(f"  - Final feature dim: {final_feature_dim}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor,
        graph_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            graph_features: Graph-level features [batch_size, graph_feature_dim]
            
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        
        # Apply all GCN layers except the last one
        for i in range(self.num_conv_layers - 1):
            x = self.conv_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Apply the last GCN layer (without dropout)
        if self.num_conv_layers > 0:
            x = self.conv_layers[-1](x, edge_index)
            x = self.bn_layers[-1](x)
            x = F.relu(x)
        
        # Global pooling: combine all node features for each graph
        x = global_mean_pool(x, batch)  # [batch_size, out_channels]
        
        # Combine with graph-level features if available and enabled
        if self.use_graph_features and graph_features is not None:
            x = self._fuse_graph_features(x, graph_features)
        
        # Final classification
        x = self.linear(x)
        
        return x
    
    def _fuse_graph_features(
        self, 
        node_features: torch.Tensor, 
        graph_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse node-level and graph-level features based on the chosen strategy.
        
        Args:
            node_features: Aggregated node features [batch_size, out_channels]
            graph_features: Graph-level features [batch_size, graph_feature_dim]
            
        Returns:
            torch.Tensor: Fused features
        """
        
        if self.graph_feature_fusion == "concat":
            # Simple concatenation
            return torch.cat([node_features, graph_features], dim=1)
        
        elif self.graph_feature_fusion == "add":
            # Project graph features and add to node features
            projected_graph_features = self.graph_feature_projection(graph_features)
            return node_features + projected_graph_features
        
        elif self.graph_feature_fusion == "gated":
            # Gated fusion - learn how much of each feature to use
            projected_graph_features = self.graph_feature_projection(graph_features)
            
            # Compute gate weights
            combined = torch.cat([node_features, graph_features], dim=1)
            gate_weights = self.gate(combined)
            
            # Apply gating
            return gate_weights * node_features + (1 - gate_weights) * projected_graph_features
        
        else:
            raise ValueError(f"Unknown fusion strategy: {self.graph_feature_fusion}")


class GraphFeaturesMLP(nn.Module):
    """
    A simpler model that only uses graph-level features (no node-level processing).
    
    This is useful for comparison and to understand the contribution of graph features alone.
    """
    
    def __init__(
        self,
        graph_feature_dim: int,
        hidden_dims: list = [128, 64, 32],
        num_classes: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = graph_feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
        
        logger.info(f"GraphFeaturesMLP initialized with {graph_feature_dim} input features")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor,
        graph_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass using only graph features.
        
        Args:
            x: Node features (ignored)
            edge_index: Graph connectivity (ignored)
            batch: Batch assignment (ignored)
            graph_features: Graph-level features [batch_size, graph_feature_dim]
            
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        return self.model(graph_features)


class HybridGraphModel(nn.Module):
    """
    Hybrid model that can switch between using only node features, only graph features,
    or both, based on configuration.
    """
    
    def __init__(
        self,
        node_input_dim: int,
        graph_feature_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 1,
        use_node_features: bool = True,
        use_graph_features: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.use_node_features = use_node_features
        self.use_graph_features = use_graph_features
        
        if not (use_node_features or use_graph_features):
            raise ValueError("At least one of use_node_features or use_graph_features must be True")
        
        # Node feature processing
        if use_node_features:
            self.node_processor = nn.Sequential(
                GCNConv(node_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                GCNConv(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            node_output_dim = hidden_dim
        else:
            self.node_processor = None
            node_output_dim = 0
        
        # Graph feature processing
        if use_graph_features:
            self.graph_processor = nn.Sequential(
                Linear(graph_feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            graph_output_dim = hidden_dim
        else:
            self.graph_processor = None
            graph_output_dim = 0
        
        # Final classifier
        final_dim = node_output_dim + graph_output_dim
        self.classifier = nn.Sequential(
            Linear(final_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, num_classes)
        )
        
        logger.info(f"HybridGraphModel initialized:")
        logger.info(f"  - Use node features: {use_node_features}")
        logger.info(f"  - Use graph features: {use_graph_features}")
        logger.info(f"  - Final dimension: {final_dim}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor,
        graph_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass that can use node features, graph features, or both.
        """
        features = []
        
        # Process node features if enabled
        if self.use_node_features:
            node_x = x
            for layer in self.node_processor:
                if isinstance(layer, GCNConv):
                    node_x = layer(node_x, edge_index)
                else:
                    node_x = layer(node_x)
            
            # Global pooling
            node_x = global_mean_pool(node_x, batch)
            features.append(node_x)
        
        # Process graph features if enabled
        if self.use_graph_features and graph_features is not None:
            graph_x = self.graph_processor(graph_features)
            features.append(graph_x)
        
        # Combine all features
        if len(features) == 1:
            combined_features = features[0]
        else:
            combined_features = torch.cat(features, dim=1)
        
        # Final classification
        return self.classifier(combined_features)


# Example usage functions
def create_enhanced_model_with_graph_features(
    node_input_dim: int,
    graph_feature_dim: int,
    model_type: str = "enhanced_gcn"
):
    """
    Factory function to create different types of models that can use graph features.
    
    Args:
        node_input_dim: Dimension of node features
        graph_feature_dim: Dimension of graph-level features
        model_type: Type of model to create
        
    Returns:
        PyTorch model
    """
    
    if model_type == "enhanced_gcn":
        return EnhancedEEGGCN(
            in_channels=node_input_dim,
            hidden_channels=64,
            out_channels=64,
            graph_feature_dim=graph_feature_dim,
            use_graph_features=True,
            graph_feature_fusion="concat"
        )
    
    elif model_type == "gated_gcn":
        return EnhancedEEGGCN(
            in_channels=node_input_dim,
            hidden_channels=64,
            out_channels=64,
            graph_feature_dim=graph_feature_dim,
            use_graph_features=True,
            graph_feature_fusion="gated"
        )
    
    elif model_type == "graph_features_only":
        return GraphFeaturesMLP(
            graph_feature_dim=graph_feature_dim,
            hidden_dims=[128, 64, 32]
        )
    
    elif model_type == "hybrid":
        return HybridGraphModel(
            node_input_dim=node_input_dim,
            graph_feature_dim=graph_feature_dim,
            use_node_features=True,
            use_graph_features=True
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example of how to use the enhanced models
    
    # Simulate some input dimensions
    node_input_dim = 3000  # e.g., time series length
    graph_feature_dim = 16  # e.g., extracted graph features
    batch_size = 8
    num_nodes_per_graph = 19  # EEG channels
    
    # Create different model types
    models = {
        "Enhanced GCN (Concat)": create_enhanced_model_with_graph_features(
            node_input_dim, graph_feature_dim, "enhanced_gcn"
        ),
        "Enhanced GCN (Gated)": create_enhanced_model_with_graph_features(
            node_input_dim, graph_feature_dim, "gated_gcn"
        ),
        "Graph Features Only": create_enhanced_model_with_graph_features(
            node_input_dim, graph_feature_dim, "graph_features_only"
        ),
        "Hybrid Model": create_enhanced_model_with_graph_features(
            node_input_dim, graph_feature_dim, "hybrid"
        )
    }
    
    # Print model information
    for name, model in models.items():
        print(f"\n{name}:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\nAll models are ready to use with graph features!")
