"""
Graph Feature Extraction Module for EEG Graphs

This module provides functionality to compute various graph-level and node-level 
features that can be used as additional inputs during training.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx
from torch_geometric.utils import to_networkx, degree
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class GraphFeatureExtractor:
    """
    Extracts various graph features from PyTorch Geometric Data objects.
    Features include both node-level and graph-level metrics.
    """
    
    def __init__(
        self,
        include_node_features: bool = True,
        include_graph_features: bool = True,
        feature_types: Optional[List[str]] = None
    ):
        """
        Initialize the graph feature extractor.
        
        Args:
            include_node_features: Whether to compute node-level features
            include_graph_features: Whether to compute graph-level features
            feature_types: List of specific feature types to compute. If None, computes all.
                          Available types: ['degree', 'clustering', 'centrality', 
                          'connectivity', 'path_length', 'efficiency']
        """
        self.include_node_features = include_node_features
        self.include_graph_features = include_graph_features
        
        # Default to all feature types if none specified
        if feature_types is None:
            self.feature_types = [
                'degree', 'clustering', 'centrality', 
                'connectivity', 'path_length', 'efficiency'
            ]
        else:
            self.feature_types = feature_types
            
        logger.info(f"GraphFeatureExtractor initialized with features: {self.feature_types}")
    
    def extract_features(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Extract graph features from a PyTorch Geometric Data object.
        
        Args:
            data: PyTorch Geometric Data object containing graph structure
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        # Convert to NetworkX for easier computation
        G = to_networkx(data, to_undirected=True)
        num_nodes = G.number_of_nodes()
        
        if num_nodes == 0:
            logger.warning("Empty graph encountered, returning zero features")
            return self._get_zero_features(num_nodes)
        
        try:
            # Node-level features
            if self.include_node_features:
                node_features = self._compute_node_features(G, data)
                features.update(node_features)
            
            # Graph-level features
            if self.include_graph_features:
                graph_features = self._compute_graph_features(G)
                features.update(graph_features)
                
        except Exception as e:
            logger.warning(f"Error computing graph features: {e}, returning zero features")
            return self._get_zero_features(num_nodes)
        
        return features
    
    def _compute_node_features(self, G: nx.Graph, data: Data) -> Dict[str, torch.Tensor]:
        """Compute node-level graph features."""
        features = {}
        num_nodes = G.number_of_nodes()
        
        if 'degree' in self.feature_types:
            # Node degrees
            degrees = torch.tensor([G.degree(n) for n in range(num_nodes)], dtype=torch.float32)
            features['node_degree'] = degrees.unsqueeze(1)  # [num_nodes, 1]
            
            # Normalized degrees
            max_degree = degrees.max().item() if degrees.numel() > 0 else 1.0
            if max_degree > 0:
                features['node_degree_normalized'] = (degrees / max_degree).unsqueeze(1)
            else:
                features['node_degree_normalized'] = degrees.unsqueeze(1)
        
        if 'clustering' in self.feature_types:
            # Clustering coefficients
            clustering = nx.clustering(G)
            clustering_values = torch.tensor([clustering[n] for n in range(num_nodes)], dtype=torch.float32)
            features['node_clustering'] = clustering_values.unsqueeze(1)  # [num_nodes, 1]
        
        if 'centrality' in self.feature_types:
            # Betweenness centrality
            try:
                betweenness = nx.betweenness_centrality(G)
                betweenness_values = torch.tensor([betweenness[n] for n in range(num_nodes)], dtype=torch.float32)
                features['node_betweenness'] = betweenness_values.unsqueeze(1)
            except:
                features['node_betweenness'] = torch.zeros(num_nodes, 1, dtype=torch.float32)
            
            # Closeness centrality
            try:
                closeness = nx.closeness_centrality(G)
                closeness_values = torch.tensor([closeness[n] for n in range(num_nodes)], dtype=torch.float32)
                features['node_closeness'] = closeness_values.unsqueeze(1)
            except:
                features['node_closeness'] = torch.zeros(num_nodes, 1, dtype=torch.float32)
            
            # Eigenvector centrality (if graph is connected)
            try:
                if nx.is_connected(G):
                    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
                    eigenvector_values = torch.tensor([eigenvector[n] for n in range(num_nodes)], dtype=torch.float32)
                    features['node_eigenvector'] = eigenvector_values.unsqueeze(1)
                else:
                    features['node_eigenvector'] = torch.zeros(num_nodes, 1, dtype=torch.float32)
            except:
                features['node_eigenvector'] = torch.zeros(num_nodes, 1, dtype=torch.float32)
        
        return features
    
    def _compute_graph_features(self, G: nx.Graph) -> Dict[str, torch.Tensor]:
        """Compute graph-level features."""
        features = {}
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        if 'connectivity' in self.feature_types:
            # Basic connectivity measures
            features['graph_num_nodes'] = torch.tensor([num_nodes], dtype=torch.float32)
            features['graph_num_edges'] = torch.tensor([num_edges], dtype=torch.float32)
            features['graph_density'] = torch.tensor([nx.density(G)], dtype=torch.float32)
            features['graph_is_connected'] = torch.tensor([float(nx.is_connected(G))], dtype=torch.float32)
            
            # Number of connected components
            features['graph_num_components'] = torch.tensor([nx.number_connected_components(G)], dtype=torch.float32)
        
        if 'clustering' in self.feature_types:
            # Global clustering coefficient
            features['graph_clustering'] = torch.tensor([nx.average_clustering(G)], dtype=torch.float32)
            features['graph_transitivity'] = torch.tensor([nx.transitivity(G)], dtype=torch.float32)
        
        if 'path_length' in self.feature_types:
            # Path length measures
            if nx.is_connected(G):
                try:
                    avg_path_length = nx.average_shortest_path_length(G)
                    diameter = nx.diameter(G)
                    radius = nx.radius(G)
                except:
                    avg_path_length = 0.0
                    diameter = 0.0
                    radius = 0.0
            else:
                avg_path_length = 0.0
                diameter = 0.0
                radius = 0.0
            
            features['graph_avg_path_length'] = torch.tensor([avg_path_length], dtype=torch.float32)
            features['graph_diameter'] = torch.tensor([diameter], dtype=torch.float32)
            features['graph_radius'] = torch.tensor([radius], dtype=torch.float32)
        
        if 'efficiency' in self.feature_types:
            # Efficiency measures
            try:
                global_efficiency = nx.global_efficiency(G)
                local_efficiency = nx.local_efficiency(G)
            except:
                global_efficiency = 0.0
                local_efficiency = 0.0
            
            features['graph_global_efficiency'] = torch.tensor([global_efficiency], dtype=torch.float32)
            features['graph_local_efficiency'] = torch.tensor([local_efficiency], dtype=torch.float32)
        
        if 'degree' in self.feature_types:
            # Degree statistics
            degrees = [G.degree(n) for n in G.nodes()]
            if degrees:
                features['graph_avg_degree'] = torch.tensor([np.mean(degrees)], dtype=torch.float32)
                features['graph_degree_std'] = torch.tensor([np.std(degrees)], dtype=torch.float32)
                features['graph_max_degree'] = torch.tensor([np.max(degrees)], dtype=torch.float32)
                features['graph_min_degree'] = torch.tensor([np.min(degrees)], dtype=torch.float32)
            else:
                features['graph_avg_degree'] = torch.tensor([0.0], dtype=torch.float32)
                features['graph_degree_std'] = torch.tensor([0.0], dtype=torch.float32)
                features['graph_max_degree'] = torch.tensor([0.0], dtype=torch.float32)
                features['graph_min_degree'] = torch.tensor([0.0], dtype=torch.float32)
        
        return features
    
    def _get_zero_features(self, num_nodes: int) -> Dict[str, torch.Tensor]:
        """Return zero features when computation fails."""
        features = {}
        
        if self.include_node_features:
            if 'degree' in self.feature_types:
                features['node_degree'] = torch.zeros(num_nodes, 1, dtype=torch.float32)
                features['node_degree_normalized'] = torch.zeros(num_nodes, 1, dtype=torch.float32)
            if 'clustering' in self.feature_types:
                features['node_clustering'] = torch.zeros(num_nodes, 1, dtype=torch.float32)
            if 'centrality' in self.feature_types:
                features['node_betweenness'] = torch.zeros(num_nodes, 1, dtype=torch.float32)
                features['node_closeness'] = torch.zeros(num_nodes, 1, dtype=torch.float32)
                features['node_eigenvector'] = torch.zeros(num_nodes, 1, dtype=torch.float32)
        
        if self.include_graph_features:
            feature_names = [
                'graph_num_nodes', 'graph_num_edges', 'graph_density', 'graph_is_connected',
                'graph_num_components', 'graph_clustering', 'graph_transitivity',
                'graph_avg_path_length', 'graph_diameter', 'graph_radius',
                'graph_global_efficiency', 'graph_local_efficiency',
                'graph_avg_degree', 'graph_degree_std', 'graph_max_degree', 'graph_min_degree'
            ]
            for name in feature_names:
                features[name] = torch.tensor([0.0], dtype=torch.float32)
        
        return features
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get the dimensions of different feature types.
        
        Returns:
            Dictionary mapping feature names to their dimensions
        """
        dims = {}
        
        if self.include_node_features:
            if 'degree' in self.feature_types:
                dims['node_degree'] = 1
                dims['node_degree_normalized'] = 1
            if 'clustering' in self.feature_types:
                dims['node_clustering'] = 1
            if 'centrality' in self.feature_types:
                dims['node_betweenness'] = 1
                dims['node_closeness'] = 1
                dims['node_eigenvector'] = 1
        
        if self.include_graph_features:
            # All graph-level features are scalar (dimension 1)
            graph_feature_names = [
                'graph_num_nodes', 'graph_num_edges', 'graph_density', 'graph_is_connected',
                'graph_num_components', 'graph_clustering', 'graph_transitivity',
                'graph_avg_path_length', 'graph_diameter', 'graph_radius',
                'graph_global_efficiency', 'graph_local_efficiency',
                'graph_avg_degree', 'graph_degree_std', 'graph_max_degree', 'graph_min_degree'
            ]
            for name in graph_feature_names:
                dims[name] = 1
        
        return dims


def augment_data_with_graph_features(
    data: Data, 
    feature_extractor: GraphFeatureExtractor,
    concat_to_node_features: bool = True
) -> Data:
    """
    Augment a PyTorch Geometric Data object with graph features.
    
    Args:
        data: Original Data object
        feature_extractor: Configured GraphFeatureExtractor instance
        concat_to_node_features: Whether to concatenate node-level graph features to x
        
    Returns:
        Augmented Data object with additional features
    """
    # Extract graph features
    graph_features = feature_extractor.extract_features(data)
    
    # Create new data object
    new_data = data.clone()
    
    # Add node-level features to x if requested
    if concat_to_node_features and feature_extractor.include_node_features:
        node_feature_tensors = []
        original_x = data.x if data.x is not None else torch.zeros(data.edge_index.max().item() + 1, 0)
        node_feature_tensors.append(original_x)
        
        # Collect node-level graph features
        node_graph_features = [v for k, v in graph_features.items() if k.startswith('node_')]
        if node_graph_features:
            node_features_concat = torch.cat(node_graph_features, dim=1)
            node_feature_tensors.append(node_features_concat)
        
        if len(node_feature_tensors) > 1:
            new_data.x = torch.cat(node_feature_tensors, dim=1)
    
    # Add graph-level features as separate attributes
    if feature_extractor.include_graph_features:
        for feature_name, feature_value in graph_features.items():
            if feature_name.startswith('graph_'):
                setattr(new_data, feature_name, feature_value)
    
    return new_data


def batch_graph_features(batch_data: List[Data], feature_names: List[str]) -> torch.Tensor:
    """
    Batch graph-level features from a list of Data objects.
    
    Args:
        batch_data: List of Data objects with graph features
        feature_names: Names of graph-level features to batch
        
    Returns:
        Batched tensor of shape [batch_size, num_features]
    """
    batch_features = []
    
    for data in batch_data:
        data_features = []
        for feature_name in feature_names:
            if hasattr(data, feature_name):
                feature_value = getattr(data, feature_name)
                if isinstance(feature_value, torch.Tensor):
                    data_features.append(feature_value.flatten())
                else:
                    data_features.append(torch.tensor([feature_value], dtype=torch.float32))
            else:
                # Default to zero if feature not found
                data_features.append(torch.tensor([0.0], dtype=torch.float32))
        
        if data_features:
            batch_features.append(torch.cat(data_features))
        else:
            batch_features.append(torch.tensor([0.0], dtype=torch.float32))
    
    return torch.stack(batch_features) if batch_features else torch.empty(0, 0)
