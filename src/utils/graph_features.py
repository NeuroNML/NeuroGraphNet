"""
Graph Feature Extraction Module for EEG Graphs

This module provides functionality to compute various graph-level and node-level
features that can be used as additional inputs during training. This enhanced
version includes advanced metrics relevant to brain network analysis, such as
modularity, assortativity, and spectral features.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional
import networkx as nx
from networkx.algorithms import community
from torch_geometric.utils import to_networkx
import logging
import warnings # Add this line

# Set up logger with console handler
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


class GraphFeatureExtractor:
    """
    Extracts various graph features from PyTorch Geometric Data objects.
    Features include both node-level and graph-level metrics, with a focus
    on those relevant for EEG seizure detection.
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
            include_node_features: Whether to compute node-level features.
            include_graph_features: Whether to compute graph-level features.
            feature_types: List of specific feature types to compute. If None,
                           computes all available types.
                           Available types: ['degree', 'clustering', 'centrality',
                                           'connectivity', 'path_length', 'efficiency',
                                           'assortativity', 'modularity',
                                           'laplacian_spectrum', 'k_core']
        """
        self.include_node_features = include_node_features
        self.include_graph_features = include_graph_features

        # Default to all feature types if none specified
        if feature_types is None:
            self.feature_types = [
                'degree', 'clustering', 'centrality', 'connectivity',
                'path_length', 'efficiency', 'assortativity', 'modularity',
                'laplacian_spectrum', 'k_core'
            ]
        else:
            self.feature_types = feature_types

        logger.info(f"GraphFeatureExtractor initialized with features: {self.feature_types}")

    def extract_features(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Extract graph features from a PyTorch Geometric Data object.

        Args:
            data: PyTorch Geometric Data object containing graph structure.

        Returns:
            Dictionary containing extracted features.
        """
        features = {}

        # Convert to NetworkX for easier computation
        G = to_networkx(data, to_undirected=True)
        num_nodes = G.number_of_nodes()

        if num_nodes == 0:
            logger.warning("Empty graph encountered, returning zero features.")
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
            logger.warning(f"Error computing graph features: {e}, returning zero features.")
            return self._get_zero_features(num_nodes)

        return features

    def _compute_node_features(self, G: nx.Graph, data: Data) -> Dict[str, torch.Tensor]:
        """Compute node-level graph features."""
        features = {}
        num_nodes = G.number_of_nodes()
        # It's crucial to handle cases where G.nodes() might not be a simple 0..N-1 range.
        # However, the original code and tensor creation assumes 0..N-1 indexing for features.
        # We'll proceed with this assumption but be mindful if node IDs are arbitrary.
        # If node IDs are not 0..N-1, a mapping would be needed.

        if 'degree' in self.feature_types:
            # G.degree() returns a DegreeView, which is an iterator of (node, degree) pairs or a dict-like object.
            # We need a list of degrees corresponding to nodes 0 to num_nodes-1.
            degree_view = G.degree() # type: ignore
            degrees_list = [0] * num_nodes # Initialize with zeros
            for node, deg in degree_view:
                if 0 <= node < num_nodes: # Ensure node index is within expected range for the tensor
                    degrees_list[node] = deg
            degrees = torch.tensor(degrees_list, dtype=torch.float32)
            features['node_degree'] = degrees.unsqueeze(1)
            max_degree = degrees.max().item() if degrees.numel() > 0 and degrees.max().numel() > 0 else 1.0
            features['node_degree_normalized'] = (degrees / max_degree if max_degree > 0 else degrees).unsqueeze(1)

        if 'clustering' in self.feature_types:
            try:
                clustering_dict = nx.clustering(G) # Returns a dict {node: clustering_coefficient}
                # Ensure we get clustering values for nodes 0 to num_nodes-1
                clustering_values_list = [clustering_dict.get(n, 0.0) for n in range(num_nodes)]
                clustering_values = torch.tensor(clustering_values_list, dtype=torch.float32)
                features['node_clustering'] = clustering_values.unsqueeze(1)
            except Exception as e: # nx.clustering can return a single float for some graph types if all nodes have same clustering.
                                   # However, the more common return is a dict.
                logger.warning(f"Could not compute node clustering as dict: {e}. Trying to interpret as single value or defaulting.")
                try:
                    # Attempt to see if a single value was returned (less common for general graphs)
                    clustering_val = nx.clustering(G)
                    if isinstance(clustering_val, (float, int)):
                         clustering_values = torch.full((num_nodes,), float(clustering_val), dtype=torch.float32)
                         features['node_clustering'] = clustering_values.unsqueeze(1)
                    else: # If it's not a dict and not a single float/int, default to zeros
                        logger.warning(f"Node clustering returned unexpected type: {type(clustering_val)}. Defaulting to zeros.")
                        features['node_clustering'] = torch.zeros(num_nodes, 1, dtype=torch.float32)
                except Exception as e_inner:
                    logger.warning(f"Further error computing node clustering: {e_inner}. Defaulting to zeros.")
                    features['node_clustering'] = torch.zeros(num_nodes, 1, dtype=torch.float32)

        if 'centrality' in self.feature_types:
            self._safe_compute(features, 'node_betweenness', lambda: nx.betweenness_centrality(G), num_nodes)
            self._safe_compute(features, 'node_closeness', lambda: nx.closeness_centrality(G), num_nodes)
            self._safe_compute(features, 'node_load_centrality', lambda: nx.load_centrality(G), num_nodes)
            self._safe_compute(features, 'node_pagerank', lambda: nx.pagerank(G), num_nodes)
            if nx.is_connected(G):
                 self._safe_compute(features, 'node_eigenvector', lambda: nx.eigenvector_centrality(G, max_iter=1000), num_nodes)
            else:
                 features['node_eigenvector'] = torch.zeros(num_nodes, 1, dtype=torch.float32)

        if 'k_core' in self.feature_types:
            self._safe_compute(features, 'node_core_number', lambda: nx.core_number(G), num_nodes)
            
        return features

    def _compute_graph_features(self, G: nx.Graph) -> Dict[str, torch.Tensor]:
        """Compute graph-level features."""
        features = {}
        num_nodes = G.number_of_nodes()
        is_connected = nx.is_connected(G)

        if 'connectivity' in self.feature_types:
            features['graph_num_nodes'] = torch.tensor([num_nodes], dtype=torch.float32)
            features['graph_num_edges'] = torch.tensor([G.number_of_edges()], dtype=torch.float32)
            features['graph_density'] = torch.tensor([nx.density(G)], dtype=torch.float32)
            features['graph_is_connected'] = torch.tensor([float(is_connected)], dtype=torch.float32)
            features['graph_num_components'] = torch.tensor([nx.number_connected_components(G)], dtype=torch.float32)

        if 'clustering' in self.feature_types:
            features['graph_clustering'] = torch.tensor([nx.average_clustering(G)], dtype=torch.float32)
            features['graph_transitivity'] = torch.tensor([nx.transitivity(G)], dtype=torch.float32)

        if 'path_length' in self.feature_types:
            if is_connected:
                features['graph_avg_path_length'] = torch.tensor([nx.average_shortest_path_length(G)], dtype=torch.float32)
                features['graph_diameter'] = torch.tensor([nx.diameter(G)], dtype=torch.float32)
                features['graph_radius'] = torch.tensor([nx.radius(G)], dtype=torch.float32)
            else:
                for key in ['graph_avg_path_length', 'graph_diameter', 'graph_radius']:
                    features[key] = torch.tensor([0.0], dtype=torch.float32)

        if 'efficiency' in self.feature_types:
            features['graph_global_efficiency'] = torch.tensor([nx.global_efficiency(G)], dtype=torch.float32)
            features['graph_local_efficiency'] = torch.tensor([nx.local_efficiency(G)], dtype=torch.float32)

        if 'degree' in self.feature_types:
            degrees = [d for n, d in G.degree()] # type: ignore
            if degrees:
                features['graph_avg_degree'] = torch.tensor([np.mean(degrees)], dtype=torch.float32)
                features['graph_degree_std'] = torch.tensor([np.std(degrees)], dtype=torch.float32)
                features['graph_max_degree'] = torch.tensor([np.max(degrees)], dtype=torch.float32)
                features['graph_min_degree'] = torch.tensor([np.min(degrees)], dtype=torch.float32)
            else:
                for key in ['graph_avg_degree', 'graph_degree_std', 'graph_max_degree', 'graph_min_degree']:
                    features[key] = torch.tensor([0.0], dtype=torch.float32)

        if 'assortativity' in self.feature_types:
            try:
                # Temporarily ignore the specific RuntimeWarning from networkx's assortativity calculation
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning, message="invalid value encountered in scalar divide")
                    # The original np.errstate is also good practice for numerical stability
                    with np.errstate(divide='ignore', invalid='ignore'):
                        assortativity = nx.degree_assortativity_coefficient(G)
                        # Handle NaN/inf values that might result from the computation
                        if np.isnan(assortativity) or np.isinf(assortativity):
                            assortativity = 0.0
                        features['graph_assortativity'] = torch.tensor([assortativity], dtype=torch.float32)
            except Exception as e:
                logger.warning(f"Could not compute assortativity: {e}. Defaulting to 0.0.")
                features['graph_assortativity'] = torch.tensor([0.0], dtype=torch.float32)

        if 'modularity' in self.feature_types and num_nodes > 0:
            try:
                # community.louvain_communities returns a list of frozensets (each frozenset is a community)
                communities_list_of_sets = community.louvain_communities(G) # This is already a list of iterables
                features['graph_num_communities'] = torch.tensor([len(communities_list_of_sets)], dtype=torch.float32)
                if communities_list_of_sets: # Only compute modularity if communities were found
                    # modularity function expects the graph and the list of communities (sets/frozensets)
                    modularity_val = community.modularity(G, communities_list_of_sets)
                    features['graph_modularity'] = torch.tensor([modularity_val], dtype=torch.float32)
                else:
                    features['graph_modularity'] = torch.tensor([0.0], dtype=torch.float32)
            except Exception as e: 
                logger.warning(f"Could not compute modularity: {e}. Defaulting to 0.0.")
                features['graph_num_communities'] = torch.tensor([0.0], dtype=torch.float32)
                features['graph_modularity'] = torch.tensor([0.0], dtype=torch.float32)


        if 'laplacian_spectrum' in self.feature_types and num_nodes > 0:
            if is_connected:
                features['graph_algebraic_connectivity'] = torch.tensor([nx.algebraic_connectivity(G)], dtype=torch.float32)
                spec = nx.linalg.spectrum.adjacency_spectrum(G)
                # Spectral radius is the maximum absolute value of eigenvalues
                # Handle complex eigenvalues by taking their magnitude
                spectral_radius = float(np.abs(spec).max())
                features['graph_spectral_radius'] = torch.tensor([spectral_radius], dtype=torch.float32)
            else:
                features['graph_algebraic_connectivity'] = torch.tensor([0.0], dtype=torch.float32)
                features['graph_spectral_radius'] = torch.tensor([0.0], dtype=torch.float32)
        
        if 'k_core' in self.feature_types and num_nodes > 0:
            core_numbers = nx.core_number(G)
            features['graph_degeneracy'] = torch.tensor([max(core_numbers.values()) if core_numbers else 0], dtype=torch.float32)

        return features

    def _get_zero_features(self, num_nodes: int) -> Dict[str, torch.Tensor]:
        """Return zero features when computation fails or for an empty graph."""
        features = {}
        dims = self.get_feature_dimensions()

        if self.include_node_features:
            for name, dim in dims.items():
                if name.startswith('node_'):
                    features[name] = torch.zeros(num_nodes, dim, dtype=torch.float32)
        
        if self.include_graph_features:
            for name, dim in dims.items():
                if name.startswith('graph_'):
                    features[name] = torch.zeros(dim, dtype=torch.float32)
        
        return features
    
    def _safe_compute(self, features_dict, key, func, num_nodes):
        """Safely compute a feature and handle exceptions."""
        try:
            result_dict = func() # result is expected to be a dict {node: value}
            # Ensure values are extracted for nodes 0 to num_nodes-1
            # This assumes nodes are 0-indexed. If not, mapping is needed.
            values_list = [result_dict.get(n, 0.0) for n in range(num_nodes)]
            values = torch.tensor(values_list, dtype=torch.float32)
            features_dict[key] = values.unsqueeze(1)
        except Exception as e:
            logger.warning(f"Could not compute {key}: {e}. Defaulting to zeros.")
            features_dict[key] = torch.zeros(num_nodes, 1, dtype=torch.float32)


    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get the dimensions of all possible features.

        Returns:
            Dictionary mapping feature names to their dimensions.
        """
        node_features = {
            'node_degree': 1, 'node_degree_normalized': 1, 'node_clustering': 1,
            'node_betweenness': 1, 'node_closeness': 1, 'node_eigenvector': 1,
            'node_load_centrality': 1, 'node_pagerank': 1, 'node_core_number': 1
        }
        
        graph_features = {
            'graph_num_nodes': 1, 'graph_num_edges': 1, 'graph_density': 1,
            'graph_is_connected': 1, 'graph_num_components': 1,
            'graph_clustering': 1, 'graph_transitivity': 1,
            'graph_avg_path_length': 1, 'graph_diameter': 1, 'graph_radius': 1,
            'graph_global_efficiency': 1, 'graph_local_efficiency': 1,
            'graph_avg_degree': 1, 'graph_degree_std': 1, 'graph_max_degree': 1, 'graph_min_degree': 1,
            'graph_assortativity': 1, 'graph_num_communities': 1, 'graph_modularity': 1,
            'graph_algebraic_connectivity': 1, 'graph_spectral_radius': 1, 'graph_degeneracy': 1
        }
        
        dims = {}
        if self.include_node_features:
            dims.update(node_features)
        if self.include_graph_features:
            dims.update(graph_features)
            
        return dims


def augment_data_with_graph_features(
    data: Data,
    feature_extractor: GraphFeatureExtractor,
    concat_to_node_features: bool = True
) -> Data:
    """
    Augment a PyTorch Geometric Data object with graph features.

    Args:
        data: Original Data object.
        feature_extractor: Configured GraphFeatureExtractor instance.
        concat_to_node_features: If True, concatenate node-level graph
                                 features to `data.x`.

    Returns:
        Augmented Data object with additional features.
    """
    graph_features = feature_extractor.extract_features(data)
    new_data = data.clone()

    if concat_to_node_features and feature_extractor.include_node_features:
        node_feature_tensors = []
        if data.x is not None:
            node_feature_tensors.append(data.x)
        
        node_graph_features = [v for k, v in graph_features.items() if k.startswith('node_')]
        if node_graph_features:
            node_features_concat = torch.cat(node_graph_features, dim=1)
            node_feature_tensors.append(node_features_concat)
        
        if node_feature_tensors:
            new_data.x = torch.cat(node_feature_tensors, dim=1)

    if feature_extractor.include_graph_features:
        for name, value in graph_features.items():
            if name.startswith('graph_'):
                setattr(new_data, name, value)

    return new_data


def batch_graph_features(batch_data: List[Data], feature_names: List[str]) -> torch.Tensor:
    """
    Batch graph-level features from a list of Data objects.

    Args:
        batch_data: List of Data objects with graph features.
        feature_names: Names of graph-level features to batch.

    Returns:
        Batched tensor of shape [batch_size, num_features].
    """
    batch_features = []
    for data in batch_data:
        data_features = []
        for name in feature_names:
            feature_value = getattr(data, name, torch.tensor([0.0], dtype=torch.float32))
            data_features.append(feature_value.flatten())
        
        if data_features:
            batch_features.append(torch.cat(data_features))

    return torch.stack(batch_features) if batch_features else torch.empty(0, 0)