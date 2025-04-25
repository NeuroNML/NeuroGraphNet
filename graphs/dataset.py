import os.path as osp
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from typing import Optional, Callable, List, Union, Dict
import networkx as nx


class GraphEEGDataset(Dataset):
    def __init__(
        self, 
        root: str,
        metadata_file: str,
        signal_folder: str,
        edge_strategy: str = 'spatial',
        spatial_distance_file: Optional[str] = None,
        correlation_threshold: float = 0.7,
        target_length: Optional[int] = None,
        transform: Optional[Callable] = None, 
        pre_transform: Optional[Callable] = None, 
        pre_filter: Optional[Callable] = None,
        force_reprocess=False
    ):
        """
        Custom PyTorch Geometric dataset for EEG data.
        
        Args:
            root: Root directory where the dataset should be saved
            metadata_file: Path to the parquet file containing metadata
            signal_folder: Path to the file/folder mentioned in metadata_file
            edge_strategy: Strategy to create edges ('spatial' or 'correlation')
            spatial_distance_file: Path to file containing spatial distances (required if edge_strategy='spatial')
            correlation_threshold: Threshold for creating edges based on correlation (used if edge_strategy='correlation')
            transform: Transform to be applied to each data object
            pre_transform: Pre-transform to be applied to each data object
            pre_filter: Pre-filter to be applied to each data object
        """
        self.metadata_file = metadata_file
        self.signal_folder = signal_folder
        self.edge_strategy = edge_strategy
        self.spatial_distance_file = spatial_distance_file
        self.correlation_threshold = correlation_threshold
        self.target_length = target_length
        self.force_reprocess = force_reprocess
        
        # EEG channels - standard 10-20 system
        self.channels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                         'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
                         'FZ', 'CZ', 'PZ']
        
        # Load metadata
        self.metadata = pd.read_parquet(metadata_file)
        
        # Load spatial distances if applicable
        if edge_strategy == 'spatial' and spatial_distance_file is not None:
            self.spatial_distances = self._load_spatial_distances(spatial_distance_file)
        
        super().__init__(root, transform, pre_transform, pre_filter)

        if self.force_reprocess:
            self.process()

    def _load_spatial_distances(self, file_path: str) -> Dict:
        """
        Load spatial distances between electrodes.
        Expected format: a file that can be loaded into a dictionary or matrix
        representing distances between electrodes.
        
        Returns:
            Dictionary of distances or adjacency matrix
        """
        # Implement based on your specific spatial distance file format
        # This is a placeholder - modify according to your file format
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)#, index_col=0)
            return {(ch1, ch2): df[(df['from']==ch1) & (df['to']==ch2)]['distance']
                    for ch1 in self.channels 
                    for ch2 in self.channels if ch1 != ch2}
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            return {(ch1, ch2): df[(df['from']==ch1) & (df['to']==ch2)]['distance']
                    for ch1 in self.channels 
                    for ch2 in self.channels if ch1 != ch2}
        else:
            # Default: create a placeholder distance matrix
            # In a real scenario, replace this with your actual distance loading logic
            distances = {}
            for i, ch1 in enumerate(self.channels):
                for j, ch2 in enumerate(self.channels):
                    if i != j:
                        # Example: random distances (replace with real distances)
                        distances[(ch1, ch2)] = np.random.rand()
            return distances

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns the names of all downloaded files in the raw directory.
        """
        # Return the metadata file and all signal files
        signal_files = self.metadata['signals_path'].tolist()
        return [self.metadata_file] + signal_files

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns the names of all processed files.
        """
        return [f'data_{i}.pt' for i in range(len(self.metadata))]

    def download(self):
        """
        Downloads the dataset to the raw directory.
        Not needed if data is already available locally.
        """
        # Skip if data is already local
        pass

    def _resize_signal(self, signal_data: pd.DataFrame) -> pd.DataFrame:
        """
        Resize the signal data to the target length
        
        Args:
            signal_data: DataFrame containing EEG signals
            
        Returns:
            Resized signal data
        """
        current_length = signal_data.shape[0]
        
        if current_length == self.target_length:
            return signal_data
        elif current_length > self.target_length:
            # Truncate to target length
            return signal_data.iloc[:self.target_length]
        else:
            # This case shouldn't happen if we determine target_length correctly
            # But we'll pad with zeros just in case
            print(f"Warning: Signal length {current_length} is less than target length {self.target_length}")
            padding = pd.DataFrame(0, 
                                  index=range(current_length, self.target_length),
                                  columns=signal_data.columns)
            return pd.concat([signal_data, padding])
        
    def _determine_target_length(self):

        """

        Determine the minimum length of all signals in dataset to use as target length

        """

        print("Determining minimum signal length in dataset...")

        min_length = float('inf')

        sample_count = min(len(self.metadata), 10)  # Check first 10 samples to save time

        

        for i in range(sample_count):

            signal_path = self.metadata.iloc[i]['signals_path']

            signal_data = pd.read_parquet(signal_path)

            length = signal_data.shape[0]

            min_length = min(min_length, length)

        

        self.target_length = min_length

        print(f"Target length set to: {self.target_length}")

    def process(self):
        """
        Processes raw data into PyTorch Geometric Data objects.
        """
        for idx, (_, row) in enumerate(self.metadata.iterrows()):
            # Load signal data
            signal_data = pd.read_parquet(f"{self.signal_folder}/{row['signals_path']}")
            
            # Resize signal to target length
            signal_data = self._resize_signal(signal_data)

            # Extract signals as features (nodes x features)
            # Each channel is a node, and its time series is its feature vector
            x = torch.tensor(signal_data[self.channels].values.T, dtype=torch.float)
            
            # Create edges based on the selected strategy
            edge_index = self._create_edges(signal_data)
            
            # Create label tensor
            y = torch.tensor([row['label']], dtype=torch.long)
            
            # Create additional metadata
            metadata = {
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'date': row['date'],
                'sampling_rate': row['sampling_rate']
            }
            
            # Create Data object
            data = Data(
                x=x,  # Node features: channels x time points
                edge_index=edge_index,  # Edges between channels
                y=y,  # Label
                **metadata  # Additional metadata
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Save processed data
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))

    def _create_edges(self, signal_data: pd.DataFrame) -> torch.Tensor:
        """
        Creates edges between EEG channels based on the specified strategy.
        
        Args:
            signal_data: DataFrame containing EEG signals
            
        Returns:
            torch.Tensor: Edge index tensor of shape [2, num_edges]
        """
        if self.edge_strategy == 'spatial':
            return self._create_spatial_edges()
        elif self.edge_strategy == 'correlation':
            return self._create_correlation_edges(signal_data)
        else:
            raise ValueError(f"Unknown edge strategy: {self.edge_strategy}")
    
    def _create_spatial_edges(self) -> torch.Tensor:
        """
        Creates edges based on spatial distances between electrodes.
        
        Returns:
            torch.Tensor: Edge index tensor
        """
        # Create a graph
        G = nx.Graph()
        
        # Add nodes
        for i, channel in enumerate(self.channels):
            G.add_node(i, name=channel)
        
        # Add edges based on distances
        edge_list = []
        for i, ch1 in enumerate(self.channels):
            for j, ch2 in enumerate(self.channels):
                if i < j:  # Avoid duplicate edges
                    # Get distance if available, otherwise use a default
                    distance = self.spatial_distances.get((ch1, ch2), 1.0)
                    
                    # Add edge if distance is within threshold (you can adjust this logic)
                    # Here we're adding all edges but you might want to threshold
                    G.add_edge(i, j, weight=distance)
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # Add in both directions for PyG
        
        # Convert to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index
    
    def _create_correlation_edges(self, signal_data: pd.DataFrame) -> torch.Tensor:
        """
        Creates edges based on correlation between channel signals.
        
        Args:
            signal_data: DataFrame containing EEG signals
            
        Returns:
            torch.Tensor: Edge index tensor
        """
        # Calculate correlation matrix
        corr_matrix = signal_data[self.channels].corr().abs().values
        
        # Create edges where correlation exceeds threshold
        edge_list = []
        for i in range(len(self.channels)):
            for j in range(len(self.channels)):
                if i != j and corr_matrix[i, j] >= self.correlation_threshold:
                    edge_list.append([i, j])
        
        # Convert to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index

    def len(self) -> int:
        """
        Returns the number of examples in the dataset.
        """
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        """
        Returns the data object at index idx.
        """
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data