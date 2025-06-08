import logging
from torch_geometric.data import DataLoader
from src.utils.graph_features import batch_graph_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeoDataLoader:
    """
    Custom data loader that handles both node features and graph-level features.
    """
    
    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = False, sampler=None, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Standard PyG DataLoader for node-level data
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            follow_batch=['x'],  # This helps with batching
            shuffle=shuffle,
            sampler=sampler,
            drop_last=drop_last
        )
        
        # Get graph feature names from the first sample (if available)
        self.graph_feature_names = []
        if len(dataset) > 0:
            sample = dataset[0]
            self.graph_feature_names = [
                attr for attr in dir(sample) 
                if attr.startswith('graph_') and not attr.startswith('__')
            ]
        
        logger.info(f"Found {len(self.graph_feature_names)} graph-level features: {self.graph_feature_names}")
    
    def __iter__(self):
        for batch in self.dataloader:
            # Extract graph-level features and batch them
            if self.graph_feature_names:
                # Get individual data objects from the batch
                batch_data = batch.to_data_list()
                # Batch graph-level features
                graph_features = batch_graph_features(batch_data, self.graph_feature_names)
                # Add batched graph features to the batch object
                batch.graph_features = graph_features
            
            yield batch
    
    def __len__(self):
        return len(self.dataloader)
