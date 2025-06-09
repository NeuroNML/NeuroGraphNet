from typing import Optional
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from src.data.geodataloader import GeoDataLoader
import numpy as np
import torch

from src.utils.general_funcs import labels_stats

class LazyDataLoaderManager:
    """
    Manages the lazy loading of multiple dataset types (e.g., timeseries, graph).
    
    This class centralizes the configuration and creation of DataLoader instances.
    It splits datasets and creates dataloaders only when they are first requested
    for a specific dataset type, saving memory and processing time.
    """

    def __init__(self, datasets_config, train_ratio=0.8, oversampling_power=1.0, batch_size=64):
        """
        Initializes the data manager.
        """
        self.datasets_config = datasets_config
        self.train_ratio = train_ratio
        self.oversampling_power = oversampling_power
        self.batch_size = batch_size
        
        # Internal caches to store created subsets and loaders, preventing re-computation
        self._data_subsets = {}  # Caches train/val subsets
        self._loaders = {}       # Caches DataLoader instances

    def _create_subsets(self, dataset_type):
        """
        Splits the training data for a given dataset type into train and validation subsets.
        This is called internally on the first request for a dataset type.
        """
        if dataset_type in self._data_subsets:
            return

        print(f"\n--- Creating train/val subsets for '{dataset_type}' dataset ---")
        
        config = self.datasets_config[dataset_type]
        dataset_tr, clips_tr = config['dataset_tr'], config['clips_tr']
        
        total_samples = len(dataset_tr)
        train_size = int(self.train_ratio * total_samples)
        
        indices = torch.randperm(total_samples)
        train_indices, val_indices = indices[:train_size].tolist(), indices[train_size:].tolist()

        print(f"Splitting {total_samples} samples -> Train: {len(train_indices)}, Val: {len(val_indices)}")
        labels_stats(clips_tr["label"].values, train_indices, val_indices)

        train_dataset, val_dataset = Subset(dataset_tr, train_indices), Subset(dataset_tr, val_indices)
        
        # --- Oversampling Logic ---
        train_labels = clips_tr.iloc[train_indices]["label"].values
        class_counts = np.bincount(train_labels)
        class_weights = np.where(class_counts > 0, (1. / class_counts) ** self.oversampling_power, 0)
        sample_weights = [float(class_weights[label]) for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        print(f"Class weights: {class_weights}")
        print(f"Train set class distribution: {np.bincount(train_labels)}")

        self._data_subsets[dataset_type] = {'train': train_dataset, 'val': val_dataset, 'sampler': sampler}

    def get_loaders(self, dataset_type):
        """
        Retrieves the train, validation, and test dataloaders for a given dataset type.
        """
        if dataset_type in self._loaders:
            return self._loaders[dataset_type]

        print(f"\n--- Lazily creating loaders for '{dataset_type}' ---")
        self._create_subsets(dataset_type)
        
        subsets, config = self._data_subsets[dataset_type], self.datasets_config[dataset_type]
        LoaderClass = GeoDataLoader if dataset_type not in ['signal', 'feature'] else DataLoader

        self._loaders[dataset_type] = {
            'train': LoaderClass(subsets['train'], batch_size=self.batch_size, sampler=subsets['sampler'], drop_last=True),
            'val': LoaderClass(subsets['val'], batch_size=self.batch_size, shuffle=False, drop_last=False),
            'test': LoaderClass(config['dataset_te'], batch_size=self.batch_size, shuffle=False, drop_last=False)
        }
        return self._loaders[dataset_type]

    def get_dataset_info(self, dataset_type):
        """
        Provides a summary of dimensions and properties for a given dataset type.
        """
        config = self.datasets_config[dataset_type]
        sample = config['dataset_tr'][0]
        
        info = {'type': dataset_type, 'total_train_samples': len(config['dataset_tr'])}
        
        if dataset_type in ['spatial', 'correlation', 'absdiff_correlation']:
            info.update({'feature_dim': len([k for k in sample if k.startswith('graph_')])})
        elif dataset_type == 'feature':
            info.update({'feature_dim': sample[0].shape[-1], 'sequence_length': sample[0].shape[0]})
        elif dataset_type == 'signal':
            info.update({'channels': sample[0].shape[0], 'sequence_length': sample[0].shape[1]})
            
        return info


class TrainingContext:
    """
    A stateful wrapper for LazyDataLoaderManager that provides a simplified
    interface for switching between datasets.
    
    This class holds the currently active loaders and dataset information,
    matching the user's desired workflow.
    """
    def __init__(self, data_manager: LazyDataLoaderManager):
        self.data_manager = data_manager
        self.dataset_type = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.clips: Optional[object] = None
        self.info = {}
        print("✅ TrainingContext initialized. Use .switch_to('dataset_type') to begin.")

    def switch_to(self, dataset_type):
        """
        Switches the active dataset, updating all public loader and info attributes.
        """
        print(f"\n🌐 Switching context to '{dataset_type.upper()}' dataset...")

        # clean up CUDA memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # Ensure the dataset type exists in the manager's configuration FIRST
        if dataset_type not in self.data_manager.datasets_config:
            raise ValueError(f"Dataset type '{dataset_type}' is not recognized. Available types: {list(self.data_manager.datasets_config.keys())}")
        
        # Ensure the dataset type is one of the specifically supported types for this context
        # Corrected 'features' to 'feature' in the error message to match the list in the check
        supported_context_types = ['spatial', 'correlation', 'absdiff_correlation', 'feature', 'signal']
        if dataset_type not in supported_context_types:
            raise ValueError(f"Invalid dataset type '{dataset_type}'. Supported types: {', '.join(supported_context_types)}.")
        
        # Get the dictionary of loaders from the manager (creates them if they don't exist)
        # This is now safe because dataset_type has been validated against datasets_config
        loaders = self.data_manager.get_loaders(dataset_type)
        
        # Overwrite the context's loader attributes
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.test_loader = loaders['test']
        self.clips = self.data_manager.datasets_config[dataset_type]['clips_tr']

        # ensure that all loaders are of the same type
        if not all(isinstance(loader, type(self.train_loader)) for loader in [self.val_loader, self.test_loader]):
            raise ValueError("All loaders must be of the same type (e.g., DataLoader or GeoDataLoader).")
        
        # Update the dataset type and informational attributes
        self.dataset_type = dataset_type
        self.info = self.data_manager.get_dataset_info(dataset_type)
        
        print(f"🚀 Context ready for '{dataset_type}'.")
        print(f"   Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        for key, value in self.info.items():
            if value is not None:
                print(f"   {key.replace('_', ' ').title()}: {value}")
        return self