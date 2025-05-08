import torch
from torch_geometric.loader import DataLoader
from util.dataset import GraphEEGDataset

# Initialize dataset with spatial edge strategy
dataset_spatial = GraphEEGDataset(
    root='data/graph_dataset',
    metadata_file='data/train/segments.parquet',
    signal_folder='data/train',
    edge_strategy='spatial',
    spatial_distance_file='data/distances_3d.csv',
    target_length=100000,  # Specify a fixed length for all signals
    force_reprocess=True,
)

# # Or initialize with correlation edge strategy
# dataset_correlation = GraphEEGDataset(
#     root='path/to/dataset_folder',
#     metadata_file='path/to/metadata.parquet',
#     edge_strategy='correlation',
#     correlation_threshold=0.7,
# )

# Create data loaders
train_loader = DataLoader(dataset_spatial, batch_size=32, shuffle=True)

# Example of accessing a single data point
data = dataset_spatial[0]
print(f"Node features shape: {data.x.shape}")
print(f"Edge index shape: {data.edge_index.shape}")
print(f"Label: {data.y.item()}")
print(f"Sampling rate: {data.sampling_rate}")

# Example of iterating through the data loader
for batch in train_loader:
    # Process batch
    print(f"Batch size: {batch.num_graphs}")
    print(f"Batch node features: {batch.x.shape}")
    print(f"Batch edge index: {batch.edge_index.shape}")
    break  # Just show the first batch