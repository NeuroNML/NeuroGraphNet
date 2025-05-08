import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from util.dataset import GraphEEGDataset
from models.gcn import EEGGCN


def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = GraphEEGDataset(
        root='data/graph_dataset',
        metadata_file='data/train/segments.parquet',
        signal_folder='data/train',
        edge_strategy='spatial',
        spatial_distance_file='data/distances_3d.csv',
        target_length=100000,  # Specify a fixed length for all signals
        force_reprocess=True,
        )
    
    # Get number of time points (in_channels) from first sample
    sample_data = dataset[0]
    in_channels = sample_data.x.shape[1]
    
    # Get number of classes from your dataset
    # Assuming labels are 0, 1, 2, ... num_classes-1
    num_classes = len(torch.unique(torch.cat([data.y for data in dataset])))
    
    print(f"Dataset info:")
    print(f" - Number of samples: {len(dataset)}")
    print(f" - Number of time points (in_channels): {in_channels}")
    print(f" - Number of classes: {num_classes}")
    
    # Split dataset into train/val/test sets (70%/15%/15%)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = EEGGCN(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=32,
        num_classes=num_classes
    ).to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    epochs = 50
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 10
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                loss = F.cross_entropy(out, data.y)
                val_loss += loss.item() * data.num_graphs
        val_loss /= len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f'Early stopping at epoch {epoch}')
                break
    
    # Load best model for testing
    model.load_state_dict(torch.load('best_model.pt'))
    
    # Test the model
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            y_true.extend(data.y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print("Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == '__main__':
    train_model()