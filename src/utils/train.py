from collections import OrderedDict, defaultdict # Added defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # type: ignore
from torchmetrics.functional import accuracy, f1_score, auroc 
from torch_geometric.utils import to_dense_batch
from torch.utils.data import WeightedRandomSampler

# Assuming Data and Batch are from PyTorch Geometric if used with GraphEEGDataset
from torch_geometric.data import Batch as PyGBatch 


def _save(
    model: nn.Module,
    path: Path,
    train_history: Dict[str, List[float]], 
    val_history: Dict[str, List[float]],   
    optimizer_state_dict: Optional[Dict] = None, 
    epoch: Optional[int] = None,                 
    best_score: Optional[float] = None           
):
    """Save .state_dict() of the underlying model (handles DP/DDP)
    along with training and validation histories and other optional info."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'model_state_dict': model_state_dict,
        'train_history': train_history, 
        'val_history': val_history,     
        'optimizer_state_dict': optimizer_state_dict,
        'epoch': epoch,
        'best_score': best_score,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _load(
    model: nn.Module,
    path: Path,
    device: torch.device,
    optimizer: Optional[optim.Optimizer] = None 
) -> Dict[str, Any]:
    """Loads model state_dict and optimizer state. Returns the full checkpoint dictionary."""
    print(f"   - Loading checkpoint from: {path}")
    loaded_data = torch.load(path, map_location=device, weights_only=False) 
    
    state_dict_to_load = None
    full_checkpoint_data = {} # Initialize for return

    if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
        print("   - Detected full checkpoint dictionary.")
        state_dict_to_load = loaded_data['model_state_dict']
        full_checkpoint_data = loaded_data # Return the original full checkpoint
        if optimizer and loaded_data.get('optimizer_state_dict'):
            try:
                optimizer.load_state_dict(loaded_data['optimizer_state_dict'])
                print("   - Optimizer state loaded from checkpoint.")
            except Exception as e:
                print(f"   - Warning: Could not load optimizer state from checkpoint: {e}")
    elif isinstance(loaded_data, OrderedDict): # Likely a raw state_dict
        print("   - Warning: Loaded a raw state_dict. Optimizer state and training history not in this file.")
        state_dict_to_load = loaded_data
        # Reconstruct a minimal checkpoint structure for consistent return type
        full_checkpoint_data = {'model_state_dict': state_dict_to_load, 'train_history': {}, 'val_history': {}}
    else:
        raise TypeError(f"Checkpoint file at {path} is not a recognized format (expected dict or OrderedDict).")

    if state_dict_to_load is None:
        raise ValueError(f"Could not extract model_state_dict from checkpoint at {path}.")

    # Handle 'module.' prefix from DataParallel/DDP if necessary
    wrapped = isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
    has_module_prefix = any(k.startswith("module.") for k in state_dict_to_load.keys())

    if wrapped and not has_module_prefix:
        model.module.load_state_dict(state_dict_to_load)
    elif not wrapped and has_module_prefix:
        new_state_dict = OrderedDict()
        for k, v in state_dict_to_load.items():
            name = k.replace("module.", "", 1)
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict_to_load)
            
    print("   - Model state successfully loaded.")
    return full_checkpoint_data # Return the loaded (or reconstructed) checkpoint data


# --- Training and Evaluation Loop ---
def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    save_path: Path,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, # type: ignore
    monitor: str = "val_f1",
    patience: int = 15,
    num_epochs: int = 100,
    grad_clip: float = 1.0,
    overwrite: bool = False,
    use_gnn: bool = True,
    use_oversampling: bool = False, # New parameter for toggling oversampling
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Generic training loop.
    If use_gnn=False, it assumes data comes from a standard DataLoader (e.g., with TensorDataset)
    and batch_data is a tuple (input_features, labels).
    If use_gnn=True, it expects PyG Batch objects.
    """
    start_epoch = 0
    train_history = defaultdict(list)
    val_history = defaultdict(list)

    if monitor in ["val_loss", "loss"]:
        best_score = float("inf")
    else:
        best_score = -float("inf")

    if save_path.exists() and not overwrite:
        print(f"🚀 Attempting to load checkpoint from {save_path}...")
        try:
            checkpoint = _load(model, save_path, device, optimizer)
            # Restore histories
            loaded_train_hist = checkpoint.get('train_history', {})
            loaded_val_hist = checkpoint.get('val_history', {})
            for key, val_list in loaded_train_hist.items(): train_history[key].extend(val_list)
            for key, val_list in loaded_val_hist.items(): val_history[key].extend(val_list)

            start_epoch = checkpoint.get('epoch', 0)
            loaded_best_score = checkpoint.get('best_score')
            if loaded_best_score is not None: best_score = loaded_best_score
            
            if start_epoch >= num_epochs:
                 print(f" 🏁 Training already completed up to epoch {start_epoch}. Final best '{monitor}': {best_score:.4f}")
                 return dict(train_history), dict(val_history)
            print(f" ✅ Checkpoint loaded. Resuming from epoch {start_epoch + 1}. Best '{monitor}' score: {best_score:.4f}")

        except Exception as e:
            print(f" ⚠️ Could not load checkpoint: {e}. Starting training from scratch.")
            save_path.unlink(missing_ok=True)
            start_epoch = 0 # Ensure start_epoch is reset
            # Re-initialize best_score based on monitor
            if monitor in ["val_loss", "loss"]: best_score = float("inf")
            else: best_score = -float("inf")

    elif overwrite and save_path.exists():
        print(f" 🗑️ Overwrite enabled: Removed existing checkpoint at {save_path}")
        save_path.unlink()

    train_loader_to_iterate = train_loader

    if use_oversampling:
        print("🔍 Applying oversampling to the training data...")
        original_train_dataset = train_loader.dataset

        all_labels_list = []
        labels_source_attribute_name = None
        # Attempt to get all labels at once from common attributes (optimization)
        if hasattr(original_train_dataset, 'y') and isinstance(original_train_dataset.y, torch.Tensor):
            if original_train_dataset.y.ndim == 1 and len(original_train_dataset.y) == len(original_train_dataset):
                all_labels_list = [int(l.item()) for l in original_train_dataset.y.cpu()]
                labels_source_attribute_name = "dataset.y"
            elif original_train_dataset.y.ndim == 2 and original_train_dataset.y.shape[1] == 1 and len(original_train_dataset.y) == len(original_train_dataset):
                all_labels_list = [int(l.item()) for l in original_train_dataset.y.cpu().squeeze()]
                labels_source_attribute_name = "dataset.y (squeezed)"
        elif use_gnn and hasattr(original_train_dataset, 'data') and hasattr(original_train_dataset.data, 'y'):
            # Common for PyG InMemoryDataset
            potential_labels = original_train_dataset.data.y
            if isinstance(potential_labels, torch.Tensor) and potential_labels.ndim == 1 and len(potential_labels) == len(original_train_dataset):
                all_labels_list = [int(l.item()) for l in potential_labels.cpu()]
                labels_source_attribute_name = "dataset.data.y (PyG InMemoryDataset)"
        
        if labels_source_attribute_name:
            print(f"  Extracted {len(all_labels_list)} labels directly from '{labels_source_attribute_name}'.")
        else:
            print(f"  Attribute-based label extraction not applicable or failed. Iterating through {len(original_train_dataset)} dataset samples...")
            for i in tqdm(range(len(original_train_dataset)), desc="Collecting labels for oversampling", ncols=100, leave=False):
                data_sample = original_train_dataset[i]
                label_value = None
                if use_gnn: # Expect PyG Data object
                    if hasattr(data_sample, 'y') and data_sample.y is not None: label_value = data_sample.y
                else: # Not use_gnn
                    if isinstance(data_sample, (list, tuple)) and len(data_sample) >= 2: label_value = data_sample[1]
                    elif hasattr(data_sample, 'y') and data_sample.y is not None: label_value = data_sample.y

                if label_value is not None:
                    if isinstance(label_value, torch.Tensor):
                        if label_value.numel() == 1: all_labels_list.append(int(label_value.item()))
                    elif isinstance(label_value, (int, float)): all_labels_list.append(int(label_value))
            print(f"  Finished iterating. Collected {len(all_labels_list)} labels.")

        if not all_labels_list:
            print("  ⚠️ Could not extract any labels for oversampling. Proceeding without it.")
        else:
            labels_tensor = torch.tensor(all_labels_list, dtype=torch.long)
            class_counts = torch.bincount(labels_tensor)

            if len(class_counts) < 2:
                print(f"  ⚠️ Only {len(class_counts)} class(es) found (counts: {class_counts.tolist()}). Oversampling ineffective. Proceeding without it.")
            elif class_counts.min() == 0: # Should not happen if all_labels_list is populated from these classes
                print(f"  ⚠️ One class has zero samples (counts: {class_counts.tolist()}). Oversampling cannot be applied. Proceeding without it.")
            else:
                print(f"  Class counts before oversampling: {class_counts.tolist()}")
                num_samples_total = len(labels_tensor)
                weight_per_class = 1. / class_counts.float()
                sample_weights = torch.zeros(num_samples_total, dtype=torch.float)
                for i, label_idx in enumerate(labels_tensor):
                    sample_weights[i] = weight_per_class[label_idx]
                
                # Convert tensor to list for WeightedRandomSampler
                sample_weights_list = sample_weights.tolist()
                sampler = WeightedRandomSampler(weights=sample_weights_list, num_samples=num_samples_total, replacement=True)
                
                train_loader_to_iterate = torch.utils.data.DataLoader(
                    original_train_dataset,
                    batch_size=train_loader.batch_size,
                    sampler=sampler, # Key change
                    num_workers=train_loader.num_workers,
                    pin_memory=train_loader.pin_memory,
                    collate_fn=train_loader.collate_fn,
                    drop_last=train_loader.drop_last
                )
                print(f"  Successfully created an oversampled DataLoader.")

    bad_epochs = 0
    print(f"💪 Starting training from epoch {start_epoch + 1} to {num_epochs}...")
    # Corrected initial for tqdm if resuming
    pbar = tqdm(range(start_epoch + 1, num_epochs + 1), desc="Epochs", ncols=120, initial=start_epoch + 1, total=num_epochs)

    for epoch in pbar:
        model.train()
        epoch_train_loss = 0.0
        all_train_preds, all_train_targets = [], []

        for data_batch_item in train_loader_to_iterate: # Use the potentially oversampled loader
            optimizer.zero_grad(set_to_none=True)

            if use_gnn:
                if not PyGBatch or not to_dense_batch:
                    raise ImportError("PyTorch Geometric is required for GNN mode.")
                curr_batch = data_batch_item.to(device)
                y_targets = curr_batch.y
                if y_targets is None: continue
                y_targets = y_targets.float().unsqueeze(1)
                if not (hasattr(curr_batch, 'x') and hasattr(curr_batch, 'edge_index')):
                    raise ValueError("For GNN mode, batch_data must have 'x' and 'edge_index'.")
                logits = model(curr_batch.x.float(),
                               curr_batch.edge_index,
                               curr_batch.batch if hasattr(curr_batch, 'batch') else None)
            else: # Non-GNN
                if isinstance(data_batch_item, (tuple, list)) and len(data_batch_item) == 2:
                    x_batch, y_batch = data_batch_item
                    x_batch = x_batch.to(device)
                    y_targets = y_batch.to(device) if isinstance(y_batch, torch.Tensor) else torch.tensor(y_batch, device=device)
                    y_targets = y_targets.float().unsqueeze(1) if y_targets.ndim == 1 else y_targets.float()
                    logits = model(x_batch.float())
                elif hasattr(data_batch_item, 'x') and hasattr(data_batch_item, 'y'):
                    if not to_dense_batch: # Should be caught earlier if PyG not installed but good check
                         raise ImportError("PyTorch Geometric 'to_dense_batch' is required for this non-GNN path if data is PyG Batch.")
                    curr_batch = data_batch_item.to(device)
                    y_targets = curr_batch.y
                    if y_targets is None: continue
                    y_targets = y_targets.float().unsqueeze(1)
                    
                    n_channels = 19 # This was hardcoded, consider making it a parameter or inferring
                    assert n_channels > 0, "n_channels must be a positive integer."
                    if curr_batch.num_graphs > 0:
                        node_features_tensor = curr_batch.x
                        if node_features_tensor.ndim == 1: node_features_tensor = node_features_tensor.unsqueeze(-1)
                        num_features_per_channel = node_features_tensor.size(1)
                        expected_total_nodes = curr_batch.num_graphs * n_channels
                        if node_features_tensor.size(0) == expected_total_nodes:
                            input_features = node_features_tensor.view(curr_batch.num_graphs, n_channels, num_features_per_channel)
                        else:
                            input_features, _ = to_dense_batch(node_features_tensor, curr_batch.batch, max_num_nodes=n_channels)
                    elif curr_batch.num_graphs == 0: raise ValueError("Empty batch encountered.")
                    else: raise ValueError(f"Unexpected num_graphs: {curr_batch.num_graphs}.")
                    logits = model(input_features.float())
                else:
                    raise ValueError("Batch data must be a tuple (x, y) or have 'x' and 'y' attributes for non-GNN mode.")

            loss = criterion(logits, y_targets)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_train_loss += loss.item()
            all_train_preds.append(logits.sigmoid().detach().cpu())
            all_train_targets.append(y_targets.detach().cpu())

        avg_train_loss = epoch_train_loss / max(1, len(train_loader_to_iterate))
        train_history['loss'].append(avg_train_loss)

        if all_train_preds and all_train_targets:
            preds_cat_train = torch.cat(all_train_preds)
            targets_cat_train = torch.cat(all_train_targets).int()
            try:
                train_history['acc'].append(accuracy(preds_cat_train, targets_cat_train, task="binary").item())
                train_history['f1'].append(f1_score(preds_cat_train, targets_cat_train, task="binary").item())
                train_history['auroc'].append(auroc(preds_cat_train, targets_cat_train, task="binary").item())
            except ValueError as e:
                print(f"Warning (Train Metrics, Epoch {epoch}): {e}. Appending 0.0 for metrics.")
                train_history['acc'].append(0.0); train_history['f1'].append(0.0); train_history['auroc'].append(0.0)
        else: # Handles empty train_loader_to_iterate or if all batches were skipped
             train_history['acc'].append(0.0); train_history['f1'].append(0.0); train_history['auroc'].append(0.0)


        # Validation
        model.eval()
        epoch_val_loss = 0.0
        all_val_preds, all_val_targets = [], []
        with torch.no_grad():
            for data_batch_item_val in val_loader: # Use a different variable name
                if use_gnn:
                    curr_batch = data_batch_item_val.to(device)
                    y_targets_val = curr_batch.y
                    if y_targets_val is None: continue
                    y_targets_val = y_targets_val.float().unsqueeze(1)
                    if not (hasattr(curr_batch, 'x') and hasattr(curr_batch, 'edge_index')):
                         raise ValueError("For GNN mode, batch_data must have 'x' and 'edge_index'.")
                    logits_val = model(curr_batch.x.float(),
                                   curr_batch.edge_index,
                                   curr_batch.batch if hasattr(curr_batch, 'batch') else None)
                else: # Non-GNN
                    if isinstance(data_batch_item_val, (tuple, list)) and len(data_batch_item_val) == 2:
                        x_batch_val, y_batch_val = data_batch_item_val
                        x_batch_val = x_batch_val.to(device)
                        if y_batch_val is not None:
                            y_targets_val = y_batch_val.to(device) if isinstance(y_batch_val, torch.Tensor) else torch.tensor(y_batch_val, device=device)
                            y_targets_val = y_targets_val.float().unsqueeze(1) if y_targets_val.ndim == 1 else y_targets_val.float()
                        else: continue # Skip if no labels in validation
                        logits_val = model(x_batch_val.float())
                    elif hasattr(data_batch_item_val, 'x') and hasattr(data_batch_item_val, 'y'):
                        curr_batch = data_batch_item_val.to(device)
                        y_targets_val = curr_batch.y
                        if y_targets_val is None: continue
                        y_targets_val = y_targets_val.float().unsqueeze(1)
                        
                        n_channels = 19 # This was hardcoded
                        assert n_channels > 0
                        if curr_batch.num_graphs > 0:
                            node_features_tensor = curr_batch.x
                            if node_features_tensor.ndim == 1: node_features_tensor = node_features_tensor.unsqueeze(-1)
                            num_features_per_channel = node_features_tensor.size(1)
                            expected_total_nodes = curr_batch.num_graphs * n_channels
                            if node_features_tensor.size(0) == expected_total_nodes:
                                input_features_val = node_features_tensor.view(curr_batch.num_graphs, n_channels, num_features_per_channel)
                            else:
                                input_features_val, _ = to_dense_batch(node_features_tensor, curr_batch.batch, max_num_nodes=n_channels)
                        elif curr_batch.num_graphs == 0: continue # Skip empty val batch
                        else: raise ValueError(f"Unexpected num_graphs in val: {curr_batch.num_graphs}.")
                        logits_val = model(input_features_val.float())
                    else:
                        raise ValueError("Val batch data must be a tuple (x, y) or have 'x' and 'y' attributes for non-GNN mode.")

                epoch_val_loss += criterion(logits_val, y_targets_val).item()
                all_val_preds.append(logits_val.sigmoid().cpu())
                all_val_targets.append(y_targets_val.cpu())
        
        avg_val_loss = epoch_val_loss / max(1, len(val_loader))
        val_history['loss'].append(avg_val_loss)
        current_score_val = avg_val_loss # Default for monitor

        if all_val_preds and all_val_targets:
            preds_cat_val = torch.cat(all_val_preds)
            targets_cat_val = torch.cat(all_val_targets).int()
            try:
                val_acc = accuracy(preds_cat_val, targets_cat_val, task="binary").item()
                val_f1 = f1_score(preds_cat_val, targets_cat_val, task="binary").item()
                val_auroc = auroc(preds_cat_val, targets_cat_val, task="binary").item()
                val_history['acc'].append(val_acc)
                val_history['f1'].append(val_f1)
                val_history['auroc'].append(val_auroc)

                if monitor == "val_auroc": current_score_val = val_auroc
                elif monitor == "val_f1": current_score_val = val_f1
                elif monitor == "val_acc": current_score_val = val_acc
                # else monitor is "val_loss", already set
            except ValueError as e:
                print(f"Warning (Validation Metrics, Epoch {epoch}): {e}. Using val_loss for monitoring.")
                val_history['acc'].append(0.0); val_history['f1'].append(0.0); val_history['auroc'].append(0.0)
                # current_score_val remains avg_val_loss
        else: # Handles empty val_loader or if all batches were skipped
            val_history['acc'].append(0.0); val_history['f1'].append(0.0); val_history['auroc'].append(0.0)
            # current_score_val remains avg_val_loss

        improved = False
        if monitor in ["val_loss", "loss"]:
            if current_score_val < best_score:
                best_score = current_score_val
                improved = True
        else:
            if current_score_val > best_score:
                best_score = current_score_val
                improved = True

        if improved:
            bad_epochs = 0
            _save(model, save_path, dict(train_history), dict(val_history), optimizer.state_dict(), epoch, best_score)
        else:
            bad_epochs += 1

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_score_val)
            else:
                scheduler.step()
        
        postfix_dict = {
            "train_loss": f"{train_history['loss'][-1]:.4f}" if train_history['loss'] else "N/A",
            "val_loss": f"{val_history['loss'][-1]:.4f}" if val_history['loss'] else "N/A",
            f"best_{monitor}": f"{best_score:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}", # Check if optimizer.param_groups exists
            "bad_epochs": f"{bad_epochs}/{patience}",
            "save": "✓" if improved else "-"
        }
        if val_history.get('auroc'): postfix_dict["val_auroc"] = f"{val_history['auroc'][-1]:.4f}"
        if val_history.get('f1'): postfix_dict["val_f1"] = f"{val_history['f1'][-1]:.4f}"
        pbar.set_postfix(postfix_dict)

        if patience > 0 and bad_epochs >= patience:
            pbar.write(f"🛑 Early stopping: no '{monitor}' improvement in {patience} epochs.")
            break
    
    pbar.close()
    print("\n✅ Training complete.")
    
    if save_path.exists():
        print(f"↩️ Loading best model state from {save_path} for return.")
        # Load only model weights, not optimizer or epoch for the final returned model
        final_checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(final_checkpoint['model_state_dict'])
    else:
        print(" ⚠️ No checkpoint was saved during training (or an error occurred). Model reflects last epoch state.")

    return dict(train_history), dict(val_history)

def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device,
    checkpoint_path: Path, 
    submission_path: Path,
    id_attribute: str = 'original_clips_df_idx', 
    threshold: float = 0.5,
    use_gnn: bool = True, # Added: For consistency with train_model
    input_type: str = 'signal' # Added: For consistency with train_model (feature, signal, embedding)
) -> pd.DataFrame:
    """
    Loads checkpoint, runs inference, and writes submission CSV.
    
    Args:
        use_gnn: If True, uses GNN mode with edge_index
        input_type: Type of input processing for non-GNN models:
            - 'feature': expects (batch_size, features) -> unsqueezed to (batch_size, 1, features)
            - 'signal': expects (batch_size, sensors, time_steps) e.g., (512, 19, 3000)
            - 'embedding': not implemented yet
            - For GNN mode: reshapes PyG Batch data for standard sequence models
    """
    print(f"⚙️ Evaluating model. Loading model from: {checkpoint_path}")
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}. Cannot evaluate.")
        return pd.DataFrame()

    try:
        _ = _load(model, checkpoint_path, device)
    except Exception as e:
        print(f"❌ Error loading model checkpoint for evaluation: {e}")
        return pd.DataFrame()
        
    model.to(device).eval()

    all_ids, all_binary_preds = [], []
    print("🧪 Performing inference on the test set...")
    with torch.no_grad():
        # for batch_data in tqdm(test_loader, desc="Evaluating"):
        for batch_data in test_loader:
            print(f"BATCH: {batch_data}, feature: {batch_data.x}")
            # Handle different batch formats for device transfer
            if isinstance(batch_data, (tuple, list)):
                # For tuple/list format, move elements to device individually
                batch_data = tuple(item.to(device) if hasattr(item, 'to') else item for item in batch_data)
            else:
                # For PyG Batch objects
                batch_data = batch_data.to(device)
            
            current_ids = []
            # Handle ID extraction based on batch format
            if isinstance(batch_data, (tuple, list)):
                # For tuple/list format from standard DataLoader, use sequential IDs
                batch_size = batch_data[0].size(0) if len(batch_data) > 0 else 0
                start_id = len(all_ids)
                current_ids = list(range(start_id, start_id + batch_size))
            elif not hasattr(batch_data, id_attribute):
                print(f"Warning: Batch object missing ID attribute '{id_attribute}'. Using sequential IDs.")
                num_items_in_batch = batch_data.num_graphs if isinstance(batch_data, PyGBatch) else batch_data.x.size(0)
                start_id = len(all_ids)
                current_ids = list(range(start_id, start_id + num_items_in_batch))
            else:
                batch_ids_tensor = getattr(batch_data, id_attribute)
                current_ids = batch_ids_tensor.cpu().tolist()

            if use_gnn:
                if not (hasattr(batch_data, 'x') and hasattr(batch_data, 'edge_index')):
                     raise ValueError("For GNN mode, batch_data must have 'x' and 'edge_index'.")
                logits = model(batch_data.x.float(), 
                               batch_data.edge_index, 
                               batch_data.batch if hasattr(batch_data, 'batch') else None)
            else:
                # Handle non-GNN models based on input_type
                if input_type == 'feature':
                    # Expected input shape: (batch_size, features)
                    if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                        # Handle tuple format (x, y) from standard DataLoader
                        x_batch, _ = batch_data
                        x_batch = x_batch.to(device).float()
                        logits = model(x_batch)
                    elif hasattr(batch_data, 'x'):
                        # Handle PyG Batch format - convert to dense and aggregate
                        if not hasattr(batch_data, 'x'):
                            raise ValueError("Batch data missing 'x' attribute for feature mode.")
                        
                        # Convert PyG batch to dense format for feature processing
                        x_dense, mask = to_dense_batch(batch_data.x.float(), 
                                                     batch_data.batch if hasattr(batch_data, 'batch') else None, 
                                                     fill_value=0)
                        # For feature mode, we need to aggregate node features per graph
                        # x_dense shape: (batch_size, max_nodes, features)
                        # Aggregate across nodes (mean pooling) to get (batch_size, features)
                        x_aggregated = torch.mean(x_dense, dim=1)  # (batch_size, features)
                        logits = model(x_aggregated)
                    else:
                        raise ValueError("Batch data format not supported for feature input type.")
                        
                elif input_type == 'signal':
                    # Expected input shape: (batch_size, sensors, time_steps) e.g., (512, 19, 3000)
                    if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                        # Handle tuple format (x, y) from TimeseriesEEGDataset
                        x_batch, _ = batch_data
                        x_batch = x_batch.to(device).float()
                        # x_batch should already be in shape (batch_size, sensors, time_steps)
                        logits = model(x_batch)
                    elif hasattr(batch_data, 'x'):
                        # Handle PyG Batch format - reshape to signal format
                        if not hasattr(batch_data, 'x'):
                            raise ValueError("Batch data missing 'x' attribute for signal mode.")
                        
                        # Convert PyG batch to dense format
                        x_dense, mask = to_dense_batch(batch_data.x.float(), 
                                                     batch_data.batch if hasattr(batch_data, 'batch') else None, 
                                                     fill_value=0)
                        # x_dense shape: (batch_size, max_nodes, features)
                        # For signal mode, we expect features to be time steps
                        # Permute to get (batch_size, sensors, time_steps) where sensors=max_nodes
                        x_permuted = x_dense.permute(0, 2, 1)  # (batch_size, features, max_nodes)
                        logits = model(x_permuted)
                    else:
                        raise ValueError("Batch data format not supported for signal input type.")
                        
                elif input_type == 'embedding':
                    raise NotImplementedError("Embedding input type is not implemented yet.")
                else:
                    raise ValueError(f"Unknown input_type: {input_type}. Must be one of ['feature', 'signal', 'embedding'].")

            probs = logits.sigmoid().cpu() 
            
            if probs.ndim > 1 and probs.shape[1] == 1:
                probs = probs.squeeze(1)
            elif probs.ndim == 0: 
                 probs = probs.unsqueeze(0)

            binary_preds = (probs > threshold).int().tolist()
            
            all_binary_preds.extend(binary_preds)
            all_ids.extend(current_ids)

    if not all_ids:
        print("❌ No IDs were collected during evaluation. Cannot create submission file.")
        return pd.DataFrame()

    print(f"   Generated {len(all_binary_preds)} predictions for {len(all_ids)} IDs.")
    # Ensure all_ids and all_binary_preds have the same length
    min_len = min(len(all_ids), len(all_binary_preds))
    if len(all_ids) != len(all_binary_preds):
        print(f"Warning: Mismatch in length of IDs ({len(all_ids)}) and predictions ({len(all_binary_preds)}). Truncating to {min_len}.")
    
    sub_df = pd.DataFrame({"id": all_ids[:min_len], "label": all_binary_preds[:min_len]})
    
    try:
        sub_df['id_numeric'] = pd.to_numeric(sub_df['id'], errors='coerce')
        if not sub_df['id_numeric'].isna().any(): 
             sub_df = sub_df.sort_values(by='id_numeric').drop(columns=['id_numeric'])
        else: 
             sub_df = sub_df.drop(columns=['id_numeric'])
             sub_df = sub_df.sort_values(by="id", key=lambda col: col.astype(str)) # Robust string sort
    except Exception: 
        print("   ⚠️ Warning: Could not sort submission by ID.")

    try:
        submission_path.parent.mkdir(parents=True, exist_ok=True)
        sub_df.to_csv(submission_path, index=False)
        print(f"📄 Saved submission ({len(sub_df)} rows) → {submission_path}")
    except Exception as e:
        print(f"❌ Error saving submission file to {submission_path}: {e}")

    return sub_df
