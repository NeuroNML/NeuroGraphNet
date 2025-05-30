from collections import OrderedDict, defaultdict # Added defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # type: ignore
from torchmetrics.functional import accuracy, f1_score, auroc 

# Assuming Data and Batch are from PyTorch Geometric if used with GraphEEGDataset
from torch_geometric.data import Batch as PyGBatch 
from torch_geometric.utils import to_dense_batch # For handling non-GNN models


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
    checkpoint = torch.load(path, map_location=device, weights_only=False) 
    state_dict = checkpoint['model_state_dict']

    wrapped = isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

    if wrapped and not has_module_prefix:
        model.module.load_state_dict(state_dict)
    elif not wrapped and has_module_prefix:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "", 1)
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    if optimizer and checkpoint.get('optimizer_state_dict'):
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("   - Optimizer state loaded.")
        except Exception as e:
            print(f"   - Warning: Could not load optimizer state: {e}")
            
    print("   - Model state loaded.")
    return checkpoint


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
    use_gnn: bool = True, # Added: True if GNN model, False for LSTM/CNN using PyG Batch
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Generic training loop.
    If use_gnn=False, it reshapes PyG Batch data for standard sequence models like LSTMs.
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
            loaded_train_hist = checkpoint.get('train_history', {})
            loaded_val_hist = checkpoint.get('val_history', {})
            for key, val_list in loaded_train_hist.items(): train_history[key].extend(val_list)
            for key, val_list in loaded_val_hist.items(): val_history[key].extend(val_list)

            start_epoch = checkpoint.get('epoch', 0) # No +1 here, loop range handles it
            loaded_best_score = checkpoint.get('best_score')
            if loaded_best_score is not None: best_score = loaded_best_score
            
            print(f"   Checkpoint loaded. Resuming from epoch {start_epoch + 1}. Best '{monitor}' score: {best_score:.4f}")
            print(f"   ⚠️ Warning: Training already completed up to epoch {start_epoch + 1}.")
            return dict(train_history), dict(val_history)
        except Exception as e:
            print(f"   ⚠️ Could not load checkpoint: {e}. Starting training from scratch.")
            save_path.unlink(missing_ok=True) 
    elif overwrite and save_path.exists():
         print(f"   Overwrite enabled: Removed existing checkpoint at {save_path}")
         save_path.unlink()

    bad_epochs = 0
    print(f"💪 Starting training from epoch {start_epoch + 1} to {num_epochs}...")
    # Corrected initial for tqdm if resuming
    pbar = tqdm(range(start_epoch + 1, num_epochs + 1), desc="Epochs", ncols=120, initial=start_epoch + 1, total=num_epochs) 
    
    for epoch in pbar:
        model.train()
        epoch_train_loss = 0.0
        all_train_preds, all_train_targets = [], []

        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            
            y_targets = batch_data.y 
            if y_targets is None: 
                 continue 
            y_targets = y_targets.float().unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            if use_gnn:
                # GNN model call: assumes model.forward(x, edge_index, batch_vector)
                # or model.forward(batch_data_object) if the model handles it internally
                if not (hasattr(batch_data, 'x') and hasattr(batch_data, 'edge_index')):
                     raise ValueError("For GNN mode, batch_data must have 'x' and 'edge_index'.")
                logits = model(batch_data.x.float(), 
                               batch_data.edge_index, 
                               batch_data.batch if hasattr(batch_data, 'batch') else None)
            else:
                # Non-GNN model (e.g., LSTM, CNN) using data from PyG Batch
                # batch_data.x is [total_nodes_in_batch, num_node_features]
                # For EEG, num_nodes = num_channels, num_node_features = num_timesteps
                # So, batch_data.x is [batch_size * num_channels, num_timesteps]
                # We need to reshape to (batch_size, num_channels, num_timesteps) using to_dense_batch
                # then permute for LSTM to (batch_size, num_timesteps, num_channels)
                if not hasattr(batch_data, 'x'):
                    raise ValueError("Batch data missing 'x' attribute for non-GNN mode.")
                
                # x_dense shape: (batch_size, max_nodes_in_a_graph, node_features)
                # For GraphEEGDataset: (batch_size, num_channels, num_timesteps)
                x_dense, mask = to_dense_batch(batch_data.x.float(), 
                                               batch_data.batch if hasattr(batch_data, 'batch') else None, 
                                               fill_value=0)
                
                # Permute for LSTM/CNN: (batch_size, num_timesteps, num_channels)
                # This assumes your LSTM's input_dim matches num_channels
                x_permuted = x_dense.permute(0, 2, 1) 
                logits = model(x_permuted)

            loss = criterion(logits, y_targets)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_train_loss += loss.item()
            all_train_preds.append(logits.sigmoid().detach().cpu())
            all_train_targets.append(y_targets.detach().cpu())

        avg_train_loss = epoch_train_loss / max(1, len(train_loader)) # Avoid division by zero
        train_history['loss'].append(avg_train_loss)
        
        if all_train_preds:
            preds_cat_train = torch.cat(all_train_preds)
            targets_cat_train = torch.cat(all_train_targets).int()
            try:
                train_history['acc'].append(accuracy(preds_cat_train, targets_cat_train, task="binary").item())
                train_history['f1'].append(f1_score(preds_cat_train, targets_cat_train, task="binary").item())
                train_history['auroc'].append(auroc(preds_cat_train, targets_cat_train, task="binary").item())
            except ValueError as e: 
                print(f"Warning (Train Metrics, Epoch {epoch}): {e}. Appending 0.0 for metrics.")
                train_history['acc'].append(0.0)
                train_history['f1'].append(0.0)
                train_history['auroc'].append(0.0)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        all_val_preds, all_val_targets = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                y_targets = batch_data.y
                if y_targets is None: continue
                y_targets = y_targets.float().unsqueeze(1)

                if use_gnn:
                    if not (hasattr(batch_data, 'x') and hasattr(batch_data, 'edge_index')):
                         raise ValueError("For GNN mode, batch_data must have 'x' and 'edge_index'.")
                    logits = model(batch_data.x.float(), 
                                   batch_data.edge_index, 
                                   batch_data.batch if hasattr(batch_data, 'batch') else None)
                else:
                    x_dense, mask = to_dense_batch(batch_data.x.float(), 
                                                   batch_data.batch if hasattr(batch_data, 'batch') else None, 
                                                   fill_value=0)
                    x_permuted = x_dense.permute(0, 2, 1)
                    logits = model(x_permuted)

                epoch_val_loss += criterion(logits, y_targets).item()
                all_val_preds.append(logits.sigmoid().cpu())
                all_val_targets.append(y_targets.cpu())

        avg_val_loss = epoch_val_loss / max(1, len(val_loader)) # Avoid division by zero
        val_history['loss'].append(avg_val_loss)
        current_score = avg_val_loss 

        if all_val_preds: 
            preds_cat_val = torch.cat(all_val_preds)
            targets_cat_val = torch.cat(all_val_targets).int()
            try:
                val_acc = accuracy(preds_cat_val, targets_cat_val, task="binary").item()
                val_f1 = f1_score(preds_cat_val, targets_cat_val, task="binary").item()
                val_auroc = auroc(preds_cat_val, targets_cat_val, task="binary").item()
                val_history['acc'].append(val_acc)
                val_history['f1'].append(val_f1)
                val_history['auroc'].append(val_auroc)

                if monitor == "val_auroc": current_score = val_auroc
                elif monitor == "val_f1": current_score = val_f1
                elif monitor == "val_acc": current_score = val_acc
            except ValueError as e:
                print(f"Warning (Validation Metrics, Epoch {epoch}): {e}. Using val_loss for monitoring.")
                val_history['acc'].append(0.0)
                val_history['f1'].append(0.0)
                val_history['auroc'].append(0.0)
                current_score = avg_val_loss 
        else: 
            current_score = avg_val_loss

        improved = False
        if monitor in ["val_loss", "loss"]:
            if current_score < best_score:
                best_score = current_score
                improved = True
        else: 
            if current_score > best_score:
                best_score = current_score
                improved = True

        if improved:
            bad_epochs = 0
            _save(model, save_path, dict(train_history), dict(val_history), optimizer.state_dict(), epoch, best_score)
        else:
            bad_epochs += 1

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_score) 
            else:
                scheduler.step()
        
        postfix_dict = {
            "train_loss": f"{train_history['loss'][-1]:.4f}" if train_history['loss'] else "N/A",
            "val_loss": f"{val_history['loss'][-1]:.4f}" if val_history['loss'] else "N/A",
            f"best_{monitor}": f"{best_score:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "bad_epochs": f"{bad_epochs}/{patience}",
            "save": "✓" if improved else "-"
        }
        if val_history['auroc']: postfix_dict["val_auroc"] = f"{val_history['auroc'][-1]:.4f}"
        if val_history['f1']: postfix_dict["val_f1"] = f"{val_history['f1'][-1]:.4f}"
        pbar.set_postfix(postfix_dict)

        if patience > 0 and bad_epochs >= patience:
            pbar.write(f"🛑 Early stopping: no '{monitor}' improvement in {patience} epochs.")
            break
    
    pbar.close()
    print("\n✅ Training complete.")
    
    if save_path.exists():
        print(f"↩️ Loading best model state from {save_path} for return.")
        _load(model, save_path, device) 
    else:
        print("   ⚠️ No checkpoint was saved during training.")

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
) -> pd.DataFrame:
    """
    Loads checkpoint, runs inference, and writes submission CSV.
    If use_gnn=False, reshapes PyG Batch data for standard sequence models.
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
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            batch_data = batch_data.to(device)
            
            current_ids = []
            if not hasattr(batch_data, id_attribute):
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
                if not hasattr(batch_data, 'x'):
                    raise ValueError("Batch data missing 'x' attribute for non-GNN mode.")
                x_dense, mask = to_dense_batch(batch_data.x.float(), 
                                               batch_data.batch if hasattr(batch_data, 'batch') else None, 
                                               fill_value=0)
                x_permuted = x_dense.permute(0, 2, 1)
                logits = model(x_permuted)

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
        if sub_df['id_numeric'].notna().all(): 
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
