from collections import OrderedDict, defaultdict # Added defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import time  # Add time import
import logging  # Add logging import
import inspect  # Add inspect for checking function signatures
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchmetrics.functional import accuracy, f1_score, auroc, precision, recall
from torch_geometric.utils import to_dense_batch
import numpy as np  # For median/IQR calculations
from src.data.dataset_graph import GraphEEGDataset 

# import required dataloaders
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import DataLoader

# k-fold
from sklearn.model_selection import KFold, StratifiedKFold
from src.utils.timeseries_eeg_dataset import TimeseriesEEGDataset

# Import wandb with optional fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.") 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
    train_loader: Union[torch.utils.data.DataLoader, GeoDataLoader],
    val_loader: Union[torch.utils.data.DataLoader, GeoDataLoader],
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
    wandb_config: Optional[Dict[str, Any]] = None,  # Wandb configuration
    wandb_project: Optional[str] = None,  # Wandb project name
    wandb_run_name: Optional[str] = None,  # Wandb run name
    log_wandb: bool = True,  # Whether to log to wandb
    try_load_checkpoint: bool = True,
    # pass original database to log information to wandb
    original_dataset: Optional[GraphEEGDataset] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Generic training loop.
    If use_gnn=False, it assumes data comes from a standard DataLoader (e.g., with TensorDataset)
    and batch_data is a tuple (input_features, labels).
    If use_gnn=True, it expects PyG Batch objects.
    """
    logger.info("Starting training setup...")
    logger.info(f"Model type: {'GNN' if use_gnn else 'Standard'}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {train_loader.batch_size}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Patience: {patience}")
    logger.info(f"Monitor metric: {monitor}")
    
    # Initialize wandb if requested and available
    wandb_run = None
    if log_wandb and WANDB_AVAILABLE:
        logger.info("Initializing wandb...")
        
        # Use checkpoint file name as run name if not provided
        if wandb_run_name is None:
            wandb_run_name = save_path.stem  # Get filename without extension
        
        # Prepare wandb config
        config = {
            'model_type': 'GNN' if use_gnn else 'Standard',
            'num_epochs': num_epochs,
            'patience': patience,
            'monitor': monitor,
            'grad_clip': grad_clip,
            'batch_size': train_loader.batch_size,
            'optimizer': type(optimizer).__name__,
            'lr': optimizer.param_groups[0]['lr'],
            'scheduler': type(scheduler).__name__ if scheduler else None,
        }
        
        # Add user-provided config
        if wandb_config:
            config.update(wandb_config)

        try:
            wandb_run = wandb.init( # type: ignore
                project=wandb_project or "neuro-graph-net",
                name=wandb_run_name,
                config=config,
                reinit=True
            )
            # Watch the model for gradient and parameter tracking
            wandb.watch(model, log='all', log_freq=100) # type: ignore
            print(f"🔗 Wandb initialized: {wandb.run.name}") # type: ignore
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize wandb: {e}")
            log_wandb = False
    elif log_wandb and not WANDB_AVAILABLE:
        print("⚠️ Warning: wandb logging requested but wandb not available")
        log_wandb = False
    
    start_epoch = 0
    train_history = defaultdict(list)
    val_history = defaultdict(list)

    if monitor in ["val_loss", "loss"]:
        best_score = float("inf")
    else:
        best_score = -float("inf")

    if save_path.exists() and not overwrite and try_load_checkpoint:
        print(f"🚀 Attempting to load checkpoint from {save_path}...")
        try:
            checkpoint = _load(model, save_path, device, optimizer)
            # Restore histories
            loaded_train_hist = checkpoint.get('train_history', {})
            loaded_val_hist = checkpoint.get('val_history', {})
            for key, val_list in loaded_train_hist.items():
                train_history[key].extend(val_list)
            for key, val_list in loaded_val_hist.items():
                val_history[key].extend(val_list)

            start_epoch = checkpoint.get('epoch', 0)
            loaded_best_score = checkpoint.get('best_score')
            if loaded_best_score is not None:
                best_score = loaded_best_score
            
            if start_epoch >= num_epochs:
                 print(f" 🏁 Training already completed up to epoch {start_epoch}. Final best '{monitor}': {best_score:.4f}")
                 return dict(train_history), dict(val_history)
            print(f" ✅ Checkpoint loaded. Resuming from epoch {start_epoch + 1}. Best '{monitor}' score: {best_score:.4f}")
        except Exception as e:
            print(f" ⚠️ Could not load checkpoint: {e}. Starting training from scratch.")
            save_path.unlink(missing_ok=True)
            start_epoch = 0 # Ensure start_epoch is reset
            # Re-initialize best_score based on monitor
            best_score = float("inf") if monitor in ["val_loss", "loss"] else -float("inf")
    elif overwrite and save_path.exists():
        print(f" 🗑️ Overwrite enabled: Removed existing checkpoint at {save_path}")
        save_path.unlink()

    train_loader_to_iterate = train_loader
    total_batches = len(train_loader_to_iterate)
    logger.info(f"Total training batches per epoch: {total_batches}")

    bad_epochs = 0
    epoch = start_epoch  # Initialize epoch variable
    logger.info(f"Starting training from epoch {start_epoch + 1} to {num_epochs}")
    pbar = tqdm(range(start_epoch + 1, num_epochs + 1), desc="Epochs", ncols=120, initial=start_epoch + 1, total=num_epochs)

    for epoch in pbar:
        epoch_start_time = time.time()
        model.train()
        epoch_train_loss = 0.0
        all_train_preds, all_train_targets = [], []
        
        logger.info(f"\nEpoch {epoch}/{num_epochs} - Training phase")
        batch_times = []
        
        for batch_idx, data_batch_item in enumerate(train_loader_to_iterate):
            batch_start_time = time.time()
            
            if batch_idx % 10 == 0:  # Log every 10 batches
                logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
            
            optimizer.zero_grad(set_to_none=True)

            if use_gnn:
                curr_batch = data_batch_item.to(device)
                y_targets = curr_batch.y.reshape(-1, 1)
                if y_targets is None:
                    logger.warning(f"Batch {batch_idx} has no targets, skipping...")
                    continue
                    
                if batch_idx == 0:  # Log shapes only for first batch
                    logger.info(f"Batch shapes - x: {curr_batch.x.shape}, edge_index: {curr_batch.edge_index.shape}, y: {y_targets.shape}")
                
                if not (hasattr(curr_batch, 'x') and hasattr(curr_batch, 'edge_index')):
                    raise ValueError("For GNN mode, batch_data must have 'x' and 'edge_index'.")
                
                try:
                    if hasattr(curr_batch, 'graph_features'):
                        logits = model(curr_batch.x.float(), curr_batch.edge_index, curr_batch.batch, curr_batch.graph_features)
                    else:
                        logits = model(curr_batch.x.float(), curr_batch.edge_index, curr_batch.batch)

                    # unsqueeze y_targets to match the shape of the logits. used later!
                except Exception as e:
                    logger.error(f"Error in forward pass for batch {batch_idx}: {str(e)}")
                    logger.error(f"Edge index shape: {curr_batch.edge_index.shape}")
                    logger.error(f"Edge index content: {curr_batch.edge_index}")
                    raise
            else:
                curr_batch = data_batch_item.to(device)
                y_targets = curr_batch.y.reshape(-1, 1)
                if y_targets is None:
                    continue
                logits = model(curr_batch.x.float())

            # Compute loss on raw logits
            loss = criterion(logits, y_targets)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # convert logits to binary predictions
            probs = logits.sigmoid().squeeze() # [batch_size, 1] -> [batch_size]
            
            # Validate probability ranges
            if torch.any(probs < 0) or torch.any(probs > 1):
                logger.warning(f"Batch {batch_idx}: Probabilities outside [0,1] range detected. Clamping values.")
                probs = torch.clamp(probs, 0, 1)

            # store round results
            epoch_train_loss += loss.item()
            # Store probabilities instead of binary predictions
            all_train_preds.append(probs.cpu())
            all_train_targets.append(y_targets.squeeze(1).int().cpu())  # shape: (batch_size)

            # assert that all_train_preds and all_train_targets have the same shape
            assert all_train_preds[0].shape == all_train_targets[0].shape, f"all_train_preds and all_train_targets have different shapes: {all_train_preds[0].shape} != {all_train_targets[0].shape}"
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            if batch_idx % 10 == 0:  # Log every 10 batches
                avg_batch_time = sum(batch_times[-10:]) / min(10, len(batch_times))
                logger.info(f"Batch {batch_idx + 1}/{total_batches} - Loss: {loss.item():.4f} - Avg batch time: {avg_batch_time:.2f}s")

        # Epoch training metrics
        avg_train_loss = epoch_train_loss / max(1, len(train_loader_to_iterate))
        train_history['loss'].append(avg_train_loss)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"\nEpoch {epoch} training completed in {epoch_time:.2f}s")
        logger.info(f"Average training loss: {avg_train_loss:.4f}")

        if all_train_preds and all_train_targets:
            preds_cat_train = torch.cat(all_train_preds)
            targets_cat_train = torch.cat(all_train_targets).int()
            try:
                # Compute metrics with explicit threshold
                threshold = 0.5
                binary_preds = (preds_cat_train > threshold).int()
                
                # Binary classification metrics on binary predictions
                train_acc = accuracy(binary_preds, targets_cat_train, task="binary").item()
                train_f1 = f1_score(binary_preds, targets_cat_train, task="binary").item()
                train_precision = precision(binary_preds, targets_cat_train, task="binary").item()
                train_recall = recall(binary_preds, targets_cat_train, task="binary").item()
                
                # AUROC on probabilities
                train_auroc_result = auroc(preds_cat_train, targets_cat_train, task="binary")
                train_auroc = train_auroc_result.item() if train_auroc_result is not None else 0.0
                
                # For macro F1 (same as binary F1 for binary classification, but explicit)
                train_macro_f1 = f1_score(binary_preds, targets_cat_train, task="binary", average="macro").item()
                
                # Store all metrics
                train_history['acc'].append(train_acc)
                train_history['f1'].append(train_f1)
                train_history['precision'].append(train_precision)
                train_history['recall'].append(train_recall)
                train_history['auroc'].append(train_auroc)
                train_history['macro_f1'].append(train_macro_f1)
                
                # Log metric values for debugging
                print(
                    f"Epoch {epoch} training metrics - Acc: {train_acc:.4f}, F1: {train_f1:.4f}, "
                    f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, "
                    f"AUROC: {train_auroc:.4f}, Macro F1: {train_macro_f1:.4f}"
                )
            except ValueError as e:
                logger.error(f"Error computing metrics (Epoch {epoch}): {str(e)}")
                logger.error(f"Preds shape: {preds_cat_train.shape}, Targets shape: {targets_cat_train.shape}")
                logger.error(f"Preds range: [{preds_cat_train.min():.4f}, {preds_cat_train.max():.4f}]")
                logger.error(f"Targets unique values: {torch.unique(targets_cat_train).tolist()}")
                # Append zeros for all metrics
                train_history['acc'].append(0.0)
                train_history['f1'].append(0.0)
                train_history['precision'].append(0.0)
                train_history['recall'].append(0.0)
                train_history['auroc'].append(0.0)
                train_history['macro_f1'].append(0.0)
        else: # Handles empty train_loader_to_iterate or if all batches were skipped
             train_history['acc'].append(0.0)
             train_history['f1'].append(0.0)
             train_history['precision'].append(0.0)
             train_history['recall'].append(0.0)
             train_history['auroc'].append(0.0)
             train_history['macro_f1'].append(0.0)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        all_val_preds, all_val_targets = [], []
        # Add tracking of per-patient predictions and targets
        patient_preds = defaultdict(list)
        patient_targets = defaultdict(list)
        
        with torch.no_grad():
            for data_batch_item_val in val_loader: # Use a different variable name
                if use_gnn:
                    curr_batch = data_batch_item_val.to(device)
                    y_targets = curr_batch.y.reshape(-1, 1)
                    if y_targets is None: continue
                    if not (hasattr(curr_batch, 'x') and hasattr(curr_batch, 'edge_index')):
                         raise ValueError("For GNN mode, batch_data must have 'x' and 'edge_index'.")
                    
                    # Extract patient information for metrics calculation
                    patient_ids = None
                    if hasattr(curr_batch, 'patient'):
                        patient_ids = curr_batch.patient.cpu().tolist() if isinstance(curr_batch.patient, torch.Tensor) else curr_batch.patient

                    # Extract graph features if available (same as training)
                    if hasattr(curr_batch, 'graph_features'):
                        logits = model(curr_batch.x.float(), curr_batch.edge_index, curr_batch.batch, curr_batch.graph_features)
                    else:
                        logits = model(curr_batch.x.float(), curr_batch.edge_index, curr_batch.batch)
                    
                else: # Non-GNN
                    # Extract patient information (if available)
                    patient_ids = None
                    if hasattr(data_batch_item_val, 'patient'):
                        patient_ids = data_batch_item_val.patient.cpu().tolist() if isinstance(data_batch_item_val.patient, torch.Tensor) else data_batch_item_val.patient
                    
                    # get current batch and targets
                    curr_batch = data_batch_item_val.to(device) # type: ignore
                    y_targets = curr_batch.y.reshape(-1, 1)

                    # inference with model
                    logits = model(curr_batch.x.float())

                # compute loss on validation set using raw logits
                epoch_val_loss += criterion(logits, y_targets).item()

                # Convert logits to binary predictions
                probs = logits.sigmoid().squeeze()  # [batch_size, 1] -> [batch_size]
                
                # Validate probability ranges
                if torch.any(probs < 0) or torch.any(probs > 1):
                    logger.warning(f"Val Batch: Probabilities outside [0,1] range detected. Clamping values.")
                    probs = torch.clamp(probs, 0, 1)

                # store round results
                probs_cpu = probs.cpu()
                targets_cpu = y_targets.squeeze(1).int().cpu()
                assert probs_cpu.shape == targets_cpu.shape, f"Probs and targets have different shapes: {probs_cpu.shape} != {targets_cpu.shape}"

                all_val_preds.append(probs_cpu)
                all_val_targets.append(targets_cpu)

                # Store predictions and targets by patient ID
                if patient_ids is not None:
                    # Ensure we have the same number of patient IDs as predictions
                    if len(patient_ids) != len(probs_cpu):
                        logger.warning(f"Mismatch between patient IDs ({len(patient_ids)}) and predictions ({len(probs_cpu)})")
                        # Use the smaller length
                        min_len = min(len(patient_ids), len(probs_cpu))
                        patient_ids = patient_ids[:min_len]
                        probs_cpu = probs_cpu[:min_len]
                        targets_cpu = targets_cpu[:min_len]
                    
                    # Store predictions and targets by patient ID
                    for i, patient_id in enumerate(patient_ids):
                        patient_preds[patient_id].append(probs_cpu[i].item())
                        patient_targets[patient_id].append(targets_cpu[i].item())
        
        avg_val_loss = epoch_val_loss / max(1, len(val_loader))
        val_history['loss'].append(avg_val_loss)
        current_score_val = avg_val_loss # Default for monitor

        if all_val_preds and all_val_targets:
            preds_cat_val = torch.cat(all_val_preds)
            targets_cat_val = torch.cat(all_val_targets).int()
            try:
                # Compute metrics with explicit threshold
                threshold = 0.5
                binary_preds = (preds_cat_val > threshold).int()
                
                # Binary classification metrics on binary predictions
                val_acc = accuracy(binary_preds, targets_cat_val, task="binary").item()
                val_f1 = f1_score(binary_preds, targets_cat_val, task="binary").item()
                val_precision = precision(binary_preds, targets_cat_val, task="binary").item()
                val_recall = recall(binary_preds, targets_cat_val, task="binary").item()
                
                # AUROC on probabilities
                val_auroc_result = auroc(preds_cat_val, targets_cat_val, task="binary")
                val_auroc = val_auroc_result.item() if val_auroc_result is not None else 0.0
                
                # For macro F1 (same as binary F1 for binary classification, but explicit)
                val_macro_f1 = f1_score(binary_preds, targets_cat_val, task="binary", average="macro").item()
                
                # Store all metrics
                val_history['acc'].append(val_acc)
                val_history['f1'].append(val_f1)
                val_history['precision'].append(val_precision)
                val_history['recall'].append(val_recall)
                val_history['auroc'].append(val_auroc)
                val_history['macro_f1'].append(val_macro_f1)
                
                # Compute per-patient F1 scores and median if we have patient-level data
                if patient_preds:
                    patient_f1_scores = {}
                    patient_precision_scores = {}
                    patient_recall_scores = {}
                    
                    for patient_id, predictions in patient_preds.items():
                        if patient_id in patient_targets:
                            targets = patient_targets[patient_id]
                            if len(predictions) == len(targets):
                                # Convert to tensors for torchmetrics
                                p_preds = torch.tensor(predictions)
                                p_targets = torch.tensor(targets).int()
                                
                                # Binarize predictions with threshold
                                p_binary_preds = (p_preds > threshold).int()
                                
                                # Compute patient-specific metrics
                                try:
                                    p_f1 = f1_score(p_binary_preds, p_targets, task="binary").item()
                                    p_precision = precision(p_binary_preds, p_targets, task="binary").item()
                                    p_recall = recall(p_binary_preds, p_targets, task="binary").item()
                                    
                                    patient_f1_scores[patient_id] = p_f1
                                    patient_precision_scores[patient_id] = p_precision
                                    patient_recall_scores[patient_id] = p_recall
                                except Exception as e:
                                    logger.warning(f"Could not compute F1 for patient {patient_id}: {e}")
                    
                    # Calculate median patient metrics
                    if patient_f1_scores:
                        f1_values = list(patient_f1_scores.values())
                        precision_values = list(patient_precision_scores.values())
                        recall_values = list(patient_recall_scores.values())
                        
                        median_patient_f1 = float(np.median(f1_values))
                        median_patient_precision = float(np.median(precision_values))
                        median_patient_recall = float(np.median(recall_values))
                        
                        # Store median values in validation history
                        val_history['median_patient_f1'].append(median_patient_f1)
                        val_history['median_patient_precision'].append(median_patient_precision)
                        val_history['median_patient_recall'].append(median_patient_recall)
                        
                        # Log median values
                        logger.info(f"Median patient F1: {median_patient_f1:.4f}, "
                                   f"Precision: {median_patient_precision:.4f}, "
                                   f"Recall: {median_patient_recall:.4f}")
                    else:
                        val_history['median_patient_f1'].append(0.0)
                        val_history['median_patient_precision'].append(0.0)
                        val_history['median_patient_recall'].append(0.0)

                # Log metric values for debugging
                logger.debug(f"Epoch {epoch} validation metrics - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, "
                           f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
                           f"AUROC: {val_auroc:.4f}, Macro F1: {val_macro_f1:.4f}")

                if monitor == "val_auroc": current_score_val = val_auroc
                elif monitor == "val_f1": current_score_val = val_f1
                elif monitor == "val_acc": current_score_val = val_acc
                elif monitor == "val_precision": current_score_val = val_precision
                elif monitor == "val_recall": current_score_val = val_recall
                elif monitor == "val_macro_f1": current_score_val = val_macro_f1
                # else monitor is "val_loss", already set
            except ValueError as e:
                logger.error(f"Error computing validation metrics (Epoch {epoch}): {str(e)}")
                logger.error(f"Preds shape: {preds_cat_val.shape}, Targets shape: {targets_cat_val.shape}")
                logger.error(f"Preds range: [{preds_cat_val.min():.4f}, {preds_cat_val.max():.4f}]")
                logger.error(f"Targets unique values: {torch.unique(targets_cat_val).tolist()}")
                # Append zeros for all metrics
                val_history['acc'].append(0.0)
                val_history['f1'].append(0.0)
                val_history['precision'].append(0.0)
                val_history['recall'].append(0.0)
                val_history['auroc'].append(0.0)
                val_history['macro_f1'].append(0.0)
                # current_score_val remains avg_val_loss
        else: # Handles empty val_loader or if all batches were skipped
            val_history['acc'].append(0.0)
            val_history['f1'].append(0.0)
            val_history['precision'].append(0.0)
            val_history['recall'].append(0.0)
            val_history['auroc'].append(0.0)
            val_history['macro_f1'].append(0.0)
            # current_score_val remains avg_val_loss

        # Log metrics to wandb
        if log_wandb and wandb_run:
            wandb_metrics = {
                'epoch': epoch,
                'train/loss': train_history['loss'][-1] if train_history['loss'] else 0.0,
                'train/accuracy': train_history['acc'][-1] if train_history['acc'] else 0.0,
                'train/f1': train_history['f1'][-1] if train_history['f1'] else 0.0,
                'train/precision': train_history['precision'][-1] if train_history['precision'] else 0.0,
                'train/recall': train_history['recall'][-1] if train_history['recall'] else 0.0,
                'train/auroc': train_history['auroc'][-1] if train_history['auroc'] else 0.0,
                'train/macro_f1': train_history['macro_f1'][-1] if train_history['macro_f1'] else 0.0,
                'val/loss': val_history['loss'][-1] if val_history['loss'] else 0.0,
                'val/accuracy': val_history['acc'][-1] if val_history['acc'] else 0.0,
                'val/f1': val_history['f1'][-1] if val_history['f1'] else 0.0,
                'val/precision': val_history['precision'][-1] if val_history['precision'] else 0.0,
                'val/recall': val_history['recall'][-1] if val_history['recall'] else 0.0,
                'val/auroc': val_history['auroc'][-1] if val_history['auroc'] else 0.0,
                'val/macro_f1': val_history['macro_f1'][-1] if val_history['macro_f1'] else 0.0,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'bad_epochs': bad_epochs,
                f'best_{monitor}': best_score,
            }
            
            # Add patient-level metrics if available
            if 'median_patient_f1' in val_history and val_history['median_patient_f1']:
                wandb_metrics['val/median_patient_f1'] = val_history['median_patient_f1'][-1]
                wandb_metrics['val/median_patient_precision'] = val_history['median_patient_precision'][-1] if val_history.get('median_patient_precision') else 0.0
                wandb_metrics['val/median_patient_recall'] = val_history['median_patient_recall'][-1] if val_history.get('median_patient_recall') else 0.0
            
            try:
                wandb.log(wandb_metrics) # type: ignore
            except Exception as e:
                print(f"⚠️ Warning: Could not log to wandb: {e}")

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
            "train_loss": f"{train_history['loss'][-1]:.4f}",
            "val_loss": f"{val_history['loss'][-1]:.4f}",
            f"best_{monitor}": f"{best_score:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "bad_epochs": f"{bad_epochs}/{patience}",
            "save": "✓" if improved else "-"
        }
        if val_history.get('auroc'): postfix_dict["val_auroc"] = f"{val_history['auroc'][-1]:.4f}"
        if val_history.get('f1'): postfix_dict["val_f1"] = f"{val_history['f1'][-1]:.4f}"
        if val_history.get('precision'): postfix_dict["val_prec"] = f"{val_history['precision'][-1]:.4f}"
        if val_history.get('recall'): postfix_dict["val_rec"] = f"{val_history['recall'][-1]:.4f}"
        if val_history.get('median_patient_f1'): postfix_dict["med_pat_f1"] = f"{val_history['median_patient_f1'][-1]:.4f}"
        pbar.set_postfix(postfix_dict)

        if patience > 0 and bad_epochs >= patience:
            logger.info(f"Early stopping triggered: no '{monitor}' improvement in {patience} epochs")
            break
    
    pbar.close()
    logger.info("Training completed successfully!")
    
    # Log final results
    final_metrics = {
        "final_best_score": best_score,
        "final_epoch": epoch,
        "total_bad_epochs": bad_epochs
    }
    logger.info(f"Final results: {final_metrics}")
    
    return dict(train_history), dict(val_history)

def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device,
    checkpoint_path: Path, 
    submission_path: Path,
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
        print(f"   Please check the path and ensure the model was saved correctly.")
        return pd.DataFrame({"id": [], "label": []})  # Return empty DataFrame with correct columns

    try:
        _ = _load(model, checkpoint_path, device)
    except Exception as e:
        print(f"❌ Error loading model checkpoint for evaluation: {e}")
        print(f"   The checkpoint file might be corrupted or incompatible with the model.")
        return pd.DataFrame({"id": [], "label": []})  # Return empty DataFrame with correct columns
        
    model.to(device).eval()

    all_ids, all_binary_preds = [], []
    print("🧪 Performing inference on the test set...")
    with torch.no_grad():
        # for batch_data in tqdm(test_loader, desc="Evaluating"):
        for batch_data in test_loader:
            # Handle different batch formats for device transfer
            if isinstance(batch_data, (tuple, list)):
                # For tuple/list format, move elements to device individually
                batch_data = tuple(item.to(device) if hasattr(item, 'to') else item for item in batch_data)
            else:
                # For PyG Batch objects
                batch_data = batch_data.to(device)
            
            # Handle ID extraction based on batch format
            current_ids = []
            if isinstance(batch_data, (tuple, list)):
                # For tuple/list format, we need to handle this differently
                # This suggests the test loader might not include IDs in tuple format
                raise ValueError("Test batch data is in tuple format but ID extraction requires object with 'id' attribute. Check test dataset configuration.")
            elif not hasattr(batch_data, "id"):
                raise ValueError(f"Batch object missing ID attribute 'id'.")
            else:
                if isinstance(batch_data.id, torch.Tensor):
                    current_ids.extend(batch_data.id.cpu().tolist())
                else:
                    current_ids.extend(batch_data.id)

            if use_gnn:
                if not (hasattr(batch_data, 'x') and hasattr(batch_data, 'edge_index')):
                     raise ValueError("For GNN mode, batch_data must have 'x' and 'edge_index'.")
                
                # Extract graph features if available
                if hasattr(batch_data, 'graph_features'):
                    logits = model(batch_data.x.float(), batch_data.edge_index, batch_data.batch, batch_data.graph_features)
                else:
                    logits = model(batch_data.x.float(), batch_data.edge_index, batch_data.batch)
            else:
                # Handle non-GNN models based on input_type
                if input_type == 'feature':
                    # Expected input shape: (batch_size, features)
                    if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                        # Handle tuple format (x, y) from standard DataLoader
                        x_batch, _ = batch_data
                        x_batch = x_batch.to(device).float()
                        logits = model(x_batch)
                    elif hasattr(batch_data, 'x') and not isinstance(batch_data, (tuple, list)):
                        # Handle PyG Batch format - convert to dense and aggregate
                        # Type guard: ensure it's not a tuple/list before accessing attributes
                        
                        # Convert PyG batch to dense format for feature processing
                        batch_attr = getattr(batch_data, 'batch', None)
                        x_dense, mask = to_dense_batch(batch_data.x.float(), 
                                                     batch_attr,
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
                    elif hasattr(batch_data, 'x') and not isinstance(batch_data, (tuple, list)):
                        # Handle PyG Batch format - reshape to signal format
                        # Type guard: ensure it's not a tuple/list before accessing attributes
                        
                        # Convert PyG batch to dense format
                        batch_attr = getattr(batch_data, 'batch', None)
                        x_dense, mask = to_dense_batch(batch_data.x.float(), 
                                                     batch_attr, 
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
            # Flatten the binary predictions to get single integers instead of lists
            binary_preds = (probs > threshold).int().flatten().tolist()
            all_binary_preds.extend(binary_preds)
            all_ids.extend(current_ids)
    if not all_ids:
        print("❌ No IDs were collected during evaluation. Cannot create submission file.")
        return pd.DataFrame()

    print(f"   Generated {len(all_binary_preds)} predictions for {len(all_ids)} IDs.")
    # Ensure all_ids and all_binary_preds have the same length
    min_len = min(len(all_ids), len(all_binary_preds))
    if len(all_ids) != len(all_binary_preds):
        print(f"⚠️ Warning: Mismatch in total length of IDs ({len(all_ids)}) and predictions ({len(all_binary_preds)})")
        print(f"   Truncating both to {min_len} entries to ensure consistency")
        # This shouldn't happen if we handled batch-level mismatches correctly, but just in case
        all_ids = all_ids[:min_len]
        all_binary_preds = all_binary_preds[:min_len]
    
    # create submission dataframe + sort by id to maintain the same order as the test set
    sub_df = pd.DataFrame({"id": all_ids, "label": all_binary_preds})
    
    # Sort by ID to maintain the same order as the test set
    sub_df = sub_df.sort_values(by="id", key=lambda col: col.astype(str)) # Robust string sort

    # save submission file
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    
    if len(sub_df) == 0:
        print(f"⚠️ Warning: No predictions were generated. Creating an empty submission file.")
        # Still save an empty file with the correct columns
        sub_df = pd.DataFrame({"id": [], "label": []})
    
    sub_df.to_csv(submission_path, index=False)
    print(f"📄 Saved submission ({len(sub_df)} rows) → {submission_path}")

    # return submission dataframe
    return sub_df

def train_k_fold(
    dataset: GraphEEGDataset | TimeseriesEEGDataset,
    labels: List[int],
    model_class: type,
    model_kwargs: Dict[str, Any],
    criterion: nn.Module,
    optimizer_class: type,
    optimizer_kwargs: Dict[str, Any],
    device: torch.device,
    save_dir: Path,
    k_folds: int = 5,
    stratified: bool = True,
    scheduler_class: Optional[type] = None,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    monitor: str = "val_f1",
    patience: int = 15,
    num_epochs: int = 100,
    grad_clip: float = 1.0,
    batch_size: int = 32,
    use_gnn: bool = True,
    wandb_config: Optional[Dict[str, Any]] = None,
    wandb_project: Optional[str] = None,
    log_wandb: bool = True,
    random_state: int = 42,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], List[Dict[str, Any]]]:
    """
    Perform k-fold cross-validation training.
    
    Args

        dataset: The dataset to split into k folds
        model_class: Class of the model to instantiate for each fold
        model_kwargs: Keyword arguments for model instantiation
        criterion: Loss function
        optimizer_class: Optimizer class (e.g., torch.optim.Adam)
        optimizer_kwargs: Keyword arguments for optimizer
        device: Device to train on
        save_dir: Directory to save fold checkpoints
        k_folds: Number of folds for cross-validation
        stratified: Whether to use stratified k-fold (maintains class distribution)
        scheduler_class: Optional learning rate scheduler class
        scheduler_kwargs: Keyword arguments for scheduler
        monitor: Metric to monitor for early stopping
        patience: Early stopping patience
        num_epochs: Maximum number of epochs per fold
        grad_clip: Gradient clipping value
        batch_size: Batch size for data loaders
        use_gnn: Whether using GNN models
        wandb_config: Configuration for wandb logging
        wandb_project: Wandb project name
        log_wandb: Whether to log to wandb
        random_state: Random seed for reproducible splits
        
    Returns:
        Tuple of (aggregated_train_history, aggregated_val_history, fold_results)
    """
    logger.info(f"Starting {k_folds}-fold cross-validation")
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Stratified: {stratified}")
    logger.info(f"Batch size: {batch_size}")
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize k-fold splitter
    if stratified and labels is not None:
        logger.info("Using stratified k-fold with label distribution")
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        # Convert to numpy array for sklearn compatibility
        indices_array = np.arange(len(dataset))
        labels_array = np.array(labels)
        splits = list(kfold.split(indices_array, labels_array))
    else:
        logger.info("Using regular k-fold (non-stratified)")
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        # Convert to numpy array for sklearn compatibility
        indices_array = np.arange(len(dataset))
        splits = list(kfold.split(indices_array))
    
    # Storage for results
    fold_results = []
    aggregated_train_history = defaultdict(list)
    aggregated_val_history = defaultdict(list)
    
    logger.info(f"Created {len(splits)} folds")
    logger.info("Folds: %s", splits)

    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx + 1}/{k_folds}")
        logger.info(f"{'='*60}")
        logger.info(f"Train samples: {len(train_indices)}")
        logger.info(f"Val samples: {len(val_indices)}")
        
        # Calculate class distribution for this fold
        train_labels = [labels[i] for i in train_indices]
        val_labels = [labels[i] for i in val_indices]
        train_pos_ratio = sum(train_labels) / len(train_labels) if train_labels else 0
        val_pos_ratio = sum(val_labels) / len(val_labels) if val_labels else 0
        logger.info(f"Train positive ratio: {train_pos_ratio:.3f}")
        logger.info(f"Val positive ratio: {val_pos_ratio:.3f}")
        
        # Create subset datasets
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        
        # Create data loaders
        if use_gnn:
            train_loader = GeoDataLoader(
                train_subset, 
                batch_size=batch_size, 
                follow_batch=['x'],
                shuffle=True,
                pin_memory=True,
                num_workers=4
            )
            val_loader = GeoDataLoader(
                val_subset, 
                batch_size=batch_size, 
                follow_batch=['x'],
                shuffle=False,
                pin_memory=True,
                num_workers=4
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_subset, 
                batch_size=batch_size, 
                shuffle=True,
                pin_memory=True,
                num_workers=4
            )
            val_loader = torch.utils.data.DataLoader(
                val_subset, 
                batch_size=batch_size, 
                shuffle=False,
                pin_memory=True,
                num_workers=4
            )
        
        # Initialize model for this fold
        model = model_class(**model_kwargs).to(device)
        logger.info(f"Initialized model: {model.__class__.__name__}")
        
        # Initialize optimizer
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        logger.info(f"Initialized optimizer: {optimizer.__class__.__name__}")
        
        # Initialize scheduler if provided
        scheduler = None
        if scheduler_class is not None:
            scheduler_kwargs_safe = scheduler_kwargs or {}
            scheduler = scheduler_class(optimizer, **scheduler_kwargs_safe)
            logger.info(f"Initialized scheduler: {scheduler.__class__.__name__}")
        
        # Set up paths for this fold
        fold_save_path = save_dir / f"fold_{fold_idx + 1}_best_model.pth"
        
        # Prepare wandb config for this fold
        fold_wandb_config = {
            'fold': fold_idx + 1,
            'k_folds': k_folds,
            'train_samples': len(train_indices),
            'val_samples': len(val_indices),
            'train_pos_ratio': train_pos_ratio,
            'val_pos_ratio': val_pos_ratio,
        }
        if wandb_config:
            fold_wandb_config.update(wandb_config)
        
        # Set wandb run name for this fold
        fold_wandb_run_name = f"fold_{fold_idx + 1}"
        if wandb_config and 'run_name' in wandb_config:
            fold_wandb_run_name = f"{wandb_config['run_name']}_fold_{fold_idx + 1}"
        
        try:
            # Train this fold
            logger.info(f"Starting training for fold {fold_idx + 1}")
            train_history, val_history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                save_path=fold_save_path,
                scheduler=scheduler,
                monitor=monitor,
                patience=patience,
                num_epochs=num_epochs,
                grad_clip=grad_clip,
                overwrite=True, # Always overwrite for k-fold
                use_gnn=use_gnn,
                wandb_config=fold_wandb_config,
                wandb_project=wandb_project,
                wandb_run_name=fold_wandb_run_name,
                log_wandb=log_wandb,
                try_load_checkpoint=False,  # Don't load checkpoint for k-fold
            )
            
            # Get best scores for this fold
            best_train_score = max(train_history.get(monitor.replace('val_', ''), [0])) if 'val_' in monitor else min(train_history.get('loss', [float('inf')]))
            best_val_score = max(val_history.get(monitor.replace('val_', ''), [0])) if 'val_' not in monitor else (
                max(val_history.get(monitor.replace('val_', ''), [0])) if monitor not in ['val_loss'] else min(val_history.get('loss', [float('inf')]))
            )
            
            # Store fold results
            fold_result = {
                'fold': fold_idx + 1,
                'train_samples': len(train_indices),
                'val_samples': len(val_indices),
                'train_pos_ratio': train_pos_ratio,
                'val_pos_ratio': val_pos_ratio,
                'best_train_score': best_train_score,
                'best_val_score': best_val_score,
                'train_history': train_history,
                'val_history': val_history,
                'checkpoint_path': fold_save_path,
            }
            fold_results.append(fold_result)
            
            # Aggregate histories (for averaging across folds)
            for metric, values in train_history.items():
                if values:  # Only if we have values
                    # Pad or truncate to match the length of existing aggregated history
                    if aggregated_train_history[metric]:
                        max_len = max(len(aggregated_train_history[metric]), len(values))
                        # Extend current aggregated list if needed
                        while len(aggregated_train_history[metric]) < max_len:
                            aggregated_train_history[metric].append(aggregated_train_history[metric][-1] if aggregated_train_history[metric] else 0)
                        # Extend current fold values if needed
                        values_extended = values[:]
                        while len(values_extended) < max_len:
                            values_extended.append(values_extended[-1] if values_extended else 0)
                        # Average with existing values
                        for i in range(max_len):
                            if i < len(aggregated_train_history[metric]):
                                aggregated_train_history[metric][i] = (
                                    aggregated_train_history[metric][i] * fold_idx + values_extended[i]
                                ) / (fold_idx + 1)
                            else:
                                aggregated_train_history[metric].append(values_extended[i])
                    else:
                        aggregated_train_history[metric] = values[:]
            
            for metric, values in val_history.items():
                if values:  # Only if we have values
                    if aggregated_val_history[metric]:
                        max_len = max(len(aggregated_val_history[metric]), len(values))
                        # Extend current aggregated list if needed
                        while len(aggregated_val_history[metric]) < max_len:
                            aggregated_val_history[metric].append(aggregated_val_history[metric][-1] if aggregated_val_history[metric] else 0)
                        # Extend current fold values if needed
                        values_extended = values[:]
                        while len(values_extended) < max_len:
                            values_extended.append(values_extended[-1] if values_extended else 0)
                        # Average with existing values
                        for i in range(max_len):
                            if i < len(aggregated_val_history[metric]):
                                aggregated_val_history[metric][i] = (
                                    aggregated_val_history[metric][i] * fold_idx + values_extended[i]
                                ) / (fold_idx + 1)
                            else:
                                aggregated_val_history[metric].append(values_extended[i])
                    else:
                        aggregated_val_history[metric] = values[:]
            
            logger.info(f"Fold {fold_idx + 1} completed successfully")
            logger.info(f"Best {monitor}: {best_val_score:.4f}")
            
        except Exception as e:

            logger.error(f"Error training fold {fold_idx + 1}: {str(e)}")
            # Store failed fold result
            fold_result = {
                'fold': fold_idx + 1,
                'train_samples': len(train_indices),
                'val_samples': len(val_indices),
                'train_pos_ratio': train_pos_ratio,
                'val_pos_ratio': val_pos_ratio,
                'best_train_score': 0.0,
                'best_val_score': 0.0,
                'train_history': {},
                'val_history': {},
                'checkpoint_path': fold_save_path,
                'error': str(e),
            }
            fold_results.append(fold_result)
            raise # Re-raise to stop further processing of folds
    
    # Calculate and log summary statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"K-FOLD CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    successful_folds = [f for f in fold_results if 'error' not in f]
    failed_folds = [f for f in fold_results if 'error' in f]
    
    logger.info(f"Successful folds: {len(successful_folds)}/{k_folds}")
    if failed_folds:
        logger.warning(f"Failed folds: {len(failed_folds)}")
        for fold in failed_folds:
            logger.warning(f"  Fold {fold['fold']}: {fold.get('error', 'Unknown error')}")
    
    if successful_folds:
        # Extract all metrics from successful folds
        metrics_to_analyze = ['f1', 'auroc', 'precision', 'recall', 'macro_f1', 'acc', 'median_patient_f1', 'median_patient_precision', 'median_patient_recall']
        fold_metrics = {}
        
        for metric in metrics_to_analyze:
            values = []
            for fold in successful_folds:
                val_history = fold.get('val_history', {})
                if metric in val_history and val_history[metric]:
                    # Get the best value for this metric in this fold
                    best_val = max(val_history[metric]) if metric != 'loss' else min(val_history[metric])
                    values.append(best_val)
                else:
                    # Fallback: check if best_val_score corresponds to this metric
                    if monitor.replace('val_', '') == metric:
                        values.append(fold['best_val_score'])
            
            if values:
                fold_metrics[metric] = values
        
        # Calculate comprehensive statistics for each metric
        logger.info(f"\n📊 COMPREHENSIVE METRICS SUMMARY:")
        logger.info(f"{'='*70}")
        
        summary_stats = {}
        for metric, values in fold_metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                median_val = np.median(values)
                q25 = np.percentile(values, 25)
                q75 = np.percentile(values, 75)
                iqr = q75 - q25
                min_val = np.min(values)
                max_val = np.max(values)
                
                summary_stats[metric] = {
                    'values': values,
                    'mean': mean_val,
                    'std': std_val,
                    'median': median_val,
                    'q25': q25,
                    'q75': q75,
                    'iqr': iqr,
                    'min': min_val,
                    'max': max_val
                }
                
                logger.info(f"\n{metric.upper()} Results:")
                logger.info(f"  Mean ± Std:     {mean_val:.4f} ± {std_val:.4f}")
                logger.info(f"  Median [IQR]:   {median_val:.4f} [{q25:.4f}, {q75:.4f}]")
                logger.info(f"  Range:          [{min_val:.4f}, {max_val:.4f}]")
                logger.info(f"  Individual folds: {[f'{v:.4f}' for v in values]}")
        
        # Report in your requested format
        logger.info(f"\n📋 REPORT SUMMARY (Copy to your report):")
        logger.info(f"{'='*70}")
        
        if 'macro_f1' in summary_stats:
            stats = summary_stats['macro_f1']
            logger.info(f"• Macro F1-score: {stats['mean']:.4f} ± {stats['std']:.4f} (Med: {stats['median']:.4f}, IQR: {stats['iqr']:.4f})")
        
        if 'auroc' in summary_stats:
            stats = summary_stats['auroc']
            logger.info(f"• AUC: {stats['mean']:.4f} ± {stats['std']:.4f} (Med: {stats['median']:.4f}, IQR: {stats['iqr']:.4f})")
        
        if 'f1' in summary_stats:
            stats = summary_stats['f1']
            logger.info(f"• F1 (seizure): {stats['mean']:.4f} ± {stats['std']:.4f} (Med: {stats['median']:.4f}, IQR: {stats['iqr']:.4f})")
        
        if 'recall' in summary_stats:
            stats = summary_stats['recall']
            logger.info(f"• Recall (seizure): {stats['mean']:.4f} ± {stats['std']:.4f} (Med: {stats['median']:.4f}, IQR: {stats['iqr']:.4f})")
        
        if 'precision' in summary_stats:
            stats = summary_stats['precision']
            logger.info(f"• Precision (seizure): {stats['mean']:.4f} ± {stats['std']:.4f} (Med: {stats['median']:.4f}, IQR: {stats['iqr']:.4f})")
            
        # Add median patient F1 score if available
        patient_metrics = [metric for metric in summary_stats.keys() if 'median_patient' in metric]
        if patient_metrics:
            logger.info(f"\n📊 PATIENT-LEVEL METRICS:")
            if 'median_patient_f1' in summary_stats:
                stats = summary_stats['median_patient_f1']
                logger.info(f"• Median patient F1: {stats['mean']:.4f} ± {stats['std']:.4f} (Med: {stats['median']:.4f}, IQR: {stats['iqr']:.4f})")
            if 'median_patient_precision' in summary_stats:
                stats = summary_stats['median_patient_precision']
                logger.info(f"• Median patient precision: {stats['mean']:.4f} ± {stats['std']:.4f} (Med: {stats['median']:.4f}, IQR: {stats['iqr']:.4f})")
            if 'median_patient_recall' in summary_stats:
                stats = summary_stats['median_patient_recall']
                logger.info(f"• Median patient recall: {stats['mean']:.4f} ± {stats['std']:.4f} (Med: {stats['median']:.4f}, IQR: {stats['iqr']:.4f})")
        
        # Legacy summary for the monitored metric
        val_scores = [f['best_val_score'] for f in successful_folds]
        mean_score = sum(val_scores) / len(val_scores)
        std_score = (sum((x - mean_score) ** 2 for x in val_scores) / len(val_scores)) ** 0.5
        
        logger.info(f"\n{monitor} Results (Primary Metric):")
        logger.info(f"  Mean: {mean_score:.4f} ± {std_score:.4f}")
        logger.info(f"  Min:  {min(val_scores):.4f}")
        logger.info(f"  Max:  {max(val_scores):.4f}")
        
        # Log individual fold results
        for fold in successful_folds:
            logger.info(f"  Fold {fold['fold']}: {fold['best_val_score']:.4f}")
        
        # Log final summary to wandb if enabled
        if log_wandb and WANDB_AVAILABLE:
            try:
                summary_config = {
                    'k_fold_summary/mean_score': mean_score,
                    'k_fold_summary/std_score': std_score,
                    'k_fold_summary/min_score': min(val_scores),
                    'k_fold_summary/max_score': max(val_scores),
                    'k_fold_summary/successful_folds': len(successful_folds),
                    'k_fold_summary/failed_folds': len(failed_folds),
                }
                
                # Add all metrics to wandb summary
                for metric, stats in summary_stats.items():
                    summary_config.update({
                        f'k_fold_summary/{metric}_mean': stats['mean'],
                        f'k_fold_summary/{metric}_std': stats['std'],
                        f'k_fold_summary/{metric}_median': stats['median'],
                        f'k_fold_summary/{metric}_iqr': stats['iqr'],
                        f'k_fold_summary/{metric}_min': stats['min'],
                        f'k_fold_summary/{metric}_max': stats['max'],
                    })
                
                # Log summary in a separate run
                summary_run = wandb.init( # type: ignore
                    project=wandb_project or "neuro-graph-net",
                    name=f"k_fold_summary_{k_folds}folds",
                    config=summary_config,
                    reinit=True
                )
                wandb.log(summary_config) # type: ignore
                wandb.finish() # type: ignore
                logger.info("Logged comprehensive k-fold summary to wandb")
                
            except Exception as e:
                logger.warning(f"Could not log k-fold summary to wandb: {e}")
    
    # Save summary results
    summary_path = save_dir / "k_fold_summary.json"
    summary_data = {
        'k_folds': k_folds,
        'successful_folds': len(successful_folds),
        'failed_folds': len(failed_folds),
        'fold_results': fold_results,
        'monitor': monitor,
    }
    if successful_folds:
        val_scores = [f['best_val_score'] for f in successful_folds]
        summary_data.update({
            'primary_metric': {
                'mean_score': sum(val_scores) / len(val_scores),
                'std_score': (sum((x - sum(val_scores) / len(val_scores)) ** 2 for x in val_scores) / len(val_scores)) ** 0.5,
                'min_score': min(val_scores),
                'max_score': max(val_scores),
            }
        })
        
        # Add comprehensive metrics summary to JSON
        summary_stats = locals().get('summary_stats', {})
        if summary_stats:
            summary_data['comprehensive_metrics'] = summary_stats
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    logger.info(f"Saved comprehensive k-fold summary to {summary_path}")
    logger.info("K-fold cross-validation completed!")
    
    return dict(aggregated_train_history), dict(aggregated_val_history), fold_results
