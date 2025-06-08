from collections import OrderedDict, defaultdict # Added defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import time  # Add time import
import logging  # Add logging import
import inspect  # Add inspect for checking function signatures

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchmetrics.functional import accuracy, f1_score, auroc 
from torch_geometric.utils import to_dense_batch
from src.data.geodataloader import GeoDataLoader
from src.data.dataset_graph import GraphEEGDataset 

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


def _safe_model_call(model, x, edge_index=None, batch=None, graph_features=None):
    """
    Safely call a model with appropriate parameters based on its forward method signature.
    
    Args:
        model: The PyTorch model to call
        x: Node features
        edge_index: Graph edge indices (for GNN models)
        batch: Batch indices (for GNN models)
        graph_features: Graph-level features (optional)
    
    Returns:
        Model output
    """
    # Get the model's forward method signature
    forward_signature = inspect.signature(model.forward)
    params = forward_signature.parameters
    
    # Base call arguments for GNN models
    if edge_index is not None and batch is not None:
        call_args = [x, edge_index, batch]
        call_kwargs = {}
        
        # Check if model accepts graph_features parameter
        if 'graph_features' in params and graph_features is not None:
            call_kwargs['graph_features'] = graph_features
            
        return model(*call_args, **call_kwargs)
    else:
        # Non-GNN models - just pass x
        call_args = [x]
        call_kwargs = {}
        
        # Some non-GNN models might still accept graph_features
        if 'graph_features' in params and graph_features is not None:
            call_kwargs['graph_features'] = graph_features
            
        return model(*call_args, **call_kwargs)


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

            logger.info(f"🔗 Wandb run initialized: {wandb_run.name}")

            # Watch the model for gradient and parameter tracking
            wandb.watch(model, log='all', log_freq=100) # type: ignore
            print(f"🔗 Wandb initialized: {wandb.run.name}") # type: ignore
            
            # Log dataset information to wandb if available
            if original_dataset is not None:
                logger.info("Logging dataset configuration to wandb...")
                dataset_config = {}
                
                # Helper function to safely get attribute
                def safe_get_attr(obj, attr_name, default=None):
                    try:
                        return getattr(obj, attr_name, default)
                    except:
                        return default
                
                # Signal processing settings
                dataset_config['dataset/bandpass_frequencies'] = safe_get_attr(original_dataset, 'bandpass_frequencies', (0.5, 50))
                dataset_config['dataset/segment_length'] = safe_get_attr(original_dataset, 'segment_length', 3000)
                dataset_config['dataset/sampling_rate'] = safe_get_attr(original_dataset, 'sampling_rate', 250)
                
                # Calculate segment duration
                segment_length = dataset_config['dataset/segment_length']
                sampling_rate = dataset_config['dataset/sampling_rate']
                if segment_length and sampling_rate:
                    dataset_config['dataset/segment_duration_seconds'] = segment_length / sampling_rate
                
                # Preprocessing flags
                dataset_config['dataset/apply_filtering'] = safe_get_attr(original_dataset, 'apply_filtering', False)
                dataset_config['dataset/apply_rereferencing'] = safe_get_attr(original_dataset, 'apply_rereferencing', False)
                dataset_config['dataset/apply_normalization'] = safe_get_attr(original_dataset, 'apply_normalization', False)
                
                # Graph construction settings
                dataset_config['dataset/edge_strategy'] = safe_get_attr(original_dataset, 'edge_strategy', 'unknown')
                dataset_config['dataset/top_k'] = safe_get_attr(original_dataset, 'top_k', None)
                dataset_config['dataset/correlation_threshold'] = safe_get_attr(original_dataset, 'correlation_threshold', 0.7)
                
                # Feature settings
                dataset_config['dataset/use_embeddings'] = safe_get_attr(original_dataset, 'embeddings_train', False)
                dataset_config['dataset/use_selected_features'] = safe_get_attr(original_dataset, 'selected_features_train', False)
                dataset_config['dataset/extract_graph_features'] = safe_get_attr(original_dataset, 'extract_graph_features', False)
                
                # Dataset metadata
                dataset_config['dataset/n_channels'] = safe_get_attr(original_dataset, 'n_channels', 19)
                dataset_config['dataset/is_test'] = safe_get_attr(original_dataset, 'is_test', False)
                dataset_config['dataset/force_reprocess'] = safe_get_attr(original_dataset, 'force_reprocess', False)
                
                # Dataset size
                try:
                    dataset_config['dataset/total_samples'] = len(original_dataset)
                except:
                    dataset_config['dataset/total_samples'] = 'unknown'
                
                # Add graph feature types if available
                graph_feature_types = safe_get_attr(original_dataset, 'graph_feature_types', None)
                if graph_feature_types:
                    dataset_config['dataset/graph_feature_types'] = graph_feature_types
                
                # Add graph feature extractor info if available
                graph_feature_extractor = safe_get_attr(original_dataset, 'graph_feature_extractor', None)
                if graph_feature_extractor and hasattr(graph_feature_extractor, 'feature_types'):
                    dataset_config['dataset/graph_feature_extractor_types'] = graph_feature_extractor.feature_types
                
                # Add channel information
                channels = safe_get_attr(original_dataset, 'channels', None)
                if channels:
                    dataset_config['dataset/channels'] = channels
                    dataset_config['dataset/n_channels_actual'] = len(channels)
                
                # Add clips information if available
                clips = safe_get_attr(original_dataset, 'clips', None)
                if clips is not None:
                    try:
                        dataset_config['dataset/n_clips'] = len(clips)
                        
                        # Add label distribution for training data
                        if not safe_get_attr(original_dataset, 'is_test', True) and hasattr(clips, 'columns') and 'label' in clips.columns:
                            label_counts = clips['label'].value_counts().to_dict()
                            dataset_config['dataset/label_distribution'] = label_counts
                            dataset_config['dataset/n_positive'] = label_counts.get(1, 0)
                            dataset_config['dataset/n_negative'] = label_counts.get(0, 0)
                            if len(clips) > 0:
                                dataset_config['dataset/positive_ratio'] = label_counts.get(1, 0) / len(clips)
                        
                        # Add patient/session information if available
                        if hasattr(clips, 'columns'):
                            if 'patient' in clips.columns:
                                dataset_config['dataset/n_patients'] = clips['patient'].nunique()
                            if 'session' in clips.columns:
                                dataset_config['dataset/n_sessions'] = clips['session'].nunique()
                    except Exception as e:
                        logger.warning(f"Could not extract clips information: {e}")
                
                # Log the dataset configuration to wandb
                wandb.config.update(dataset_config) # type: ignore
                logger.info(f"Logged {len(dataset_config)} dataset parameters to wandb")
                
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
    total_batches = len(train_loader_to_iterate)
    logger.info(f"Total training batches per epoch: {total_batches}")

    bad_epochs = 0
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
                    # get graph features if available
                    if hasattr(curr_batch, 'graph_features'):
                        graph_features = curr_batch.graph_features
                        if graph_features is not None:
                            logger.info(f"Graph features shape: {graph_features.shape}")
                    else:
                        graph_features = None

                    # Use safe model call that handles graph_features parameter automatically
                    logits = _safe_model_call(model, 
                                            curr_batch.x.float(),
                                            curr_batch.edge_index,
                                            curr_batch.batch,
                                            graph_features)
                    # unsqueeze y_targets to match the shape of the logits. used later!
                except Exception as e:
                    logger.error(f"Error in forward pass for batch {batch_idx}: {str(e)}")
                    logger.error(f"Edge index shape: {curr_batch.edge_index.shape}")
                    logger.error(f"Edge index content: {curr_batch.edge_index}")
                    raise
            else:
                if isinstance(data_batch_item, (tuple, list)) and len(data_batch_item) == 2:
                    x_batch, y_batch = data_batch_item
                    x_batch = x_batch.to(device)
                    y_targets = y_batch.to(device) if isinstance(y_batch, torch.Tensor) else torch.tensor(y_batch, device=device)
                    y_targets = y_targets.reshape(-1, 1)
                    # Use safe model call for non-GNN models
                    logits = _safe_model_call(model, x_batch.float())
                elif hasattr(data_batch_item, 'x') and hasattr(data_batch_item, 'y'):
                    if not to_dense_batch: # Should be caught earlier if PyG not installed but good check
                         raise ImportError("PyTorch Geometric 'to_dense_batch' is required for this non-GNN path if data is PyG Batch.")
                    curr_batch = data_batch_item.to(device) # type: ignore
                    y_targets = curr_batch.y.reshape(-1, 1)
                    if y_targets is None: continue
                    
                    # Extract graph features if available
                    if hasattr(curr_batch, 'graph_features'):
                        graph_features = curr_batch.graph_features
                    else:
                        graph_features = None
                    
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
                    # Use safe model call that can handle graph_features if the model supports it
                    logits = _safe_model_call(model, input_features.float(), graph_features=graph_features)
                else:
                    raise ValueError("Batch data must be a tuple (x, y) or have 'x' and 'y' attributes for non-GNN mode.")

            # Compute loss on raw logits
            loss = criterion(logits, y_targets)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # convert logits to binary predictions
            probs = logits.sigmoid().squeeze()  # [batch_size, 1] -> [batch_size]
            
            # Validate probability ranges
            if torch.any(probs < 0) or torch.any(probs > 1):
                logger.warning(f"Batch {batch_idx}: Probabilities outside [0,1] range detected. Clamping values.")
                probs = torch.clamp(probs, 0, 1)

            # store round results
            epoch_train_loss += loss.item()
            all_train_preds.append(probs.cpu()) # Store probabilities instead of binary predictions
            all_train_targets.append(y_targets.squeeze(1).int().cpu()) # shape: (batch_size)

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
                
                # Accuracy and F1 on binary predictions
                train_history['acc'].append(accuracy(binary_preds, targets_cat_train, task="binary").item())
                train_history['f1'].append(f1_score(binary_preds, targets_cat_train, task="binary").item())
                
                # AUROC on probabilities
                train_history['auroc'].append(auroc(preds_cat_train, targets_cat_train, task="binary").item()) # type: ignore
                
                # Log metric values for debugging
                logger.debug(f"Epoch {epoch} metrics - Acc: {train_history['acc'][-1]:.4f}, F1: {train_history['f1'][-1]:.4f}, AUROC: {train_history['auroc'][-1]:.4f}")
            except ValueError as e:
                logger.error(f"Error computing metrics (Epoch {epoch}): {str(e)}")
                logger.error(f"Preds shape: {preds_cat_train.shape}, Targets shape: {targets_cat_train.shape}")
                logger.error(f"Preds range: [{preds_cat_train.min():.4f}, {preds_cat_train.max():.4f}]")
                logger.error(f"Targets unique values: {torch.unique(targets_cat_train).tolist()}")
                train_history['acc'].append(0.0)
                train_history['f1'].append(0.0)
                train_history['auroc'].append(0.0)
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
                    y_targets = curr_batch.y.reshape(-1, 1)
                    if y_targets is None: continue
                    if not (hasattr(curr_batch, 'x') and hasattr(curr_batch, 'edge_index')):
                         raise ValueError("For GNN mode, batch_data must have 'x' and 'edge_index'.")
                    
                    # Extract graph features if available (same as training)
                    if hasattr(curr_batch, 'graph_features'):
                        graph_features = curr_batch.graph_features
                    else:
                        graph_features = None
                    
                    # Use safe model call that handles graph_features parameter automatically
                    logits = _safe_model_call(model,
                                            curr_batch.x.float(),
                                            curr_batch.edge_index,
                                            curr_batch.batch,
                                            graph_features)
                else: # Non-GNN
                    if isinstance(data_batch_item_val, (tuple, list)) and len(data_batch_item_val) == 2:
                        x_batch_val, y_batch_val = data_batch_item_val
                        x_batch_val = x_batch_val.to(device)
                        if y_batch_val is not None:
                            y_targets = y_batch_val.to(device) if isinstance(y_batch_val, torch.Tensor) else torch.tensor(y_batch_val, device=device)
                            y_targets = y_targets.reshape(-1, 1)
                        else: continue # Skip if no labels in validation
                        # Use safe model call for non-GNN models too
                        logits = _safe_model_call(model, x_batch_val.float())
                    elif hasattr(data_batch_item_val, 'x') and hasattr(data_batch_item_val, 'y'):
                        curr_batch = data_batch_item_val.to(device) # type: ignore
                        y_targets = curr_batch.y.reshape(-1, 1)
                        if y_targets is None: continue
                        
                        # Extract graph features if available
                        if hasattr(curr_batch, 'graph_features'):
                            graph_features = curr_batch.graph_features
                        else:
                            graph_features = None
                        
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
                                input_features_val, _ = to_dense_batch(node_features_tensor, curr_batch, max_num_nodes=n_channels)
                        elif curr_batch.num_graphs == 0: continue # Skip empty val batch
                        else: raise ValueError(f"Unexpected num_graphs in val: {curr_batch.num_graphs}.")
                        # Use safe model call that can handle graph_features if the model supports it
                        logits = _safe_model_call(model, input_features_val.float(), graph_features=graph_features)
                    else:
                        raise ValueError("Val batch data must be a tuple (x, y) or have 'x' and 'y' attributes for non-GNN mode.")

                # compute loss on validation set using raw logits
                epoch_val_loss += criterion(logits, y_targets).item()

                # Convert logits to binary predictions
                probs = logits.sigmoid().squeeze()  # [batch_size, 1] -> [batch_size]
                
                # Validate probability ranges
                if torch.any(probs < 0) or torch.any(probs > 1):
                    logger.warning(f"Batch {batch_idx}: Probabilities outside [0,1] range detected. Clamping values.")
                    probs = torch.clamp(probs, 0, 1)

                # store round results
                all_val_preds.append(probs.cpu()) # Store probabilities instead of binary predictions
                all_val_targets.append(y_targets.squeeze(1).int().cpu())
        
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
                
                # Accuracy and F1 on binary predictions
                val_acc = accuracy(binary_preds, targets_cat_val, task="binary").item()
                val_f1 = f1_score(binary_preds, targets_cat_val, task="binary").item()
                
                # AUROC on probabilities
                val_auroc = auroc(preds_cat_val, targets_cat_val, task="binary").item()
                
                val_history['acc'].append(val_acc)
                val_history['f1'].append(val_f1)
                val_history['auroc'].append(val_auroc)

                # Log metric values for debugging
                logger.debug(f"Epoch {epoch} validation metrics - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUROC: {val_auroc:.4f}")

                if monitor == "val_auroc": current_score_val = val_auroc
                elif monitor == "val_f1": current_score_val = val_f1
                elif monitor == "val_acc": current_score_val = val_acc
                # else monitor is "val_loss", already set
            except ValueError as e:
                logger.error(f"Error computing validation metrics (Epoch {epoch}): {str(e)}")
                logger.error(f"Preds shape: {preds_cat_val.shape}, Targets shape: {targets_cat_val.shape}")
                logger.error(f"Preds range: [{preds_cat_val.min():.4f}, {preds_cat_val.max():.4f}]")
                logger.error(f"Targets unique values: {torch.unique(targets_cat_val).tolist()}")
                val_history['acc'].append(0.0)
                val_history['f1'].append(0.0)
                val_history['auroc'].append(0.0)
                # current_score_val remains avg_val_loss
        else: # Handles empty val_loader or if all batches were skipped
            val_history['acc'].append(0.0); val_history['f1'].append(0.0); val_history['auroc'].append(0.0)
            # current_score_val remains avg_val_loss

        # Log metrics to wandb
        if log_wandb and wandb_run:
            wandb_metrics = {
                'epoch': epoch,
                'train/loss': train_history['loss'][-1] if train_history['loss'] else 0.0,
                'train/accuracy': train_history['acc'][-1] if train_history['acc'] else 0.0,
                'train/f1': train_history['f1'][-1] if train_history['f1'] else 0.0,
                'train/auroc': train_history['auroc'][-1] if train_history['auroc'] else 0.0,
                'val/loss': val_history['loss'][-1] if val_history['loss'] else 0.0,
                'val/accuracy': val_history['acc'][-1] if val_history['acc'] else 0.0,
                'val/f1': val_history['f1'][-1] if val_history['f1'] else 0.0,
                'val/auroc': val_history['auroc'][-1] if val_history['auroc'] else 0.0,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'bad_epochs': bad_epochs,
                f'best_{monitor}': best_score,
            }
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
                    graph_features = batch_data.graph_features
                else:
                    graph_features = None
                
                # Use safe model call for evaluation too
                logits = _safe_model_call(model,
                                        batch_data.x.float(), 
                                        batch_data.edge_index, 
                                        batch_data.batch,
                                        graph_features)
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
        print(f"Warning: Mismatch in length of IDs ({len(all_ids)}) and predictions ({len(all_binary_preds)}). Truncating to {min_len}.")
    
    # create submission dataframe + sort by id to maintain the same order as the test set
    sub_df = pd.DataFrame({"id": all_ids[:min_len], "label": all_binary_preds[:min_len]})
    sub_df = sub_df.sort_values(by="id", key=lambda col: col.astype(str)) # Robust string sort

    # save submission file
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(submission_path, index=False)
    print(f"📄 Saved submission ({len(sub_df)} rows) → {submission_path}")

    # return submission dataframe
    return sub_df
