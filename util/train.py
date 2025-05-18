from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchmetrics.functional import auroc # Added for auroc metric


def train_model(
    model: nn.Module,  # model to train
    train_loader: torch.utils.data.DataLoader,  # training data loader
    val_loader: torch.utils.data.DataLoader,  # validation data loader
    criterion: nn.Module,  # loss function
    optimizer: optim.Optimizer,  # optimizer (e.g. Adam)
    device: torch.device,  # device to use (CPU or GPU)
    save_path: Path,  # path to save the model
    # learning rate scheduler (optional)
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    # ---- early‑stop settings ------------------------------------------
    monitor: str = "val_loss",  # "val_loss" or "auroc", used for early stopping as well
    # number of epochs with no improvement after which training will be stopped
    patience: int = 15,
    # ---- runtime tweaks ---------------------------------------------------
    num_epochs: int = 100,  # number of epochs to train
    # gradient clipping value (used to avoid exploding gradients)
    grad_clip: float = 1.0,
    # ---- checkpoint settings -------------------------------------------
    overwrite: bool = False,  # overwrite existing checkpoint
) -> Tuple[List[float], List[float]]:
    """
    Generic training loop with early stopping.
    Saves the *unwrapped* model state (supports DataParallel / DDP)
    along with training and validation loss history.
    Returns (train_losses, val_losses).
    """
    # ── checkpoint handling ──────────────────────────────────────────────
    if save_path.exists():
        if overwrite:
            save_path.unlink()
            print(f"Overwrite enabled: Removed existing checkpoint at {save_path}")
        else:
            print(f"Attempting to load checkpoint from {save_path}...")
            checkpoint_data = _load(model, save_path, device)
            print("Checkpoint found → model loaded, skipping training.")
            loaded_train_losses = checkpoint_data.get('train_losses', [])
            loaded_val_losses = checkpoint_data.get('val_losses', [])
            # If losses were not in checkpoint for some reason, return empty lists
            return loaded_train_losses, loaded_val_losses

    if monitor == "val_loss":
        best_score = float("inf")       # we want the *smallest* loss
    else:                               # e.g. AUROC, F1 => larger is better
        best_score = -float("inf")

    bad_epochs = 0
    train_hist, val_hist = [], []

    pbar = tqdm(range(1, num_epochs + 1), desc="Epochs", ncols=120)
    for epoch in pbar:
        # train model
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.float32).unsqueeze(1)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        train_hist.append(avg_train)

        # validation for early stopping
        model.eval()
        val_loss, preds, targets = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, dtype=torch.float32)
                yb = yb.to(device, dtype=torch.float32).unsqueeze(1)

                logits = model(xb)
                val_loss += criterion(logits, yb).item()

                preds.append(logits.sigmoid().cpu())
                targets.append(yb.cpu())

        avg_val = val_loss / len(val_loader)
        val_hist.append(avg_val)

        preds_tensor = torch.cat(preds)
        targets_tensor = torch.cat(targets)

        # compute metric for early stopping
        if monitor == "val_loss":
            score = avg_val
            improved = score < best_score      # lower is better
        elif monitor == "auroc":
            # NOTE: task=binary since we have only 2 classes (0, 1)
            try:
                score = auroc(preds_tensor, targets_tensor.int(), task="binary").item()
            except ValueError as e:
                print(f"Warning: Could not compute AUROC for epoch {epoch}. Error: {e}. Using 0.0 as score.")
                score = 0.0 # Or handle as appropriate, e.g. best_score or -float('inf')
            improved = score > best_score      # higher is better
        else:
            raise ValueError(f"Unknown monitor '{monitor}'")

        # check for improvement
        if improved:
            best_score = score
            bad_epochs = 0
            _save(model, save_path, train_hist, val_hist) # Pass histories to _save
        else:
            bad_epochs += 1

        # if scheduler is passed, we need to step it in order to update the
        # learning rate (or other optimizer params like weight decay)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # give scheduler the metric it monitors
                scheduler.step(score if monitor != "val_loss" else avg_val)
            else:
                scheduler.step()

        pbar.set_postfix({
            "train_loss": f"{avg_train:.4f}",
            "val_loss":   f"{avg_val:.4f}",
            f"best_{monitor}": f"{best_score:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "bad_epochs": f"{bad_epochs}/{patience}",
            "save": "✓" if improved else "-",
        })

        if patience > 0 and bad_epochs >= patience:
            pbar.write(
                f"Early stopping: no {monitor} improvement in {patience} epochs.")
            break

    print("Training complete.")
    # If training completed without ever saving (e.g. num_epochs=1 and no improvement),
    # ensure the latest (and only) history is available if checkpoint wasn't created.
    # However, the best model according to 'monitor' is what's saved.
    # The returned history is the full history of this training run.
    return train_hist, val_hist


def _save(
    model: nn.Module,
    path: Path,
    train_losses: List[float],
    val_losses: List[float]
):
    """Save .state_dict() of the underlying model (handles DP/DDP)
    along with training and validation loss histories."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'model_state_dict': model_state_dict,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    torch.save(checkpoint, path)


def _load(
    model: nn.Module,
    path: Path,
    device: torch.device,
) -> Dict[str, Any]: # Returns the loaded checkpoint dictionary
    """Loads model state_dict and returns the full checkpoint dictionary."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    wrapped = isinstance(
        model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

    if wrapped and not has_module_prefix:
        # Loading a non-DataParallel state_dict into a DataParallel model
        model.module.load_state_dict(state_dict)
    elif not wrapped and has_module_prefix:
        # Loading a DataParallel state_dict into a non-DataParallel model
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "", 1) # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    return checkpoint


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: Path, # Path to the saved checkpoint
    submission_path: Path,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Loads the best checkpoint, runs inference on `test_loader`, and writes
    a CSV with columns ['id', 'label'] to `submission_path`.

    Returns the DataFrame so you can inspect it directly.
    """
    # load checkpoint (model state is loaded in-place by _load)
    print(f"Loading model for evaluation from: {save_path}")
    _ = _load(model, save_path, device) # We don't need the returned dict here, model is loaded.
    model.to(device).eval()

    # perform inference on the test set
    all_ids, all_preds = [], []
    with torch.no_grad():
        for xb, xids in tqdm(test_loader, desc="Evaluating"):
            xb = xb.to(device, dtype=torch.float32)
            logits = model(xb)
            probs = logits.sigmoid().cpu().squeeze() # Squeeze to remove single dimension if batch_size=1

            # Handle cases where probs might be a 0-dim tensor (single item batch) or 1-dim tensor
            if probs.ndim == 0:
                preds = [(probs.item() > threshold)]
            else:
                preds = (probs > threshold).int().tolist()
            
            all_preds.extend(preds)

            # xids may be list[str] or tensor[int]
            if isinstance(xids, list): # Assuming xids is a list of IDs from the DataLoader
                all_ids.extend(xids)
            elif torch.is_tensor(xids):
                all_ids.extend(xids.cpu().tolist())
            else:
                # Handle cases where xids might be a tuple of tensors, e.g. if DataLoader returns multiple ID components
                try:
                    # Attempt to extend if it's an iterable of items
                    all_ids.extend(list(xids))
                except TypeError:
                    # If not iterable or other unhandled type, convert to list of strings
                    all_ids.extend([str(x) for x in xids])


    # write predictions to CSV file (submission format)
    sub_df = pd.DataFrame({"id": all_ids, "label": all_preds})
    
    # Attempt to sort by ID if IDs are sortable (e.g. numeric or consistent strings)
    try:
        # If IDs are numeric or can be converted to numeric for sorting
        sub_df['id_sort_temp'] = pd.to_numeric(sub_df['id'], errors='ignore')
        sub_df = sub_df.sort_values(by='id_sort_temp').drop(columns=['id_sort_temp'])
    except Exception:
        # Fallback to string sort or no sort if conversion/complex type
        try:
            sub_df = sub_df.sort_values("id")
        except TypeError:
            print("Warning: Could not sort submission by ID due to mixed types or unsortable IDs.")

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(submission_path, index=False)
    print(f"Saved submission → {submission_path}")

    return sub_df