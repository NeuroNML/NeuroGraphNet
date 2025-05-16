from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


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
    Saves the *unwrapped* model state (supports DataParallel / DDP).
    Returns (train_losses, val_losses).
    """
    # ── checkpoint handling ──────────────────────────────────────────────
    if overwrite and save_path.exists():
        save_path.unlink()
    if save_path.exists():
        _load(model, save_path, device)
        print("Checkpoint found → model loaded, skipping training.")
        return [], []

    if monitor == "val_loss":
        best_score = float("inf")       # we want the *smallest* loss
    else:                               # e.g. AUROC, F1 => larger is better
        best_score = -float("inf")

    bad_epochs = 0
    train_hist, val_hist = [], []

    pbar = tqdm(range(1, num_epochs + 1), desc="Epochs", ncols=120)
    for _ in pbar:
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

        preds = torch.cat(preds)
        targets = torch.cat(targets)

        # compute metric for early stopping
        if monitor == "val_loss":
            score = avg_val
            improved = score < best_score        # lower is better
        elif monitor == "auroc":
            # NOTE: task=binary since we have only 2 classes (0, 1)
            score = auroc(preds, targets.int(), task="binary").item()
            improved = score > best_score        # higher is better
        else:
            raise ValueError(f"Unknown monitor '{monitor}'")

        # check for improvement
        if improved:
            best_score = score
            bad_epochs = 0
            _save(model, save_path)
        else:
            bad_epochs += 1

        # if scheduler is passed, we need to step it in order to update the
        # weight decay
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
            "save": "✓" if improved else "-",
        })

        if patience > 0 and bad_epochs >= patience:
            pbar.write(
                f"Early stopping: no {monitor} improvement in {patience} epochs.")
            break

    print("Training complete.")
    return train_hist, val_hist


def _save(model: nn.Module, path: Path):
    """Save .state_dict() of the underlying model (handles DP/DDP)."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        state = model.module.state_dict()
    else:
        state = model.state_dict()
    torch.save(state, path)


def _load(
    model: nn.Module,
    path: Path,
    device: torch.device,
):
    state_dict = torch.load(path, map_location=device)
    wrapped = isinstance(
        model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
    has_module = any(k.startswith("module.") for k in state_dict.keys())

    # support 3 different cases (allows to mix between one / multi GPU):
    # 1. model is wrapped (DataParallel) and state_dict has "module."
    # 2. model is NOT wrapped and state_dict has "module."
    # 3. both are wrapped or both are NOT wrapped
    #    (in this case, we can just load the state_dict as is)
    if wrapped and not has_module:
        model.module.load_state_dict(state_dict)
    elif not wrapped and has_module:
        state_dict = OrderedDict((k.replace("module.", "", 1), v)
                                 for k, v in state_dict.items())
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: Path,
    submission_path: Path,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Loads the best checkpoint, runs inference on `test_loader`, and writes
    a CSV with columns ['id', 'label'] to `submission_path`.

    Returns the DataFrame so you can inspect it directly.
    """
    # load checkpoint
    _load(model, save_path, device)
    model.to(device).eval()

    # perform inference on the test set
    all_ids, all_preds = [], []
    with torch.no_grad():
        for xb, xids in test_loader:
            xb = xb.to(device, dtype=torch.float32)
            logits = model(xb)
            probs = logits.sigmoid().cpu().squeeze()

            preds = (probs > threshold).int().tolist()
            all_preds.extend(preds)

            # xids may be list[str] or tensor[int]
            if isinstance(xids, list):
                all_ids.extend(xids)
            else:
                all_ids.extend(xids.cpu().tolist())

    # write predictions to CSV file (submission format)
    sub = pd.DataFrame({"id": all_ids, "label": all_preds}).sort_values("id")
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(submission_path, index=False)
    print(f"Saved submission → {submission_path}")

    return sub