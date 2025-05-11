# -------- Import libraries -------- #
# --------------------- General imports --------------------- #
import argparse
import io
import yaml
import sys
from pathlib import Path
from datetime import datetime


import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# --------------------- Custom imports --------------------- #
# Add the root directory to path
project_root = (
    Path(__file__).resolve().parents[1]
)  # 1 levels up from scripts/ -> repository root
sys.path.append(str(project_root))
from src.data.dataset import GraphEEGDataset
from src.utils.models_funcs import build_model
from src.utils.general_funcs import log


def main():

    # --- Parse config file path from command line --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # --- Load config --- #
    with open(args.config, "r") as f:
        default_config = yaml.safe_load(f)

    # --- Initialize W&B run --- #
    wandb.init(project="eeg-seizure", config=default_config)
    config = wandb.config  # Replaces variables with W&B-managed config

    # -------- Define directories -------- #

    DATA_ROOT = Path("data")
    train_dir = DATA_ROOT / "train"
    train_dir_signals = train_dir / "signals"
    train_dir_metadata = train_dir / "segments.parquet"
    train_dataset_dir = DATA_ROOT / "train_dataset"
    spatial_distance_file = pd.read_csv(DATA_ROOT / "distances_3d.csv")

    # ----------------- Prepare training data -----------------#

    clips_tr = pd.read_parquet(train_dir_metadata)
    clips_tr = clips_tr[~clips_tr.label.isna()]  # Filter NaN values out of clips_tr
    # Extract the sessions
    sessions = list(
        clips_tr.groupby(["patient", "session"])
    )  # List of tuples: ((patient_id, session_id), session_df)

    # -------------- Dataset definition -------------- #
    train_dataset = GraphEEGDataset(
        root=train_dataset_dir,
        sessions=sessions,
        signal_folder=train_dir_signals,
        edge_strategy=config.edge_strategy,
        spatial_distance_file=(
            str(spatial_distance_file) if config.edge_strategy == "spatial" else None
        ),
        correlation_threshold=config.correlation_threshold,
        force_reprocess=True,
        bandpass_frequencies=(
            config.low_bandpass_frequency,
            config.high_bandpass_frequency,
        ),
        segment_length=3000,
        apply_filtering=True,
        apply_rereferencing=True,
        apply_normalization=True,
        sampling_rate=250,
    )

    # --------------- Split dataset intro train/val/test --------------- #
    train_ids, val_ids = train_test_split(
        np.arange(len(clips_tr)), test_size=0.2, random_state=config.seed, shuffle=True
    )

    train_subset = Subset(train_dataset, train_ids)
    val_subset = Subset(train_dataset, val_ids)

    # -------Compute sample weights for oversampling ------------------#
    train_labels = [clips_tr.iloc[i]["label"] for i in train_ids]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (
        class_counts**config.oversampling_power
    )  # Higher weights for not frequent classes
    sample_weights = [
        class_weights[label] for label in train_labels
    ]  # Assign weight to each sample based on its class

    # -------------  Define oversampler ------------------------------#
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )  # Still train on N samples per epoch, but instead of sampling uniformly takes more from minority class

    # -------- Create DataLoader -------- #
    train_loader = DataLoader(
        train_subset, batch_size=config.batch_size, sampler=sampler, shuffle=False
    )  # Since we use specific sampler: shuffle=False
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=True)

    # -------- Initialize model  -------- #
    model = build_model(config)
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -------- Initialize optimizer and loss function -------- #
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate
    )  # Adam with weight decay
    loss_fn = nn.BCEWithLogitsLoss()  # Applies sigmoid implicitly

    # -------- Training loop -------- #
    best_val_loss = float("inf")
    best_val_f1 = 0
    patience = 20
    counter = 0

    for epoch in range(1, config.epochs + 1):
        # ------- Training ------- #
        model.train()
        total_loss = 0
        # for batch in tqdm(train_loader,desc=f"Epoch {epoch} — Training" ):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)  # Average loss per batch

        # ------- Validation ------- #
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(
                    batch.x, batch.edge_index, batch.batch
                )  # batch.batch: [num_nodes_batch] = 19*batch_size -> tells the model which graph each node belongs to
                loss = loss_fn(out, batch.y)
                val_loss += loss.item()

                probs = torch.sigmoid(out).squeeze()  # [batch_size, 1] -> [batch_size]
                preds = (probs > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(
                    batch.y.int().cpu().numpy()
                )  # Labels: stored as float in dataset

        avg_val_loss = val_loss / len(val_loader)  # Average loss per batch
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        # Monitor progress
        log(f"Finished Epoch {epoch} | Avg Val Loss: {avg_val_loss:.4f}")

        # W&B
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_f1": val_f1,
            }
        )

        # ------- Early Stopping ------- #
        if avg_val_loss < best_val_loss:

            # Save best statistics and model
            best_val_loss = avg_val_loss
            best_val_f1 = val_f1
            best_val_f1_epoch = epoch
            counter = 0
            best_state_dict = model.state_dict().copy()  # Save the best model state
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best Validation F1: {best_val_f1:.4f}")

    # Loads best stats in W&B
    wandb.run.summary["best_val_f1"] = best_val_f1
    wandb.run.summary["best_val_f1_epoch"] = best_val_f1_epoch

    #  -------------- Save best model ------------------#
    buffer = io.BytesIO()
    torch.save(best_state_dict, buffer)
    buffer.seek(0)

    # Save the model as an artifact
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(buffer, name="model.pt")
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()
