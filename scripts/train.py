# -------- Import libraries -------- #
# --------------------- General imports --------------------- #
import argparse
import io
import os
import yaml
import sys
from pathlib import Path
from datetime import datetime


import numpy as np
import pandas as pd
from tqdm import tqdm

import wandb
from omegaconf import OmegaConf


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, WeightedRandomSampler

from torch_geometric.loader import DataLoader

# --------------------- Custom imports --------------------- #
# Add the root directory to path
project_root = (
    Path(__file__).resolve().parents[1]
)  # 1 levels up from scripts/ -> repository root ; should directly see src, data, configs
sys.path.append(str(project_root))
from src.data.dataset import GraphEEGDataset
from src.utils.models_funcs import build_model
from src.utils.general_funcs import log, generate_run_name


# --------------------------------------------- Main function ---------------------------------------------------------#a


def main():

    # --- Parse config file path from command line --- #

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="gcn.yaml"
    )  # config attribute for args object
    args = parser.parse_args()

    # --- Load config --- #
    config = OmegaConf.load(
        # "configs/gcn.yaml"
        f"configs/{args.config}"
    )  # Accepts nested structure and dot notation
    run_name = generate_run_name(config)  # Generate run name
    log(f"Run name: {run_name}")
    # --- Initialize W&B run --- #
    wandb.init(
        project="eeg-seizure",
        config=OmegaConf.to_container(config, resolve=True),
        name=run_name,
    )

    log("wandb login successful")

    # -------- Define directories -------- #

    DATA_ROOT = Path("data")
    train_dir = DATA_ROOT / "train"
    train_dir_metadata = train_dir / "segments.parquet"
    train_dataset_dir = DATA_ROOT / "graph_dataset_train"
    spatial_distance_file = DATA_ROOT / "distances_3d.csv"

    # ----------------- Prepare training data -----------------#

    clips_tr = pd.read_parquet(train_dir_metadata)
    clips_tr = clips_tr[~clips_tr.label.isna()]  # Filter NaN values out of clips_tr

    # -------------- Dataset definition -------------- #
    train_dataset = GraphEEGDataset(
        root=train_dataset_dir,
        clips=clips_tr,
        signal_folder=train_dir,
        edge_strategy=config.edge_strategy,
        spatial_distance_file=(
            spatial_distance_file if config.edge_strategy == "spatial" else None
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

    # Check the length of the dataset
    log(f"Length of train_dataset: {len(train_dataset)}")

    # --------------- Split dataset intro train/val/test --------------- #
    train_ids, val_ids = train_test_split(
        np.arange(len(train_dataset)),
        test_size=0.2,
        random_state=config.seed,
        stratify=clips_tr.label,  # Stratified split based on labels -> to ensure same distribution in train and val sets
        shuffle=True,
    )

    train_subset = Subset(train_dataset, train_ids)
    val_subset = Subset(train_dataset, val_ids)

    # -------Compute sample weights for oversampling ------------------#
    train_labels = [clips_tr.iloc[i]["label"] for i in train_ids]
    class_counts = np.bincount(train_labels)
    # class_weights = 1.0 / (
    #    class_counts**config.oversampling_power
    # )  # Higher weights for not frequent classes
    class_weights = np.where(
        class_counts > 0, 1.0 / (class_counts**config.oversampling_power), 0.0
    )
    sample_weights = [
        class_weights[label] for label in train_labels
    ]  # Assign weight to each sample based on its class

    # -------------  Define oversampler ------------------------------#
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_subset), replacement=True
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
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )  # Adam with weight decay
    loss_fn = nn.BCEWithLogitsLoss()  # Applies sigmoid implicitly

    # -------- Training loop -------- #
    best_val_loss = float("inf")
    best_val_f1 = 0
    patience = 10
    counter = 0
    log("Training started")

    for epoch in range(1, config.epochs + 1):
        # ------- Training ------- #
        model.train()
        total_loss = 0
        # for batch in tqdm(train_loader,desc=f"Epoch {epoch} — Training" ):
        for batch in train_loader:
            batch = batch.to(device)  # Move batch to GPU
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(
                out, batch.y.reshape(-1, 1)
            )  # y: [batch_size] ->[batch_size, 1]
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
                batch = batch.to(device)  # Move batch to GPU
                out = model(
                    batch.x, batch.edge_index, batch.batch
                )  # batch.batch: [num_nodes_batch] = 19*batch_size -> tells the model which graph each node belongs to
                loss = loss_fn(out, batch.y.reshape(-1, 1))
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
        # ------- Record best F1 score ------- #
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_f1_epoch = epoch
        # ------- Early Stopping ------- #
        if avg_val_loss < best_val_loss:
            # Save best statistics and model
            best_val_loss = avg_val_loss
            counter = 0
            best_state_dict = model.state_dict().copy()  # Save the best model state
        else:
            counter += 1
            if counter >= patience:
                log("Early stopping triggered.")
                break

    log(f"Best validation F1: {best_val_f1:.4f} at epoch {best_val_f1_epoch}")

    # Loads best stats in W&B
    log("Saving best metrics and model")
    wandb.run.summary["best_f1"] = best_val_f1
    wandb.run.summary["best_f1_epoch"] = best_val_f1_epoch

    #  -------------- Save best model ------------------#
    # -------------- Save best model ------------------ #
    model_path = "model.pt"
    torch.save(best_state_dict, model_path)

    # Save the model as a W&B artifact
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    #   Delete file after logging
    os.remove(model_path)

    wandb.finish()  # Close the W&B run


if __name__ == "__main__":
    main()
