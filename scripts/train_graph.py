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
import matplotlib.pyplot as plt

import wandb
from omegaconf import OmegaConf


from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
from src.data.dataset_graph import GraphEEGDataset
from src.utils.models_funcs import build_model
from src.utils.general_funcs import *


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
    extracted_features_dir = DATA_ROOT / "extracted_features"
    embeddings_dir =  DATA_ROOT / "embeddings"

    # ----------------- Prepare training data -----------------#

    clips_tr = pd.read_parquet(train_dir_metadata)
    clips_tr = clips_tr[~clips_tr.label.isna()].reset_index()  # Filter NaN values out of clips_tr

    # -------------- Dataset definition -------------- #
    dataset = GraphEEGDataset(
        root=train_dataset_dir,
        clips=clips_tr,
        signal_folder=train_dir,
        extracted_features_dir=extracted_features_dir,
        selected_features_train=config.selected_features,
        embeddings_dir = embeddings_dir,
        embeddings_train = config.embeddings,
        edge_strategy=config.edge_strategy,
        spatial_distance_file=(
            spatial_distance_file if config.edge_strategy == "spatial" else None
        ),
        top_k=config.top_k,
        correlation_threshold=config.correlation_threshold,
        force_reprocess=False,
        bandpass_frequencies=(
            config.low_bandpass_frequency,
            config.high_bandpass_frequency,
        ),
        segment_length=3000,
        apply_filtering=True,
        apply_rereferencing=False,
        apply_normalization=False,
        sampling_rate=250,
    )

    # Check the length of the dataset
    log(f"Length of train_dataset: {len(dataset)}")
    log(f' Eliminated IDs:{dataset.ids_to_eliminate}')

    # Eliminate ids that did not have electrodes above correlation threshols
    clips_tr = clips_tr[~clips_tr.index.isin(dataset.ids_to_eliminate)].reset_index(drop=True)

    # --------------- Split dataset intro train/val --------------- #
    '''
    y = clips_tr.label.values
    train_ids, val_ids = train_test_split(
    np.arange(len(y)),
    test_size=0.2,
    random_state=config.seed
)
    '''

    
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=config.seed)
    groups = clips_tr.patient.values
    y = clips_tr["label"].values
    X = np.zeros(len(y))  # Dummy X (not used); just placeholder for the Kfold
    train_ids, val_ids = next(cv.split(X, y, groups=groups))  # Just select one split
    print('Labels before Kfold', flush=True)
    print(y,flush=True)

    # Print stats for class 0 and 1
    labels_stats(y, trains_ids=train_ids,val_ids= val_ids)
    
   


    # 2. From dataset generate train and val datasets
    train_dataset = Subset(dataset, train_ids)
    val_dataset = Subset(dataset, val_ids)



    # 3. Compute sample weights for oversampling
    train_labels = [clips_tr.iloc[i]["label"] for i in train_ids]
    class_counts = np.bincount(train_labels)
    class_weights = (1. / class_counts) ** config.oversampling_power# Higher weights for not frequent classes
    sample_weights = [class_weights[label] for label in train_labels] # Assign weight to each sample based on its class

    # 4. Define sampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True) # Still train on N samples per epoch, but instead of sampling uniformly takes more from minority class
    
    # Define dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    

    # -------- Initialize model  -------- #
    model = build_model(config)
    print(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -------- Initialize optimizer and loss function -------- #
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )  # Adam with weight decay

    '''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    '''

    #loss_fn = nn.BCEWithLogitsLoss()  # Applies sigmoid implicitly
    adjusted_pos_weight = torch.tensor([1.5], dtype=torch.float32).to(device)
    log(f'pos_weigth:{adjusted_pos_weight}')
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=adjusted_pos_weight)

    # -------- Training loop -------- #
    best_val_loss = float("inf")
    best_val_f1 = 0
    best_val_auc = 0
    best_val_f1_epoch = 0
    patience = 30
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
            '''
            # For debugging
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad norm = {param.grad.norm().item():.2e}")
            '''
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)  # Average loss per batch

        # ------- Validation ------- #
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

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
                all_probs.extend(probs.cpu().numpy().ravel())
                all_preds.extend(preds.cpu().numpy().ravel())
                all_labels.extend(
                    batch.y.int().cpu().numpy().ravel()
                )  # Labels: stored as float in dataset
                #log(f"Val logits stats — min: {out.min().item():.4f}, max: {out.max().item():.4f}, mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")


                #log(f"Predictions:{preds.cpu().numpy()}")
                log(f"Sigmoid outputs: { torch.sigmoid(out).detach().cpu().numpy()}")
                #log(f"Labels:{batch.y}")
                


        avg_val_loss = val_loss / len(val_loader)  # Average loss per batch
        #scheduler.step(avg_val_loss)
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        all_labels = np.array(all_labels).astype(int)
        all_preds = np.array(all_preds).astype(int)
        all_probs = np.array(all_probs)

        
        for name, param in model.named_parameters():
            if param.grad is not None:
                log(f"{name} grad mean: {param.grad.abs().mean()}")
        

       
        val_auc = roc_auc_score(all_labels, all_probs)
        # Monitor progress
        log(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}| Val AUC:  {val_auc:.4f} ")
        
        # Additional metrics
         # Confusion matrix
        confusion_matrix_plot(all_preds, all_labels)
        # Compute metrics per class (0 and 1)
        precision = precision_score(all_labels, all_preds, average=None)
        recall = recall_score(all_labels, all_preds, average=None)
        f1 = f1_score(all_labels, all_preds, average=None)
        

        # Print only for class 1
        log(f"Class 1 — Precision: {precision[1]:.2f}, Recall: {recall[1]:.2f}, F1: {f1[1]:.2f}")
        

        # W&B
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "val_f1_class_1":f1[1],
                 "val_f1_class_0":f1[0]
            }
        )
        # ------- Record best F1 score ------- #
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_f1_epoch = epoch
            best_state_dict = model.state_dict().copy()
            best_preds = all_preds.copy()
            best_labels = all_labels.copy()
            # Load best stats in wandb
            wandb.summary["best_f1_score"] = val_f1
            wandb.summary["f1_score_epoch"] = epoch
        # ------------ Best AUC -----------------#
        if val_auc > best_val_auc: 
            best_val_auc = val_auc
            best_val_auc_epoch = epoch
            wandb.summary["best_auc_score"] = val_auc
            wandb.summary["best_auc_epoch"] = epoch
        # ------- Early Stopping ------- #
        if avg_val_loss < best_val_loss:
            # Save best statistics and model
            best_val_loss = avg_val_loss
            counter = 0
            #best_state_dict = model.state_dict().copy()  # Save the best model state
        else:
            counter += 1
            if counter >= patience:
                log("Early stopping triggered.")
                break

    log(f"Best validation F1: {best_val_f1:.4f} at epoch {best_val_f1_epoch}")
    log(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_val_auc_epoch}")

    # -------------- Sve confusion matrix --------------------#
    cm = confusion_matrix(best_labels, best_preds, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Create a figure for the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Best Confusion Matrix (Epoch {best_val_f1_epoch}, F1-score: {best_val_f1:.3f})")
    # Log to W&B
    wandb.log({"best_confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

 
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
