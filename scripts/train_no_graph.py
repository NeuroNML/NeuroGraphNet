# -------- Import libraries -------- #
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import importlib

import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, WeightedRandomSampler, DataLoader

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import wandb
from omegaconf import OmegaConf

# --------------------- Custom imports --------------------- #
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from src.data.dataset_no_graph import EEGTimeSeriesDataset
from src.utils.general_funcs import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cnnbilstm.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(f"configs/{args.config}")
    run_name = generate_run_name(config)
    log(f"Run name: {run_name}")

    wandb.init(
        project="eeg-seizure",
        config=OmegaConf.to_container(config, resolve=True),
        name=run_name,
    )

    log("wandb login successful")

    # -------- Directories -------- #
    DATA_ROOT = Path("data")
    clips_df = pd.read_parquet(DATA_ROOT / "train/segments.parquet")
    extracted_features_dir = DATA_ROOT / "extracted_features"
    embeddings_dir =  DATA_ROOT / "embeddings"

    # ------------- Prepare training data -----------------#
    clips_df = clips_df[~clips_df.label.isna()].reset_index()


    dataset = EEGTimeSeriesDataset(
        clips=clips_df,
        signal_folder="data/train",
        extracted_features_dir=extracted_features_dir,
        selected_features_train=config.selected_features,
        embeddings_dir = embeddings_dir,
        embeddings_train = config.embeddings,
        segment_length=3000,
        apply_filtering=True,
        apply_rereferencing=False,
        apply_normalization=False,
        sampling_rate=250,
    )

    # -------- Split data -------- #
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.seed)
    y = clips_df.label.values
    groups = clips_df.patient.values
    X = np.zeros(len(y))
    train_ids, val_ids = next(cv.split(X, y, groups))
    """
    y = clips_df.label.values
    train_ids, val_ids = train_test_split(
    np.arange(len(y)),
    test_size=0.2,
    stratify = y,
    random_state=config.seed
)   """


    # 2. From dataset generate train and val datasets
    train_dataset = Subset(dataset, train_ids)
    val_dataset = Subset(dataset, val_ids)

    
    
    # 3. Compute sample weights for oversampling
    train_labels = [clips_df.iloc[i]["label"] for i in train_ids]
    class_counts = np.bincount(train_labels)
    class_weights = (1. / class_counts)  ** config.oversampling_power# Higher weights for not frequent classes
    sample_weights = [class_weights[label] for label in train_labels] # Assign weight to each sample based on its class
    
    # 4. Define sampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True) # Still train on N samples per epoch, but instead of sampling uniformly takes more from minority class
    
    # Define dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    '''
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    '''
   

    # -------- Load model -------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config.model
    module = importlib.import_module(f"src.models.{model_cfg.type}")
    model_class = getattr(module, model_cfg.name)
    model = model_class(**model_cfg.params).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # Penalize more positive samples
    
    imbalance_ratio = class_counts[0] / class_counts[1]
    #adjusted_pos_weight = torch.tensor([math.sqrt(imbalance_ratio)], dtype=torch.float32).to(device)
    adjusted_pos_weight = torch.tensor([1.5], dtype=torch.float32).to(device)
    log(f'pos_weigth:{adjusted_pos_weight}')
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=adjusted_pos_weight)
    #loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_val_f1 = 0
    best_val_auc = 0
    counter = 0
    patience = 30
    log("Training started")

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x) 
            #logits = model(x)   
            loss = loss_fn(logits, y.reshape(-1, 1))
            #loss = loss_fn(out, y.reshape(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)  
                #logits = model(x)  
                loss = loss_fn(logits, y.reshape(-1, 1))
                val_loss += loss.item()
                probs = torch.sigmoid(logits).squeeze()  # [batch_size, 1] -> [batch_size]
                all_probs.extend(probs.cpu().numpy().ravel())
                preds = (probs > 0.5).int()
                all_preds.extend(preds.cpu().numpy().ravel())
                probs = torch.sigmoid(logits).squeeze()
                all_labels.extend(y.int().cpu().numpy().ravel())

        avg_val_loss = val_loss / len(val_loader)

        # Sweep for best threshold
        all_preds = np.array(all_preds).astype(int)
        all_labels = np.array(all_labels).astype(int)
        #best_thresh, _ = sweep_thresholds(all_probs, all_labels, metric="f1")
        #best_preds = (all_probs > best_thresh).astype(int)

        # Compute metrics
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        precision = precision_score(all_labels, all_preds, average=None)
        recall = recall_score(all_labels, all_preds, average=None)
        f1 = f1_score(all_labels, all_preds, average=None)
        val_auc = roc_auc_score(all_labels, all_probs)

        log(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}| Val AUC: {val_auc:.4f}")
        #log(f"[Threshold sweep] Best threshold: {best_thresh:.2f}")

        # Confusion matrix + class 1 metrics
        confusion_matrix_plot(all_preds, all_labels)
        log(f"Class 1 — Precision: {precision[1]:.2f}, Recall: {recall[1]:.2f}, F1: {f1[1]:.2f}")

        # W&B logging
        wandb.log({
            "epoch": epoch,
           
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_f1": val_f1,
            "val_auc": val_auc,
            "val_f1_class_1": f1[1],
            "val_f1_class_0": f1[0]
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_f1_epoch = epoch
            best_state_dict = model.state_dict().copy()
            best_preds = all_preds.copy()
            best_labels = all_labels.copy()
            wandb.summary["best_f1_score"] = val_f1
            wandb.summary["f1_score_epoch"] = epoch
        if val_auc > best_val_auc: 
            best_val_f1 = val_f1
            best_val_f1_epoch = epoch
            wandb.summary["best_auc_score"] = val_auc
            wandb.summary["best_auc_epoch"] = epoch

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                log("Early stopping triggered.")
                break
    # -------------- Sve confusion matrix --------------------#
    cm = confusion_matrix(best_labels, best_preds,  normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Create a figure for the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Best Confusion Matrix (Epoch {best_val_f1_epoch}), F1-score: {best_val_f1:.3f})")
    # Log to W&B
    wandb.log({"best_confusion_matrix": wandb.Image(fig)})
    plt.close(fig)



    # Save  embeddings
    '''
    model.load_state_dict(best_state_dict)
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    # New DataLoader with all samples
    embedding_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in embedding_loader:
            x, y = batch
            x = x.to(device)
            _, embeddings = model(x) # [B, 128]

            for i in range(x.size(0)):
                all_embeddings.append(embeddings[i].cpu().numpy())  # shape: [19, 128]
                all_labels.append(y[i].item())
    # Save
    log(np.array(all_embeddings).shape)
    np.save(embeddings_dir /"embeddings.npy", np.array(all_embeddings))  # shape: [N, 19, 128]
    np.save(embeddings_dir /"labels_embeddings.npy", np.array(all_labels))          # shape: [N]
    '''

    log(f"Best validation F1: {best_val_f1:.4f}")
    model_path = "model.pt"
    torch.save(best_state_dict, model_path)
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    os.remove(model_path)
    wandb.finish()


if __name__ == "__main__":
    main()
