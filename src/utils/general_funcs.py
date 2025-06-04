from datetime import datetime
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def log(msg):
    """Log message with timestamp to monitor training."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
    
def labels_stats(y, trains_ids, val_ids):
    # Get train/val labels
    y_train, y_val = y[trains_ids], y[val_ids]

    # Compute class counts
    train_counts = np.bincount(y_train)
    val_counts = np.bincount(y_val)

    # Handle case where one class might be missing (e.g., only 0s or 1s in val set)
    train_0 = train_counts[0] 
    train_1 = train_counts[1] 
    val_0 = val_counts[0] 
    val_1 = val_counts[1] 

    # Log results
    log(f"Train labels: 0 -> {train_0}, 1 -> {train_1}")
    log(f"Val labels:   0 -> {val_0}, 1 -> {val_1}")

def confusion_matrix_plot(all_preds, all_labels):
    
    TP = np.sum((all_preds == 1) & (all_labels == 1))  # True Positives
    TN = np.sum((all_preds == 0) & (all_labels == 0))  # True Negatives
    FP = np.sum((all_preds == 1) & (all_labels == 0))  # False Positives
    FN = np.sum((all_preds == 0) & (all_labels == 1))  # False Negatives


    print("\nConfusion Matrix:", flush=True)
    print("               Predicted", flush=True)
    print("              0       1", flush=True)
    print(f"Actual 0    {TN:5d}   {FP:5d}", flush=True)
    print(f"Actual 1    {FN:5d}   {TP:5d}", flush=True)

   



def generate_run_name(config):
    """Generates a run name from the configuration file."""
    name = []

    model_type = config.model.type
    name.append(f"{model_type}")
    if model_type == "encoder_gnn":
        encoder = config.model.encoder
        gnn = config.model.gnn

        # Add encoder
        name.append(f"{encoder.type}")
        # Add model parameters
        for key, value in encoder.params.items():
            name.append(f"{key}_{value}")

        # Add GNN
        name.append(f"{gnn.type}")
        # Add model parameters
        for key, value in gnn.params.items():
            name.append(f"{key}_{value}")

    else:

        # Simple model
        model_params = config.model.params
        # Add model parameters
        for key, value in model_params.items():
            name.append(f"{key}_{value}")

    # Add all other parameters not related to model
    for key, value in config.items():
        if key != "model":
            name.append(f"{key}_{value}")

    return "_".join(name)


def sweep_thresholds(probs, true_labels, metric="f1"):
    best_score = 0
    best_thresh = 0.5
    for threshold in np.arange(0.1, 0.91, 0.05):
        preds = (probs > threshold).astype(int)
        if metric == "f1":
            score = f1_score(true_labels, preds)
        elif metric == "recall":
            score = recall_score(true_labels, preds)
        elif metric == "precision":
            score = precision_score(true_labels, preds)
        else:
            raise ValueError("Unsupported metric")
        if score > best_score:
            best_score = score
            best_thresh = threshold
    return best_thresh, best_score


