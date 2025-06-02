from datetime import datetime
import numpy as np



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

   

def log(msg):
    """Log message with timestamp to monitor training."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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
