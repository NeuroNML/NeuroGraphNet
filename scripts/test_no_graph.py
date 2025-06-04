import wandb
import torch
import sys
from pathlib import Path
import os
import pandas as pd
from torch.utils.data import  DataLoader
# --------------------- Custom imports --------------------- #
# Add the root directory to path
project_root = (
    Path(__file__).resolve().parents[1]
)  # 1 levels up from scripts/ -> repository root ; should directly see src, data, configs
sys.path.append(str(project_root))
from src.data.dataset_no_graph import EEGTimeSeriesDataset
from src.models.cnnlstm_embeddings import CNNLSTMEmb
from src.utils.general_funcs import *

# -------- Directories -------- #
DATA_ROOT = Path("data")
test_dir_metadata = DATA_ROOT / "test/segments.parquet"
extracted_features_dir = DATA_ROOT / "extracted_features"
embeddings_dir =  DATA_ROOT / "embeddings"
# ------------------------------------------------#
clips_te= pd.read_parquet(test_dir_metadata).reset_index()
clips_te['label'] = 0


# Download model file from W&B
api = wandb.Api()
artifact = api.artifact("bizquier1-epfl/eeg-seizure/best_model:v64", type="model")
artifact_dir = artifact.download()
model_path = os.path.join(artifact_dir, "model.pt")


# Load model
model = CNNLSTMEmb(embedding_dim=64, hidden_dim=128)
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
print('Model ready')

# Load test set
dataset_te = EEGTimeSeriesDataset(
        clips=clips_te,
        signal_folder="data/test" ,
        extracted_features_dir=extracted_features_dir,
        selected_features_train=False,
        embeddings_dir = embeddings_dir,
        embeddings_train = False,
        segment_length=3000,
        apply_filtering=True,
        apply_rereferencing=False,
        apply_normalization=False,
        sampling_rate=250,
    )
print(f'Length dataset: {len(dataset_te)}')


# Star evaluation on test set
model.eval()
print('Start evaluation')
# Create test DataLoader
test_loader = DataLoader(dataset_te, batch_size=64)

# 3. Make predictions
model.eval()
all_preds = []
with torch.no_grad():
        for batch in test_loader:
                x, _ = batch
                logits, _ = model(x)  
                probs = torch.sigmoid(logits).squeeze()  # [batch_size, 1] -> [batch_size]
                preds = (probs > 0.5).int()
                all_preds.extend(preds.numpy().ravel())
# Convert to binary predictions
predictions = np.array(all_preds)

# Combine with IDs
# Test-ids

# Create new 'id' column by joining index parts with '_'
clips_te["id"] = clips_te["patient"] + "_" + clips_te["session"] + "_" + clips_te["segment"].astype(str)


submission_df = pd.DataFrame({
    "id": clips_te['id'],
    "label": predictions
})

# 3. Save to CSV (no index)
submission_df.to_csv("submissions/submission.csv", index=False)
print(" Submission file saved as 'submission.csv'")
