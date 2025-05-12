#!/bin/bash

# Configuration
ENV_NAME="nml_env"
ENV_FILE="environment.yml"
PYG_URL="https://data.pyg.org/whl/torch-2.1.0+cu117.html"
CONDA_DIR="$HOME/miniconda3"
CONDA_SH="$CONDA_DIR/etc/profile.d/conda.sh"



# Install Miniconda if not found
if [ ! -d "$CONDA_DIR" ]; then
  echo "Miniconda not found, installing..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
  bash ~/miniconda.sh -b -p "$CONDA_DIR"
  rm ~/miniconda.sh
fi

# Initialize conda
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
else
  echo "conda.sh not found. Exiting."
  exit 1
fi

# Create environment
if conda info --envs | grep -q "$ENV_NAME"; then
  echo "Environment $ENV_NAME already exists"
else
  echo "Creating environment from $ENV_FILE"
  conda env create -f "$ENV_FILE"
fi

# Activate environment
conda activate "$ENV_NAME"

# Install PyTorch Geometric packages
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f "$PYG_URL"

# Test install
python -c "import torch; import torch_geometric; print('Torch version:', torch.__version__)"

echo "Environment setup complete"
