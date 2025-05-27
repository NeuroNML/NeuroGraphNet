#!/bin/bash
#SBATCH --job-name=eeg_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Parse config argument
CONFIG=$1
LOG_NAME=$(basename "$CONFIG" .yaml)

# Activate conda
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nml_env

# Run and redirect output manually
python scripts/train.py --config "$CONFIG" > logs/${LOG_NAME}.out 2> logs/${LOG_NAME}.err
