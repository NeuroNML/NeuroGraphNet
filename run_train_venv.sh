#!/bin/usr/env bash
#SBATCH --partition=gpu
#SBATCH --account=ee-452
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16gb
#SBATCH --time=03:00:00

# Parse config argument
CONFIG=$1
LOG_NAME=$(basename "$CONFIG" .yaml)

echo "Loading modules…"
module load gcc python

echo "Activating virtualenv…"
source "$HOME/venvs/neuro/bin/activate"

# Enter project directory
cd ~/NeuroGraphNet || exit

# Start 
python scripts/train_no_graph.py --config "$CONFIG" > logs/${LOG_NAME}.out 2> logs/${LOG_NAME}.err