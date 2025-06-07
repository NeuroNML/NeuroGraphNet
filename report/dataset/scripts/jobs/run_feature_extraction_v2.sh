#!/usr/bin/env bash

#SBATCH --job-name=feature_extraction_v2
#SBATCH --output=feature_extraction_v2_%j.log
#SBATCH --partition=gpu
#SBATCH --account=ee-452
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --time=03:00:00

echo "Loading modules…"
module load gcc python

echo "Activating virtualenv…"
source "$HOME/venvs/neuro/bin/activate"

# Enter project directory
cd ~/NeuroGraphNet || exit

# Run feature extraction script
python3 feature_extraction_v2.py