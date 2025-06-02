#!/usr/bin/env bash

#SBATCH --job-name=feature_extraction
#SBATCH --output=feature_extraction_%j.log
#SBATCH --partition=gpu
#SBATCH --account=ee-452
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SBATCH --time=00:30:00

echo "Loading modules…"
module load gcc python

echo "Activating virtualenv…"
source "$HOME/venvs/neuro/bin/activate"

# Enter project directory
cd ~/NeuroGraphNet || exit

# Run feature extraction script
python3 feature_extraction.py