#!/usr/bin/env bash

#sbatch --job-name=feature_extraction
#sbatch --output=feature_extraction_%j.log
#sbatch --account=ee-452
#sbatch --nodes=1
#sbatch --ntasks=1
#sbatch --mem=16gb
#sbatch --time=00:30:00

echo "Loading modules…"
module load gcc python

echo "Activating virtualenv…"
source "$HOME/venvs/neuro/bin/activate"

# Enter project directory
cd ~/NeuroGraphNet || exit

# Run feature extraction script
python3 feature_extraction.py