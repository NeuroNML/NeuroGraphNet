#!/usr/bin/env bash

#SBATCH --job-name=eeg_jupyter
#SBATCH --output=eeg_jupyter_%j.log
#SBATCH --partition=gpu
#SBATCH --account=ee-452
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16gb
#SBATCH --time=03:00:00

echo "Loading modules…"
module load gcc python

echo "Activating virtualenv…"
source "$HOME/venvs/neuro/bin/activate"

# Enter project directory
cd ~/NeuroGraphNet || exit

# Setup Jupyter Notebook job
NODE=$(hostname)
PORT=$(shuf -i 8000-9999 -n 1)

# Print useful information
cat <<EOF
===============================================
Job ${SLURM_JOB_ID} on ${NODE}
Tunnel: ssh -N -L 8888:$(hostname -i):${PORT} ${USER}@izar.hpc.epfl.ch
Browse: http://localhost:8888
===============================================
EOF

# Start jupyter
jupyter notebook --no-browser --ip=0.0.0.0 --port="${PORT}" \
  --NotebookApp.token='' --NotebookApp.password=''
