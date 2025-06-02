#!/usr/bin/env bash

#sbatch --job-name=eeg_jupyter
#sbatch --output=eeg_jupyter_%j.log
#sbatch --partition=gpu
#sbatch --account=ee-452
#sbatch --gres=gpu:1
#sbatch --nodes=1
#sbatch --ntasks=1
#sbatch --cpus-per-task=20
#sbatch --mem=32gb
#sbatch --time=02:00:00

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
