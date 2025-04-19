#!/bin/bash

#SBATCH --job-name=jupyter_notebook  # Job name
#SBATCH --output=jupyter_notebook_%j.log # Standard output and error log
#SBATCH --partition=gpu             # Specify the partition
#SBATCH --qos=ee-452                # Specify the Quality of Service if needed
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --time=02:00:00             # Time limit hrs:min:sec (2 hours)

# --- Job Steps ---

# 1. Load necessary modules
echo "Loading modules..."
module load gcc python

# 2. Get the hostname of the compute node where the job is running
NODE_HOSTNAME=$(hostname)
echo "Job will run on node: ${NODE_HOSTNAME}"

# 3. Generate a random port number between 8000-9999
JUPYTER_PORT=$(shuf -i 8000-9999 -n 1)
echo "Jupyter will run on port: ${JUPYTER_PORT}"

# 4. Print instructions for SSH tunnel (BEFORE starting Jupyter)
#    Replace 'ldibello@izar.hpc.epfl.ch' with your actual username and login node address
#    Replace '8888' with your preferred local port if you don't want to use 8888
echo "---------------------------------------------------------------------"
echo "Job Started: ${SLURM_JOB_ID} on ${NODE_HOSTNAME}"
echo ""
echo "To connect to the Jupyter Notebook:"
echo "1. Open a NEW terminal on your LOCAL machine (not on izar)."
echo "2. Run the following SSH tunnel command:"
echo "   ssh -N -L 8888:${NODE_HOSTNAME}:${JUPYTER_PORT} <username>@izar.hpc.epfl.ch"
echo ""
echo "   (Replace '8888' if you want to use a different local port)"
echo "   (You might need to enter your password for izar.hpc.epfl.ch)"
echo ""
echo "3. Once the tunnel is running, open a web browser on your LOCAL machine."
echo "4. Go to the following address:"
echo "   http://localhost:8888"
echo ""
echo "   (If you used a different local port in step 2, use that port number here)"
echo "---------------------------------------------------------------------"
echo ""
echo "Starting Jupyter Notebook..."

# 5. Start the Jupyter Notebook server (disable authentication to avoid looking at logs to extract token)
jupyter notebook --no-browser --port="${JUPYTER_PORT}" --ip="${NODE_HOSTNAME}" --NotebookApp.token='' --NotebookApp.password=''

# The script will remain active here while Jupyter Notebook is running.
# The job will end when Jupyter stops or the time limit is reached.

echo "Jupyter Notebook server stopped."
