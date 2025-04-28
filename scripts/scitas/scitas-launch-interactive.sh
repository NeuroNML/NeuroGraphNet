#!/usr/bin/env bash

# Connect to the remote server
ssh ldibello@izar.hpc.epfl.ch

# Execture the following commands on the remote server

# launch interactive session on SCITAS (this will connect to a new node)
Sinteract -q ee-452 -p gpu -g gpu:1

# Load gcc and python modules
module load gcc python

# Load additional modules (not required for development but only for developer experience)
module load neovim

# Start jupyter notebook headless on remote server
ipnport=$(shuf -i8000-9999 -n1)
jupyter notebook --no-browser --port="${ipnport}" --ip="$(hostname -i)"

# Now, on your local machine, you can create an SSH tunnel to the remote server
ssh -L 8306:10.91.27.54:8306 -l ldibello izar.hpc.epfl.ch -f -N
