#!/usr/bin/env bash

# Create a new virtual environment in home directory
python3 -m venv --system-site-packages venvs/neuro

# Enter the virtual environment
source venvs/neuro/bin/activate

# Install the required packages
pip3 install -r requirements.txt
