#!/usr/bin/env bash
# sync_models.sh
# Sync .checkpoints and .submissions from the HCP cluster to the local machine.
#
# Usage:  ./sync.sh <username> [server=izar.hpc.epfl.ch] [remote_dir=~/NeuroGraphNet]
# Example: ./sync.sh jdoe

set -euo pipefail

# ---------- Argument parsing ----------
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <username> [server] [remote_dir]"
  exit 1
fi

USER="$1"
SERVER="${2:-izar.hpc.epfl.ch}"
REMOTE_DIR="${3:-~/NeuroGraphNet}"

# ---------- Sync checkpoints ----------
rsync -avz "${USER}@${SERVER}:${REMOTE_DIR}/.checkpoints" .

# ---------- Sync submissions ----------
mkdir -p submissions
rsync -avz "${USER}@${SERVER}:${REMOTE_DIR}/.submissions/" submissions
