#!/usr/bin/env bash
# Sync .checkpoints and .submissions directories from the HCP cluster to the local machine.
#
# Usage:  ./sync.sh <username> [server=izar.hpc.epfl.ch] [remote_dir=~/NeuroGraphNet]
# Example: ./sync.sh ldibello

set -euo pipefail

# ---------- Argument parsing ----------
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <username> [server] [remote_dir]"
  exit 1
fi

USER="$1"
SERVER="${2:-izar.hpc.epfl.ch}"
REMOTE_DIR="${3:-~/NeuroGraphNet}"

# ---------- Sync ----------
rsync -avz "${USER}@${SERVER}:${REMOTE_DIR}/.checkpoints"   .
rsync -avz "${USER}@${SERVER}:${REMOTE_DIR}/.submissions"   .
