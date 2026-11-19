#!/usr/bin/env bash
# Runs after the smoke test (tiny CPU job): if it looks healthy, launch the long run.
#SBATCH --job-name=gate
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=oscar/logs/%x_%j.out
set -euo pipefail
cd "${REPO_DIR:-$HOME/IsaacDrone}"
if python3 oscar/check_smoke.py runs/smoke_s0; then
    echo "Launching long run: ITERS=${ITERS:-6000} SEEDS=${SEEDS:-0-2} NUM_ENVS=${NUM_ENVS:-1024}"
    ONLY=isaac bash oscar/overnight.sh
else
    echo "Smoke test failed the checks; long run NOT started."
    exit 1
fi
