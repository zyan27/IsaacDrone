#!/usr/bin/env bash
# Export a checkpoint to deploy/policy.npz for the drone (numpy only on the Pi).
#   CKPT=runs/dd24_hover_1115_1834_s0/best.pt sbatch oscar/export_policy.sbatch
#SBATCH --job-name=export_policy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000|geforce3090|a6000|a40
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=oscar/logs/%x_%j.out
#SBATCH --error=oscar/logs/%x_%j.err
set -euo pipefail
source "${REPO_DIR:-$HOME/IsaacDrone}/oscar/common.sh"
cd "$REPO_DIR"
: "${CKPT:?set CKPT=runs/<run>/best.pt}"
OUT="${OUT:-deploy/policy.npz}"
isaac_exec "cd '$REPO_DIR' && /isaac-sim/python.sh deploy/export_policy.py $CKPT $OUT"
echo "Copy to your Mac with:"
echo "  scp $USER@ssh.ccv.brown.edu:$(readlink -m "$OUT") ~/IsaacDrone/deploy/"
