#!/usr/bin/env bash
# Convert the DD24 URDF (built with assets/dd24/build_dd24_urdf.py) to USD on Oscar.
#   sbatch oscar/convert_dd24.sbatch
#SBATCH --job-name=dd24_urdf2usd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=oscar/logs/%x_%j.out
#SBATCH --error=oscar/logs/%x_%j.err
set -euo pipefail
source "${REPO_DIR:-$HOME/IsaacDrone}/oscar/common.sh"

isaac_exec "cd '$REPO_DIR' && ISAACLAB='$ISAACLAB_DIR' bash assets/dd24/convert_to_usd.sh"
ls -la "$REPO_DIR/assets/dd24/"
