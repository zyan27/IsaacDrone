#!/usr/bin/env bash
#SBATCH --job-name=dd24_frozen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a6000|a40|a5000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=bitwise_improvement/logs/%x_%j.out
#SBATCH --error=bitwise_improvement/logs/%x_%j.err

# The bitwise_att_oracle weights, frozen (no training): the 6 oracle inputs are filled from
# the wall-map estimator (ToF + IMU) instead of ground truth. Same seed, a ground-truth control.
#     sbatch bitwise_improvement/frozen.sbatch      (from ~/IsaacDrone)
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
CKPT="${CKPT:-bitwise_improvement/runs/bitwise_att_oracle_8012144/bitwise_att_oracle.pt}"
SEED="${SEED:-0}"
B=bitwise_improvement
cd "$REPO_DIR"
mkdir -p $B/logs $B/videos

for SRC in wallmap true; do
    echo "== frozen oracle, oracle inputs from $SRC $(date)"
    bash oscar/run_isaac.sh play_px4.py --headless --enable_cameras --deterministic --action attitude --obs_oracle --oracle_source $SRC --seed "$SEED" --seconds 15 --hold 20 \
        --checkpoint "$CKPT" --out $B/videos/frozen_oracle_$SRC.mp4
done
echo "done $(date) -> $REPO_DIR/$B/videos/frozen_oracle_{wallmap,true}.mp4"
