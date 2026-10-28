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
# the wall-map estimator (ToF + IMU) instead of ground truth, with a ground-truth control.
# WORLD makes the room harder (--wall_rough, --net_noise); each seed draws a new room.
#     sbatch bitwise_improvement/frozen.sbatch      (from ~/IsaacDrone)
#     TAG=rough WORLD=--wall_rough SEEDS="0 1 2" sbatch bitwise_improvement/frozen.sbatch
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
CKPT="${CKPT:-bitwise_improvement/runs/bitwise_att_oracle_8012144/bitwise_att_oracle.pt}"
SEEDS="${SEEDS:-0}"
SRCS="${SRCS:-wallmap true}"
WORLD="${WORLD:-}"
TAG="${TAG:-}"
B=bitwise_improvement
cd "$REPO_DIR"
mkdir -p $B/logs $B/videos

for SEED in $SEEDS; do
    for SRC in $SRCS; do
        OUT=$B/videos/frozen_oracle${TAG:+_$TAG}_${SRC}${SEED/#/_s}.mp4
        [ "$SEEDS" = 0 ] && OUT=$B/videos/frozen_oracle${TAG:+_$TAG}_$SRC.mp4
        echo "== frozen oracle, oracle inputs from $SRC, seed $SEED, world: ${WORLD:-flat, clean} $(date)"
        bash oscar/run_isaac.sh play_px4.py --headless --enable_cameras --deterministic --action attitude --obs_oracle --oracle_source $SRC --seed "$SEED" $WORLD --seconds 15 --hold 20 \
            --checkpoint "$CKPT" --out "$OUT"
    done
done
echo "done $(date)"
