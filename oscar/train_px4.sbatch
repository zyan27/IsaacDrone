#!/usr/bin/env bash
#SBATCH --job-name=dd24_px4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a6000|a40|a5000
#SBATCH --cpus-per-task=4   # norm-gpu QOS caps a user at 12 CPUs total
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=oscar/logs/%x_%j.out
#SBATCH --error=oscar/logs/%x_%j.err

# DD24 through the PX4 inner loop, sensor-only policy (5 ToF + IMU):
#   1. 5 s scripted flight video (checks env, room and ToF before the long run)
#   2. PPO training (train_rl_px4.py)
#   3. 20 s video of the trained policy
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
NUM_ENVS="${NUM_ENVS:-1024}"
ITERS="${ITERS:-600}"
cd "$REPO_DIR"
mkdir -p oscar/logs checkpoints videos

echo "== 1/3 scripted check $(date)"
bash oscar/run_isaac.sh play_px4.py --headless --enable_cameras --seconds 5 --out videos/px4_room_check.mp4

echo "== 2/3 train $(date)"
bash oscar/run_isaac.sh train_rl_px4.py --headless --num_envs "$NUM_ENVS" --iters "$ITERS" \
    --save_path checkpoints/px4_hover.pt

echo "== 3/3 policy video $(date)"
bash oscar/run_isaac.sh play_px4.py --headless --enable_cameras --seconds 20 \
    --checkpoint checkpoints/px4_hover.pt --out videos/px4_policy.mp4

echo "done $(date) -> $REPO_DIR/videos/px4_policy.mp4"
