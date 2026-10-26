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

# DD24 through the PX4 inner loop, policy on down ToF + IMU only:
#   1. scripted takeoff/hold/land video (checks the env before the long run)
#   2. PPO training (train_rl_px4.py)
#   3. video of the trained policy flying the real-drone test profile
#   4. export to deploy/policy.npz for the Pi (deploy/policy_node.py)
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
NUM_ENVS="${NUM_ENVS:-1024}"
ITERS="${ITERS:-1000}"
cd "$REPO_DIR"
mkdir -p oscar/logs checkpoints videos

echo "== 1/4 scripted check $(date)"
bash oscar/run_isaac.sh play_px4.py --headless --enable_cameras --seconds 12 --out videos/px4_scripted_updown.mp4

echo "== 2/4 train $(date)"
bash oscar/run_isaac.sh train_rl_px4.py --headless --num_envs "$NUM_ENVS" --iters "$ITERS" \
    --save_path checkpoints/px4_updown.pt

echo "== 3/4 policy video $(date)"
bash oscar/run_isaac.sh play_px4.py --headless --enable_cameras --seconds 15 \
    --checkpoint checkpoints/px4_updown.pt --out videos/px4_policy_updown.mp4

echo "== 4/4 export $(date)"
bash oscar/run_isaac.sh deploy/export_policy.py checkpoints/px4_updown.pt deploy/policy.npz

echo "done $(date) -> $REPO_DIR/videos/px4_policy_updown.mp4, $REPO_DIR/deploy/policy.npz"
