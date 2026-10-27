#!/usr/bin/env bash
#SBATCH --job-name=dd24_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a6000|a40|a5000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=oscar/logs/%x_%j.out
#SBATCH --error=oscar/logs/%x_%j.err
# Fine-tune from a saved policy, alongside another run (own checkpoint and videos).
# Inside an interactive GPU job:
#     bash oscar/finetune_px4.sh                                     # from run 1's best
#     INIT=runs/<run>/best_policy.pt NAME=px4_ft2 bash oscar/finetune_px4.sh
# Or as a batch job: INIT=... NAME=... REWARD=field sbatch oscar/finetune_px4.sh
# Watch it: bash oscar/snapshot.sh checkpoints/<NAME>.pt
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
INIT="${INIT:-runs/run1_7979979/best_policy.pt}"
NAME="${NAME:-px4_finetune}"
NUM_ENVS="${NUM_ENVS:-1024}"
ITERS="${ITERS:-600}"
REWARD="${REWARD:-hover}"
cd "$REPO_DIR"
mkdir -p oscar/logs checkpoints videos
source oscar/run_record.sh
echo "init $INIT  reward $REWARD" >> "$RUN_DIR/git.txt"

echo "== 1/3 fine-tune $INIT -> checkpoints/$NAME.pt $(date)"
bash oscar/run_isaac.sh train_rl_px4.py --headless --num_envs "$NUM_ENVS" --iters "$ITERS" --steps 32 \
    --init_checkpoint "$INIT" --reward "$REWARD" --save_path "checkpoints/$NAME.pt"

echo "== 2/3 policy video $(date)"
bash oscar/run_isaac.sh play_px4.py --headless --enable_cameras --seconds 15 --hold 20 \
    --checkpoint "checkpoints/$NAME.pt" --out "videos/${NAME}.mp4"

echo "== 3/3 export $(date)"
bash oscar/run_isaac.sh deploy/export_policy.py "checkpoints/$NAME.pt" "deploy/${NAME}.npz"

record_finish "checkpoints/$NAME.pt" "videos/${NAME}.mp4" "deploy/${NAME}.npz"
echo "done $(date) -> $REPO_DIR/videos/${NAME}.mp4, $REPO_DIR/deploy/${NAME}.npz"
