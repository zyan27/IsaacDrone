#!/usr/bin/env bash
# Fine-tune from a saved policy, alongside another run (own checkpoint and videos).
# Inside an interactive GPU job:
#     bash oscar/finetune_px4.sh                                     # from run 1's best
#     INIT=runs/<run>/best_policy.pt NAME=px4_ft2 bash oscar/finetune_px4.sh
# Watch it: bash oscar/snapshot.sh checkpoints/<NAME>.pt
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
INIT="${INIT:-runs/run1_7979979/best_policy.pt}"
NAME="${NAME:-px4_finetune}"
NUM_ENVS="${NUM_ENVS:-1024}"
ITERS="${ITERS:-600}"
cd "$REPO_DIR"
mkdir -p checkpoints videos

echo "== 1/3 fine-tune $INIT -> checkpoints/$NAME.pt $(date)"
bash oscar/run_isaac.sh train_rl_px4.py --headless --num_envs "$NUM_ENVS" --iters "$ITERS" --steps 32 \
    --init_checkpoint "$INIT" --save_path "checkpoints/$NAME.pt"

echo "== 2/3 policy video $(date)"
bash oscar/run_isaac.sh play_px4.py --headless --enable_cameras --seconds 15 --hold 20 \
    --checkpoint "checkpoints/$NAME.pt" --out "videos/${NAME}.mp4"

echo "== 3/3 export $(date)"
bash oscar/run_isaac.sh deploy/export_policy.py "checkpoints/$NAME.pt" "deploy/${NAME}.npz"

echo "done $(date) -> $REPO_DIR/videos/${NAME}.mp4, $REPO_DIR/deploy/${NAME}.npz"
