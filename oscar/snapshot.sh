#!/usr/bin/env bash
# Video of the policy so far, while training keeps running. From the login node:
#     bash oscar/snapshot.sh                                   # best so far: checkpoints/px4_hold.pt
#     bash oscar/snapshot.sh checkpoints/px4_hold_latest.pt    # current policy
#     JOB=7978456 bash oscar/snapshot.sh                       # a specific job
# Runs play_px4.py inside your newest running GPU job (batch or interactive), on the
# same node and GPU (srun --overlap), so training doesn't stop. Takes ~5-20 min
# (Isaac start-up shares the job's CPUs) and writes videos/live/<job>_<time>.mp4 (+ .csv).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
cd "$REPO_DIR"
CKPT="${1:-checkpoints/px4_hold.pt}"
JOB="${JOB:-$(squeue --me -h -t R -p gpu -o %i | sort -n | tail -1)}"
[ -n "$JOB" ] || { echo "no running GPU job"; exit 1; }
[ -f "$CKPT" ] || { echo "$CKPT not saved yet"; exit 1; }
SNAP="checkpoints/snapshot_${JOB}.pt"
cp "$CKPT" "$SNAP"   # training rewrites $CKPT; play a fixed copy
OUT="videos/live/${JOB}_$(date +%H%M%S).mp4"
mkdir -p videos/live
echo "job $JOB: $CKPT -> $OUT"

srun --jobid="$JOB" --overlap -n 1 -c 2 --mem=0 --gres=gpu:1 \
    bash oscar/run_isaac.sh play_px4.py --headless --enable_cameras --seconds 15 --hold 20 \
    --checkpoint "$SNAP" --out "$OUT" 2>&1 | grep -E "^policy from|^t=|RESET|wrote|Error|error"
echo "video: $REPO_DIR/$OUT"
