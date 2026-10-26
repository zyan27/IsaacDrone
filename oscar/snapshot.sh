#!/usr/bin/env bash
# Video of the best policy so far, while training keeps running.
# From the login node, while a dd24_px4 training job is running:
#     bash oscar/snapshot.sh            # newest running dd24_px4 job
#     bash oscar/snapshot.sh 7977545    # a specific job
# Runs play_px4.py inside that job's allocation (same node and GPU, srun --overlap),
# so it needs no new job and doesn't stop training. Takes ~3-4 min and writes
# videos/live/<job>_<time>.mp4 (+ .csv): take off, climb to 0.5 m, hold.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
cd "$REPO_DIR"
JOB="${1:-$(squeue --me -h -t R -n dd24_px4 -o %i | sort -n | tail -1)}"
[ -n "$JOB" ] || { echo "no running dd24_px4 job"; exit 1; }

# The checkpoint that job trains (its --save_path), copied so training can keep rewriting it.
CKPT="$(scontrol write batch_script "$JOB" - | grep -o 'checkpoints/[^ ]*\.pt' | head -1)"
[ -f "$CKPT" ] || { echo "job $JOB has not saved $CKPT yet"; exit 1; }
SNAP="checkpoints/snapshot_${JOB}.pt"
cp "$CKPT" "$SNAP"
OUT="videos/live/${JOB}_$(date +%H%M%S).mp4"
mkdir -p videos/live
echo "job $JOB: $CKPT -> $OUT"

srun --jobid="$JOB" --overlap -n 1 -c 2 --mem=0 --gres=gpu:1 \
    bash oscar/run_isaac.sh play_px4.py --headless --enable_cameras --seconds 15 --hold 20 \
    --checkpoint "$SNAP" --out "$OUT" 2>&1 | grep -E "^policy from|^t=|RESET|wrote|Error|error"
echo "video: $REPO_DIR/$OUT"
