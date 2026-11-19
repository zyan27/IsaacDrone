#!/usr/bin/env bash
# Record MP4s of a trained policy flying in Isaac Sim.
#
#   sbatch oscar/record_video.sbatch                                  # seed 0 of dd24_hover_1115
#   CKPT=runs/dd24_hover_1115_s1/best.pt sbatch oscar/record_video.sbatch
#   VIEWS="chase" SECONDS_LONG=40 sbatch oscar/record_video.sbatch
#
# Makes, next to the checkpoint in videos/:
#   arena.mp4, chase.mp4, top.mp4   normal conditions
#   arena_net70.mp4                 ToFs miss the net 70% of the time (worst case)
#SBATCH --job-name=hover_video
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000|geforce3090|a6000|a40
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=oscar/logs/%x_%j.out
#SBATCH --error=oscar/logs/%x_%j.err
set -euo pipefail
source "${REPO_DIR:-$HOME/IsaacDrone}/oscar/common.sh"

CKPT="${CKPT:-runs/dd24_hover_1115_s0/best.pt}"
VIEWS="${VIEWS:-arena chase top}"
SECONDS_LONG="${SECONDS_LONG:-20}"
cd "$REPO_DIR"
[ -f "$CKPT" ] || { echo "No checkpoint at $CKPT"; exit 1; }

# MP4 writer (one-time, into the scratch user site; skipped once installed).
isaac_exec "/isaac-sim/python.sh -c 'import imageio, imageio_ffmpeg' 2>/dev/null \
    || /isaac-sim/python.sh -m pip install --user --quiet imageio imageio-ffmpeg"

rec() {   # rec <view> [extra args...]
    local view="$1"; shift
    ISAAC_TIMEOUT=15m isaac_exec "cd '$REPO_DIR' && '$ISAACLAB_DIR/isaaclab.sh' -p record_video.py \
        --headless --enable_cameras --checkpoint $CKPT --view $view --seconds $SECONDS_LONG $*" \
        || echo "WARNING: recording '$view $*' failed or timed out"
}

for v in $VIEWS; do rec "$v"; done
rec arena --net_dropout 0.7

ls -lh "$(dirname "$CKPT")/videos/"
echo "Copy to your Mac with:"
echo "  scp -r $USER@ssh.ccv.brown.edu:$REPO_DIR/$(dirname "$CKPT")/videos ~/Downloads/"
