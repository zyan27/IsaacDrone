#!/usr/bin/env bash
#SBATCH --job-name=dd24_hover
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# Isaac Sim NEEDS an RTX (RT-core) GPU. On Oscar that means l40s / a6000 / a40 /
# a5000 — NOT a100/v100/h100/b200 (no RT cores, Isaac Sim rendering unsupported,
# so the gpu-he V100 partition will NOT work). The OR-constraint grabs any RTX card.
#SBATCH --constraint=l40s|a6000|a40|a5000
#SBATCH --cpus-per-task=4   # norm-gpu QOS caps a user at 12 CPUs total
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=oscar/logs/%x_%j.out
#SBATCH --error=oscar/logs/%x_%j.err

set -euo pipefail

# --- EDIT THESE (or export before sbatch) ---
SIF="${SIF:-$HOME/scratch/isaac/isaac-sim-5.0.0.sif}"   # from build_sif.sh
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/scratch/isaac/IsaacLab}"  # cloned + installed
REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"                 # this repo, on Oscar
NUM_ENVS="${NUM_ENVS:-256}"
ITERS="${ITERS:-2000}"
# ---------------------------------------------

mkdir -p "$REPO_DIR/oscar/logs" "$REPO_DIR/checkpoints"

# Isaac Sim caches assets/kit under $HOME by default; point them at scratch so
# home quota isn't blown and startup is faster. Also accept the EULAs headlessly.
# ~/scratch is a symlink to /oscar/scratch/$USER, which apptainer does NOT mount
# by default, so bind the real scratch path or every ~/scratch path is missing.
SCRATCH_REAL="$(readlink -f "$HOME/scratch")"
ISAAC_DIR="$SCRATCH_REAL/isaac"
SIF="$(readlink -f "$SIF")"
ISAACLAB_DIR="$(readlink -f "$ISAACLAB_DIR")"
export ISAAC_CACHE="${ISAAC_CACHE:-$ISAAC_DIR/cache}"
mkdir -p "$ISAAC_CACHE"/{kit,kitdata,kitlogs,home}

# The container runs as $USER (not root), so Isaac Sim's caches land under $HOME.
# --home points the container's HOME at scratch so they don't fill the home quota.
# The .sif is read-only: Isaac Lab is pip-installed into PYTHONUSERBASE on scratch
# (see README), and kit's cache/data/logs dirs are bound to writable scratch dirs.
# ACCEPT_EULA / OMNI_KIT_ACCEPT_EULA keep the container from blocking on prompts.
# PYTHONUNBUFFERED: simulation_app.close() hard-exits, which drops buffered prints.
apptainer exec --nv --cleanenv \
    --home "$ISAAC_CACHE/home" \
    --bind "$SCRATCH_REAL" \
    --bind "$REPO_DIR" \
    --bind "$ISAAC_CACHE/kit:/isaac-sim/kit/cache" \
    --bind "$ISAAC_CACHE/kitdata:/isaac-sim/kit/data" \
    --bind "$ISAAC_CACHE/kitlogs:/isaac-sim/kit/logs" \
    --env ACCEPT_EULA=Y \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env PYTHONUNBUFFERED=1 \
    --env PYTHONUSERBASE="$ISAAC_DIR/pyuser" \
    "$SIF" \
    bash -c "cd '$REPO_DIR' && \
        '$ISAACLAB_DIR/isaaclab.sh' -p train_rl_hover.py \
            --headless \
            --num_envs $NUM_ENVS \
            --iters $ITERS \
            --save_path checkpoints/rl_hover.pt"

echo "done -> $REPO_DIR/checkpoints/rl_hover.pt"
