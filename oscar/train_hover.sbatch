#!/usr/bin/env bash
#SBATCH --job-name=dd24_hover
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# Isaac Sim NEEDS an RTX (RT-core) GPU. On Oscar that means l40s / a6000 / a40 /
# a5000 — NOT a100/v100/h100/b200 (no RT cores, Isaac Sim rendering unsupported,
# so the gpu-he V100 partition will NOT work). The OR-constraint grabs any RTX card.
#SBATCH --constraint=l40s|a6000|a40|a5000
#SBATCH --cpus-per-task=8
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
export ISAAC_CACHE="${ISAAC_CACHE:-$HOME/scratch/isaac/cache}"
mkdir -p "$ISAAC_CACHE"/{kit,ov,pip,glcache,computecache,logs,data,documents}

# apptainer --nv exposes the GPU + driver. Bind the repo, Isaac Lab, and caches.
# ACCEPT_EULA / OMNI_KIT_ACCEPT_EULA keep the container from blocking on prompts.
apptainer exec --nv \
    --env ACCEPT_EULA=Y \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --bind "$REPO_DIR:$REPO_DIR" \
    --bind "$ISAACLAB_DIR:$ISAACLAB_DIR" \
    --bind "$ISAAC_CACHE/kit:/isaac-sim/kit/cache" \
    --bind "$ISAAC_CACHE/ov:/root/.cache/ov" \
    --bind "$ISAAC_CACHE/pip:/root/.cache/pip" \
    --bind "$ISAAC_CACHE/glcache:/root/.cache/nvidia/GLCache" \
    --bind "$ISAAC_CACHE/computecache:/root/.nv/ComputeCache" \
    --bind "$ISAAC_CACHE/logs:/root/.nvidia-omniverse/logs" \
    --bind "$ISAAC_CACHE/data:/root/.local/share/ov/data" \
    --bind "$ISAAC_CACHE/documents:/root/Documents" \
    "$SIF" \
    bash -lc "cd '$REPO_DIR' && \
        '$ISAACLAB_DIR/isaaclab.sh' -p train_rl_hover.py \
            --headless \
            --num_envs $NUM_ENVS \
            --iters $ITERS \
            --save_path checkpoints/rl_hover.pt"

echo "done -> $REPO_DIR/checkpoints/rl_hover.pt"
