#!/usr/bin/env bash
# Run any repo script inside the Isaac Sim container, same binds as train_hover.sbatch.
# Use on a GPU node (salloc), e.g.:
#   bash oscar/run_isaac.sh play_video.py --headless --enable_cameras
set -euo pipefail

SCRATCH_REAL="$(readlink -f "$HOME/scratch")"
ISAAC_DIR="$SCRATCH_REAL/isaac"
SIF="$(readlink -f "${SIF:-$ISAAC_DIR/isaac-sim-5.0.0.sif}")"
ISAACLAB_DIR="$(readlink -f "${ISAACLAB_DIR:-$ISAAC_DIR/IsaacLab}")"
REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
ISAAC_CACHE="${ISAAC_CACHE:-$ISAAC_DIR/cache}"
mkdir -p "$ISAAC_CACHE"/{kit,kitdata,kitlogs,home}

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
    bash -c 'cd "$0" && "$1/isaaclab.sh" -p "${@:2}"' "$REPO_DIR" "$ISAACLAB_DIR" "$@"
