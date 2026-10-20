#!/usr/bin/env bash
# Build the Isaac Sim Apptainer image ON OSCAR (not on the Mac — Apple Silicon
# can't build or run the x86/CUDA Isaac Sim image).
#
# Run this on an Oscar LOGIN node or an interactive GPU session. Apptainer is
# available on Oscar's GPU nodes without `module load` (per CCV docs); if the
# command is missing, try `module load apptainer`.
#
# Isaac Sim images live on NVIDIA NGC (nvcr.io). You need a (free) NGC account
# and API key the first time:
#     apptainer remote login --username '$oauthtoken' docker://nvcr.io
#     (paste your NGC API key as the password)
#
# VERSION: Dylan's repo targets Isaac Sim 5.0 / Isaac Lab ~2.x. The .sif MUST
# match the Isaac Lab version the code was written against or the isaaclab /
# isaaclab_assets imports (CRAZYFLIE_CFG, DirectRLEnv) will break. Confirm the
# exact tag against the repo before building.
set -euo pipefail

# --- EDIT THESE ---
ISAAC_SIM_TAG="5.0.0"                        # confirm vs repo
SIF_DIR="${SIF_DIR:-$HOME/scratch/isaac}"    # scratch, NOT home (image is ~10+ GB)
# ------------------

# apptainer's layer cache + build tmp default to ~/.apptainer and /tmp; keep the
# ~10+ GB of layers off the home quota.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SIF_DIR/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SIF_DIR/apptainer-tmp}"
mkdir -p "$SIF_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
cd "$SIF_DIR"

echo "Building isaac-sim-${ISAAC_SIM_TAG}.sif in $SIF_DIR ..."
apptainer build "isaac-sim-${ISAAC_SIM_TAG}.sif" \
    "docker://nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_TAG}"

echo
echo "Built: $SIF_DIR/isaac-sim-${ISAAC_SIM_TAG}.sif"
echo "Next: clone Isaac Lab, install it into the container's python, then submit"
echo "      oscar/train_hover.sbatch. See oscar/README.md."
