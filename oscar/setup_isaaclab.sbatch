#!/usr/bin/env bash
# ONE-TIME: clone Isaac Lab (pinned) and install it into the container's Python.
# Run after build_sif.sh:   sbatch oscar/setup_isaaclab.sbatch
#SBATCH --job-name=isaaclab_setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=oscar/logs/%x_%j.out
#SBATCH --error=oscar/logs/%x_%j.err
set -euo pipefail
source "${REPO_DIR:-$HOME/IsaacDrone}/oscar/common.sh"

if [ ! -d "$ISAACLAB_DIR/.git" ]; then
    git clone --depth 1 --branch "$ISAACLAB_TAG" \
        https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
ln -sfn /isaac-sim "$ISAACLAB_DIR/_isaac_sim"   # point Isaac Lab at the in-container sim

isaac_exec "
    set -e
    PY=/isaac-sim/python.sh
    \$PY -c 'import site,sys; print(\"user site enabled:\", site.ENABLE_USER_SITE, site.getusersitepackages())'
    # isaaclab.sh tries 'sudo apt-get install cmake' unless cmake is already on PATH.
    # No sudo on Oscar, so install cmake from pip into \$PYUSER/bin (on PATH via common.sh).
    \$PY -m pip install --user --quiet cmake
    echo \"cmake: \$(command -v cmake)\"
    echo \"torch seen by pip: \$(\$PY -m pip show torch 2>/dev/null | grep -E '^(Version|Location)' | tr '\\n' ' ')\"
    # isaaclab.sh UNINSTALLS any torch that isn't exactly 2.7.0+cu128 - impossible in
    # the read-only image. Put that exact build in the user site first (it shadows the
    # image's copy), so the installer finds it and leaves things alone.
    TV=\$(\$PY -m pip show torch 2>/dev/null | awk '/^Version:/{print \$2}')
    if [ \"\$TV\" != \"2.7.0+cu128\" ]; then
        echo \"Installing torch 2.7.0+cu128 into the user site (found: \${TV:-none})\"
        \$PY -m pip install --user torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    fi
    # Isaac Lab pins flatdict==4.0.1, whose setup.py imports pkg_resources. Recent
    # setuptools removed pkg_resources, and pip builds packages in a fresh env with
    # the LATEST setuptools, so flatdict fails and core isaaclab never installs.
    # Pre-build it here with an older setuptools and no build isolation; the
    # installer then finds flatdict already satisfied and skips the broken build.
    \$PY -m pip install --user --quiet 'setuptools<81' wheel
    \$PY -m pip install --user --no-build-isolation flatdict==4.0.1
    cd '$ISAACLAB_DIR'
    # 'none' = core Isaac Lab only (we bring our own PPO); much faster install.
    ./isaaclab.sh --install none
    # isaaclab_assets needs the Kit app running (omni.client), so the check starts
    # Isaac Sim headless rather than importing in plain python.
    ./isaaclab.sh -p '$REPO_DIR/oscar/check_isaaclab.py'
"
echo "Isaac Lab installed into $PYUSER"
