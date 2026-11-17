#!/usr/bin/env bash
# One-time: create the Python venv for the RotorPy track + analysis tools.
# Run on an Oscar LOGIN node (needs internet for pip):
#     bash oscar/setup_venv.sh
#
# Isaac Sim is NOT installed here: its pip wheels need GLIBC >= 2.35 and Oscar's
# OS is older, so Isaac runs inside the Apptainer container (see README).
set -euo pipefail

VENV="${VENV:-$HOME/envs/dd24}"
REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"

# RotorPy/torch need Python >= 3.10. Oscar's system python3 is older, so load a
# newer module. Pick the first 3.11/3.12 module available.
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    MOD="$(module -t avail python 2>&1 | grep -E '^python/3\.1[12]' | sort -V | tail -1 || true)"
    if [ -z "$MOD" ]; then
        echo "No python 3.11/3.12 module found. Run 'module avail python' and set PY_MODULE."; exit 1
    fi
    PY_MODULE="${PY_MODULE:-$MOD}"
    echo "Loading $PY_MODULE"
    module load "$PY_MODULE"
fi
python3 --version

# Remember which module built the venv: the job scripts load the same one.
mkdir -p "$(dirname "$VENV")"
python3 -m venv "$VENV"
echo "${PY_MODULE:-}" > "$VENV/.python_module"
source "$VENV/bin/activate"
pip install --upgrade pip wheel

# CPU-only torch: RotorPy runs on CPU, and the CPU wheel is ~10x smaller.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install rotorpy==3.0.0 stable-baselines3 gymnasium roma torchdiffeq \
    matplotlib tensorboard pandas

python - <<'EOF'
import rotorpy, stable_baselines3, torch
print("venv OK: torch", torch.__version__, "| sb3", stable_baselines3.__version__)
EOF
echo
echo "Done. Activate with:  source $VENV/bin/activate"
