#!/usr/bin/env bash
# Why/when does a policy fail? Fly it on 512 drones, log everything, then write a
# report, flight-recorder plots and animations of failures and the worst drift.
#
#   CKPT=runs/dd24_hover_1115_1834_s0/best.pt sbatch oscar/flight_recorder.sbatch
#   CKPT=... SCENARIO=fixed_net sbatch oscar/flight_recorder.sbatch     # just the 3x3 net
#   CKPT=... LAG=1 sbatch oscar/flight_recorder.sbatch                  # worst-case lag
#
# Output: <checkpoint dir>/flights_<scenario>[_lag]/ report.txt, plots/, videos/
#SBATCH --job-name=flights
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
cd "$REPO_DIR"
: "${CKPT:?set CKPT=runs/<run>/best.pt}"
[ -f "$CKPT" ] || { echo "No checkpoint at $CKPT"; exit 1; }
SCENARIO="${SCENARIO:-random}"; LAG="${LAG:-0}"; NUM_ENVS="${NUM_ENVS:-512}"
FAILS="${FAILS:-12}"; DRIFTS="${DRIFTS:-4}"
OUT="$(dirname "$CKPT")/flights_${SCENARIO}"
LAGARG=""
if [ "$LAG" = 1 ]; then OUT="${OUT}_lag"; LAGARG="--lag"; fi
mkdir -p "$OUT"
echo "== $(date) $CKPT scenario=$SCENARIO lag=$LAG -> $OUT"

# Plot / video libraries in the Isaac Python (one-time, user site on scratch).
isaac_exec "/isaac-sim/python.sh -c 'import matplotlib, imageio, imageio_ffmpeg' 2>/dev/null \
    || /isaac-sim/python.sh -m pip install --user --quiet matplotlib imageio imageio-ffmpeg"

ISAAC_TIMEOUT=25m isaac_exec "cd '$REPO_DIR' && '$ISAACLAB_DIR/isaaclab.sh' -p flight_recorder.py --headless \
    --checkpoint $CKPT --num_envs $NUM_ENVS --scenario $SCENARIO $LAGARG --out $OUT/flights.npz"
ISAAC_TIMEOUT=25m isaac_exec "cd '$REPO_DIR' && /isaac-sim/python.sh analyze_flights.py $OUT/flights.npz \
    --fails $FAILS --drifts $DRIFTS"

echo
echo "Copy to your Mac with:"
echo "  scp -r $USER@ssh.ccv.brown.edu:$(readlink -f "$OUT") ~/Downloads/"
echo "== $(date) done"
