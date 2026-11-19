#!/usr/bin/env bash
# Submit an overnight batch: Isaac (GPU) and RotorPy (CPU) training, each with
# 3 seeds, each job training then testing itself. Run from a login node:
#
#   bash oscar/overnight.sh                       # both sims, 3 seeds each
#   SEEDS=0-4 bash oscar/overnight.sh             # 5 seeds
#   ONLY=rotorpy bash oscar/overnight.sh          # just one track
#   MAIL_USER=you@brown.edu bash oscar/overnight.sh   # email when each job ends/fails
#
# In the morning:  python oscar/summarize_runs.py
set -euo pipefail
cd "${REPO_DIR:-$HOME/IsaacDrone}"
mkdir -p oscar/logs

SEEDS="${SEEDS:-0-2}"
ONLY="${ONLY:-all}"
# Month-day-hour-minute, so two submissions never share run folders.
STAMP="$(date +%m%d_%H%M)"

# Refuse to start a second batch on top of one that's still queued/running.
# Only queued/running jobs count; jobs stuck "completing" (CG) after a cancel don't.
EXISTING="$(squeue -u "$USER" -h -t PENDING,RUNNING -o "%j" | grep -cE '^(isaac_hover|rotorpy_hover)' || true)"
if [ "$EXISTING" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
    echo "You already have $EXISTING training job(s) queued or running:"
    squeue -u "$USER" -t PENDING,RUNNING -o "%.14i %.16j %.8T %.10M %.20R" | grep -E 'JOBID|isaac_ho|rotorpy_' || true
    echo "Cancel them first (scancel -u $USER) or rerun with FORCE=1 to add another batch."
    exit 1
fi
MAIL=()
[ -n "${MAIL_USER:-}" ] && MAIL=(--mail-type=END,FAIL --mail-user="$MAIL_USER")

# Exploratory accounts: 2 GPUs + 12 CPU cores at once across all GPU jobs.
# %2 runs at most 2 Isaac seeds together (6 cores each); the 3rd starts when one ends.
if [ "$ONLY" = "all" ] || [ "$ONLY" = "isaac" ]; then
    RUN_TAG="dd24_hover_${STAMP}" sbatch --array="${SEEDS}%2" "${MAIL[@]}" \
        --export=ALL,RUN_TAG="dd24_hover_${STAMP}" oscar/train_eval_isaac.sbatch
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "rotorpy" ]; then
    sbatch --array="${SEEDS}" "${MAIL[@]}" \
        --export=ALL,RUN_TAG="rotorpy_hover_${STAMP}" oscar/train_eval_rotorpy.sbatch
fi
echo
squeue -u "$USER" -o "%.12i %.16j %.8T %.10M %.20R"
echo "Logs: oscar/logs/   Results: runs/*_${STAMP}_s*/eval.json"
