#!/usr/bin/env bash
# One command for a big run: 40-min smoke test -> automatic check -> long run.
#
#   ITERS=10000 SEEDS=0-2 MAIL_USER=you@brown.edu bash oscar/smoke_then_overnight.sh
#
# The long run only starts if the smoke test passes oscar/check_smoke.py (all four
# tests ran, learning rate not pinned, KL sane). Check what happened with:
#   cat oscar/logs/gate_*.out
set -euo pipefail
cd "${REPO_DIR:-$HOME/IsaacDrone}"
mkdir -p oscar/logs
MAIL=()
[ -n "${MAIL_USER:-}" ] && MAIL=(--mail-type=END,FAIL --mail-user="$MAIL_USER")

RUNNING="$(squeue -u "$USER" -h -t PENDING,RUNNING -o "%j" | grep -cE '^(isaac_hover|rotorpy_hover|gate)' || true)"
if [ "$RUNNING" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
    echo "You already have training jobs queued/running. Cancel them (scancel -u $USER) or use FORCE=1."
    exit 1
fi

rm -rf runs/smoke_s0
SMOKE=$(sbatch --parsable --time=00:40:00 \
    --export=ALL,RUN_TAG=smoke,SEED=0,ITERS=20,WALL_MIN=40,RESUBMITS=0 oscar/train_eval_isaac.sbatch)
GATE=$(sbatch --parsable --dependency=afterany:"$SMOKE" "${MAIL[@]}" \
    --export=ALL,ITERS="${ITERS:-6000}",SEEDS="${SEEDS:-0-2}",NUM_ENVS="${NUM_ENVS:-1024}" oscar/gate.sbatch)
echo "smoke test: job $SMOKE  ->  check + launch: job $GATE"
echo "long run: ITERS=${ITERS:-6000} SEEDS=${SEEDS:-0-2} NUM_ENVS=${NUM_ENVS:-1024} (only if the smoke test passes)"
squeue -u "$USER" -t PENDING,RUNNING
