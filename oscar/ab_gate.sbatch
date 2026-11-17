#!/usr/bin/env bash
# After the A/B smoke test: if healthy, launch variant A and/or B (see oscar/ab_test.sh).
#SBATCH --job-name=ab_gate
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=oscar/logs/%x_%j.out
set -euo pipefail
cd "${REPO_DIR:-$HOME/IsaacDrone}"
if ! python3 oscar/check_smoke.py runs/smoke_s0; then
    echo "Smoke test failed the checks; A/B runs NOT started."
    exit 1
fi
MAIL=(); [ -n "${MAIL_USER:-}" ] && MAIL=(--mail-type=END,FAIL --mail-user="$MAIL_USER")
if [ "${SKIP_A:-0}" != "1" ]; then
    EXTRA="--reward v1 --gamma 0.99" sbatch "${MAIL[@]}" --job-name=isaac_hover \
        --export=ALL,SEED=0,RUN_TAG="${STAMP}_A_imufix",ITERS="$ITERS" oscar/train_eval_isaac.sbatch
fi
EXTRA="--reward calm --gamma 0.995" sbatch "${MAIL[@]}" --job-name=isaac_hover \
    --export=ALL,SEED=0,RUN_TAG="${STAMP}_B_calm",ITERS="$ITERS" oscar/train_eval_isaac.sbatch
echo "Launched: $([ "${SKIP_A:-0}" != 1 ] && echo "runs/${STAMP}_A_imufix_s0 (accelerometer fix, original reward) and ")runs/${STAMP}_B_calm_s0 (accelerometer fix + calm reward, gamma 0.995), ${ITERS} iterations"
