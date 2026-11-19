#!/usr/bin/env bash
# Compare two training variants side by side (one GPU each):
#   A = accelerometer fix, original reward (v1, gamma 0.99)
#   B = accelerometer fix + calm reward (crash 40, horizontal-speed + tilt + smoothness
#       penalties, gamma 0.995)
# A 40-min smoke test (with the calm reward) runs first; A and B only start if it passes.
#
#   ITERS=2000 MAIL_USER=you@brown.edu bash oscar/ab_test.sh
#   SKIP_A=1 ITERS=2000 bash oscar/ab_test.sh     # A already running elsewhere: only B
#
# Compare afterwards:
#   python3 oscar/pick_best.py runs/<stamp>_A_imufix_s0 runs/<stamp>_B_calm_s0 runs/dd24_hover_1115_1834_s0
set -euo pipefail
cd "${REPO_DIR:-$HOME/IsaacDrone}"; mkdir -p oscar/logs
STAMP="$(date +%m%d_%H%M)"
ITERS="${ITERS:-2000}"
MAIL=(); [ -n "${MAIL_USER:-}" ] && MAIL=(--mail-type=END,FAIL --mail-user="$MAIL_USER")
RUNNING="$(squeue -u "$USER" -h -t PENDING,RUNNING -o "%j" | grep -cE '^(isaac_hover|gate|ab_gate)' || true)"
if [ "$RUNNING" -gt 0 ] && [ "${SKIP_A:-0}" != "1" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "Training jobs are already queued/running:"; squeue -u "$USER" -t PENDING,RUNNING
    echo "Cancel them (scancel <id>), or if that's your accelerometer-fix run, use SKIP_A=1 to add only B."
    exit 1
fi
rm -rf runs/smoke_s0
SMOKE=$(EXTRA="--reward calm --gamma 0.995" sbatch --parsable --time=00:40:00 \
    --export=ALL,RUN_TAG=smoke,SEED=0,ITERS=20,WALL_MIN=40,RESUBMITS=0 oscar/train_eval_isaac.sbatch)
GATE=$(sbatch --parsable --dependency=afterany:"$SMOKE" "${MAIL[@]}" \
    --export=ALL,STAMP="$STAMP",ITERS="$ITERS",SKIP_A="${SKIP_A:-0}",MAIL_USER="${MAIL_USER:-}" oscar/ab_gate.sbatch)
echo "smoke test (calm reward): job $SMOKE  ->  check + launch A/B: job $GATE"
echo "runs will be: runs/${STAMP}_A_imufix_s0 $([ "${SKIP_A:-0}" = 1 ] && echo '(skipped)')  and  runs/${STAMP}_B_calm_s0"
squeue -u "$USER" -t PENDING,RUNNING
