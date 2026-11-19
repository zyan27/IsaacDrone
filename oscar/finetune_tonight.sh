#!/usr/bin/env bash
# Test each seed's best.pt, pick the best seed, fine-tune it with a decaying
# learning-rate ceiling, then test the fine-tuned policy. One command:
#
#   RUN=dd24_hover_1115_1834 MAIL_USER=you@brown.edu bash oscar/finetune_tonight.sh
#
# Cancel the training run first (scancel <jobid>). Compare at the end with:
#   python3 oscar/pick_best.py runs/${RUN}_s0 runs/${RUN}_s1 runs/${RUN}_ft_s*
set -euo pipefail
cd "${REPO_DIR:-$HOME/IsaacDrone}"; mkdir -p oscar/logs
: "${RUN:?set RUN, e.g. RUN=dd24_hover_1115_1834}"
SEED_LIST="${SEED_LIST:-0 1}"
MAIL=(); [ -n "${MAIL_USER:-}" ] && MAIL=(--mail-type=END,FAIL --mail-user="$MAIL_USER")

if [ "$(squeue -u "$USER" -h -t PENDING,RUNNING -o "%j" | grep -cE '^isaac_' || true)" -gt 0 ]; then
    echo "Isaac jobs still queued/running - cancel the training run first:"; squeue -u "$USER" -t PENDING,RUNNING; exit 1
fi
DEPS=""
for s in $SEED_LIST; do
    d="runs/${RUN}_s$s"
    [ -f "$d/best.pt" ] || { echo "missing $d/best.pt"; exit 1; }
    j=$(sbatch --parsable --export=ALL,CKPT="$d/best.pt",OUT_DIR="$d" oscar/eval_isaac.sbatch)
    echo "testing $d/best.pt: job $j"; DEPS="$DEPS:$j"
done
G=$(sbatch --parsable --dependency=afterany"$DEPS" "${MAIL[@]}" \
    --export=ALL,RUN="$RUN",SEED_LIST="$SEED_LIST",FT_ITERS="${FT_ITERS:-1500}",FT_LR="${FT_LR:-3e-4}",FT_LR_FINAL="${FT_LR_FINAL:-1e-5}",FT_TIME="${FT_TIME:-03:30:00}",FT_WALL_MIN="${FT_WALL_MIN:-210}" \
    oscar/finetune_gate.sbatch)
echo "pick best seed + launch fine-tune: job $G"
squeue -u "$USER" -t PENDING,RUNNING
