#!/usr/bin/env bash
# After the tests of a run's seeds: pick the best seed and fine-tune its best.pt.
#SBATCH --job-name=ft_gate
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=oscar/logs/%x_%j.out
set -euo pipefail
cd "${REPO_DIR:-$HOME/IsaacDrone}"
: "${RUN:?}"; : "${SEED_LIST:?}"
DIRS=""; for s in $SEED_LIST; do DIRS="$DIRS runs/${RUN}_s$s"; done
python3 oscar/pick_best.py $DIRS | tee /dev/stderr > /tmp/pick_$$.txt
BEST=$(tail -1 /tmp/pick_$$.txt | awk '$1=="BEST"{print $2}')
if [ -z "$BEST" ]; then echo "No seed has all four tests - fine-tune NOT started."; exit 1; fi
SEED=${BEST##*_s}
echo "Fine-tuning $BEST/best.pt for ${FT_ITERS} iterations -> runs/${RUN}_ft_s${SEED}"
# New random seed (seed+100) so the fine-tune doesn't replay the original run's random draws.
export EXTRA="--init_from $BEST/best.pt --lr ${FT_LR} --lr_max ${FT_LR} --lr_max_final ${FT_LR_FINAL} --seed $((SEED + 100))"
sbatch --time="${FT_TIME}" --job-name=isaac_hover \
    --export=ALL,SEED="$SEED",RUN_TAG="${RUN}_ft",ITERS="$FT_ITERS",WALL_MIN="$FT_WALL_MIN",RESUBMITS=1 \
    oscar/train_eval_isaac.sbatch
