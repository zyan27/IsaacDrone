#!/usr/bin/env bash
# Run the four standard tests on an existing checkpoint (no training).
#   CKPT=runs/x_s0/best.pt OUT_DIR=runs/x_s0 sbatch oscar/eval_isaac.sbatch
# Writes eval.json, eval_net70.json, eval_fixednet.json, eval_lag.json to OUT_DIR.
#SBATCH --job-name=isaac_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000|geforce3090|a6000|a40
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=oscar/logs/%x_%j.out
#SBATCH --error=oscar/logs/%x_%j.err
set -euo pipefail
source "${REPO_DIR:-$HOME/IsaacDrone}/oscar/common.sh"
cd "$REPO_DIR"
: "${CKPT:?set CKPT}"; OUT_DIR="${OUT_DIR:-$(dirname "$CKPT")}"; EVAL_ENVS="${EVAL_ENVS:-512}"
echo "== $(date) testing $CKPT -> $OUT_DIR"
run_eval() {  # $1 = output name, rest = extra args
    local out="$1"; shift
    ISAAC_TIMEOUT=20m isaac_exec "cd '$REPO_DIR' && '$ISAACLAB_DIR/isaaclab.sh' -p eval_rl_hover.py --headless \
        --checkpoint $CKPT --num_envs $EVAL_ENVS $* --out $OUT_DIR/$out" || echo "WARNING: $out failed or timed out"
}
run_eval eval.json
run_eval eval_net70.json --net_dropout 0.7
run_eval eval_fixednet.json --scenario fixed_net
run_eval eval_lag.json --lag
echo "== $(date) done"
