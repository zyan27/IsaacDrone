#!/usr/bin/env bash
# Train (resumable) + evaluate a RotorPy DD24 policy on CPU nodes (no GPU needed,
# so these start quickly). Array index = seed.
#
#   sbatch --array=0-2 oscar/train_eval_rotorpy.sbatch
#   TASK=takeoff TIMESTEPS=30000000 sbatch --array=0-2 oscar/train_eval_rotorpy.sbatch
#SBATCH --job-name=rotorpy_hover
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=oscar/logs/%x_%A_%a.out
#SBATCH --error=oscar/logs/%x_%A_%a.err
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
VENV="${VENV:-$HOME/envs/dd24}"
SEED="${SLURM_ARRAY_TASK_ID:-${SEED:-0}}"
TASK="${TASK:-hover}"
RUN_TAG="${RUN_TAG:-rotorpy_${TASK}}"
RUN_DIR="runs/${RUN_TAG}_s${SEED}"
TIMESTEPS="${TIMESTEPS:-20000000}"
NUM_ENVS="${NUM_ENVS:-16}"
RESUBMITS="${RESUBMITS:-2}"
WALL_MIN="${WALL_MIN:-600}"
TRAIN_MIN=$(( WALL_MIN - 15 ))

type module >/dev/null 2>&1 || source /etc/profile   # make `module` available in batch shells
PYMOD="$(cat "$VENV/.python_module" 2>/dev/null || true)"
[ -n "$PYMOD" ] && module load "$PYMOD"
source "$VENV/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$REPO_DIR"
mkdir -p "$RUN_DIR" oscar/logs
echo "== $(date) job ${SLURM_JOB_ID:-local} task $TASK seed $SEED -> $RUN_DIR"

python rotorpy_dd24/train_paths.py --task "$TASK" --net --seed "$SEED" \
    --timesteps "$TIMESTEPS" --num_envs "$NUM_ENVS" \
    --run_dir "$RUN_DIR" --resume --time_limit_min "$TRAIN_MIN"

if [ ! -f "$RUN_DIR/DONE" ] && [ "$RESUBMITS" -gt 0 ]; then
    echo "Not finished - resubmitting (RESUBMITS left: $((RESUBMITS-1)))"
    sbatch --export=ALL,SEED="$SEED",TASK="$TASK",RUN_TAG="$RUN_TAG",RESUBMITS=$((RESUBMITS-1)) \
        --job-name="${SLURM_JOB_NAME:-rotorpy_hover}" "$REPO_DIR/oscar/train_eval_rotorpy.sbatch"
fi

python rotorpy_dd24/eval_policy.py --model "$RUN_DIR/latest.zip" --task "$TASK" --net \
    --episodes 256 --out "$RUN_DIR/eval.json"
echo "== $(date) done"
