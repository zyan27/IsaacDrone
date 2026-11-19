#!/usr/bin/env bash
#SBATCH --job-name=dd24_hover
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# Isaac Sim needs an RTX-class GPU (RT cores). Oscar's GPU feature tags are
# lowercase model names. Check what YOUR partition actually has before the first
# submit - Slurm rejects a job that names a feature no node has:
#     sinfo -p gpu -o "%f" | tr ',' '\n' | sort -u
# Override on the command line, e.g.  sbatch --constraint="a6000|a40" ...
#SBATCH --constraint=a5000|geforce3090|a6000|a40
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=oscar/logs/%x_%j.out
#SBATCH --error=oscar/logs/%x_%j.err
set -euo pipefail
source "${REPO_DIR:-$HOME/IsaacDrone}/oscar/common.sh"

AIRFRAME="${AIRFRAME:-dd24}"          # dd24 | crazyflie
NUM_ENVS="${NUM_ENVS:-1024}"
ITERS="${ITERS:-2000}"
STEPS="${STEPS:-128}"
SAVE="${SAVE:-checkpoints/${AIRFRAME}_hover_${SLURM_JOB_ID:-local}.pt}"

mkdir -p "$REPO_DIR/oscar/logs" "$REPO_DIR/checkpoints"
nvidia-smi --query-gpu=name,memory.total --format=csv

isaac_exec "cd '$REPO_DIR' && \
    '$ISAACLAB_DIR/isaaclab.sh' -p train_rl_hover.py \
        --headless \
        --airframe $AIRFRAME \
        --num_envs $NUM_ENVS \
        --iters $ITERS \
        --steps $STEPS \
        --save_path $SAVE"

echo "done -> $REPO_DIR/$SAVE"
