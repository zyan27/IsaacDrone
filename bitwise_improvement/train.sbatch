#!/usr/bin/env bash
#SBATCH --job-name=dd24_bitwise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a6000|a40|a5000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=bitwise_improvement/logs/%x_%j.out
#SBATCH --error=bitwise_improvement/logs/%x_%j.err

# No randomisation at all (deterministic.py): take off from the centre, heading 0, climb
# straight to 0.5 m and hold it. Field reward, from scratch. Everything it writes stays in
# bitwise_improvement/ (checkpoints, videos, logs, runs/<name>_<job>/).
#     sbatch bitwise_improvement/train.sbatch      (from ~/IsaacDrone)
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
NUM_ENVS="${NUM_ENVS:-1024}"
ITERS="${ITERS:-600}"
NAME="${NAME:-bitwise}"
B=bitwise_improvement
cd "$REPO_DIR"
mkdir -p $B/logs $B/checkpoints $B/videos
RUNS_ROOT=$B/runs source oscar/run_record.sh
echo "reward field, deterministic" >> "$RUN_DIR/git.txt"

echo "== 1/3 train $(date)"
bash oscar/run_isaac.sh train_rl_px4.py --headless --deterministic --reward field \
    --num_envs "$NUM_ENVS" --iters "$ITERS" --steps 32 --save_path $B/checkpoints/$NAME.pt

echo "== 2/3 policy video $(date)"
bash oscar/run_isaac.sh play_px4.py --headless --enable_cameras --deterministic --seconds 15 --hold 20 \
    --checkpoint $B/checkpoints/$NAME.pt --out $B/videos/$NAME.mp4

echo "== 3/3 export $(date)"
bash oscar/run_isaac.sh deploy/export_policy.py $B/checkpoints/$NAME.pt $B/$NAME.npz

record_finish $B/checkpoints/$NAME.pt $B/videos/$NAME.mp4 $B/videos/$NAME.csv $B/$NAME.npz
[ -n "${SLURM_JOB_ID:-}" ] && cp -p $B/logs/*_"$SLURM_JOB_ID".out "$RUN_DIR/" 2>/dev/null || true
echo "done $(date) -> $REPO_DIR/$B/videos/$NAME.mp4"
