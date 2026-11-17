#!/usr/bin/env bash
# Validate the whole Matias pipeline with --real_hover on one GPU, end to end:
#   1. DD24 URDF -> USD (if missing)   2. short training run (ITERS, default 60)
#   3. eval (the standard test)        4. export -> npz (Pi format)
#   5. deploy/validate_deploy.py: the Pi's numpy Core flies drones in Isaac from the npz;
#      its inputs/outputs are compared with the sim's and the torch actor's.
#     sbatch oscar/validate_deploy.sbatch
#SBATCH --job-name=validate_deploy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000|geforce3090|a6000|a40|l40s
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=oscar/logs/%x_%j.out
#SBATCH --error=oscar/logs/%x_%j.err
set -euo pipefail
# oscar/run_isaac.sh (apptainer --cleanenv): common.sh's isaac_exec leaks the host's conda
# into the container and isaaclab.sh then runs miniconda's python (job 9214470).
REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
cd "$REPO_DIR"
ISAACLAB_DIR="$(readlink -f "$HOME/scratch")/isaac/IsaacLab"
R() { timeout --kill-after=60 "$1" bash oscar/run_isaac.sh "${@:2}"; }
RUN=runs/validate_realhover_s0
ITERS="${ITERS:-60}"
mkdir -p "$RUN"
echo "== 1/5 USD $(date)"
[ -f assets/dd24/dd24.usd ] || R 20m "$ISAACLAB_DIR/scripts/tools/convert_urdf.py" "$REPO_DIR/assets/dd24/dd24.urdf" "$REPO_DIR/assets/dd24/dd24.usd" \
    --joint-target-type none --joint-stiffness 0.0 --joint-damping 0.0 --headless
ls -la assets/dd24/dd24.usd
echo "== 2/5 train $ITERS iters $(date)"
R 40m train_rl_hover.py --headless --airframe dd24 --real_hover \
    --num_envs 1024 --iters $ITERS --steps 128 --seed 0 --run_dir $RUN || echo "WARNING: train exited non-zero"
ls -la $RUN
echo "== 3/5 eval $(date)"
R 15m eval_rl_hover.py --headless --checkpoint $RUN/best.pt --num_envs 256 --out $RUN/eval.json || echo "WARNING: eval exited non-zero"
echo "== 4/5 export $(date)"
R 10m deploy/export_policy.py $RUN/best.pt $RUN/policy.npz
echo "== 5/5 Pi core in the loop $(date)"
R 15m deploy/validate_deploy.py --headless --checkpoint $RUN/best.pt \
    --npz $RUN/policy.npz --num_envs 64 --core_envs 16 --out $RUN/validate.json
echo "== done $(date)"
