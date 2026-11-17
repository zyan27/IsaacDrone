#!/usr/bin/env bash
# Fine-tune a policy to also take off from the floor, with the DD24's real IMU vibration:
# --init_from FROM, --real_hover --real_noise --ground_start_prob 0.3, calm reward.
# Then export + all tests (oscar/test_policy_full.sbatch's list).
#     FROM=runs/dd24_realhover_calm_s0/best.pt sbatch oscar/finetune_takeoff.sbatch
#SBATCH --job-name=dd24_takeoff_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000|geforce3090|a6000|a40|l40s
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=oscar/logs/%x_%j.out
#SBATCH --error=oscar/logs/%x_%j.err
set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
cd "$REPO_DIR"
FROM="${FROM:-runs/dd24_realhover_calm_s0/best.pt}"
RUN="runs/${RUN_TAG:-dd24_takeoff_noise}_s0"
ITERS="${ITERS:-900}"
R() { timeout --kill-after=60 "$1" bash oscar/run_isaac.sh "${@:2}"; }
mkdir -p $RUN
echo "== train $(date)"
R 190m train_rl_hover.py --headless --airframe dd24 --real_hover --real_noise --ground_start_prob 0.3 \
    --reward calm --gamma 0.995 --init_from $FROM --num_envs 1024 --iters $ITERS --steps 128 --seed 0 \
    --run_dir $RUN --time_limit_min 170 || echo "WARNING: train exited non-zero"
echo "== standard test (as trained: 30% floor starts) $(date)"
R 15m eval_rl_hover.py --headless --checkpoint $RUN/best.pt --num_envs 256 --out $RUN/eval.json
R 10m deploy/export_policy.py $RUN/best.pt $RUN/policy.npz
RUN=$RUN bash oscar/test_policy_full.sbatch
