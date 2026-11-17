#!/usr/bin/env bash
# All tests for one trained policy (Matias pipeline): standard, ToFs miss the net 70%,
# fixed net, worst-case lag, floor starts, plus the Pi core in the loop (parity, and the
# scripted takeoff -> policy hand-over from the floor).
#     RUN=runs/dd24_realhover_calm_s0 sbatch oscar/test_policy_full.sbatch
#SBATCH --job-name=dd24_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000|geforce3090|a6000|a40|l40s
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=oscar/logs/%x_%j.out
#SBATCH --error=oscar/logs/%x_%j.err
set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
cd "$REPO_DIR"
RUN="${RUN:?RUN=runs/<tag>_s0}"
R() { timeout --kill-after=60 "$1" bash oscar/run_isaac.sh "${@:2}"; }
E="eval_rl_hover.py --headless --checkpoint $RUN/best.pt --num_envs 256"
[ -f $RUN/policy.npz ] || R 10m deploy/export_policy.py $RUN/best.pt $RUN/policy.npz
echo "== takeoff (Pi core) $(date)";  R 15m deploy/validate_deploy.py --headless --checkpoint $RUN/best.pt --npz $RUN/policy.npz --num_envs 64 --core_envs 32 --takeoff --out $RUN/validate_takeoff.json
echo "== parity (Pi core) $(date)";   [ -f $RUN/validate.json ] || R 15m deploy/validate_deploy.py --headless --checkpoint $RUN/best.pt --npz $RUN/policy.npz --num_envs 64 --core_envs 16 --out $RUN/validate.json
echo "== lag $(date)";                R 15m $E --lag --out $RUN/eval_lag.json
echo "== net70 $(date)";              R 15m $E --net_dropout 0.7 --out $RUN/eval_net70.json
echo "== fixed net $(date)";          R 15m $E --scenario fixed_net --out $RUN/eval_fixednet.json
echo "== floor starts $(date)";       R 15m $E --ground_start_prob 1.0 --out $RUN/eval_ground.json
echo "== done $(date)"
