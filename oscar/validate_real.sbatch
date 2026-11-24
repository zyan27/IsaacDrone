#!/usr/bin/env bash
# Fly exported .npz policies in Isaac the way the real drone runs them: the Pi's numpy Core
# (deploy/dd24_policy_core.py) flies every drone; sim = DD24 with the measured 0.47 hover throttle
# and in-flight vibration. Matias's policies are centred on 0.27, so they get thrust_gain 0.47/0.27.
#     sbatch oscar/validate_real.sbatch      -> deploy/eval/real_*.json
#SBATCH --job-name=dd24_valreal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000|geforce3090|a6000|a40|l40s
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=oscar/logs/%x_%j.out
#SBATCH --error=oscar/logs/%x_%j.err
set -uo pipefail
cd "${REPO_DIR:-$HOME/IsaacDrone}"
R() { timeout --kill-after=60 20m bash oscar/run_isaac.sh deploy/validate_deploy.py --headless --npz_only --num_envs 128 "$@"; }
mkdir -p deploy/eval
G=1.7407   # 0.47 / 0.27
for spec in "ours_calm deploy/policy.npz 1.0" "ours_v1 deploy/policy_realhover_v1.npz 1.0" \
            "matias_675 deploy/policy_matias_675.npz $G" "matias_A369 deploy/policy_matias_A_369.npz $G" \
            "matias_675_nogain deploy/policy_matias_675.npz 1.0"; do
    set -- $spec
    echo "== $1 hover (air start) $(date)"
    R --npz $2 --thrust_gain $3 --out deploy/eval/real_$1_hover.json | tail -3
    echo "== $1 takeoff from floor $(date)"
    R --npz $2 --thrust_gain $3 --takeoff --out deploy/eval/real_$1_takeoff.json | tail -3
done
echo "== summary"
for f in deploy/eval/real_*.json; do
    python3 -c "import json,sys; d=json.load(open('$f')); print(f\"{'$f'.split('/')[-1]:38s} survive {d['core_success_rate']:.3f} z {d['core_z_err_mean_m']} xy {d['core_xy_err_mean_m']} takeoff_s {d.get('takeoff_s_mean')}\")"
done
