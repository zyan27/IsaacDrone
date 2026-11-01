#!/usr/bin/env bash
#SBATCH --job-name=dd24_deploy_check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a6000|a40|a5000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=bitwise_improvement/logs/%x_%j.out
#SBATCH --error=bitwise_improvement/logs/%x_%j.err

# Before flying bitwise_att_oracle on the real DD24:
#  1. export it (attitude mode) to deploy/dd24_oracle_wallmap.npz, numpy checked against torch;
#  2. replay the 10-25 flight logs through the Pi's code (deploy/oracle_core.py);
#  3. fly it in the sim with every randomisation on (no --deterministic: sensor noise, delays,
#     dropouts, hover thrust 0.42-0.52, IMU mounted +-2 deg off, CG offset), rugged walls and the
#     noisy net, wall-map inputs, 3 seeds.
#     sbatch bitwise_improvement/deploy_check.sbatch      (from ~/IsaacDrone)
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/IsaacDrone}"
CKPT=bitwise_improvement/runs/bitwise_att_oracle_8012144/bitwise_att_oracle.pt
NPZ=deploy/dd24_oracle_wallmap.npz
B=bitwise_improvement
cd "$REPO_DIR"
mkdir -p $B/logs $B/videos

echo "== export $(date)"
bash oscar/run_isaac.sh deploy/export_policy.py "$CKPT" "$NPZ" attitude

echo "== replay real logs $(date)"
LOGS=$B/replay_logs_$SLURM_JOB_ID   # inside the repo: the container binds it (34 MB, removed below)
mkdir -p "$LOGS"
for f in data/flight_logs/updown_*.tar.xz; do tar xJf "$f" -C "$LOGS"; done
bash oscar/run_isaac.sh deploy/replay_check.py "$NPZ" "$LOGS"/updown_*_csv
rm -rf "$LOGS"

for SEED in 0 1 2; do
    echo "== sim, all randomisation, rugged walls + net, wall-map inputs, seed $SEED $(date)"
    bash oscar/run_isaac.sh play_px4.py --headless --enable_cameras --action attitude --obs_oracle --oracle_source wallmap \
        --seed "$SEED" --wall_rough --net_noise --seconds 15 --hold 20 \
        --checkpoint "$CKPT" --out $B/videos/deploy_check_s$SEED.mp4
done
echo "done $(date)"
