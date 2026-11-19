#!/usr/bin/env bash
# Run the policy node on the drone's Pi, inside the ROS 2 MAVROS container.
#   bash run_on_drone.sh                                   # dry run (default): nothing sent to PX4
#   bash run_on_drone.sh -p target_height:=0.8             # extra ROS parameters
#   bash run_on_drone.sh -p dry_run:=false                 # FLY: streams setpoints to PX4
# Run from the folder holding dd24_policy_core.py, dd24_policy_node.py and policy.npz.
set -euo pipefail
cd "$(dirname "$0")"
C="${CONTAINER:-ros2-mavros}"
for f in dd24_policy_core.py dd24_policy_node.py policy.npz; do
    [ -f "$f" ] || { echo "missing $f here"; exit 1; }
done
docker exec "$C" mkdir -p /root/dd24
for f in dd24_policy_core.py dd24_policy_node.py policy.npz; do docker cp "$f" "$C:/root/dd24/$f"; done
docker exec "$C" python3 -c "import numpy" 2>/dev/null || { echo "numpy is missing in $C: docker exec $C pip install numpy"; exit 1; }
echo "Starting dd24_policy in $C (stop with Ctrl-C; if it hangs, Ctrl-\\)"
docker exec -it "$C" bash -c "source /opt/ros/jazzy/setup.bash && cd /root/dd24 && \
    exec python3 dd24_policy_node.py --ros-args -p policy:=/root/dd24/policy.npz $*"
