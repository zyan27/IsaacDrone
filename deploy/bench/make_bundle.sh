#!/usr/bin/env bash
# Pack the Pi side into one self-extracting script: deploy/dd24_bundle.sh
#     bash deploy/bench/make_bundle.sh      (from ~/IsaacDrone)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=dd24_bundle.sh
{
cat <<'HEAD'
#!/usr/bin/env bash
# DD24 policy on the Pi. Run where ROS 2 + mavros are sourced (ros2 topic list shows /mavros/*).
#   bash dd24_bundle.sh bench   fake ToFs (down 0.0, sides 1.0) + policy node + monitor  (flight controller only)
#   bash dd24_bundle.sh fly     policy node + monitor, real ToFs                         (whole drone)
# Files go to ~/dd24. Evidence: ~/dd24/evidence.csv (monitor, 1 line/s) and ~/dd24/policy_log.csv (node, 20 Hz).
# Then arm and switch to OFFBOARD: the monitor's "sent" and "px4" columns should match and the motors move.
set -euo pipefail
MODE="${1:-bench}"
D="$HOME/dd24"
mkdir -p "$D"
cd "$D"
HEAD
for f in policy_node.py oracle_core.py bench/fake_tof.py bench/monitor.py; do
    echo "cat > $(basename "$f") <<'DD24_EOF'"
    cat "$f"
    echo "DD24_EOF"
done
echo "base64 -d > dd24_oracle_wallmap.npz <<'DD24_EOF'"
base64 dd24_oracle_wallmap.npz
echo "DD24_EOF"
cat <<'TAIL'
python3 -c "import numpy, rclpy, mavros_msgs.msg, sensor_msgs.msg" || { echo "need numpy + rclpy + mavros_msgs here (source ROS 2 / enter the ROS container)"; exit 1; }
ros2 topic list | grep -q /mavros/imu/data || echo "WARNING: /mavros/imu/data not visible: is mavros running and connected to the flight controller?"
pids=()
trap 'kill "${pids[@]}" 2>/dev/null' EXIT
if [ "$MODE" = bench ]; then
    python3 fake_tof.py --down 0.0 --side 1.0 & pids+=($!)
fi
python3 policy_node.py --policy dd24_oracle_wallmap.npz --log policy_log.csv & pids+=($!)
sleep 2
python3 monitor.py --csv evidence.csv
TAIL
} > "$OUT"
chmod +x "$OUT"
echo "wrote deploy/$OUT ($(du -h "$OUT" | cut -f1))"
