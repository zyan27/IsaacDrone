#!/usr/bin/env bash
# Fly the DD24 policy on Duckietown's VIRTUAL Duckiedrone (Duckiematrix), from your Mac.
# Needs: Docker running, Duckietown Shell `dts` logged in, this repo pulled (deploy/policy.npz).
# Background + full test plan: deploy/VIRTUAL_DRONE_HANDOFF.md. UNTESTED end to end (written
# on Oscar, which has no Docker): if a step fails, the message says which; run it by hand.
#
#   bash deploy/dts_virtual.sh setup      # once: create + start the virtual robot (asks for a password)
#   bash deploy/dts_virtual.sh matrix     # start the Duckiematrix world + attach (leave it running)
#   bash deploy/dts_virtual.sh dry        # copy files in, start stand-in sensors, node in dry run (logs only)
#   bash deploy/dts_virtual.sh fly        # same, node flies: arms, OFFBOARD, takes off, policy, lands after FLIGHT_S
#
# Env: NAME (virtual robot, default vdd24), TARGET (m, 0.8), FLIGHT_S (20), POLICY (deploy/policy.npz)
set -euo pipefail
NAME="${NAME:-vdd24}"
TARGET="${TARGET:-0.8}"
FLIGHT_S="${FLIGHT_S:-20}"
HERE="$(cd "$(dirname "$0")" && pwd)"
POLICY="${POLICY:-$HERE/policy.npz}"
V="dts-virtual-$NAME"          # the virtual robot's container (the "Pi")
IN() { docker exec "$V" bash -c "$*"; }                          # on the virtual robot
ROS() { IN "docker exec ros2-mavros bash -c 'source /opt/ros/jazzy/setup.bash && $*'"; }   # in its MAVROS container

case "${1:-}" in
setup)
    dts duckiebot virtual create -t duckiedrone -c DD24 "$NAME"
    dts duckiebot virtual start "$NAME"
    ;;
matrix)
    dts matrix run --standalone --embedded --map sandbox_drone &
    sleep 20
    dts matrix attach "$NAME" map_0/vehicle_0
    echo "Duckiematrix running (pid $!). Wait until 'docker exec $V docker ps' lists ros2-mavros."
    wait
    ;;
dry|fly)
    docker ps --format '{{.Names}}' | grep -qx "$V" || { echo "$V is not running: bash $0 setup / matrix first"; exit 1; }
    IN "docker ps --format '{{.Names}}' | grep -qx ros2-mavros" || { echo "ros2-mavros not running inside $V yet"; exit 1; }
    echo "== copy node, core, policy, stand-in sensors into $V:/root/deploy"
    IN "mkdir -p /root/deploy"
    for f in dd24_policy_core.py dd24_policy_node.py sim_sensor_bridge.py run_on_drone.sh; do docker cp "$HERE/$f" "$V:/root/deploy/$f"; done
    docker cp "$POLICY" "$V:/root/deploy/policy.npz"
    IN "docker exec ros2-mavros mkdir -p /root/dd24 && docker cp /root/deploy/sim_sensor_bridge.py ros2-mavros:/root/dd24/"
    echo "== MAVROS state / topics"
    ROS "ros2 topic echo --once /mavros/state" || echo "WARNING: no /mavros/state (MAVROS not connected to PX4 SITL?)"
    if [ "$1" = fly ]; then
        # Virtual drone only: no RC in simulation, so allow OFFBOARD/arming without it and land on
        # offboard loss. NEVER run these on the real drone.
        echo "== PX4 params (virtual only)"
        for kv in "COM_RCL_EXCEPT 4" "NAV_RCL_ACT 0" "COM_OF_LOSS_T 0.5" "COM_OBL_RC_ACT 4" "COM_OBL_ACT 1"; do
            set -- $kv
            ROS "ros2 param set /mavros/param $1 $2" || echo "WARNING: could not set $1 (set it in QGroundControl)"
        done
        ROS "ros2 param get /mavros/param MPC_THR_HOVER" || true
    fi
    echo "== stand-in ToFs + optical flow (the virtual drone has none)"
    IN "docker exec -d ros2-mavros bash -c 'source /opt/ros/jazzy/setup.bash && python3 /root/dd24/sim_sensor_bridge.py --ros-args -p robot:=$NAME -p room_half:=1.5 -p ceiling:=2.5 > /root/dd24/bridge.log 2>&1'"
    sleep 3
    ARGS="-p robot:=$NAME -p target_height:=$TARGET"
    [ "$1" = fly ] && ARGS="$ARGS -p dry_run:=false -p auto_start:=true -p takeoff:=true -p flight_s:=$FLIGHT_S"
    echo "== node: $ARGS   (Ctrl-C stops it; record a bag in another terminal, see the handoff doc)"
    docker exec -it "$V" bash -c "cd /root/deploy && bash run_on_drone.sh $ARGS"
    ;;
*)
    sed -n '2,14p' "$0"; exit 1 ;;
esac
