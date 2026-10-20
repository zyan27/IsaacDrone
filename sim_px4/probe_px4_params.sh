#!/usr/bin/env bash
# Boot dt-px4's PX4 SITL (SIH airframe, no external simulator) and print the
# effective controller params, to diff against sim_px4/px4_defaults.py.
#   bash sim_px4/probe_px4_params.sh [dt-px4.sif]
set -euo pipefail

SIF="$(readlink -f "${1:-$HOME/scratch/px4/dt-px4-ente-amd64.sif}")"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/px4probe.XXXX")"

apptainer exec --cleanenv --bind "$WORK" \
    --env PX4_SIMULATOR=sihsim --env PX4_SYS_AUTOSTART=10040 \
    "$SIF" bash -c '
set -u
B=/workspaces/PX4-Autopilot/build/px4_sitl_default
cd "$0"
"$B/bin/px4" -d -w "$0" -s etc/init.d-posix/rcS "$B/etc" > px4.log 2>&1 &
PID=$!
for _ in $(seq 60); do grep -q "Startup script returned successfully\|pxh> \|Ready for takeoff" px4.log && break; sleep 0.5; done
sleep 3
export PATH="$B/bin:$PATH"
px4-ver all 2>&1 | grep -i -E "ver|hw|git|os" || true
px4-param show -q SYS_AUTOSTART SYS_HITL
for p in THR_MDL_FAC MC_AIRMODE MPC_THR_HOVER \
         MC_ROLLRATE_P MC_ROLLRATE_I MC_ROLLRATE_D MC_ROLLRATE_FF MC_ROLLRATE_K MC_RR_INT_LIM \
         MC_PITCHRATE_P MC_PITCHRATE_I MC_PITCHRATE_D MC_PITCHRATE_FF MC_PITCHRATE_K MC_PR_INT_LIM \
         MC_YAWRATE_P MC_YAWRATE_I MC_YAWRATE_D MC_YAWRATE_FF MC_YAWRATE_K MC_YR_INT_LIM \
         MC_ROLL_P MC_PITCH_P MC_YAW_P MC_YAW_WEIGHT \
         MC_ROLLRATE_MAX MC_PITCHRATE_MAX MC_YAWRATE_MAX \
         IMU_GYRO_CUTOFF IMU_DGYRO_CUTOFF IMU_GYRO_RATEMAX CA_ROTOR0_CT CA_ROTOR0_KM; do
    printf "%-18s %s\n" "$p" "$(px4-param show -q "$p" 2>/dev/null | tr -d "\n")"
done
kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo "--- px4.log tail"; tail -5 px4.log
' "$WORK"
rm -rf "$WORK"
