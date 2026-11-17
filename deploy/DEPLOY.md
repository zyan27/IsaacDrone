# Flying the DD24 policy through MAVROS

Files (this folder):

| file | what |
|---|---|
| `dd24_policy_node.py` | ROS 2 Jazzy node: sensors in, `/mavros/setpoint_raw/attitude` out |
| `dd24_policy_core.py` | input building, policy (numpy), safety logic; tested against the simulator's own inputs |
| `export_policy.py` | checkpoint (`best.pt`) -> `policy.npz` (no PyTorch needed on the Pi) |
| `run_on_drone.sh` | copies the node into the `ros2-mavros` container and starts it |
| `INTERFACE.md` | exact input / output definition |
| `sim_sensor_bridge.py`, `VIRTUAL_DRONE_HANDOFF.md` | virtual Duckiedrone (Duckiematrix) testing: stand-in ToF/flow sensors and the full test plan |
| `NOTES.md` | sensor quirks found on the real drone and what to fix next |

## 1. Export the policy (Oscar) and copy it over

```bash
# Oscar
CKPT=runs/dd24_hover_1115_1834_s0/best.pt sbatch oscar/export_policy.sbatch
# Mac (after the job finishes)
scp mcbronne@ssh.ccv.brown.edu:/users/mcbronne/IsaacDrone/deploy/policy.npz ~/IsaacDrone/deploy/
scp -r ~/IsaacDrone/deploy duckie@noducksgiven.local:~/
```

## 2. PX4 settings (QGroundControl), once

* `COM_OF_LOSS_T` = 0.5 s: if setpoints stop for this long, PX4 leaves offboard.
* The offboard-loss action (`COM_OBL_RC_ACT` on recent PX4): Position/Altitude hold or Land.
* `MPC_THR_HOVER` (hover throttle): note the value. The policy's thrust is centred on
  **0.27** and can command 0.07-0.47. If your real hover throttle is near 0.4 or
  above, tell us before flying (option: `-p thrust_gain:=...`, or retrain).
* An RC switch that changes flight mode out of OFFBOARD. **This is your kill path.**

## 3. Bench test: dry run, PROPS OFF

```bash
ssh duckie@noducksgiven.local
cd ~/deploy && bash run_on_drone.sh -p target_height:=0.8
```
Nothing is sent to PX4 in dry run. Twice a second it prints e.g.
`[RUN DRY] ... | ToF D0.80 U4.00 F1.20 L0.90 R4.00 | flow vx +0.00 vy +0.00 ok | tilt 0.0 deg | thrust 0.27 rates [...]`

Check, holding the drone by hand about 0.8 m above a textured floor:

1. **ToFs:** D = height; F/L/R/U = distances to walls / ceiling, 4.00 = nothing in range
   (the driver's 999 is shown as 4.00). Cover each sensor: only that letter drops.
2. **Tilt:** 0 when level; grows when you tilt it.
3. **Flow signs (important, not verifiable without the real sensor):**
   carry the drone level **forward** (camera side) -> `vx` positive;
   carry it **left** -> `vy` positive; tilt it in place without moving -> both stay near 0.
   If a sign is wrong, add `-p flow_sign_x:=-1.0` and/or `-p flow_sign_y:=-1.0`
   (if forward shows up in `vy`: `-p flow_swap_xy:=true`). `INVALID` close to the floor
   (< 8 cm) or over a blank surface is expected.
4. **Policy reacts the right way.** The log shows signed `roll` (+ = right side down) and
   `pitch` (+ = nose down) and the commanded rates. Tilt **right side down** (roll +10)
   -> roll rate **negative**; **nose down** (pitch +10) -> pitch rate **negative** (it tries
   to level). In dry run nothing is sent, so the drone never responds and commands can
   saturate within 1-2 s: judge only the first moment after you change the tilt, use
   small tilts (5-10 deg), and expect small built-in offsets (level: pitch rate ~ +0.2). Hold the drone by its legs, away from the
   side/top ToFs (they will see your hands).
   If the top ToF reads a few cm with nothing above (own structure in view), run with
   `-p tof_up:=none` (always "no target", like the open-ceiling training scenes).
5. **No `inputs far outside training` warnings** while held level near the target height.
   One means a unit / sign / axis is off: send us the line.
6. Unplug a ToF or stop the flow container: the node keeps running (treats it as
   "no target" / invalid flow) and warns.

Stop with Ctrl-C (if it hangs on shutdown, Ctrl-\).

## 4. First flights (props on, inside the net)

1. Start the node for real: `bash run_on_drone.sh -p dry_run:=false -p target_height:=0.8`.
   Until OFFBOARD it streams calm setpoints (zero rates, thrust 0.27) that PX4 ignores
   in other modes; the log says `WAIT`.
2. Arm and take off **manually** in Altitude / Position mode to about the target height,
   away from the net, level and still.
3. Flip the RC switch to **OFFBOARD**. Log: `mode WAIT -> RUN`. The policy starts from a
   fresh history, like a training episode.
4. **Flip back immediately** if it oscillates, climbs/sinks steadily, or heads for the net.
   Keep the first flights to 5-10 s.
5. The node stops sending (-> PX4 failsafe) if IMU or down ToF data is > 0.5 s old, tilt
   > 60 deg, or height > 2.2 m. Late data (> 0.1 / 0.2 s) sends the calm setpoint instead.

Expect slow horizontal drift (tens of cm over 20 s in simulation): the policy has no
position input, only flow velocity.

## Parameters (`-p name:=value`)

| name | default | |
|---|---|---|
| `dry_run` | `true` | `false` to send setpoints |
| `target_height` | 0.8 | metres; trained 0.6-1.5 |
| `flow_sign_x`, `flow_sign_y`, `flow_swap_xy` | 1, 1, false | from the bench flow check |
| `flow_quality_min` | 1 | flow `quality` below this = invalid |
| `flow_use_gyro_fields` | true | subtract `integrated_*gyro` (PX4 convention) |
| `max_tilt_deg`, `max_height_m` | 60, 2.2 | stop-sending limits |
| `thrust_gain` | 1.0 | scales the thrust setpoint (only if hover throttle is far from training) |
| `robot`, `tof_topic`, `tof_down/up/front/left/right` | noducksgiven, ... | ToF topics: `/{robot}/{name}_tof_driver_node/range` with driver names bottom/top/front/left/right |
| `imu_topic`, `flow_topic`, `setpoint_topic` | MAVROS / Duckietown defaults | |
