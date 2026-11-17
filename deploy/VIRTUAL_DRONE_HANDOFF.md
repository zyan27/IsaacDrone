# Handoff: test the DD24 hover policy on a VIRTUAL Duckiedrone (Duckiematrix)

You are picking up a project mid-way. Read this whole file before running anything.
Your job: set up Duckietown's virtual DD24, run our policy node against it (real ROS 2,
real MAVROS, PX4 in simulation), rehearse the full flight sequence, and report what
happened. You are NOT asked to change the policy, the training code or the node's
conventions; if something looks wrong, report it with evidence instead of "fixing" it.

## 0. Safety rules (non-negotiable)

* **Only the virtual drone.** The user's real drone is `noducksgiven` (a Raspberry Pi on
  the same network, reachable as `duckie@noducksgiven.local`). Never run anything from
  this file on it, and never start `dd24_policy_node.py` with `dry_run:=false` or
  `sim_sensor_bridge.py` anywhere except inside the virtual robot's container
  (`dts-virtual-<NAME>`). Before every command that sends setpoints or publishes fake
  sensors, check you are in the virtual container (`hostname`, `docker ps` on the host
  shows `dts-virtual-<NAME>`).
* Don't change PX4 parameters on the real drone. In the virtual one, change only what
  section 4 lists, and write down every change.
* Never commit or push to the user's GitHub repo; give the user patches/files instead.

## 1. Project context

* Repo: https://github.com/zyan27/IsaacDrone, branch `net-geofence-dd24-oscar`. The user's
  Mac copy is at `~/Desktop/robotics_final/IsaacDrone`. Relevant folder: `deploy/`.
* Drone: DD24 (Duckietown Duckiedrone, 0.828 kg quad X, camera side = front), Raspberry Pi
  running Duckietown's ROS 2 **Jazzy** stack in Docker (RMW = **rmw_zenoh_cpp**, a
  `zenoh-router` container), PX4 flight controller over USB via `mavlink-proxy` + MAVROS
  (container `ros2-mavros`). PX4 has no optical-flow fusion (EKF2_OF_CTRL=0); height
  reference is the range finder.
* Policy: PPO MLP (176 inputs -> 4 outputs) trained in Isaac Lab on Brown's Oscar
  cluster. Task: hover at a target height (0.6-1.5 m, we use 0.8) and hold the spawn
  position, inside a 3x3 m safety net / rooms / open space, using ONLY onboard sensors.
  Output = collective thrust + body rates (CTBR) sent to PX4 OFFBOARD through
  `/mavros/setpoint_raw/attitude` (`AttitudeTarget`, type_mask 128 = ignore attitude).
* Current policy: `deploy/policy.npz`, variant B ("calm" reward + accelerometer fix),
  **training iteration 675**. Isaac tests: 97.3% of 20 s flights survive (98.0% in the
  fixed 3x3 m net, 83% under worst-case lag), ~24 cm horizontal drift, ~4 cm height error.
  Backup: `policy_1116_A_s1.npz` (iteration 369). Check which file you have without numpy:
  `python3 -c "import zipfile,re; d=zipfile.ZipFile('deploy/policy.npz').read('meta_json.npy').decode('latin1').replace(chr(0),''); print(re.search(r'train_iter\W+(\d+)', d).group(1))"`

## 2. The node (deploy/)

| file | role |
|---|---|
| `dd24_policy_node.py` | ROS 2 node, 50 Hz. DEFAULT dry_run=true (computes, publishes nothing to PX4) |
| `dd24_policy_core.py` | numpy-only core: input building, policy, ToF cleaning, flow conversion, safety |
| `run_on_drone.sh` | copies core+node+policy.npz into the `ros2-mavros` container and starts the node there |
| `sim_sensor_bridge.py` | **virtual only**: fake ToFs + optical flow from PX4 SITL ground truth |
| `DEPLOY.md`, `INTERFACE.md`, `NOTES.md` | deploy guide, exact I/O spec, real-drone findings |

Inputs (each 22-number frame, history of 8 = 176): `/mavros/imu/data` (accel incl.
gravity, gyro, orientation -> gravity direction), 5 ToFs
`/{robot}/{bottom,top,front,left,right}_tof_driver_node/range` (`sensor_msgs/Range`;
driver publishes **999 = nothing in range**, max_range 3.6 m; node maps that to "no
target" = 4.0 m), optical flow `/{robot}/optical_flow/raw` (`mavros_msgs/OpticalFlowRad`,
raw angular flow; node converts: vx = (integrated_y - ygyro)/dt * h_down,
vy = (integrated_x - xgyro)/dt * h_down; quality 0 -> invalid), the node's previous
command, target height. `/mavros/state` gives mode/armed.

Behaviour: not OFFBOARD+armed -> streams a calm setpoint (zero rates, thrust 0.27) and
logs `WAIT`; OFFBOARD+armed -> runs the policy (`RUN`, history restarts at the switch);
IMU > 100 ms or down ToF > 200 ms stale -> calm setpoint (`NEUTRAL`); > 500 ms stale,
tilt > 60 deg or height > 2.2 m -> stops publishing (`STOP`, PX4 offboard-loss failsafe).
Log line at 2 Hz: ToFs (D U F L R), flow vx/vy, signed roll (+ = right side down) and
pitch (+ = nose down), commanded thrust and rates (FLU rad/s).
Parameters: `-p dry_run:=true|false -p target_height:=0.8 -p robot:=<name>
-p tof_up:=none` (disable a sensor) `-p flow_sign_x:=-1.0` etc. (see DEPLOY.md).

## 3. What is verified, what is not

Verified: the node's inputs equal the simulator's inputs exactly (64 random states);
numpy policy == PyTorch; dry run on the real drone reads all sensors (IMU 50 Hz, ToFs
30 Hz, flow 15.6 Hz, no stalls > 100 ms on IMU/ToF); offline, the real policy's height and
leveling responses point the right way. Not verified (this is why you exist):
1. The whole loop through real MAVROS + PX4: OFFBOARD acceptance, setpoint frame
   conventions (FLU body rates -> PX4 FRD), thrust scaling, mode switching, failsafes.
2. Closed-loop behaviour outside Isaac. A simplified Python stand-in simulator could
   NOT reproduce Isaac's results (the policy oscillated vertically, thrust bang-bang
   0.07 <-> 0.47, crashes in 1-3 s). Unexplained. The virtual drone is an independent
   physics + real PX4, so its result matters a lot.
3. Hover throttle: the policy's thrust is centred on 0.27 (range 0.07-0.47); unknown how
   that maps on the virtual (or real) airframe. Check `MPC_THR_HOVER`.

## 4. Setup (from Duckietown's docs, "Virtual Duckiedrone Takeoff Demo", ente)

Needs the Duckietown Shell `dts` (the user's teammate has it on his Mac) and Docker.
```bash
dts duckiebot virtual create -t duckiedrone -c DD24 <NAME>     # prompts for a duckie password
dts duckiebot virtual start <NAME>
dts matrix run --standalone --embedded --map sandbox_drone      # (docs also show: --standalone -d)
dts matrix attach <NAME> map_0/vehicle_0                        # wait until all containers run
docker exec -it dts-virtual-<NAME> docker exec -it ros2-mavros bash
```
Inside `ros2-mavros` (source `/opt/ros/jazzy/setup.bash` if `ros2` is missing):
1. `ros2 topic list` and `ros2 topic hz /mavros/imu/data`. Record which ToF / flow
   topics exist and the robot namespace (the real drone uses `/noducksgiven/...`).
2. `ros2 topic echo --once /mavros/state` (connected? mode?).
3. PX4 params to read (and, virtual only, set if needed), via `/mavros/param/...`
   (or QGroundControl connected to the virtual drone):
   `MPC_THR_HOVER` (read), `COM_OF_LOSS_T` -> 0.5, `MIS_TAKEOFF_ALT` -> 1.0 (the demo
   default 2.5 m is above the node's 2.2 m stop limit). If PX4 refuses OFFBOARD or arming
   because there is no RC in simulation, look at `COM_RCL_EXCEPT` / `NAV_RCL_ACT`;
   change only in the virtual drone and write it down.

## 5. Getting the node into the virtual robot

`run_on_drone.sh` expects to run on the robot host and `docker exec` into `ros2-mavros`.
In the virtual setup the "robot host" is the `dts-virtual-<NAME>` container:
```bash
docker cp ~/Desktop/robotics_final/IsaacDrone/deploy dts-virtual-<NAME>:/root/deploy
docker exec -it dts-virtual-<NAME> bash -c 'cd /root/deploy && bash run_on_drone.sh -p robot:=<NAME> -p target_height:=0.8'
```
If `docker` isn't available inside `dts-virtual-<NAME>`, copy the three files (core,
node, policy.npz) into the virtual `ros2-mavros` container another way and start
`python3 dd24_policy_node.py --ros-args -p policy:=<path> ...` there. numpy must be
importable in that container (`run_on_drone.sh` checks).

If the virtual drone has no ToF / flow topics (likely), run the stand-in sensors in a
second shell in the same virtual `ros2-mavros` container:
```bash
python3 /root/dd24/sim_sensor_bridge.py --ros-args -p robot:=<NAME> -p room_half:=1.5 -p ceiling:=2.5
```
(copy it there like the node). It publishes the 5 ToFs (30 Hz, 999 for nothing in range)
and OpticalFlowRad (15 Hz) from `/mavros/local_position/pose` + `velocity_body` in the
same format and conventions as the real drone (tested: it round-trips through the node's
flow conversion; ToF geometry = training CAD). It does NOT model the real sensors'
noise/latency/dropouts or the real flow pipeline. `room_half:=0 -p ceiling:=0` = open space.
The map's real geometry is not known to the bridge: walls are virtual (+-1.5 m around the
takeoff point, like the 3x3 m net).

## 6. Test plan (do in order, stop and report at the first failure)

1. **Dry run on the ground** (`dry_run` default true): all five ToFs + flow + IMU show up
   in the log, `D` ~0.05-0.1 on the ground, no "inputs far outside training" warnings
   except while on the ground.
2. **Takeoff without the node**: `AUTO.TAKEOFF` + arm (demo commands) to ~1 m. Log
   `MPC_THR_HOVER` and hover throttle (`/mavros/target_actuator_control` or
   `/mavros/rc/out` if available).
3. **Dry run in the air** (drone holding in PX4 hold mode at ~1 m): D ~ height, roll/pitch
   ~0, thrust near hover (~0.24-0.27) in the first second.
4. **Real run**: start the node with `-p dry_run:=false -p target_height:=1.0` while PX4
   holds at ~1 m (log shows WAIT). Switch to OFFBOARD:
   `ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{base_mode: 0, custom_mode: 'OFFBOARD'}"`.
   Expect `mode WAIT -> RUN`. Watch 20-60 s: height vs target, drift (from
   `/mavros/local_position/pose`), thrust (does it bang between 0.07 and 0.47?), tilt,
   yaw rotation. Switch back with `AUTO.LOITER` (or `AUTO.LAND`) if it diverges.
   Record `/mavros/local_position/pose`, `/mavros/imu/data`, `~/command`, `~/frame`
   (`ros2 bag record ...`) for every OFFBOARD attempt.
5. **Failsafe checks** (virtual only): kill the node during OFFBOARD -> PX4 should leave
   OFFBOARD after ~COM_OF_LOSS_T; stop the bridge's ToFs -> node goes NEUTRAL then STOP.
6. Repeat 4 with the backup policy (`policy_1116_A_s1.npz` copied over `policy.npz`) and
   with `room_half:=0 ceiling:=0` (open space).

## 7. Report back to the user (short, concrete)

* Setup: what worked, every command that differed from this file, every PX4 param changed.
* Topics present on the virtual drone; whether the bridge was needed.
* `MPC_THR_HOVER` / measured hover throttle vs the policy's 0.27 centre.
* Per OFFBOARD run: did OFFBOARD engage, seconds flown, mean |height error|, max drift,
  max tilt, thrust behaviour (smooth vs bang-bang), how it ended. Attach the bag summary
  (`ros2 bag info`) or a short CSV, and a plot if you can.
* Anything suggesting a sign/frame error (e.g. tilts away from level, climbs when above
  target, spins up in yaw): quote the log lines. Do not patch conventions yourself.
