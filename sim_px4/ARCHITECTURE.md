# Part 3 — PX4 (dt-px4) in the loop: architecture

Goal: the policy stops commanding 4 motor forces and instead commands
**collective thrust + body rates** (roll, pitch, yaw rate). PX4 — the same firmware
the DD24 flies — turns those into motor outputs. Isaac Sim stays the physics.

## What dt-px4 actually is (inspected 2026-10-20)
`duckietown/dt-px4:ente-amd64` (Ubuntu 20.04, ~1.3 GB):
- Prebuilt **PX4 SITL** (`/workspaces/PX4-Autopilot/build/px4_sitl_default/bin/px4`).
- `PX4_SIMULATOR=rotorpy`, `PX4_SYS_AUTOSTART=10040` (`10040_sihsim_quadx`, a
  generic quad-X — **no DD24 tuning**: no hover throttle, stock rate gains).
- "rotorpy" mode = PX4's internal physics disabled (`SYS_HITL=1`) + the standard
  `simulator_mavlink` link: **PX4 listens on TCP `4560+i`**, the simulator connects
  and streams sensors; PX4 replies with motor commands. Lockstep: PX4 time only
  advances when the simulator sends sensor data, so it can run faster or slower
  than real time.
- Offboard/API MAVLink: **UDP local `14580+i`, remote `14540+i`**. GCS: UDP `18570+i`.
- Entrypoint gate: only starts PX4 if `/data/config/robot_hardware` == `virtual`.
  On Oscar we bypass it and call `px4_entrypoint.sh` / the `px4` binary directly.
- `px4_entrypoint.sh` **kills PX4 if it logs "poll timeout"** after startup, i.e.
  if the simulator stops sending sensors. Isaac Sim startup and PPO update pauses
  can trip this, so the bridge must keep sending, or we run the binary without the
  wrapper.

## What the real DD24 runs
- **PX4 1.15.4** (`diatone_mamba-f405-mk2_default.bin`), autostart 4001 (quad-X).
- `fc_firmware/duckiedrone-px4-v4.params` sets geometry, sensors and battery,
  and **no rate/attitude gains**, so the DD24 flies PX4's compiled-in v1.15.4
  defaults (recorded in `sim_px4/px4_defaults.py`). The only controller-relevant
  override is `IMU_DGYRO_CUTOFF` 20 Hz (default 30).
- **No GPS, mag, or optical flow** (`SYS_HAS_GPS/MAG 0`, `EKF2_GPS_CTRL 0`,
  `EKF2_OF_CTRL 0`). Height comes from the downward ToF (`EKF2_HGT_REF 2`,
  `EKF2_RNG_CTRL 2`). PX4 knows attitude and height only, which is enough for
  rate+thrust offboard. So the bridge sends **HIL_SENSOR (IMU) + DISTANCE_SENSOR**,
  not HIL_GPS, and loads the DD24 param file into each SITL instance.

So dt-px4 is not tied to RotorPy: anything speaking the `simulator_mavlink`
protocol can be its physics. That's what the Isaac Sim bridge below does.

## Data flow (one env ↔ one PX4 instance `i`)
```
 Isaac Lab DronePX4Env (GPU, N envs, physics 250 Hz)
   │  IMU/mag/baro from body state  ──HIL_SENSOR 250 Hz──▶ ┐
   │  ToF height                    ──DISTANCE_SENSOR ───▶ │ TCP 4560+i   PX4 SITL i
   │  ◀── HIL_ACTUATOR_CONTROLS (4 motors, 0..1) ───────── ┘              (dt-px4.sif)
   │  motor cmd → per-motor thrust → force/torque on body                  ▲
   │                                                                       │
 Policy (50 Hz) ── SET_ATTITUDE_TARGET{body rates, thrust} ── UDP 14540+i ─┘
   ▲ obs = PX4's estimate (ATTITUDE, LOCAL_POSITION_NED), not sim ground truth
```
Deploying on the real DD24 is the same message through `ros2-mavros`
(`/mavros/setpoint_raw/attitude`), so the policy interface is unchanged.

## Training plan (hybrid; PX4-only RL is too slow)
One PX4 per env, each a CPU process, so tens of envs, not thousands.
1. **3a: fast pretraining, no PX4.** Same CTBR action space, but a torch
   reimplementation of PX4's `mc_rate_control` (rate PID) + quad-X control
   allocation runs on GPU in the env, with gains copied from the PX4 params.
   Thousands of envs, same speed as today. Add domain randomization (mass, thrust,
   latency, sensor noise).
2. **3b: PX4-in-the-loop fine-tune and eval.** Swap the emulator for real
   dt-px4 instances (N≈8–32, one node, lockstep). Same action/obs spec, so the 3a
   checkpoint loads directly.
3. **3c: hardware.** Same setpoint via mavros on the DD24.

## Code to write
| File | What |
|---|---|
| `environments/drone_ctbr_env.py` | `DirectRLEnv`: 4-dim action = (thrust, p, q, r) normalized; `controller="emulated" \| "px4"`. Physics dt 1/250, decimation 5 → 50 Hz policy. |
| `environments/px4_rate_emulator.py` | Batched torch PX4 rate PID + allocator (3a). |
| `sim_px4/mavlink_bridge.py` | pymavlink, per-instance TCP sim link + UDP offboard link; packs HIL_SENSOR/HIL_GPS from env tensors, unpacks HIL_ACTUATOR_CONTROLS; arm + OFFBOARD at reset. Reference: Pegasus `px4_mavlink_backend.py`. |
| `sim_px4/launch_px4.sh` | Starts N `apptainer exec dt-px4.sif px4 -i k` with DD24 params, waits for TCP 4560+k. |
| `sim_px4/px4_defaults.py` | **Done.** v1.15.4 rate/attitude gains, filters, quad-X geometry. |
| (params) | Reuse `fc_firmware/duckiedrone-px4-v4.params`, loaded into each SITL instance at start. |

Pegasus (`sim_px4/pegasus_dd24_px4.py`) is **dropped as the runtime**: it targets
Isaac Sim 4.5, not 5.0, and isn't an Isaac Lab env. We keep it only as protocol
reference.

## Risks / open questions
1. **Episode reset with PX4.** PX4's estimator and arming state don't teleport.
   Options: restart the instance (~seconds), or land → disarm → reset → re-arm.
   Decide after measuring restart time on Oscar.
2. **PX4 version match.** The DD24 runs 1.15.4. dt-px4's SITL version must match,
   or the default gains differ. Check `px4 ver` in the .sif.
3. **No horizontal position on the real drone.** The EKF has no GPS/flow/vision,
   so a position-holding policy must get x/y from somewhere else (camera
   pipeline) or be limited to hover/attitude tasks. Decide before 3b.
4. **Ports on shared nodes.** Apptainer shares the host network. Another user's
   PX4 on the same node collides on 4560/14540+i, so use a job-specific instance
   offset.
5. **Thrust model.** `HIL_ACTUATOR_CONTROLS` is 0..1. Map with a measured thrust
   curve (DD24 `max_thrust_per_motor_N` is still `[EST]`). Mass 0.82704 kg is in
   `dd24_params.py`, also unverified.
6. **Roll.** CTBR needs roll rate; pitch/throttle/yaw alone can't hold position.
