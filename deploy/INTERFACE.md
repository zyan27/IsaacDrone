# Policy interface for the MAVROS node (control_mode = "ctbr")

The deployed node must build **exactly** the vector the policy was trained on and
publish its output the same way. Source of truth: `environments/drone_hover_env.py`
(`_get_observations`, `ctbr_setpoint`). Checkpoints store the numbers below.

## Timing
* Run the policy on a fixed **50 Hz** timer (`policy_hz` in the checkpoint).
* Each tick, use the **latest received message** of every topic (no waiting, no
  interpolation, no filtering). Training reproduces this: IMU ~50 Hz, ToF 20 Hz,
  flow 15 Hz with 60-120 ms latency, values repeated between updates.

## Input: 8 stacked frames x 22 numbers = 176
Frames are oldest first (`obs[0:22]` = 7 ticks ago, `obs[154:176]` = now). On the
first tick, fill all 8 with the first frame.

| idx | value | source | scaling |
|---|---|---|---|
| 0-2 | linear acceleration x, y, z (FLU body) | `/mavros/imu/data` `linear_acceleration` | m/s^2, **includes gravity** (z ~ +9.81 level) |
| 3-5 | angular velocity x, y, z (FLU body) | `/mavros/imu/data` `angular_velocity` | rad/s |
| 6-8 | gravity direction in body frame | from `/mavros/imu/data` `orientation` q: `R(q)^T * (0,0,-1)` | unit vector, (0,0,-1) when level |
| 9 | ToF down | Pi ToF topic | metres / 4.0; **no target -> 1.0** (incl. the driver's 999, inf, NaN, > max_range); clamp to [0.01, 1] |
| 10 | ToF up | " | " |
| 11 | ToF front | " | " |
| 12 | ToF left | " | " |
| 13 | ToF right | " | " |
| 14-17 | previous command a (thrust, roll, pitch, yaw) | the node's own last output | in [-1, 1] (see below) |
| 18-19 | flow velocity x, y (FLU body) | Pi flow: flow rate x down-ToF height, gyro de-rotated | m/s; **0 when invalid** |
| 20 | flow valid | Pi flow quality / height in range | 1.0 or 0.0 |
| 21 | target height | operator setting | metres above the floor (trained 0.6-1.5) |

ToF positions/directions are in `environments/dd24_params.py` (`DD24_TOF_SENSORS`).

## Normalisation (inside the checkpoint)
Before the actor, every one of the 176 inputs is normalised with statistics saved
in the checkpoint (`ck["obs_norm"]`, frozen after training):
`x_n = clamp((x - mean) / max(sqrt(var), 0.01), -10, 10)`.
Build the raw vector exactly as above; `controllers/obs_norm.load_obs_norm(ck, 176, device)`
returns the function to apply. Never re-estimate these on the drone.

## Output: 4 numbers -> `mavros_msgs/AttitudeTarget` on `/mavros/setpoint_raw/attitude`
The actor outputs s, clamped to [0,1]^4 (linear output in new checkpoints, sigmoid in old ones). Then a = 2s - 1 and:
* `thrust` = clamp(0.27 + 0.20 * a0, 0, 1)   (`ctbr_thrust_center`, `ctbr_thrust_span`)
* `body_rate.x, .y, .z` = a1 * 2.5, a2 * 2.5, a3 * 1.5 rad/s (FLU; MAVROS converts to FRD)
* `type_mask` = IGNORE_ATTITUDE (128): rates + thrust only

Motor numbering is irrelevant here (PX4 allocates). For reference only, sim
(m1 FR, m2 BR, m3 BL, m4 FL) -> PX4 motors (1 = m1, 2 = m3, 3 = m4, 4 = m2).

PX4 must be in OFFBOARD with setpoints streaming (> 2 Hz) before the switch.
Keep a manual takeover (RC switch) and a ToF-based height cutoff in the node.

Reference implementation: `deploy/dd24_policy_core.py` (tested to reproduce the
simulator's inputs exactly). Deployment steps: `DEPLOY.md`.
