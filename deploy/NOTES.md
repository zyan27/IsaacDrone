# Real-drone notes (bench session 2026-11-16)

What we found on `noducksgiven` and what still has to be fixed. Keep this list
current: every item is a place where the real drone differs from training.

## Out-of-range sensor values: TO FIX / KEEP HANDLED

* **ToF "no target" = 999.** The Duckietown VL53L1X driver publishes `range: 999`
  when nothing is in range (seen on the top ToF facing the ceiling), and its
  `max_range` is 3.6 m. Training encodes "no target" as 4.0 m (input 1.0).
  - Handled in the node: `dd24_policy_core.clean_tof` maps 999, inf, NaN, negative
    and anything beyond the message's `max_range` to 4.0 m; values below 0.04 m are
    clamped to 0.04 m, as in training (tested against the simulator before release).
  - TODO: any other sensor added later must go through the same cleaning; never feed a
    raw driver value to the policy.
  - TODO (training): set the simulated ToF max range to 3.6 m (`TOF_MAX_RANGE_M`) so
    readings 3.6-4.0 m, which the real sensor never produces, aren't trained on.
* **ToF floor 0.05 m.** Sitting on the ground the bottom ToF reads a fixed 0.05.
  Matches training's minimum closely enough; flow is invalid there anyway (< 8 cm).
* **Flow is raw angular flow, not velocity.** `/noducksgiven/optical_flow/raw` is
  `mavros_msgs/OpticalFlowRad` (radians over `integration_time_us`, gyro integrals,
  `quality`, `distance` = -1). The node converts: velocity = (flow - gyro) / dt x
  down-ToF height; `quality` 0 -> invalid. Sign convention assumed from MAVLink;
  **verify on the bench** (DEPLOY.md step 3.3) and record the result here.

* **Top ToF reads ~0.07 m constantly** in the first dry run (drone held by hand):
  likely the drone's own deck in view. If confirmed, re-aim/move it or run the node with
  `-p tof_up:=none`. A blocked top ToF looks like a ceiling 7 cm away: the policy
  commanded minimum thrust 4 cm above the target height.

## Dry-run results with the real policy (iter 713), checked offline with policy.npz

* **Directions are right:** height below target -> more thrust, above -> less; at the
  target the first command is ~0.243 (about hover). Small tilts (5-8 deg) -> rotation
  rates back towards level, roll and pitch. Checked from a fresh history, as at the
  OFFBOARD switch.
* **Dry-run outputs saturate after ~1-2 s and are NOT meaningful then:** the drone is
  held still, so commands never take effect (gyro stays 0, height never changes) and the
  policy's own "previous command" input drifts to its limits. Large held tilts (>= 15 deg
  with zero rotation) are rarer than anything in training and also give odd outputs.
  Judge only the first moment after a change.
* **Built-in offsets:** level and still it asks for pitch rate +0.18 rad/s and a little
  yaw; training drones also spun slowly in yaw (gyro_z mean -0.22 rad/s). Expect a slow
  yaw rotation and possibly a slight forward lean in flight.
* **Thrust is used bang-bang in training** (previous-thrust input std 0.56 on a +-1
  scale): watch for height "pumping" on the first flights.
* My stand-in simulator (outside Isaac) could not reproduce Isaac's 94% with this policy
  (vertical oscillation). Isaac remains the reference; the stand-in difference is still
  unexplained, so the first real hovers must be short with the RC switch ready.

## Training bugs found

* **Accelerometer input is effectively unused.** Isaac's IMU accel had std 19-87 m/s^2
  in the training statistics (spikes, most likely when episodes reset and the drone is
  teleported), so the input normalisation scales real accelerations to ~0.
  FIXED (needs a retrain): the env reports gravity-only acceleration for the first 3
  policy steps of each episode and clips every reading to +-30 m/s^2 (`imu_settle_steps`,
  `accel_clip`); the node applies the same clip (stored in the policy file). The trainer
  logs the accelerometer normalisation spread (`acc_std` in progress.csv, healthy: a
  few m/s^2) and `oscar/check_smoke.py` refuses to launch a long run above 10.

## Hardware / setup

* ToFs re-plugged to the Duckietown standard multiplexer channels: bottom CHL0
  (i2c-23), left CHL1 (24), right CHL2 (25), top CHL3 (26), front CHL4 (27).
* The bottom driver's per-robot override `/data/config/node/tof_driver_bottom/noducksgiven.yaml`
  was changed from bus 28 to 23 (backup `.bak` next to it).
* Four extra driver + ROS 2 bridge containers (`driver-tof-{left,right,top,front}`,
  `ros2-tof-*`) were started by hand. TODO: add them to the `duckiedrone.yaml` stack
  (billyan's machine) so a stack update doesn't drop them.
* The battery cable was hanging in front of the bottom ToF (constant 0.05 reading).
  Secure it: in flight it would corrupt height and flow scaling.

## Timing measured on the drone (Pi busy with camera + flow)

| topic | rate | longest gap | trained |
|---|---|---|---|
| `/mavros/imu/data` | 50 Hz | 39 ms | 40-60 Hz, freezes up to 0.05/s |
| bottom ToF | 30 Hz | 49 ms | 18-22 Hz, freezes up to 0.2/s |
| optical flow | 15.6 Hz | 144 ms | 13-17 Hz |
| setpoints to PX4 | not measured yet | | freezes up to 0.05/s |

TODO (training): ToF rate range -> 28-32 Hz; IMU/ToF freeze rates much lower than
assumed (none seen in ~1.5 min each). The flight recorder showed IMU and command-link
freezes cause most simulated crashes, so the real drone should do better than the
93.8% simulated success if the setpoint link is as steady as the IMU.

## Still to check

* Hover throttle (`MPC_THR_HOVER`) vs the policy's 0.27 thrust centre.
* Setpoint link timing PX4-side: PX4 log (.ulg) from a bench run with the node sending.
