# DD24 flight and bench logs (noducksgiven)

| file | what |
|---|---|
| `2026-10-22_static_sensors_*.tar.xz` | drone still on the bench: sensor noise floor (IMU, ToFs) |
| `updown_20261025_194027.tar.xz` | up/down flight 1 (STABILIZED, held thrust) |
| `updown_20261025_195536.tar.xz` | up/down flight 2 |
| `updown_20261025_200045.tar.xz` | up/down flight 3 (longest) |

Each archive is a set of ROS topic dumps (mavros IMU, attitude/thrust setpoints, RC out, ToFs).
In these 10-25 logs the topics `right_tof` and `bottom_tof` are SWAPPED (bottom channel was
re-plugged later; current mapping: deploy/NOTES.md, Hardware).

## Numbers taken from them (used by the sims)

* **Hover thrust setpoint 0.47** (PX4 normalised, 16 V pack). Climbs at 0.48-0.51, sinks at 0.44-0.45.
* **Max thrust per motor 9.18 N**: with thrust ~ u^2, Fmax = m g / (4 * 0.47^2) = 0.828 * 9.81 / 0.884
  -> thrust-to-weight 4.5. (Earlier bench estimate: 7.36 N/motor.) Not a thrust stand: the
  exponent is assumed. `environments/dd24_params.py: max_thrust_per_motor_N`.
  Train with `--real_hover` (Matias trainer) so the policy's thrust is centred on 0.47.
* **In-flight IMU vibration** (sample std): gyro 0.06 / 0.10 / 0.02 rad/s (x/y/z),
  accel 1.0 / 1.0 / 2.7 m/s^2. Sim default was 0.003 rad/s / 0.05 m/s^2 (`--real_imu_noise`).
* **Hover drift**: PX4 holds IMU-level to 0.02 deg, yet the ToFs saw 0.08-0.14 m/s^2 of horizontal
  acceleration, i.e. ~0.9 deg between level and the thrust axis -> run PX4 level calibration.

Read them with `analyze_flights.py` / `tools/flight_log.py`.
