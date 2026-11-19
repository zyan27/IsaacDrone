# DD24 pre-flight checklist (first RL flight)

Work top to bottom; each section assumes the ones above pass. Tick a box only
when it has been **measured or tested**, never because it is assumed. Record
results here or in `deploy/NOTES.md`.
How-to details for the drone side are in `deploy/DEPLOY.md`; this list is the
"have we done everything" view.

🔴 = blocks the first flight · 🟡 = before longer / untethered flights

---

## 1. Thrust: does the real drone match what the policy was trained on?

The policy sends **normalised** thrust centred on **0.27** (range 0.07–0.47).
In training, hover needed a throttle of roughly **0.20–0.34**. That range comes
from the assumed 10.8 N/motor, the randomised thrust curve (k = 0.2–0.6) and
±10 % motor spread. If the real hover throttle is outside that range, the policy
is flying a drone it has never seen.

| ✓ | | Check | How | Pass if | Result |
|---|---|---|---|---|---|
| [ ] | 🔴 | **Hover throttle** | Manual hover in Stabilized/Altitude mode, ~0.8 m, full battery, the flight configuration (camera, cables). Read the hover throttle from the ULog (`vehicle_thrust_setpoint` / `actuator_motors`), or read `MPC_THR_HOVER` after PX4's hover-thrust estimator (`MPC_USE_HTE`) has run. | **0.22–0.32**: OK. 0.32–0.40: tell the team, consider `thrust_gain` or a retrain. **> 0.40: don't fly the RL policy**, retrain with the measured thrust. | |
| [ ] | 🔴 | **Hover throttle at low battery** | Same, with the pack at ~3.6 V/cell. | Still inside the range above. If it isn't, only fly on fresh packs and note the cut-off voltage. | |
| [ ] | 🔴 | **Max thrust per motor** | Thrust stand (or the motor clamped on a kitchen scale), full 4S pack, real 4040 props, throttle sweep 0→100 % in 10 % steps. | Compare with `max_thrust_per_motor_N` = **10.8 N (an estimate)**. If off by > 15 %, update `dd24_params.py` and retrain. | |
| [ ] | 🟡 | **Thrust curve shape** | From the sweep: fit thrust/max = (1−k)·u + k·u². | k inside **0.2–0.6** (`thrust_curve_k_range`). PX4 has `THR_MDL_FAC = 0`, so it doesn't linearise. | |
| [ ] | | Thrust-to-weight sanity | `python environments/dd24_params.py` after updating. | ~3–6 | |
| [ ] | 🟡 | Motor time constant | Throttle step on the stand, or ESC RPM log: time to 63 %. | Inside **0.02–0.05 s** (`motor_tau_range`). | |
| [ ] | | All-up mass | Scale, flight configuration incl. battery. | Within ±10 g of 0.828 kg | |
| [ ] | | Centre of mass | Balance along each axis. | Within a few mm of centre | |

Quick reference: hover throttle the sim expects for a given real max thrust per motor:

| max thrust / motor | 6 N | 7 N | 8 N | 9 N | **10.8 N (assumed)** | 12 N |
|---|---|---|---|---|---|---|
| sim hover throttle | 0.39–0.49 | 0.34–0.44 | 0.30–0.40 | 0.27–0.37 | **0.22–0.32** | 0.20–0.29 |

---

## 2. Flight controller and hardware (props OFF unless stated)

- [ ] 🔴 Param file `fc_firmware/duckiedrone-px4-v4.params` loaded, FC rebooted; **accelerometer, gyro, level horizon calibrated**.
- [ ] 🔴 **Motor order and spin direction**: QGC → *Actuators* → motor test, one motor at a time, matches the comments in the param file.
- [ ] 🔴 **Props** on the right motors for the spin direction, tight, undamaged.
- [ ] 🔴 **Attitude display follows the drone**: tilt it by hand; QGC roll/pitch/yaw move the right way.
- [ ] 🔴 **Rate controller untouched**: `MC_ROLLRATE_*`, `MC_PITCHRATE_*`, `MC_YAWRATE_*` are PX4 defaults (that's what training simulated). If someone tuned them, write the values down and tell the team.
- [ ] 🔴 **Battery voltage** in QGC matches a multimeter on the pack within 0.1 V (else fix `BAT1_V_DIV`, currently 11.0). The low-battery landing depends on this.
- [ ] 🔴 **Battery cable secured** away from the bottom ToF (it caused a constant 0.05 m reading).
- [ ] 🟡 **Vibration**: accel clipping and vibration metrics OK in Flight Review for the manual hover log.
- [ ] Frame and arm screws tight, battery strap tight, no wires near the props.

---

## 3. Safety

- [ ] 🔴 **Kill path tested:** an RC switch that takes PX4 out of OFFBOARD (and a separate kill switch, `RC_MAP_KILL_SW`, if possible). Test both with props off. Nobody flies without a hand on that switch.
- [ ] 🔴 **`COM_OF_LOSS_T` = 0.5 s** set and confirmed in QGC (not in the param file).
- [ ] 🔴 **Offboard-loss action** (`COM_OBL_RC_ACT`) = **Land** or **Altitude** hold. **Not Position**: the drone has no horizontal position estimate (no GPS, no flow fusion in EKF2), so Position mode can't hold. Test it: in a tethered hover, kill the node (Ctrl-C) and watch PX4 take over.
- [ ] 🔴 **Node safety stops** work on the bench (dry run). Unplug the bottom ToF → the node stops sending. Tilt past 60° → it stops. These are in `dd24_policy_core.py`.
- [ ] 🔴 **Net**: intact and anchored. The inner size is written in `environments/arena.py` (currently a 3 × 3 × 2.5 m placeholder) and matches what the `fixed_net` test assumed.
- [ ] 🔴 **Tether** for the first RL flights (short line to a heavy base).
- [ ] `NAV_DLL_ACT = 0` (no action on companion-link loss) is understood. Offboard loss covers the policy dying, but not the Pi ↔ FC link dying while in another mode.
- [ ] Spectators outside the net, eye protection, LiPo bag, spare charged packs.
- [ ] QGC pre-arm checks pass with no warnings.

---

## 4. Sensors on the drone (bench, dry run; `deploy/DEPLOY.md` §3)

- [ ] 🔴 **All five ToF containers running** (`driver-tof-*`, `ros2-tof-*`), including the four started by hand. TODO from NOTES: add them to the `duckiedrone.yaml` stack so an update doesn't drop them.
- [ ] 🔴 **ToF check**: D reads the true height; cover each sensor and only that letter drops.
- [ ] 🔴 **Top ToF**: if it reads ~0.07 m with nothing above it, it's seeing the drone's own deck. Re-aim it or run with `-p tof_up:=none`. A blocked top ToF makes the policy cut thrust.
- [ ] 🔴 **Optical-flow signs**: carry the drone forward → `vx` +; carry it left → `vy` +; tilt it in place → both ≈ 0. Record the needed `flow_sign_x/y` / `flow_swap_xy` here: ______. **This can only be verified on the real sensor.**
- [ ] 🔴 **Policy reacts the right way** (first moment only): right side down → roll rate negative; nose down → pitch rate negative; below target height → more thrust.
- [ ] 🔴 **No "inputs far outside training" warnings** while held level near the target height.
- [ ] 🟡 **Setpoint link timing**: PX4 ULog from a bench run with the node sending (`dry_run:=false`, **props off**). Setpoints are steady at ~50 Hz with no gaps > 0.1 s. (Not measured yet; NOTES lists IMU/command freezes as the main cause of simulated crashes.)
- [ ] Unplug a ToF / stop the flow container: the node keeps running and warns.

---

## 5. Policy and software

- [ ] 🔴 **Right file on the drone**: iteration check (README → Model weights) prints **675** for `policy.npz`.
- [ ] 🔴 `deploy/` on the Pi matches the git commit you tested (`dd24_policy_core.py`, `dd24_policy_node.py`, `policy.npz`).
- [ ] 🟡 **Virtual drone rehearsal** done (`deploy/VIRTUAL_DRONE_HANDOFF.md`): the full arm → takeoff → OFFBOARD → back sequence against PX4 SITL + MAVROS, with no surprises.
- [ ] 🟡 Known limits are understood by whoever flies: expect **slow horizontal drift** (no position input, ~24 cm over 20 s in sim), a **slow yaw rotation**, a slight forward lean, and possible **height "pumping"** (thrust used bang-bang in training).
- [ ] 🟡 Retrain items from NOTES are decided (fix now or fly as-is): ToF max range 3.6 m, ToF rate 28–32 Hz, lower IMU/ToF freeze rates.

---

## 6. Flight progression (stop at the first bad result)

1. [ ] **Manual hover** (Stabilized/Altitude), tethered, inside the net. Proves the hardware and gives you the hover throttle for §1. Save the ULog.
2. [ ] **Node running, not in OFFBOARD** (`dry_run:=false`), manual hover: the log says `WAIT`, nothing changes.
3. [ ] **First OFFBOARD switch**, tethered, ~0.8 m, **5–10 s**, then switch back. Watch for oscillation, steady climb/sink, or heading for the net. Flip back immediately if you see any of them.
4. [ ] Longer OFFBOARD hovers (20 s), still tethered.
5. [ ] Untethered inside the net.

After every flight: pull the ULog and the node log, and compare commanded vs
actual rates and thrust, and height vs target. Note anything new in `deploy/NOTES.md`.
