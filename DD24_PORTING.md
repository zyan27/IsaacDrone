# Porting IsaacDrone → Duckietown DuckieDrone DD24

Goal: adapt this Crazyflie hover RL reference into a DD24-faithful sim, and bring
the DD24's real flight controller (PX4) into the loop so a trained policy can
transfer to the physical DD24.

This repo is a **fork of [dpallick/IsaacDrone](https://github.com/dpallick/IsaacDrone)**
(`upstream` remote). Credit for the base RL/PID hover environment is theirs.

---

## ⭐ PRIMARY STACK: RotorPy + PX4 — see `rotorpy_dd24/`

After investigating, the **DuckieDrone team uses PX4 SITL + [rotorpy](https://github.com/spencerfolk/rotorpy)**,
NOT Isaac Sim/Pegasus. RotorPy is pure-Python/CPU, has a built-in Gym RL env and
a `PX4Multirotor` (pymavlink) class, and **trains on the Mac** (PPO verified at
~4000 fps). This is now the primary path — see **`rotorpy_dd24/README.md`**.

The Isaac Sim material below (`environments/`, `sim_px4/`) is kept only as a
**fallback for vision/camera-based RL**, which rotorpy cannot render. For
state-based flight (hover, goto, paths) use the rotorpy track.

---

## The two halves of the port

### 1. Airframe → RL environment  (status: STARTED)
The reference env (`environments/drone_hover_env.py`) simulates a **Crazyflie**
(27 g) via `CRAZYFLIE_CFG`, with a simple force/torque thrust model.

Done:
- `environments/dd24_params.py` — single source of truth for DD24 physical
  params, every value tagged `[SRC]` / `[EST]` / `[TODO(measure)]`.
- `DroneHoverEnvCfgDD24` in `drone_hover_env.py` — overrides `arm_length`,
  `max_thrust_to_weight`, `yaw_coeff` from those params.

Not done (needs real data / a mesh):
- **DD24 USD asset.** No public DD24 USD exists. The env still loads the
  Crazyflie mesh, so body **mass and inertia are still the Crazyflie's**. Build
  a DD24 URDF (hub + 4 arms at `arm_length`, measured mass/inertia) and convert
  to USD, then point `robot` at it.
- **Measure the `[TODO]` params** (see checklist below).

### 2. Flight controller → simulation  (status: SCAFFOLDED, path B chosen)
The DD24-B runs **real PX4** firmware on the `mamba-f405-mk2` board. You do NOT
rewrite PX4 in Python. PX4 ships a **SITL** build; use
**[Pegasus Simulator](https://pegasussimulator.github.io/PegasusSimulator/)**, an
Isaac Sim extension that runs PX4 SITL against an Isaac Sim multirotor.
Duckietown's own sim-of-the-FC uses PX4 SITL + rotorpy, so this mirrors it.

Starter written: **`sim_px4/pegasus_dd24_px4.py`** — spawns one drone with a PX4
MAVLink backend, reads `dd24_params`. UNTESTED (needs the Linux+GPU box + Pegasus
+ PX4). TODOs in it: DD24 USD mesh, a DD24 PX4 airframe (vs stock `iris`), and
the measured thrust curve.

**Version / architecture reality:**
- Official Pegasus **v4.5.1 → Isaac Sim 4.5**. This repo is Isaac Sim **5.0 +
  Isaac Lab**. Community forks claim Isaac Sim 5.1 / 6.0 but are unverified.
  Expect to pin versions carefully (likely downgrade to the Pegasus-supported
  Isaac Sim, or use a fork).
- **Pegasus is NOT Isaac Lab.** Pegasus uses the Isaac Sim standalone/extension
  API and flies ONE PX4 drone; it does not give the batched `DirectRLEnv` this
  repo's training loop uses. So path B's code lives in `sim_px4/`, separate from
  `environments/` + `train_rl_hover.py`. The PPO logic and `dd24_params` carry
  over; the `DirectRLEnv` plumbing does not.

---

## Architecture: path B chosen (PX4 in the loop)

PX4 SITL runs **one instance at wall-clock speed**; it does not fan out to the
thousands of parallel envs IsaacLab RL wants.

- **(A) Policy IS the controller** — fast batched IsaacLab RL, validate vs PX4
  SITL after. *Not chosen.*
- **(B) Policy sits ON TOP of PX4** — policy emits setpoints, PX4 flies.
  **CHOSEN.** Built in `sim_px4/` on Pegasus, not Isaac Lab. Slow, few envs, but
  closest to the real command interface.

Action space for (B): **attitude-rate + thrust setpoints** sent to PX4 over
MAVLink (e.g. offboard `SET_ATTITUDE_TARGET`), not raw motor thrusts.

The current observation vector uses sim **ground-truth** height/velocity, which
the real DD24 does not have (it estimates from the downward camera + IMU). Add
**domain randomization** (sensor noise, latency, mass/thrust spread) regardless
of path — there is none today.

---

## Measurement checklist (you have the physical drone)

| Param | Where in code | How to get it |
|---|---|---|
| All-up weight (AUW) | `dd24_params.mass_kg` | Kitchen scale, assembled + battery. NOT the 1.76 kg box weight. |
| Arm length | `dd24_params.arm_length_m` | Motor-to-motor diagonal ÷ 2. |
| Max thrust / motor | `dd24_params.max_thrust_per_motor_N` | Thrust stand, or DX2205 2300KV + 4040 + 4S mfr chart. |
| Yaw coefficient | `dd24_params.yaw_coeff` | Identify from flight logs or a bench test. |
| Inertia (Ixx,Iyy,Izz) | DD24 USD (TODO) | CAD of the frame, or bifilar-pendulum measurement. |

Sanity check: `python environments/dd24_params.py`. If thrust-to-weight prints
`> ~7`, the estimates are off — a loaded teaching quad is usually ~4–6.

---

## Immediate next steps
1. Weigh the drone + measure wheelbase → fill `mass_kg`, `arm_length_m`.
2. Pick path **(A)** or **(B)** above.
3. On a **Linux + NVIDIA GPU** box (this cannot run on the Mac), install Isaac
   Sim 5.0 + Isaac Lab per the README, confirm the Crazyflie env trains.
4. Build/convert a DD24 USD and wire it into `DroneHoverEnvCfgDD24`.
5. Stand up Pegasus + PX4 SITL for validation (path A) or in-the-loop (path B).
