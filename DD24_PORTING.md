# Porting IsaacDrone → Duckietown DuckieDrone DD24

Goal: adapt this Crazyflie hover RL reference into a DD24-faithful sim, and bring
the DD24's real flight controller (PX4) into the loop so a trained policy can
transfer to the physical DD24.

This repo is a **fork of [dpallick/IsaacDrone](https://github.com/dpallick/IsaacDrone)**
(`upstream` remote). Credit for the base RL/PID hover environment is theirs.

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

### 2. Flight controller → simulation  (status: PLANNED)
The DD24-B runs **real PX4** firmware on the `mamba-f405-mk2` board. You do NOT
rewrite PX4 in Python. PX4 ships a **SITL** build; use
**[Pegasus Simulator](https://github.com/PhoenixAI-LLC/PegasusSimulator)**, an
Isaac Sim extension that runs PX4 SITL against an Isaac Sim multirotor.
Duckietown's own sim-of-the-FC uses PX4 SITL + rotorpy, so this mirrors it.

---

## Decide this BEFORE training — the architecture fork

PX4 SITL runs **one instance at wall-clock speed**; it does not fan out to the
thousands of parallel envs IsaacLab RL wants. Pick the pattern:

- **(A) Policy IS the controller** — train fast in this IsaacLab env (lightweight
  thrust model, thousands of envs), then **validate** the trained net against PX4
  SITL in Pegasus before flashing to hardware. *Recommended for fast iteration.*
- **(B) Policy sits ON TOP of PX4** — policy emits setpoints, PX4 flies. Keep PX4
  SITL in the loop during training → slow, few envs, but closest to the real
  command interface.

This choice sets the **action space**:
- (A) → 4 motor thrusts (what the repo does now).
- (B) → attitude-rate + thrust setpoints (what PX4 accepts over MAVLink/uORB).

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
