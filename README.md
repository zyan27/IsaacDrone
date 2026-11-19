# IsaacDrone: RL hover for the Duckietown DD24

A PPO hover policy for the **Duckietown DuckieDrone DD24-B** (0.828 kg quad X,
PX4 on a Mamba F405 MK2, Raspberry Pi companion running ROS 2 Jazzy). It is
trained in **Isaac Lab** on Brown's Oscar cluster and flown on the real drone
through **MAVROS** in PX4 OFFBOARD mode.

The policy uses **only onboard sensors**: IMU, five ToF range sensors and
Pi-camera optical flow. Its output is a collective thrust + body-rate setpoint
(`/mavros/setpoint_raw/attitude`), so PX4's own rate controller still flies the
motors. It hovers at a set height (0.6–1.5 m) and holds the spot it started at,
inside a safety net, a room or open space.

Fork of [dpallick/IsaacDrone](https://github.com/dpallick/IsaacDrone) (original Crazyflie hover env).

> **Before the first real flight, work through [`PREFLIGHT_CHECKLIST.md`](PREFLIGHT_CHECKLIST.md).**

---

## Where to go

| I want to… | Read |
|---|---|
| Understand the project | this file |
| Train, test, fine-tune or film a policy on Oscar | [`docs/oscar.md`](docs/oscar.md) |
| Fly the policy on the drone | [`deploy/DEPLOY.md`](deploy/DEPLOY.md) |
| Know the exact policy inputs/outputs | [`deploy/INTERFACE.md`](deploy/INTERFACE.md) |
| See what's different on the real drone / known issues | [`deploy/NOTES.md`](deploy/NOTES.md) |
| Test on a virtual Duckiedrone (Duckiematrix) | [`deploy/VIRTUAL_DRONE_HANDOFF.md`](deploy/VIRTUAL_DRONE_HANDOFF.md) |
| Check everything before flying | [`PREFLIGHT_CHECKLIST.md`](PREFLIGHT_CHECKLIST.md) |

---

## The pipeline

```
 Oscar (GPU)                                   Mac                    Drone (Pi, ROS 2 Jazzy)
 ──────────────────────────────────            ─────────              ─────────────────────────────
 train_rl_hover.py  → runs/<run>/best.pt
 eval_rl_hover.py   → eval*.json  (PASS?)
 flight_recorder.py → failure report / plots
 record_video.py    → videos
 deploy/export_policy.py → deploy/policy.npz ──scp──▶ deploy/ ──scp──▶ run_on_drone.sh
                                                                       dd24_policy_node.py (50 Hz)
                                                                         IMU + 5 ToF + flow ─▶ policy
                                                                         ─▶ /mavros/setpoint_raw/attitude ─▶ PX4
```

## Model weights

Exported flight policies live in `deploy/` as `.npz` files. They need only numpy,
so the Pi doesn't need PyTorch.

| File | What |
|---|---|
| `deploy/policy.npz` | **Current flight policy.** Variant B ("calm" reward + accelerometer fix), training iteration 675. Isaac tests: 97.3 % of 20 s flights survive (98.0 % in the fixed 3×3 m net, 83 % under worst-case lag), ~24 cm horizontal drift, ~4 cm height error. |
| `deploy/policy_1116_A_s1.npz` | Backup: variant A (original reward), seed 1, iteration 369. |

`run_on_drone.sh` always loads `policy.npz`. To fly the backup, copy it over
`policy.npz` on the drone. Check which iteration a file holds:
```bash
python3 -c "import zipfile,re; d=zipfile.ZipFile('deploy/policy.npz').read('meta_json.npy').decode('latin1').replace(chr(0),''); print(re.search(r'train_iter\W+(\d+)', d).group(1))"
```
Full PyTorch checkpoints (`runs/<run>/best.pt`) stay on Oscar and are git-ignored.
When you replace `policy.npz`, update this table.

---

## Quick start

### Train / evaluate (Oscar)
Full guide: [`docs/oscar.md`](docs/oscar.md). The usual flow:
```bash
cd ~/IsaacDrone && git pull
ITERS=10000 SEEDS=0-2 MAIL_USER=you@brown.edu bash oscar/smoke_then_overnight.sh   # smoke test, then long run
RUN=dd24_hover_<id> bash oscar/finetune_tonight.sh                                  # pick best seed + fine-tune
CKPT=runs/<run>/best.pt sbatch oscar/flight_recorder.sbatch                         # why does it fail?
CKPT=runs/<run>/best.pt sbatch oscar/export_policy.sbatch                           # -> deploy/policy.npz
```

### Fly (drone)
Full guide: [`deploy/DEPLOY.md`](deploy/DEPLOY.md).
```bash
scp -r deploy duckie@noducksgiven.local:~/
ssh duckie@noducksgiven.local
cd ~/deploy && bash run_on_drone.sh -p target_height:=0.8                    # dry run, props off
cd ~/deploy && bash run_on_drone.sh -p dry_run:=false -p target_height:=0.8  # real
```

### Local Isaac install (optional, Linux + RTX GPU)
Isaac Sim needs an NVIDIA GPU **with RT cores** (A6000, A40, A5000, RTX 3090/4090;
V100/A100/H100 will not work) and Linux. On Oscar it runs from an Apptainer image
(see `docs/oscar.md`). On your own machine:
```bash
conda create -n droneIsaac5.0 python=3.11 && conda activate droneIsaac5.0
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com   # Isaac Sim first
git clone -b v2.2.1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab && ./isaaclab.sh --install && cd -                                          # then Isaac Lab
ISAACLAB=~/IsaacLab bash assets/dd24/convert_to_usd.sh                                    # DD24 URDF -> USD
python train_rl_hover.py --airframe dd24 --num_envs 1024 --iters 2000 --run_dir runs/dd24_hover --headless
python eval_rl_hover.py  --checkpoint runs/dd24_hover/best.pt --headless
python main.py --RLHover --airframe dd24 --rl_checkpoint runs/dd24_hover/best.pt          # watch it
```
cuRobo is not needed (nothing imports it). Note that `train_rl_hover.py`
defaults to `--airframe crazyflie`, so always pass `--airframe dd24` (the Oscar
scripts already do).

---

## What the policy is trained on

* **Action** (`--control_mode ctbr`, default): normalised collective thrust
  `0.27 ± 0.20` + body rates (±2.5 rad/s roll/pitch, ±1.5 rad/s yaw), FLU frame.
  Under it, `environments/px4_ctbr.py` simulates PX4's rate PID (default
  `MC_*RATE` gains), quad-X allocation from the param file, motor lag and a
  thrust curve, at 200 Hz.
* **Observation** (`--obs_mode onboard`, default): 8 frames × 22 values = 176:
  IMU accel + gyro, gravity direction, 5 ToFs, previous command, flow velocity
  + valid flag, target height. Each sensor arrives at its real rate and latency.
  Exact layout: `deploy/INTERFACE.md`.
* **Domain randomisation, every episode:** rate-PID gains, motor time constant
  (0.02–0.05 s), thrust-curve shape, ±10 % per-motor thrust, sensor noise, rates
  and dropouts, link freezes. Scenes are drawn from `environments/scenes.py`:
  net, net in a room, furnished room, open hall, clutter, ghost ToF returns.
* **Reward presets:** `--reward v1` (original) or `--reward calm` (bigger crash
  penalty, plus horizontal-speed, tilt and smoothness terms; use with `--gamma 0.995`).
* **Tests:** `eval.json` (training mix), `eval_net70.json` (ToFs miss the net
  70 % of the time), `eval_fixednet.json` (3×3 m net), `--lag` (worst-case links).

## Airframe parameters

Single source of truth: **`environments/dd24_params.py`** (Isaac, RotorPy and
the URDF builder all read it). After changing it, rebuild the URDF/USD
(`docs/oscar.md`, step 2) and **retrain**.

| Parameter | Value | Status |
|---|---|---|
| All-up mass | 0.828 kg | measured |
| Arm length | 0.1133 m | CAD |
| Inertia Ixx / Iyy / Izz | 3.59e-3 / 3.90e-3 / 6.11e-3 kg·m² | bifilar pendulum |
| ToF positions / directions | `DD24_TOF_SENSORS` | CAD |
| **Max thrust per motor** | **10.8 N** | **estimate: measure it (see checklist §1)** |
| **Yaw coefficient** | **0.0125 m** | **estimate** |

## Flight controller

`fc_firmware/duckiedrone-px4-v4.params` holds all PX4 parameters for the DD24-B,
with a comment on every line. Load it in QGroundControl under *Vehicle Setup →
Parameters → Tools → Load from file*, reboot, then calibrate the accelerometer,
gyro and level horizon. It does **not** set the rate-controller gains (PX4
defaults are used, which is what training assumes), `MPC_THR_HOVER`, or the
offboard-loss failsafe. See the checklist.

---

## Repository layout

```text
IsaacDrone/
├── README.md, PREFLIGHT_CHECKLIST.md
├── docs/oscar.md              Oscar cluster: setup, training, tests, videos, export
├── deploy/                    ON THE DRONE: policy node, exported policies (.npz), deploy docs
├── environments/
│   ├── dd24_params.py         airframe + ToF sensor layout (single source of truth)
│   ├── drone_hover_env.py     Isaac Lab DirectRLEnv: obs, reward presets, randomisation
│   ├── px4_ctbr.py            simulated PX4 rate controller + allocation + motors
│   ├── scenes.py              net / room / hall scenes
│   ├── sensor_timing.py       per-sensor rates, latency, freezes
│   └── arena.py               safety-net dimensions
├── train_rl_hover.py          PPO training (adaptive LR, obs normalisation, --init_from)
├── eval_rl_hover.py           tests: success, drift, height error, per-scenario
├── flight_recorder.py         per-step logs of 512 flights → failure report
├── record_video.py            arena / chase / top-view videos
├── main.py                    interactive viewer (PID or RL)
├── analyze_flights.py, tools/ flight-log analysis
├── controllers/               PID baseline, RL actor, obs normalisation
├── assets/dd24/               CAD mesh, URDF builder, USD conversion
├── oscar/                     Slurm / Apptainer job scripts
├── fc_firmware/               PX4 parameter file
├── rotorpy_dd24/              older CPU-only RotorPy track (see below)
└── sim_px4/                   Pegasus + PX4 SITL starter (untested)
```

### RotorPy track (older, CPU-only)
`rotorpy_dd24/` trains with ground-truth state in RotorPy on any CPU. It was the
first approach and is **not** what flies: the deployed policy comes from Isaac.
It's still useful for quick experiments:
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r rotorpy_dd24/requirements.txt      # Python 3.11; 3.14 has no torch wheels
python rotorpy_dd24/train_paths.py --task hover --run_dir runs/rotorpy_hover
python rotorpy_dd24/eval_policy.py --model runs/rotorpy_hover/latest.zip --net
```

## Videos
PID hover (Crazyflie): https://github.com/user-attachments/assets/4a06922b-99e3-4d18-b24d-4a863748ade6
RL hover, 500 iterations (Crazyflie): https://github.com/user-attachments/assets/0863106f-8b25-4cbd-b159-54d819b61174
