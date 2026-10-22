# IsaacDrone

Reinforcement-learning hover for the **Duckiedrone DD24** in Isaac Lab, with the
PX4 inner loop in the sim. The policy outputs what the real drone's Pi would send
PX4 over mavros, a CTBR setpoint (collective thrust + body rates). The emulated
PX4 rate loop turns that into motor commands, and a motor model turns those into
forces on the body.

```text
policy (sensors only) --CTBR @50 Hz--> PX4 rate loop (emulated, v1.15.4 math) --> motors (lag, thrust curve) --> rigid body
```

The DD24 / PX4 / Oscar work is on branch `oscar-px4-prep`.

## Contents
- [Requirements](#requirements)
- [Quick start on Oscar (Brown CCV)](#quick-start-on-oscar-brown-ccv)
- [Local install (Linux + RTX GPU)](#local-install-linux--rtx-gpu)
- [What's in the repo](#whats-in-the-repo)
- [DD24 model: measured vs estimated](#dd24-model-measured-vs-estimated)
- [Real-drone flight logs](#real-drone-flight-logs)
- [Status and known limitations](#status-and-known-limitations)

---

## Requirements
- Linux with an **NVIDIA RTX GPU**. Isaac Sim needs RT cores, so A100/V100/H100 don't work, and macOS can't run it.
- Isaac Sim **5.0.0** + Isaac Lab **v2.2.1**. These versions are pinned; Isaac Lab `main` has moved on.
- No Linux GPU box? Use Oscar (next section).

---

## Quick start on Oscar (Brown CCV)
The one-time setup (build the Apptainer image, install Isaac Lab) is in
[`oscar/README.md`](oscar/README.md). It puts everything under `~/scratch/isaac/`:
- `isaac-sim-5.0.0.sif`
- `IsaacLab/`
- `pyuser/`
- `cache/`

Then:
```bash
git clone -b oscar-px4-prep https://github.com/zyan27/IsaacDrone.git ~/IsaacDrone
cd ~/IsaacDrone
```
`oscar/run_isaac.sh <script> [args]` runs any repo script inside the container. Use it on a GPU node only.
To point it at other paths, set the environment variables `SIF=...`, `ISAACLAB_DIR=...` and `REPO_DIR=...`.

### Train the DD24 + PX4 policy (batch job, about 40 min)
```bash
sbatch oscar/train_px4.sbatch                 # 1024 envs, 600 iters
# override:  NUM_ENVS=512 ITERS=1000 sbatch oscar/train_px4.sbatch
squeue --me
tail oscar/logs/dd24_px4_<jobid>.out
```
The job:
1. records a 5 s scripted check video (`videos/px4_room_check.mp4`)
2. trains (`checkpoints/px4_hover.pt`, saved at the best step reward)
3. records a 20 s video of the policy (`videos/px4_policy.mp4`, plus a `.csv` of the full state each step)

Per-iteration log columns:
- `step_reward`: maximum 2.75
- `mean_ep_len`: 20 s means no crashes
- `crashes`
- `std`

### Interactive (GPU node)
```bash
tmux new -s gpu
salloc -p gpu --gres=gpu:1 --constraint="l40s|a6000|a40|a5000" -n 4 --mem=32G -t 4:00:00
# inside the allocation:
bash oscar/run_isaac.sh play_px4.py --headless --enable_cameras --seconds 20 \
    --checkpoint checkpoints/px4_hover.pt --out videos/px4_policy.mp4
bash oscar/run_isaac.sh play_px4.py --headless --enable_cameras --maneuvers   # scripted commander, no policy
```
The `norm-gpu` QOS caps you at 12 CPUs across all jobs, so keep `-n` / `--cpus-per-task` small.

### Watch the videos on your laptop
```bash
scp -r YOURID@ssh.ccv.brown.edu:~/IsaacDrone/videos ./isaacdrone_videos
```
In the videos:
- The camera is fixed.
- The red ball is the 1 m hover target; the translucent boxes are the room walls.
- The side panel shows:
  - altitude
  - ToF readings
  - the PX4 setpoint
  - motor outputs
  - a top-view trace

### Older Crazyflie hover baseline (no PX4)
```bash
sbatch oscar/train_hover.sbatch               # -> checkpoints/rl_hover.pt
bash oscar/run_isaac.sh play_video.py --headless --enable_cameras \
    --rl_checkpoint checkpoints/rl_hover.pt --out videos/rl_hover.mp4
```

---

## Local install (Linux + RTX GPU)
```bash
conda create -n droneIsaac5.0 python=3.11 && conda activate droneIsaac5.0
conda install cuda
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
git clone --branch v2.2.1 https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab && ./isaaclab.sh --install && cd ..
# optional, only for main.py:
git clone https://github.com/NVlabs/curobo.git && pip install -e ./curobo --no-build-isolation
```
Run scripts from the repo root. Leave off `--headless` to get the GUI.
```bash
python train_rl_px4.py --headless --num_envs 1024 --iters 600
python play_px4.py --checkpoint checkpoints/px4_hover.pt          # GUI
python main.py --PID                                              # PID hover demo (Crazyflie)
python main.py --RLHover --rl_checkpoint checkpoints/rl_hover.pt  # RL hover demo (Crazyflie)
```

---

## What's in the repo
```text
environments/
  drone_px4_env.py       DD24 + PX4 env: room, 5 ToF rays, sensors, reward, resets   <- main env
  px4_rate_emulator.py   PX4 v1.15.4 rate controller + mixer + desaturation in torch
  dd24_params.py         DD24 physical parameters, each tagged [SRC] or [EST]
  drone_hover_env.py     original Crazyflie hover env (privileged obs, no PX4)
controllers/
  px4_policy.py          actor network for train_rl_px4.py / play_px4.py
  pid_controller.py      PID hover/waypoint controller (main.py)
  rl_hover_controller.py Crazyflie RL policy wrapper (main.py)
train_rl_px4.py          PPO, asymmetric: actor sees sensors only, critic also sees true state
play_px4.py              fixed-camera MP4 + CSV of the policy or a scripted commander
train_rl_hover.py        PPO for the Crazyflie baseline
play_video.py            MP4 of the Crazyflie baseline
visualization.py         target markers + side-panel HUD for videos
main.py                  GUI demos (PID / RL)
sim_px4/                 PX4 notes and tools: ARCHITECTURE.md, default gains, dt-px4 param probe
fc_firmware/             DD24 PX4 parameter file
rotorpy_dd24/            earlier RotorPy (non-Isaac) prototype
oscar/                   Oscar setup, sbatch scripts, run_isaac.sh
data/flight_logs/        real-drone ROS 2 recordings exported to CSV (.tar.xz)
checkpoints/             trained policies (.pt)
```

### Policy interface (`environments/drone_px4_env.py`)
- **Observation (20 values, actor):**
  - 5 ToF ranges (down, front, left, right, up)
  - vertical velocity
  - attitude quaternion
  - gyro (FRD frame)
  - accelerometer (FRD frame)
  - the previous CTBR setpoint
  - Every term has Gaussian noise.
- **Critic (26 values):** the 20 above + the true position error + the true world velocity. Used only in training.
- **Action (4 values in [-1, 1]):**
  - thrust = hover_cmd ± 0.25
  - roll/pitch rates = ±3 rad/s
  - yaw rate = ±2 rad/s (FRD frame)
  - The mapping is `policy_to_ctbr()`. The real drone must use the same mapping.
- **Timing:** physics runs at 250 Hz (PX4's rate loop rate); the policy at 50 Hz.
- **Episode:**
  - spawns at 0.3 m, at a random xy within ±0.8 m and a random yaw; target is 1 m
  - lasts 20 s
  - ends early on a crash: floor, ceiling, wall, or tilt over 1.2 rad
- **Room:** 3 × 3 × 2.5 m box. The ToF rays hit its planes analytically (no colliders).

---

## DD24 model: measured vs estimated
Everything is in `environments/dd24_params.py`, tagged `[SRC]` (measured or from a datasheet) or `[EST]` (estimate).
- **Mass:** 0.827 kg (measured)
- **Motor-to-motor diagonal:** 0.23 m (measured). Arm 0.115 m, lever arm 0.081 m.
- **Props:** 4 in (Diatone 4040 3-blade)
- **Max thrust per motor:** 7.36 N (estimated from DX2205 2300KV 4S bench data, not this drone)
- **Inertia:** 5e-3, 5e-3, 8e-3 kg m² (estimated; not yet measured)
- **Motor time constant:** 20 ms (estimated)
- **Thrust curve:** quadratic, giving a hover command of about 0.53 (estimated). The real hover throttle is still to be measured from a hover log.
- **PX4 gains:** v1.15.4 defaults, with `IMU_DGYRO_CUTOFF` 20 from the DD24 parameter file. The dt-px4 firmware uses `MC_ROLL_P` / `MC_PITCH_P` = 4.0, while the sim uses 6.5. See `sim_px4/ARCHITECTURE.md`.

---

## Real-drone flight logs
`data/flight_logs/*.tar.xz` are ROS 2 topics exported to CSV, one file per topic. Unpack with:
```bash
tar -xJf data/flight_logs/2026-10-22_static_sensors_20261022_154949.tar.xz -C /tmp
```
- `2026-10-22_static_*`: 30 s with the drone on the ground, disarmed.
  - Bottom ToF: 30 Hz, about 8 mm noise.
  - MAVROS IMU: about 45 Hz, with dropouts of up to 1 s.
  - The front, left, right and top ToF were not recorded.

To record a new log on the drone (inside the `ros2-mavros` container):
```bash
ros2 topic list                                 # find the ToF topics
ros2 bag record -o hover1 /mavros/state /mavros/imu/data /mavros/imu/data_raw \
  /mavros/local_position/pose /mavros/local_position/velocity_local \
  /mavros/rc/in /mavros/rc/out /mavros/battery <tof topics>
```

---

## Status and known limitations
- **Done:**
  - Emulated-PX4 DD24 env.
  - Sensor-only policy that holds 1 m (about 1 cm altitude error in sim).
  - Fixed-camera videos.
- **Sideways drift:** the policy holds altitude but orbits the centre at about 0.3 m. It locates itself by side-ToF ranging to solid walls, but the real flight space is netted, so that won't transfer.
- **Planned sim changes, from the first real log:**
  - Remove the side/top ToF and yaw from the observation.
  - Lower gyro noise to 0.003 rad/s.
  - ToF: 3.6 m range limit, 27° cone, 30 Hz.
  - Simulate sensor dropouts.
  - Domain randomisation.
  - Model the net boundary.
- **Not yet built:**
  - Real PX4 SITL (dt-px4) in the loop.
  - The Pi node that runs the policy and publishes `/mavros/setpoint_raw/attitude`.
- **Rendering:** the room floor renders black in videos (cosmetic).

## Videos
### PID-controlled hover (Crazyflie)
https://github.com/user-attachments/assets/4a06922b-99e3-4d18-b24d-4a863748ade6

### RL hover (Crazyflie, 500 iterations)
https://github.com/user-attachments/assets/0863106f-8b25-4cbd-b159-54d819b61174
