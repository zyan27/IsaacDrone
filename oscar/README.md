# Running IsaacDrone (Isaac Sim 5.0 / Isaac Lab 2.2) on Oscar

Isaac Sim needs Linux + an RTX NVIDIA GPU, so training runs on Oscar inside an
Apptainer container. Everything large (image, caches, pip installs) lives in
`~/scratch`, not your home quota.

```
build_sif.sh             one-time: pull Isaac Sim 5.0 image -> .sif
setup_isaaclab.sbatch    one-time: clone Isaac Lab v2.2.1, install into container python
convert_dd24.sbatch      DD24 URDF -> USD (re-run whenever mass/inertia/mesh change)
train_hover.sbatch       PPO hover training (headless)
common.sh                shared paths + the apptainer wrapper (sourced by the above)
```

## 0. GPU type: check what your account can actually get
Isaac Sim needs RT cores: A6000 / A40 / A5000 / RTX 3090 work. V100 / A100 / H100
do not. Slurm rejects a job whose `--constraint` names a feature that no node has,
so list the real tags first:
```bash
sinfo -p gpu -o "%f" | tr ',' '\n' | sort -u
```
Edit the `#SBATCH --constraint=` line in `train_hover.sbatch` to match, or override
it: `sbatch --constraint="a5000|geforce3090" oscar/train_hover.sbatch`.
(Exploratory accounts get the `gpu` partition. `gpu-he` needs a High-End priority account.)

## 1. One-time setup
```bash
git clone https://github.com/zyan27/IsaacDrone.git ~/IsaacDrone
cd ~/IsaacDrone && mkdir -p oscar/logs

# Isaac Sim image from NGC (free account + API key):
apptainer remote login --username '$oauthtoken' docker://nvcr.io   # paste NGC key
bash oscar/build_sif.sh                 # -> ~/scratch/isaac/isaac-sim-5.0.0.sif (~10+ GB)

sbatch oscar/setup_isaaclab.sbatch      # clones Isaac Lab v2.2.1 + installs it
tail -f oscar/logs/isaaclab_setup_*.out # wait for "Isaac Lab OK"
```

## 2. Build the DD24 asset (mesh + inertia -> URDF -> USD)
`assets/dd24/dd24.urdf` and `meshes/dd24_body.obj` are already generated from the
DD24 CAD and the bifilar inertia measurement. You only need the USD conversion
(`sbatch oscar/convert_dd24.sbatch`). To regenerate after changing values:
```bash
python assets/dd24/prep_cad_mesh.py DD24.stl assets/dd24/meshes/dd24_body.obj 60000   # only if CAD changed
python assets/dd24/build_dd24_urdf.py --mesh assets/dd24/meshes/dd24_body.obj \
    --prop-z 0.018 --collision-box 0.272 0.272 0.107     # mass/inertia/arm from dd24_params.py
```
Then convert on a GPU node:
```bash
sbatch oscar/convert_dd24.sbatch        # -> assets/dd24/dd24.usd
```
Mass, arm length and inertia all come from `environments/dd24_params.py`, so
Isaac and RotorPy always use the same numbers.

## 3. Train
```bash
sbatch oscar/train_hover.sbatch                                   # DD24, 1024 envs
AIRFRAME=crazyflie NUM_ENVS=512 ITERS=500 sbatch oscar/train_hover.sbatch   # sanity run
squeue -u $USER
tail -f oscar/logs/dd24_hover_*.out    # reward, ep_len (should climb toward 1000)
```
At startup the env prints the mass and body inertia that PhysX read from the
USD. Check them against your measurements.

Checkpoints are saved to `checkpoints/<airframe>_hover_<jobid>.pt`. To pull one back:
```bash
scp YOURID@ssh.ccv.brown.edu:~/IsaacDrone/checkpoints/dd24_hover_*.pt ./checkpoints/
```

## 4. Overnight runs (train -> test, resumable)

### One-time: the Python venv (for RotorPy + analysis)
```bash
bash oscar/setup_venv.sh          # on a login node; creates ~/envs/dd24
```
Isaac Sim can't live in a venv on Oscar. Its pip wheels need GLIBC >= 2.35, newer
than Oscar's OS. So Isaac runs from the Apptainer image, and the venv is for RotorPy
(CPU) and for looking at results.

### Big run in one command (recommended)
```bash
cd ~/IsaacDrone && git pull
ITERS=10000 SEEDS=0-2 MAIL_USER=you@brown.edu bash oscar/smoke_then_overnight.sh
```
Runs a 40-min smoke test, checks it (`oscar/check_smoke.py`: all four tests ran,
learning rate not pinned, KL sane) and only then launches the long Isaac run.
`cat oscar/logs/gate_*.out` shows the decision. ~4.5 s/iteration on an A5000 with
1024 envs: 10000 iterations = ~12.5 h per seed (one automatic resubmit), two seeds
at a time on a free account.

### Test the best checkpoints and fine-tune (after a run plateaus)
```bash
scancel <training job id>          # policies are already saved in each seed's best.pt
RUN=dd24_hover_1115_1834 MAIL_USER=you@brown.edu bash oscar/finetune_tonight.sh
```
Tests every seed's `best.pt` (four tests), picks the best seed (`oscar/pick_best.py`:
mean success over the four tests), then fine-tunes it for 1500 iterations from its
`best.pt` with the learning-rate ceiling decaying 3e-4 -> 1e-5 (`--init_from`,
`--lr_max_final`) and tests the result. Compare everything with
`python3 oscar/pick_best.py runs/${RUN}_s0 runs/${RUN}_s1 runs/${RUN}_ft_s*`.

### Compare two training variants (A/B)
```bash
ITERS=2000 MAIL_USER=you@brown.edu bash oscar/ab_test.sh
```
Smoke test first (calm reward), then on one GPU each: A = original reward (`--reward v1
--gamma 0.99`), B = calm reward (`--reward calm --gamma 0.995`: crash 40, horizontal-speed,
tilt and smoothness penalties, ~4 s look-ahead). Both include the accelerometer fix.
Compare with `pick_best.py` and the flight recorder. Any run can use
`EXTRA="--reward calm --gamma 0.995"`.

### Why and when does it fail? (flight recorder)
```bash
CKPT=runs/dd24_hover_1115_1834_s0/best.pt sbatch oscar/flight_recorder.sbatch
#   SCENARIO=fixed_net or LAG=1 for the 3x3 net / worst-case lag
```
Flies 512 drones for one episode each (same setup as the tests), logs every step,
then writes `<ckpt dir>/flights_<scenario>/`: `report.txt` (failure modes, when they
happen, which conditions make failure or drift more likely), `plots/` (one
flight-recorder dashboard per selected failure / worst drifter) and `videos/`
(top + side animation of each, ToF beams as the policy saw them).

### Submit everything before you leave
```bash
cd ~/IsaacDrone && git pull
bash oscar/overnight.sh                               # Isaac x3 seeds + RotorPy x3 seeds
MAIL_USER=you@brown.edu bash oscar/overnight.sh       # same, plus an email per job
```
Each job **trains, then tests itself**:

| | Isaac (GPU, `gpu` partition) | RotorPy (CPU, `batch` partition) |
|---|---|---|
| train | `train_rl_hover.py` -> `runs/<tag>_s<seed>/best.pt, latest.pt, progress.csv` | `train_paths.py` -> `latest.zip`, `ckpts/` |
| test  | `eval_rl_hover.py`: 512 fresh episodes, no exploration noise | `eval_policy.py`: 256 fresh episodes |
| result | `eval.json` | `eval.json` |

Both stop cleanly before the 10 h walltime. If training isn't finished, the job
**resubmits itself** (up to 2 times) and resumes from `latest.pt`/`latest.zip`.
Nothing is lost if Oscar kills a job: checkpoints are written atomically.

### In the morning
```bash
python3 oscar/summarize_runs.py          # table of every run: progress + test results
squeue -u $USER                          # anything still running / resubmitted
tail -n 30 oscar/logs/isaac_hover_*_0.out
```
`PASS` in the summary means the test met all of: 95%+ of episodes completed,
mean horizontal error under 15 cm, and (Isaac) mean height error under 10 cm.
These thresholds live in the eval scripts.

### Training settings (why)
* Inputs are normalised with running mean/std (saved in the checkpoint).
* 4 minibatches x 5 epochs per iteration (20 updates), as in Isaac Lab's rsl_rl.
* Initial action noise log_std = -1.6.
* Learning rate adapted once per epoch (not per minibatch), capped at 1e-3, and an
  epoch stops early if KL > 0.04. Actor output is linear (clamped by the env), so it
  can't saturate. History: smoke test 1 had lr collapse to 1e-5; smoke test 2 had it
  run away to 1e-2 and saturate the sigmoid output.

### Learning rate
Training uses an **adaptive** schedule by default: after every minibatch the
learning rate is lowered if the policy just moved too far (KL > 0.02) and raised if
it barely moved (KL < 0.005), within 1e-5..1e-3. `lr` and `kl` are printed every
iteration and saved in `progress.csv`. To compare, pass `EXTRA="--lr_schedule linear"`.

### Useful knobs (environment variables)
```bash
SEEDS=0-4 bash oscar/overnight.sh                     # more seeds
ONLY=rotorpy bash oscar/overnight.sh                  # one track only
ITERS=6000 NUM_ENVS=2048 sbatch --array=0-2 oscar/train_eval_isaac.sbatch
TASK=takeoff TIMESTEPS=40000000 sbatch --array=0-2 oscar/train_eval_rotorpy.sbatch
SEED=0 RUN_TAG=dd24_hover_1114 sbatch oscar/train_eval_isaac.sbatch   # resume one run
```
If you change `#SBATCH --time`, set `WALL_MIN` to match (minutes) so the job
stops in time to save and test.

### Testing a checkpoint by hand
```bash
# Isaac (inside the container, from a GPU node or a job)
source oscar/common.sh
isaac_exec "cd ~/IsaacDrone && $ISAACLAB_DIR/isaaclab.sh -p eval_rl_hover.py --headless \
    --checkpoint runs/dd24_hover_1114_s0/best.pt"
# RotorPy (venv, any node)
source ~/envs/dd24/bin/activate
python rotorpy_dd24/eval_policy.py --model runs/rotorpy_hover_1114_s0/latest.zip --net
python rotorpy_dd24/visualize_policy.py --model runs/rotorpy_hover_1114_s0/latest.zip   # plot
```

## What the policy is trained on (Isaac)
**Output = PX4 offboard setpoint:** collective thrust + body rates, what a MAVROS
node publishes on `/mavros/setpoint_raw/attitude`. Below it, a simulated PX4 rate
controller (default MC_*RATE gains, IMU_DGYRO_CUTOFF 20 Hz, quad-X allocation from
`fc_firmware/duckiedrone-px4-v4.params`) runs at 200 Hz with randomised gains,
motor lag and thrust curve. Policy rate 50 Hz.

**Inputs** (exact layout: `deploy/INTERFACE.md`), each at the rate the node gets it:
IMU + attitude (~50 Hz), 5 ToFs (20 Hz), Pi-camera optical flow (15 Hz, 60-120 ms
latency), last command, target height; 8 steps of history (176 numbers).

**Scenes** re-drawn every episode (`environments/scenes.py`): net alone, net inside
a room, room with furniture, open hall, plus clutter and ghost ToF returns. The
drone holds the spot it started at, at a random target height.

Each run is tested three ways: `eval.json` (training mix, per-scenario breakdown),
`eval_net70.json` (ToFs miss the net 70% of the time) and `eval_fixednet.json`.

## 5. Watching the policy fly (videos)
```bash
sbatch oscar/record_video.sbatch                                     # seed 0, ~15-25 min
CKPT=runs/dd24_hover_1115_s1/best.pt sbatch oscar/record_video.sbatch  # another run
```
Writes `arena.mp4`, `chase.mp4`, `top.mp4` and `arena_net70.mp4` (the ToFs miss the
net 70% of the time) into `<run>/videos/`. The net is drawn as a strand grid and
the hover target as a red dot. Copy them to your Mac:
```bash
scp -r YOURID@ssh.ccv.brown.edu:~/IsaacDrone/runs/dd24_hover_1115_s0/videos ~/Downloads/
```

## Recommended first run
Before the first overnight batch, run a short Isaac smoke test:
`ITERS=30 WALL_MIN=40 RESUBMITS=0 sbatch --time=00:40:00 oscar/train_eval_isaac.sbatch`.
It should finish with an `eval.json` in `runs/dd24_hover_s0/`. Or start with a 30-minute `AIRFRAME=crazyflie ITERS=200` job. It checks the container,
the Isaac Lab install, and Nucleus asset access (the Crazyflie mesh and ground
plane download on first run) before you spend an 8-hour allocation.

## Troubleshooting
* **Job rejected / "Invalid feature specification"**: the constraint names a GPU
  tag that doesn't exist. See step 0.
* **First run hangs at asset loading**: compute nodes need outbound HTTPS to
  NVIDIA's asset server for the Crazyflie USD and ground plane. Run once
  interactively (`interact -q gpu -g 1 -t 1:00:00`) to warm `~/scratch/isaac/cache`.
* **`No module named isaaclab`** during training: the setup job's user install
  didn't land in `$PYUSER`. Check its log for `user site enabled: False`. If
  you see that, ask me for the overlay-image variant.
* **`apptainer: command not found`**: CCV ships Apptainer natively on compute
  nodes. Make sure you are on a compute node, not a login node.
