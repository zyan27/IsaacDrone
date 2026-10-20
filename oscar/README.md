# Running IsaacDrone (Isaac Sim) on Oscar

Dylan's env is **Isaac Sim / Isaac Lab** — Linux + **RTX NVIDIA GPU** only, so it
can't run on the Mac. Oscar (Brown CCV) has the right hardware. This directory
ports the headless training (`train_rl_hover.py`) to a SLURM + Apptainer job.

> `main.py` uses `render_mode="human"` (needs a display) — that's for local
> visualization, **not** Oscar. On Oscar we only run `train_rl_hover.py --headless`.

## The one gotcha that will bite you: GPU class
Isaac Sim's renderer needs **RT cores**. Oscar GPUs split:

| Have RT cores — USE THESE | No RT cores — WON'T WORK |
|---|---|
| L40S, A6000, A40, A5000 | A100, V100, H100, B200 |

So the job requests `--constraint=l40s|a6000|a40|a5000` and the `gpu` partition.
The high-end `gpu-he` (V100) partition is **unusable** for Isaac Sim.

## One-time setup (on Oscar, not the Mac)
```bash
# 1. Get the code onto Oscar
git clone https://github.com/zyan27/IsaacDrone.git ~/IsaacDrone

# 2. Build the Isaac Sim image (needs a free NGC account + API key)
#    apptainer remote login --username '$oauthtoken' docker://nvcr.io
bash ~/IsaacDrone/oscar/build_sif.sh          # -> ~/scratch/isaac/isaac-sim-*.sif

# 3. Clone Isaac Lab (version MUST match the repo; Isaac Sim 5.0 / Isaac Lab 2.x)
git clone https://github.com/isaac-sim/IsaacLab.git ~/scratch/isaac/IsaacLab
cd ~/scratch/isaac/IsaacLab
ln -s /isaac-sim _isaac_sim                   # point Isaac Lab at the in-container sim
# install Isaac Lab into the container's python (run once, inside the .sif):
apptainer exec --nv --env ACCEPT_EULA=Y ~/scratch/isaac/isaac-sim-5.0.0.sif \
    bash -lc "cd ~/scratch/isaac/IsaacLab && ./isaaclab.sh -i"
```

## Train
```bash
cd ~/IsaacDrone
sbatch oscar/train_hover.sbatch
# override defaults:  NUM_ENVS=512 ITERS=4000 sbatch oscar/train_hover.sbatch
squeue -u $USER            # watch it
tail -f oscar/logs/dd24_hover_*.out
```
Checkpoint lands at `checkpoints/rl_hover.pt`.

## Things to verify for YOUR Oscar account (I can't from here)
1. **Compute-node internet.** Isaac Sim pulls assets from NVIDIA Nucleus on first
   run. If Oscar's `gpu` nodes have no outbound internet, the first run hangs/fails
   — you'll need to pre-cache assets on a login node or set a local asset root.
   (The cache binds in the sbatch help on repeat runs but not the first fetch.)
2. **Apptainer availability.** CCV docs say it's on GPU nodes with no module; if
   `apptainer: command not found`, add `module load apptainer` to the sbatch.
3. **Version match.** If `import isaaclab` / `isaaclab_assets` (CRAZYFLIE_CFG)
   errors, the .sif Isaac Sim tag and the Isaac Lab clone are mismatched to the
   repo. Pin all three to the same Isaac Sim 5.0 generation.
4. **Scratch quota.** The .sif is ~10+ GB and the asset cache grows — keep both
   under `~/scratch`, never `~` (home quota is small).

## Pull the result back to the Mac
```bash
scp YOURID@ssh.ccv.brown.edu:~/IsaacDrone/checkpoints/rl_hover.pt ./checkpoints/
```
Then visualize locally with `main.py --RLHover --rl_checkpoint ...` **only** if you
have a Linux+RTX box; the Mac still can't render Isaac Sim. For a quick look at the
learning curve, the sbatch `.out` log prints per-iter reward.
