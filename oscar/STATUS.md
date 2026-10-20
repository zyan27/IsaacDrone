# Oscar bring-up status (2026-10-20, stopped at 01:00)

On disk under `/oscar/scratch/zyan33/` (scratch is purged periodically):
- `isaac/isaac-sim-5.0.0.sif`: **built, verified** (L40S visible, VERSION 5.0.0).
- `isaac/IsaacLab`: v2.2.1 clone + `_isaac_sim -> /isaac-sim` link.
- `isaac/pyuser`: Isaac Lab install target (README step 4). Check
  `isaac/isaaclab_install.log` ends in `exit=0`; if not, re-run step 4.
- `px4/dt-px4-ente-amd64.sif`: dt-px4 converted from Docker. Check `px4/build.log`.

Next, in order:
1. Verify imports: `apptainer exec --nv --cleanenv --bind /oscar/scratch/$USER
   --env PYTHONUSERBASE=/oscar/scratch/$USER/isaac/pyuser <sif>
   /isaac-sim/python.sh -c "import isaaclab, isaaclab_assets"`.
2. Smoke train: `NUM_ENVS=64 ITERS=20 sbatch oscar/train_hover.sbatch`.
   The sbatch binds/HOME changes are **untested**.
3. dt-px4 param probe: boot SITL (airframe 10040 and 4001), `param show`
   THR_MDL_FAC, MC_*RATE_P/I/D, MC_AIRMODE, MPC_THR_HOVER, IMU_*CUTOFF; diff vs
   `sim_px4/px4_defaults.py` + `fc_firmware/duckiedrone-px4-v4.params`.
4. Visualization: `play_video.py` (headless MP4), then OnDemand Desktop + main.py.
