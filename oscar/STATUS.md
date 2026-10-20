# Oscar bring-up status (2026-10-20, stopped at 01:00)

**Working:** `NUM_ENVS=64 ITERS=5 bash oscar/train_hover.sbatch` on gpu3104 (L40S)
ran headless and saved `checkpoints/rl_hover.pt` (00:54:56). Not yet confirmed:
per-iter reward prints did not show up in the log, so reward trend is unchecked.

On disk under `/oscar/scratch/zyan33/` (scratch is purged periodically):
- `isaac/isaac-sim-5.0.0.sif`: built, verified (L40S visible, VERSION 5.0.0).
- `isaac/IsaacLab`: v2.2.1 + `_isaac_sim -> /isaac-sim`.
- `isaac/pyuser`: Isaac Lab installed (README step 4, incl. flatdict fix).
- `isaac/cache/{kit,kitdata,kitlogs,home}`: writable dirs the sbatch binds.
- `px4/dt-px4-ente-amd64.sif`: dt-px4 converted from Docker (not yet booted).

Next, in order:
1. Find why train prints are missing from the log (check
   `isaac/cache/kitlogs`, maybe add `python -u` / flush), then run ~50 iters
   and confirm reward rises. Then a full `sbatch oscar/train_hover.sbatch`.
2. dt-px4 param probe: boot SITL (airframe 10040 and 4001), `param show`
   THR_MDL_FAC, MC_*RATE_P/I/D, MC_AIRMODE, MPC_THR_HOVER, IMU_*CUTOFF; diff vs
   `sim_px4/px4_defaults.py` + `fc_firmware/duckiedrone-px4-v4.params`.
3. Visualization: `play_video.py` (headless MP4), then OnDemand Desktop + main.py.
