# DD24 RL on RotorPy (+ PX4 SITL)

RotorPy is the simulator the **DuckieDrone team actually uses** for DD24 SIL
(PX4 SITL + rotorpy). It is pure-Python/CPU, so **training runs on your Mac** —
no Isaac Sim, no NVIDIA GPU, no conda/CUDA.

## Setup (Python 3.11)
```bash
cd IsaacDrone
/opt/homebrew/bin/python3.11 -m venv .venv && source .venv/bin/activate
pip install -r rotorpy_dd24/requirements.txt
```
Python 3.14 does NOT work (no torch wheels). `roma` + `torchdiffeq` are required
even though rotorpy lists them as optional — without them the batched env fails.

## Train (fly to a target)
```bash
python rotorpy_dd24/train_paths.py --timesteps 200000 --num_envs 16 --goal 1 1 1
```
Verified: PPO runs at ~4000 fps on CPU and saves `checkpoints/dd24_goto.zip`.

## Files
- `dd24_rotorpy_params.py` — DD24 vehicle params (rotorpy schema). Mass/arm come
  from `../environments/dd24_params.py` (single source of truth). Rotor/inertia
  coeffs are `[EST]` — see that file for what to measure.
- `rewards.py` — `make_goto_reward(goal)`; note on extending to true path/
  trajectory tracking (subclass the env to put the target in the observation).
- `train_paths.py` — PPO training entry point.

## Control mode
Uses **`cmd_ctbr`** (collective thrust + body rates). Chosen because:
1. it maps directly to **PX4 offboard rate setpoints** (clean train→PX4 handoff);
2. it works in float32 (rotorpy's `cmd_vel`/`cmd_ctatt` currently hit a
   float/double bug and would need float64 initial states).

## Next: PX4 in the loop (validation)
After training, validate against the real flight controller with rotorpy's
built-in `PX4Multirotor` (pymavlink → PX4 SITL). Build PX4 SITL
(`make px4_sitl none_iris` style) and run the policy emitting rate setpoints to
PX4 over MAVLink. This is the DD24's actual FC, so it is the sim2real checkpoint
before flashing to hardware.

## Before trusting sim2real — measure on the physical drone
Arm length, max thrust/motor (thrust stand → `k_eta`), inertia (CAD / bifilar).
All are `[EST]` placeholders now. See `../environments/dd24_params.py`.
