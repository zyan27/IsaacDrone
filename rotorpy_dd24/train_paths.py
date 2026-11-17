"""Train a DD24 policy to fly to a target, in rotorpy's QuadrotorEnv with PPO.

Runs on CPU (Mac included). This is the path-A/training half; validate the
trained policy against PX4 SITL separately (see sim_px4 / rotorpy PX4Multirotor).

Usage:
    python rotorpy_dd24/train_paths.py --timesteps 200000 --num_envs 16
"""
import argparse
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecEnvWrapper

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rotorpy.world import World
from environments import arena as A
from rotorpy.learning.quadrotor_environments import QuadrotorEnv, DEFAULT_RESET_OPTIONS
from rotorpy.learning.quadrotor_reward_functions import vec_hover_reward
from dd24_rotorpy_params import quad_params
from rewards import make_goto_reward, make_precision_hover_reward
from takeoff_env import TakeoffHoverEnv

# cmd_ctbr = collective thrust + body rates. Maps directly to PX4 offboard rate
# setpoints (ideal for the PX4 handoff), and works in float32 (cmd_vel/cmd_ctatt
# currently hit a float/double bug inside rotorpy and would need float64 states).
CONTROL_MODE = "cmd_ctbr"


def hover_rotor_speed():
    import math
    return math.sqrt(
        quad_params["mass"] * 9.81
        / (quad_params["num_rotors"] * quad_params["k_eta"])
    )


def make_initial_states(num_envs):
    hover = hover_rotor_speed()
    q = np.zeros((num_envs, 4), dtype=np.float32)
    q[:, 3] = 1.0  # identity quaternion [x, y, z, w]
    return {
        "x": np.zeros((num_envs, 3), dtype=np.float32),
        "v": np.zeros((num_envs, 3), dtype=np.float32),
        "q": q,
        "w": np.zeros((num_envs, 3), dtype=np.float32),
        "wind": np.zeros((num_envs, 3), dtype=np.float32),
        "rotor_speeds": np.full((num_envs, 4), hover, dtype=np.float32),
    }


# Tuned hover weights: nonzero body-rate weight 'w' adds attitude damping that
# the rotorpy default (w=0) lacks, and a bit more velocity weight reduces drift.
from functools import partial
HOVER_WEIGHTS = {"x": 1.0, "v": 0.3, "w": 0.1, "u": 1e-4}


class HoverCenteredThrust(VecEnvWrapper):
    """Re-centre the collective-thrust action so 0 == hover.

    RotorPy maps action[0] in [-1, 1] linearly onto [0, max total thrust]. With the
    DD24's thrust-to-weight of ~5.3, action 0 is ~2.7x body weight, so a fresh PPO
    policy (actions ~N(0, 1)) rockets into the ceiling within a second and almost
    never discovers hover (action ~ -0.62). Here the policy's action a maps to
    total thrust = weight * (1 + a), i.e. a in [-1, 1] -> [0, 2x weight].
    Body-rate actions are passed through unchanged."""

    def __init__(self, venv):
        super().__init__(venv)
        tw = (quad_params["num_rotors"] * quad_params["k_eta"] * quad_params["rotor_speed_max"] ** 2
              / (quad_params["mass"] * 9.81))
        self.h = 2.0 / tw - 1.0          # RotorPy's normalised command for exactly hover
        self.span = self.h + 1.0         # so a = +1 -> 2x weight, a = -1 -> 0

    def step_async(self, actions):
        a = np.array(actions, dtype=np.float32, copy=True)
        a[..., 0] = np.clip(self.h + a[..., 0] * self.span, -1.0, 1.0)
        self.venv.step_async(a)

    def step_wait(self):
        return self.venv.step_wait()

    def reset(self):
        return self.venv.reset()


def arena_world(task, goal):
    """The net as RotorPy world bounds (leaving them ends the episode).
    hover/goto: the target is the origin-ish and the floor is 1 m below it.
    takeoff: the drone starts on the floor at z=0."""
    floor = 0.0 if task == "takeoff" else -1.0
    xmin, xmax, ymin, ymax, zmin, zmax = A.keep_in_box(floor_z=floor)
    if task == "takeoff":
        zmin = floor - 0.3   # starts AT the floor and sags before spinning up
    return World.empty((xmin, xmax, ymin, ymax, zmin, zmax))


def with_arena_shaping(reward_fn, task, alive_bonus=2.0, w_net=2.0):
    """Add an alive bonus (so leaving the arena early is never 'rewarded' by
    skipping future negative rewards) and a soft penalty near the net."""
    floor = 0.0 if task == "takeoff" else -1.0
    xmin, xmax, ymin, ymax, zmin, zmax = A.keep_in_box(floor_z=floor)
    band = A.NET_WARN_BAND

    def fn(observation, action):
        r = reward_fn(observation, action)
        p = observation[..., 0:3]
        d = np.minimum.reduce([p[..., 0] - xmin, xmax - p[..., 0],
                               p[..., 1] - ymin, ymax - p[..., 1], zmax - p[..., 2]])
        pen = np.clip((band - d) / band, 0.0, 1.0) ** 2
        return r + alive_bonus - w_net * pen

    return fn


def build_env(num_envs, goal, task="hover", pos_bound=1.0, net=False):
    """Build the training env for a task:
      'takeoff' : start at rest on the pad, climb to `goal`, hold it to cm
                  precision (TakeoffHoverEnv + make_precision_hover_reward).
      'hover'   : rotorpy's vec_hover_reward, random airborne spawn -> origin.
      'goto'    : make_goto_reward(goal), random airborne spawn -> goal.
    pos_bound = airborne spawn radius (m) for hover/goto; ignored for takeoff
    (its start distribution lives in TakeoffHoverEnv.reset_idx)."""
    if task == "takeoff":
        return HoverCenteredThrust(TakeoffHoverEnv(
            num_envs=num_envs,
            initial_states=make_initial_states(num_envs),
            control_mode=CONTROL_MODE,
            reward_fn=(with_arena_shaping(make_precision_hover_reward(goal), task)
                       if net else make_precision_hover_reward(goal)),
            quad_params=quad_params,
            max_time=10,
            goal=goal,
            **({"world": arena_world(task, goal)} if net else {}),
        ))
    if task == "hover":
        reward_fn = partial(vec_hover_reward, weights=HOVER_WEIGHTS)
    else:
        reward_fn = make_goto_reward(goal)
    if net:
        reward_fn = with_arena_shaping(reward_fn, task)
    ro = dict(DEFAULT_RESET_OPTIONS)
    if net:
        # Random spawns must start inside the arena (floor is 1 m below the target).
        pos_bound = min(pos_bound, 0.8)
    ro["pos_bound"] = pos_bound
    ro["vel_bound"] = 0
    return HoverCenteredThrust(QuadrotorEnv(
        num_envs=num_envs,
        initial_states=make_initial_states(num_envs),
        control_mode=CONTROL_MODE,
        reward_fn=reward_fn,
        quad_params=quad_params,
        max_time=10,
        reset_options=ro,
        **({"world": arena_world(task, goal)} if net else {}),
    ))


class TimeLimit(BaseCallback):
    """Stop learning cleanly before the Slurm walltime so the model gets saved."""
    def __init__(self, minutes):
        super().__init__()
        self.deadline = time.time() + minutes * 60 if minutes > 0 else None
        self.hit = False

    def _on_step(self):
        if self.deadline and time.time() > self.deadline:
            self.hit = True
            return False
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--task", choices=["hover", "goto", "takeoff"], default="hover")
    p.add_argument("--goal", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    p.add_argument("--pos_bound", type=float, default=1.0)
    p.add_argument("--net", action="store_true", help="Use the arena (net) from environments/arena.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default=None, help="Final model path (default <run_dir>/final.zip)")
    p.add_argument("--run_dir", type=str, default="runs/rotorpy_hover")
    p.add_argument("--resume", action="store_true", help="Continue from <run_dir>/latest.zip")
    p.add_argument("--checkpoint_every", type=int, default=200_000, help="timesteps")
    p.add_argument("--time_limit_min", type=float, default=0.0)
    args = p.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    env = build_env(args.num_envs, args.goal, task=args.task, pos_bound=args.pos_bound, net=args.net)
    latest = os.path.join(args.run_dir, "latest.zip")
    if args.resume and os.path.isfile(latest):
        model = PPO.load(latest, env=env, device="cpu")
        print(f"Resumed from {latest} at {model.num_timesteps} timesteps")
    else:
        try:
            import tensorboard  # noqa: F401
            tb = os.path.join(args.run_dir, "tb")
        except ImportError:
            tb = None
        model = PPO("MlpPolicy", env, verbose=1, device="cpu", seed=args.seed, tensorboard_log=tb)

    remaining = max(args.timesteps - model.num_timesteps, 0)
    tl = TimeLimit(args.time_limit_min)
    ckpt = CheckpointCallback(save_freq=max(args.checkpoint_every // args.num_envs, 1),
                              save_path=os.path.join(args.run_dir, "ckpts"), name_prefix="ppo")
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False,
                    callback=CallbackList([ckpt, tl]))
    model.save(latest)
    final = args.save or os.path.join(args.run_dir, "final.zip")
    model.save(final)
    if not tl.hit:
        open(os.path.join(args.run_dir, "DONE"), "w").write(f"{model.num_timesteps} timesteps\n")
    print(f"saved -> {final}  ({model.num_timesteps} timesteps{', stopped at time limit' if tl.hit else ''})")


if __name__ == "__main__":
    main()
