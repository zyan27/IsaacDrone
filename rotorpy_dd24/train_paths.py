"""Train a DD24 policy to fly to a target, in rotorpy's QuadrotorEnv with PPO.

Runs on CPU (Mac included). This is the path-A/training half; validate the
trained policy against PX4 SITL separately (see sim_px4 / rotorpy PX4Multirotor).

Usage:
    python rotorpy_dd24/train_paths.py --timesteps 200000 --num_envs 16
"""
import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(__file__))
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


def build_env(num_envs, goal, task="hover", pos_bound=1.0):
    """Build the training env for a task:
      'takeoff' : start at rest on the pad, climb to `goal`, hold it to cm
                  precision (TakeoffHoverEnv + make_precision_hover_reward).
      'hover'   : rotorpy's vec_hover_reward, random airborne spawn -> origin.
      'goto'    : make_goto_reward(goal), random airborne spawn -> goal.
    pos_bound = airborne spawn radius (m) for hover/goto; ignored for takeoff
    (its start distribution lives in TakeoffHoverEnv.reset_idx)."""
    if task == "takeoff":
        return TakeoffHoverEnv(
            num_envs=num_envs,
            initial_states=make_initial_states(num_envs),
            control_mode=CONTROL_MODE,
            reward_fn=make_precision_hover_reward(goal),
            quad_params=quad_params,
            max_time=10,
            goal=goal,
        )
    if task == "hover":
        reward_fn = partial(vec_hover_reward, weights=HOVER_WEIGHTS)
    else:
        reward_fn = make_goto_reward(goal)
    ro = dict(DEFAULT_RESET_OPTIONS)
    ro["pos_bound"] = pos_bound
    ro["vel_bound"] = 0
    return QuadrotorEnv(
        num_envs=num_envs,
        initial_states=make_initial_states(num_envs),
        control_mode=CONTROL_MODE,
        reward_fn=reward_fn,
        quad_params=quad_params,
        max_time=10,
        reset_options=ro,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--task", choices=["hover", "goto", "takeoff"], default="hover")
    p.add_argument("--goal", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    p.add_argument("--pos_bound", type=float, default=1.0)
    p.add_argument("--save", type=str, default="checkpoints/dd24_hover.zip")
    args = p.parse_args()

    env = build_env(args.num_envs, args.goal, task=args.task, pos_bound=args.pos_bound)
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=args.timesteps)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    model.save(args.save)
    print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
