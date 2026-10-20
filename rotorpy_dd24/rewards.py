"""Reward functions for DD24 RL tasks in rotorpy's QuadrotorEnv.

Observation layout (rotorpy QuadrotorEnv): [pos(0:3), vel(3:6), quat(6:10),
bodyrate(10:13)]. A reward_fn receives (observation, action), both batched over
envs, and returns one reward per env.
"""
import numpy as np


def make_goto_reward(goal, w_pos=1.0, w_vel=0.1, w_act=1e-4, bonus=2.0):
    """Fly to and hold a single target position `goal` (x, y, z). This is the
    building block for paths: a path is a sequence of goto targets."""
    goal = np.asarray(goal, dtype=np.float32)

    def reward_fn(observation, action):
        pos = observation[..., 0:3]
        vel = observation[..., 3:6]
        dist = np.linalg.norm(pos - goal, axis=-1)
        speed = np.linalg.norm(vel, axis=-1)
        effort = np.linalg.norm(action, axis=-1)
        return -w_pos * dist - w_vel * speed - w_act * effort + bonus

    return reward_fn


def make_precision_hover_reward(goal, w_far=1.0, w_peak=6.0, sigma=0.12,
                                 w_vel=0.3, w_rate=0.05, w_act=1e-3):
    """Climb to `goal` and hold it to centimeter precision.

    Why not make_goto_reward: that penalty is linear in distance, so near the
    goal its gradient is a tiny constant and the policy happily loiters ~0.4 m
    out. Here two terms stack:
      -w_far * dist            far-field pull (gradient everywhere -> climb from
                               the pad toward the goal)
      +w_peak * exp(-d^2/2s^2) a sharp Gaussian peak at the goal (sigma~0.12 m):
                               near zero its gradient is largest, so the last
                               few cm are strongly rewarded; drifting to 0.4 m
                               earns almost nothing (exp(-5.6)~0.004).
    Velocity and body-rate penalties kill the residual drift/wobble that would
    otherwise leave a steady-state offset. Tune sigma down to demand tighter
    hold (at the cost of harder exploration)."""
    goal = np.asarray(goal, dtype=np.float32)
    two_sigma_sq = 2.0 * sigma * sigma

    def reward_fn(observation, action):
        pos = observation[..., 0:3]
        vel = observation[..., 3:6]
        rate = observation[..., 10:13]
        dist = np.linalg.norm(pos - goal, axis=-1)
        speed = np.linalg.norm(vel, axis=-1)
        spin = np.linalg.norm(rate, axis=-1)
        effort = np.linalg.norm(action, axis=-1)
        peak = np.exp(-(dist ** 2) / two_sigma_sq)
        return (-w_far * dist + w_peak * peak
                - w_vel * speed - w_rate * spin - w_act * effort)

    return reward_fn


# NOTE on true trajectory/path tracking:
# QuadrotorEnv's observation is absolute state with the goal baked at the origin.
# For a MOVING target (a real path), the policy must SEE the target, so subclass
# QuadrotorEnv and append the target-relative error (and lookahead points) to the
# observation, then reward small tracking error. Fixed-goal goto above needs no
# subclass and is enough to validate the DD24 dynamics + PX4 handoff first.
