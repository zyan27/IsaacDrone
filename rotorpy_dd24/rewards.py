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


# NOTE on true trajectory/path tracking:
# QuadrotorEnv's observation is absolute state with the goal baked at the origin.
# For a MOVING target (a real path), the policy must SEE the target, so subclass
# QuadrotorEnv and append the target-relative error (and lookahead points) to the
# observation, then reward small tracking error. Fixed-goal goto above needs no
# subclass and is enough to validate the DD24 dynamics + PX4 handoff first.
