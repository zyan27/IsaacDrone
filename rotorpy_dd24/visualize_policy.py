"""Roll out a trained DD24 policy and plot its trajectory.

Usage:
    python rotorpy_dd24/visualize_policy.py --model checkpoints/dd24_goto.zip --goal 1 1 1
Saves a PNG (3D path + distance-to-goal) and prints whether it reached the goal.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save to file, no display needed
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(__file__))
from train_paths import build_env, CONTROL_MODE


def rollout(model, env, max_steps=1000):
    obs = env.reset()
    pos, t = [], []
    dt = env.t_step
    for k in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(action)
        pos.append(np.asarray(obs)[0, 0:3].copy())
        t.append(k * dt)
        if bool(np.asarray(done).ravel()[0]):
            break
    return np.array(pos), np.array(t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/dd24_hover.zip")
    p.add_argument("--task", choices=["hover", "goto"], default="hover")
    p.add_argument("--goal", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--pos_bound", type=float, default=1.0)
    p.add_argument("--out", default="rotorpy_dd24/last_rollout.png")
    args = p.parse_args()

    goal = np.array(args.goal)
    env = build_env(num_envs=1, goal=args.goal, task=args.task, pos_bound=args.pos_bound)
    model = PPO.load(args.model, device="cpu")
    pos, t = rollout(model, env)

    dist = np.linalg.norm(pos - goal, axis=1)
    final = dist[-1]
    reached = final < 0.2
    print(f"control_mode={CONTROL_MODE}  steps={len(t)}  "
          f"final_dist={final:.3f} m  reached(<0.2m)={reached}")
    print(f"start={pos[0].round(2)}  end={pos[-1].round(2)}  goal={goal}")

    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=1.5)
    ax.scatter(*pos[0], c="g", s=40, label="start")
    ax.scatter(*goal, c="r", marker="*", s=120, label="goal")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("DD24 policy trajectory"); ax.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, dist)
    ax2.axhline(0.2, ls="--", c="r", lw=0.8)
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("distance to goal (m)")
    ax2.set_title("convergence")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved plot -> {args.out}")


if __name__ == "__main__":
    main()
