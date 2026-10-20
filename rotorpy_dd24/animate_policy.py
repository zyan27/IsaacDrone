"""Roll out a trained DD24 policy and render it FLYING as a 3D animation.

Unlike visualize_policy.py (a static trajectory plot), this draws the DD24
quadrotor body + rotors moving through space over time, using rotorpy's
Quadrotor shape. Saves an .mp4 (ffmpeg) or .gif (pillow) you can play back.

Usage:
    python rotorpy_dd24/animate_policy.py --task hover --pos_bound 0.6
    python rotorpy_dd24/animate_policy.py --model checkpoints/dd24_hover.zip \
        --out rotorpy_dd24/hover_flight.mp4
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # render to file, no display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(__file__))
from train_paths import build_env, CONTROL_MODE
from dd24_rotorpy_params import quad_params
from rotorpy.utils.shapes import Quadrotor
from rotorpy.world import World


def rollout(model, env, max_steps=1000):
    """Step the policy and collect position + orientation history.

    obs layout (rotorpy QuadrotorEnv): [pos 0:3, vel 3:6, quat 6:10, rate 10:13],
    quaternion in scipy [x, y, z, w] order.
    """
    obs = env.reset()
    pos, quat, t = [], [], []
    dt = env.t_step
    for k in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(action)
        o = np.asarray(obs)[0]
        pos.append(o[0:3].copy())
        quat.append(o[6:10].copy())
        t.append(k * dt)
        if bool(np.asarray(done).ravel()[0]):
            break
    return np.array(pos), np.array(quat), np.array(t)


def make_world(pos, goal, margin=0.5):
    """A cube big enough to contain the whole flight + the goal."""
    pts = np.vstack([pos, goal[None, :]])
    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    # keep it from being a degenerate flat box
    span = np.maximum(hi - lo, 1.0)
    c = (hi + lo) / 2
    lo, hi = c - span / 2, c + span / 2
    extents = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]
    return World.empty(extents), extents


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/dd24_hover.zip")
    p.add_argument("--task", choices=["hover", "goto", "takeoff"], default="hover")
    p.add_argument("--goal", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--pos_bound", type=float, default=1.0)
    p.add_argument("--out", default="rotorpy_dd24/hover_flight.mp4")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--rtf", type=float, default=1.0, help="real-time factor")
    args = p.parse_args()

    goal = np.array(args.goal)
    env = build_env(num_envs=1, goal=args.goal, task=args.task, pos_bound=args.pos_bound)
    model = PPO.load(args.model, device="cpu")
    pos, quat, t = rollout(model, env)

    dist = np.linalg.norm(pos - goal, axis=1)
    print(f"control_mode={CONTROL_MODE}  steps={len(t)}  "
          f"duration={t[-1]:.2f}s  final_dist={dist[-1]:.3f} m")
    print(f"start={pos[0].round(2)}  end={pos[-1].round(2)}  goal={goal}")

    # Decimate the sim history (1 kHz-ish) down to the render fps.
    sample_t = np.arange(0, t[-1], 1.0 / args.fps * args.rtf) if t[-1] > 0 else np.zeros(1)
    idx = np.round(np.interp(sample_t, t, np.arange(t.size))).astype(int)
    pos_s, quat_s, t_s = pos[idx], quat[idx], t[idx]
    R = Rotation.from_quat(quat_s).as_matrix()  # (N, 3, 3) body->world

    world, extents = make_world(pos, goal)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    world.draw(ax)
    ax.scatter(*goal, c="r", marker="*", s=140, label="goal")
    ax.scatter(*pos[0], c="g", s=40, label="start")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], c="0.6", lw=0.8, alpha=0.6)  # ghost path
    ax.legend(loc="upper right")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    arm_len = float(np.linalg.norm(quad_params["rotor_pos"]["r1"]))
    quad = Quadrotor(ax, arm_length=arm_len, wind=False)
    title = ax.set_title("DD24 policy — t = 0.00 s")

    def update(f):
        quad.transform(position=pos_s[f], rotation=R[f], wind=np.zeros(3))
        title.set_text(f"DD24 policy — t = {t_s[f]:.2f} s  |  dist {np.linalg.norm(pos_s[f]-goal):.2f} m")
        return []

    ani = FuncAnimation(fig, update, frames=len(t_s),
                        interval=1000.0 / args.fps, repeat=False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.out.endswith(".gif"):
        ani.save(args.out, writer="pillow", fps=args.fps, dpi=90)
    else:
        ani.save(args.out, writer="ffmpeg", fps=args.fps, dpi=110)
    print(f"saved animation -> {args.out}")


if __name__ == "__main__":
    main()
