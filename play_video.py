"""Headless playback: roll out a trained hover checkpoint and save an MP4.

For clusters with no display (Oscar). Run with --headless --enable_cameras.
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--rl_checkpoint", type=str, default="checkpoints/rl_hover.pt")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--out", type=str, default="videos/rl_hover.mp4")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import imageio
import torch

from environments.drone_hover_env import DroneHoverEnv, DroneHoverEnvCfg
from controllers.rl_hover_controller import RLHoverController


def main():
    env_cfg = DroneHoverEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.hover_height = 1.0
    # Crazyflie is ~9 cm across; the default 7.5 m world viewer makes it a speck.
    # Fixed camera near env_0 (eye/lookat are offsets from its origin), looking at
    # the 1 m hover target so climb, overshoot and drift stay visible.
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (1.2, 1.2, 1.3)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.8)

    env = DroneHoverEnv(cfg=env_cfg, render_mode="rgb_array")
    controller = RLHoverController(device=env.device, checkpoint=args.rl_checkpoint)

    dt = env.cfg.sim.dt * env.cfg.decimation
    steps = int(args.seconds / dt)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = imageio.get_writer(args.out, fps=int(round(1.0 / dt)))

    obs, _ = env.reset()
    for step in range(steps):
        with torch.inference_mode():
            actions = controller.act(obs["policy"])
            obs, _, terminated, truncated, _ = env.step(actions)

        writer.append_data(env.render())

        if step % 25 == 0:
            z = env.drone.data.root_state_w[:, 2]
            print(f"t={step * dt:5.2f}s z_env0={z[0].item():.3f} z_mean={z.mean().item():.3f}")

        if torch.any(terminated | truncated):
            print(f"t={step * dt:5.2f}s reset (terminated={terminated.sum().item()})")
            obs, _ = env.reset()

    writer.close()
    print(f"wrote {args.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
