"""Fly the DD24 through the PX4 inner loop and record an MP4 + CSV.

The height command follows the real-drone test (same profile as
deploy/policy_node.py): sit on the floor, climb straight up to --height, hold,
come straight down, land.

--checkpoint: the trained policy (train_rl_px4.py) is the commander. It sees only
the down / front / left / right ToFs and the IMU (each at its own rate, late, with
dropouts) and sends PX4 a CTBR setpoint.

Without --checkpoint, a scripted commander that cheats (true height / velocity)
flies the same profile, as a reference for what the PX4 loop can do.

Run with --headless --enable_cameras (see oscar/run_isaac.sh).
"""
import argparse
import csv
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=15.0)
parser.add_argument("--height", type=float, default=0.5, help="hold height (down ToF reading), m")
parser.add_argument("--wait", type=float, default=1.0, help="seconds on the floor before climbing")
parser.add_argument("--hold", type=float, default=6.0, help="seconds at the hold height")
parser.add_argument("--out", type=str, default="videos/px4_dd24.mp4")
parser.add_argument("--checkpoint", type=str, default=None, help="train_rl_px4.py checkpoint")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import imageio
import torch

from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from controllers.px4_policy import PX4Actor
from environments.drone_px4_env import POLICY_HOVER_THRUST, DronePX4Env, DronePX4EnvCfg, policy_to_ctbr
from sim_px4.px4_defaults import ATT
from visualization import Hud, make_target_markers, show_targets

FRD = torch.tensor([1.0, -1.0, -1.0])  # FLU <-> FRD for body rates / angles


def height_goal(t: float) -> float:
    """Wait on the floor, climb, hold, land (ramp rate = the env's target_speed_mps)."""
    if t < args.wait:
        return 0.0
    if t < args.wait + args.height / 0.3 + args.hold:
        return args.height
    return 0.0


def commander(env: DronePX4Env, z_int: torch.Tensor):
    """Reference outer loop with ground truth -> CTBR setpoint for PX4."""
    d = env.drone.data
    pos = d.root_pos_w - env.scene.env_origins
    vel = d.root_lin_vel_w
    roll, pitch, yaw = (wrap_to_pi(a) for a in euler_xyz_from_quat(d.root_quat_w))

    z_err = env.target_z - env._height()
    z_int += torch.where(z_err.abs() < 0.15, z_err, torch.zeros_like(z_err)) * 0.02  # no windup on climb
    thrust = POLICY_HOVER_THRUST + 0.12 * z_err - 0.16 * vel[:, 2] + 0.08 * z_int
    thrust = torch.where(env.target_z <= 0.0, torch.zeros_like(thrust), thrust)

    # Position hold: world accel request -> tilt, rotated into the heading frame.
    ax = 1.5 * -pos[:, 0] - 2.0 * vel[:, 0]
    ay = 1.5 * -pos[:, 1] - 2.0 * vel[:, 1]
    c, s = torch.cos(yaw), torch.sin(yaw)
    fwd, left = c * ax + s * ay, -s * ax + c * ay
    pitch_d = torch.clamp(fwd / 9.81, -0.35, 0.35)   # FLU: +pitch tilts thrust forward
    roll_d = torch.clamp(-left / 9.81, -0.35, 0.35)  # FLU: -roll tilts thrust left

    # PX4 attitude P (small-angle form) -> body rates.
    rates_flu = torch.stack([
        ATT.P[0] * wrap_to_pi(roll_d - roll),
        ATT.P[1] * wrap_to_pi(pitch_d - pitch),
        torch.zeros_like(yaw),
    ], dim=-1)
    rates_frd = rates_flu * FRD.to(rates_flu.device)
    return torch.cat([thrust.clamp(0.0, 1.0).unsqueeze(-1), rates_frd], dim=-1)


def main():
    cfg = DronePX4EnvCfg()
    cfg.scene.num_envs = 1
    cfg.wall_dist_range_m = (cfg.room_half_width_m, cfg.room_half_width_m)  # match the drawn walls
    # Camera in a corner of the enclosure, looking across it at the target.
    cfg.viewer.eye = (1.1, 1.1, 1.0)
    cfg.viewer.lookat = (0.0, 0.0, 0.35)
    env = DronePX4Env(cfg=cfg, render_mode="rgb_array")

    actor = None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=env.device)
        actor = PX4Actor(cfg.observation_space).to(env.device)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        print(f"policy from {args.checkpoint} (iter {ckpt.get('iter')}, step reward {ckpt.get('best_step_reward')})")

    dt = cfg.sim.dt * cfg.decimation
    steps = int(args.seconds / dt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = imageio.get_writer(args.out, fps=int(round(1.0 / dt)))
    log = open(os.path.splitext(args.out)[0] + ".csv", "w", newline="")
    rows = csv.writer(log)
    rows.writerow(["t", "x", "y", "height", "target", "roll_deg", "pitch_deg", "yaw_deg",
                   "sp_thrust", "sp_p", "sp_q", "sp_r", "gyro_p", "gyro_q", "gyro_r",
                   "m1", "m2", "m3", "m4", "tof_down", "tof_front", "tof_left", "tof_right", "tof_down_age", "imu_age"])

    obs, _ = env.reset()
    origin = env.scene.env_origins[0].tolist()
    env.sim.set_camera_view(
        eye=[o + e for o, e in zip(origin, cfg.viewer.eye)],
        target=[o + v for o, v in zip(origin, cfg.viewer.lookat)],
    )
    markers = make_target_markers()
    title = "DD24 RL policy -> PX4 (4 ToF + IMU, 20 Hz)" if actor else "DD24 scripted reference -> PX4"
    hud = Hud(title, args.height, args.seconds, room_half=cfg.room_half_width_m)
    z_int = torch.zeros(env.num_envs, device=env.device)

    for step in range(steps):
        t = step * dt
        env.target_goal[:] = height_goal(t)
        env.target_timer[:] = 1e9  # the profile above drives the command, not the env's random one
        with torch.inference_mode():
            o = obs["policy"][0].cpu()
            sp = policy_to_ctbr(actor(obs["policy"])) if actor is not None else commander(env, z_int)
            obs, _, terminated, truncated, _ = env.step(sp)

        d = env.drone.data
        pos = (d.root_pos_w[0] - env.scene.env_origins[0]).cpu()
        height = env._height()[0].item()
        target = env.target_z[0].item()
        roll, pitch, yaw = (wrap_to_pi(a) for a in euler_xyz_from_quat(d.root_quat_w))
        att = [math.degrees(a[0].item()) for a in (roll, pitch, yaw)]
        gyro = d.root_ang_vel_b[0].cpu() * FRD
        m = env.motor_cmd[0].cpu()
        tofs = o[0:4].tolist()
        tof_seen, tof_age, imu_age = tofs[0], o[14].item(), o[18].item()
        rows.writerow([f"{t + dt:.3f}", f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{height:.4f}", f"{target:.3f}",
                       *[f"{a:.2f}" for a in att], *[f"{v:.4f}" for v in sp[0].tolist()],
                       *[f"{v:.4f}" for v in gyro.tolist()], *[f"{v:.4f}" for v in m.tolist()],
                       *[f"{v:.3f}" for v in tofs], f"{tof_age:.3f}", f"{imu_age:.3f}"])

        show_targets(markers, env.scene.env_origins, max(target, 0.01) + env.rest_z)
        hud.target_z = target
        hud.add(t + dt, torch.tensor([pos[0], pos[1], height]))
        frame = env.render()
        if frame.max() > 0:
            writer.append_data(hud.draw(frame, [
                f"command  height {target:.2f} m ({'policy' if actor else 'scripted'})",
                f"ToF (m)  dn {tofs[0]:.2f} fr {tofs[1]:.2f} l {tofs[2]:.2f} r {tofs[3]:.2f}",
                f"ages     down ToF {tof_age * 1000:3.0f} ms   IMU {imu_age * 1000:3.0f} ms",
                f"to PX4   thr {sp[0, 0]:.2f}  p {sp[0, 1]:+.2f} q {sp[0, 2]:+.2f} r {sp[0, 3]:+.2f}",
                f"PX4 out  {m[0]:.2f} {m[1]:.2f} {m[2]:.2f} {m[3]:.2f}",
                f"att      roll {att[0]:+5.1f} pitch {att[1]:+5.1f} yaw {att[2]:+5.1f}",
            ]))

        if step % 25 == 0:
            print(f"t={t:5.2f}s height={height:.3f} target={target:.2f} xy=({pos[0]:+.2f},{pos[1]:+.2f}) "
                  f"att=({att[0]:+.1f},{att[1]:+.1f}) thr={sp[0, 0]:.2f} tof={tof_seen:.2f} "
                  f"motors={[round(v, 2) for v in m.tolist()]}")
        if torch.any(terminated | truncated):
            print(f"t={t:5.2f}s RESET (crash/tilt/wall={terminated.sum().item()})")
            z_int.zero_()

    writer.close()
    log.close()
    print(f"wrote {args.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
