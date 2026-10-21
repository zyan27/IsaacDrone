"""Fly the DD24 through the PX4 inner loop and record an MP4 + CSV.

--checkpoint: the trained policy (train_rl_px4.py) is the commander. It sees only
the DD24's sensors (5 ToF, EKF vz/attitude, IMU) and sends PX4 a CTBR setpoint.

Without --checkpoint, the scripted commander below is used:

The commander here plays the role the RL policy (or QGC/mavros) will: every 20 ms
it sends PX4 a SET_ATTITUDE_TARGET-style setpoint (thrust + FRD body rates).
It holds 1 m and the spawn x/y with a PD altitude loop, a P position loop, and
PX4's own attitude P gains (MC_ROLL_P / MC_PITCH_P / MC_YAW_P), then runs two
scripted manoeuvres so the rate tracking is visible:
  t = 6-8 s    yaw to +60 deg
  t = 10-10.25 s raw roll-rate pulse of 2 rad/s, then back to level hold

Run with --headless --enable_cameras (see oscar/run_isaac.sh).
"""
import argparse
import csv
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=15.0)
parser.add_argument("--out", type=str, default="videos/px4_dd24.mp4")
parser.add_argument("--checkpoint", type=str, default=None, help="train_rl_px4.py checkpoint")
parser.add_argument("--maneuvers", action="store_true", help="scripted mode: add yaw turn + roll kick")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import imageio
import torch

from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from controllers.px4_policy import PX4Actor
from environments.drone_px4_env import DronePX4Env, DronePX4EnvCfg, policy_to_ctbr
from sim_px4.px4_defaults import ATT
from visualization import Hud, make_target_markers, show_targets

FRD = torch.tensor([1.0, -1.0, -1.0])  # FLU <-> FRD for body rates / angles


def commander(env: DronePX4Env, t: float, z_int: torch.Tensor):
    """Outer loop -> CTBR setpoint for PX4 (thrust [0,1], FRD rates rad/s)."""
    d = env.drone.data
    pos = d.root_pos_w - env.scene.env_origins
    vel = d.root_lin_vel_w
    roll, pitch, yaw = euler_xyz_from_quat(d.root_quat_w)
    roll, pitch, yaw = wrap_to_pi(roll), wrap_to_pi(pitch), wrap_to_pi(yaw)

    # Altitude PD + I around the hover command.
    z_err = env.cfg.hover_height - pos[:, 2]
    z_int += torch.where(z_err.abs() < 0.15, z_err, torch.zeros_like(z_err)) * 0.02  # no windup on climb
    thrust = env.hover_cmd + 0.12 * z_err - 0.16 * vel[:, 2] + 0.08 * z_int

    # Position hold: world accel request -> tilt, rotated into the heading frame.
    ax = 1.5 * -pos[:, 0] - 2.0 * vel[:, 0]
    ay = 1.5 * -pos[:, 1] - 2.0 * vel[:, 1]
    c, s = torch.cos(yaw), torch.sin(yaw)
    fwd, left = c * ax + s * ay, -s * ax + c * ay
    pitch_d = torch.clamp(fwd / 9.81, -0.35, 0.35)   # FLU: +pitch tilts thrust forward
    roll_d = torch.clamp(-left / 9.81, -0.35, 0.35)  # FLU: -roll tilts thrust left
    yaw_d = torch.full_like(yaw, math.radians(60.0) if args.maneuvers and 6.0 <= t < 8.0 else 0.0)

    # PX4 attitude P (small-angle form) -> body rates.
    rates_flu = torch.stack([
        ATT.P[0] * wrap_to_pi(roll_d - roll),
        ATT.P[1] * wrap_to_pi(pitch_d - pitch),
        ATT.P[2] * wrap_to_pi(yaw_d - yaw),
    ], dim=-1)
    rates_frd = rates_flu * FRD.to(rates_flu.device)
    if args.maneuvers and 10.0 <= t < 10.25:
        rates_frd[:, 0] = 2.0  # raw roll-rate pulse straight to PX4
    return torch.cat([thrust.clamp(0.0, 1.0).unsqueeze(-1), rates_frd], dim=-1), (roll, pitch, yaw)


def main():
    cfg = DronePX4EnvCfg()
    cfg.scene.num_envs = 1
    # Camera in a corner of the room, looking across it at the target.
    cfg.viewer.eye = (1.3, 1.3, 1.6)
    cfg.viewer.lookat = (0.0, 0.0, 0.7)
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
    rows.writerow(["t", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg",
                   "sp_thrust", "sp_p", "sp_q", "sp_r", "gyro_p", "gyro_q", "gyro_r",
                   "m1", "m2", "m3", "m4", "tof_down", "tof_front", "tof_left", "tof_right", "tof_up"])

    obs, _ = env.reset()
    origin = env.scene.env_origins[0].tolist()
    env.sim.set_camera_view(
        eye=[o + e for o, e in zip(origin, cfg.viewer.eye)],
        target=[o + v for o, v in zip(origin, cfg.viewer.lookat)],
    )
    markers = make_target_markers()
    show_targets(markers, env.scene.env_origins, cfg.hover_height)
    title = "DD24 RL policy -> PX4 (sensors: 5 ToF + IMU)" if actor else "DD24 scripted commander -> PX4"
    hud = Hud(title, cfg.hover_height, args.seconds, room_half=cfg.room_half_width_m)
    z_int = torch.zeros(env.num_envs, device=env.device)

    for step in range(steps):
        t = step * dt
        with torch.inference_mode():
            tof = obs["policy"][0, 0:5].cpu()
            if actor is not None:
                sp = policy_to_ctbr(actor(obs["policy"]), env.hover_cmd)
                roll, pitch, yaw = (wrap_to_pi(a) for a in euler_xyz_from_quat(env.drone.data.root_quat_w))
            else:
                sp, (roll, pitch, yaw) = commander(env, t, z_int)
            obs, _, terminated, truncated, _ = env.step(sp)

        d = env.drone.data
        pos = (d.root_pos_w[0] - env.scene.env_origins[0]).cpu()
        gyro = (d.root_ang_vel_b[0].cpu() * FRD)
        m = env.motor_cmd[0].cpu()
        att = [math.degrees(a[0].item()) for a in (roll, pitch, yaw)]
        rows.writerow([f"{t + dt:.3f}", *[f"{v:.4f}" for v in pos.tolist()], *[f"{a:.2f}" for a in att],
                       *[f"{v:.4f}" for v in sp[0].tolist()], *[f"{v:.4f}" for v in gyro.tolist()],
                       *[f"{v:.4f}" for v in m.tolist()], *[f"{v:.3f}" for v in tof.tolist()]])

        hud.add(t + dt, pos)
        if actor is not None:
            phase = "RL policy"
        elif args.maneuvers and 6.0 <= t < 8.0:
            phase = "yaw to +60 deg"
        elif args.maneuvers and 10.0 <= t < 10.25:
            phase = "roll-rate pulse 2 rad/s"
        else:
            phase = "scripted hold"
        frame = env.render()
        if frame.max() > 0:
            writer.append_data(hud.draw(frame, [
                f"command  {phase}",
                f"ToF (m)  dn {tof[0]:.2f} fr {tof[1]:.2f} l {tof[2]:.2f} r {tof[3]:.2f} up {tof[4]:.2f}",
                f"to PX4   thr {sp[0, 0]:.2f}  p {sp[0, 1]:+.2f} q {sp[0, 2]:+.2f} r {sp[0, 3]:+.2f}",
                f"gyro     p {gyro[0]:+.2f} q {gyro[1]:+.2f} r {gyro[2]:+.2f} rad/s",
                f"PX4 out  {m[0]:.2f} {m[1]:.2f} {m[2]:.2f} {m[3]:.2f}",
                f"att      roll {att[0]:+5.1f} pitch {att[1]:+5.1f} yaw {att[2]:+5.1f}",
            ]))

        if step % 25 == 0:
            print(f"t={t:5.2f}s z={pos[2]:.3f} xy=({pos[0]:+.2f},{pos[1]:+.2f}) "
                  f"att=({att[0]:+.1f},{att[1]:+.1f},{att[2]:+.1f}) tof={[round(v, 2) for v in tof.tolist()]} "
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
