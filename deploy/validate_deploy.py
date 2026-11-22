"""End-to-end check of the Pi deployment path against Isaac (run inside Isaac Lab).

The first --core_envs drones are flown by deploy/dd24_policy_core.Core (the exact numpy
code the Pi runs, loading the exported .npz), fed each tick with "messages" rebuilt
from the sim's newest sensor frame: IMU (accel, gyro, orientation quaternion), five
ToF ranges in metres, an OpticalFlowRad-style flow message, the target height. The
other drones fly the torch actor from the checkpoint, as in eval_rl_hover.py.

Checks, over each drone's first episode:
  inputs   the 176 numbers Core builds vs the sim's observation (layout, ToF scaling,
           gravity <-> quaternion, flow conversion, previous command, history order)
  outputs  Core's command vs the torch actor's on the same observation (export, norm)
  flight   survival / height error of Core-flown vs torch-flown drones

    ./isaaclab.sh -p deploy/validate_deploy.py --headless --checkpoint runs/X/best.pt \
        --npz deploy/policy.npz --num_envs 64 --core_envs 16 --out runs/X/validate.json
Exit code 1 if a parity check fails.
"""
import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--npz", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--core_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--out", default=None)
parser.add_argument("--takeoff", action="store_true",
                    help="all drones start on the floor; Core does its scripted takeoff, then the policy "
                         "(flight check only: during takeoff the sim's 'previous command' differs by design)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers.obs_norm import load_obs_norm  # noqa: E402
from controllers.rl_hover_controller import RLHoverActor  # noqa: E402
from deploy.dd24_policy_core import Core, Policy, SafetyCfg, TakeoffCfg  # noqa: E402
from environments.drone_hover_env import DroneHoverEnv, DroneHoverEnvCfgDD24  # noqa: E402

FLOW_DT = 0.066   # integration window of the synthetic flow message (any value works)


def quat_from_gravity(g):
    """A body->world quaternion (x, y, z, w), yaw 0, whose gravity direction in body is g."""
    up = -np.asarray(g, float) / np.linalg.norm(g)
    ez = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, ez)                 # rotate body 'up' onto world z
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return (0.0, 0.0, 0.0, 1.0) if up[2] > 0 else (1.0, 0.0, 0.0, 0.0)
    ang = math.atan2(s, float(up @ ez))
    a = axis / s * math.sin(ang / 2)
    return (a[0], a[1], a[2], math.cos(ang / 2))


def feed(core, t, f, tof_max):
    """Sim frame (22) -> the messages the node would have received at time t."""
    core.on_imu(t, f[0:3], f[3:6], quat_from_gravity(f[6:9]))
    for i, name in enumerate(core.tof_names):
        core.on_tof(t, name, float(f[9 + i]) * tof_max)
    h = float(f[9]) * tof_max
    vx, vy, valid = float(f[18]), float(f[19]), float(f[20])
    msg = {"integration_time_us": int(FLOW_DT * 1e6), "integrated_xgyro": 0.0, "integrated_ygyro": 0.0,
           "integrated_x": vy / h * FLOW_DT if h > 0 else 0.0, "integrated_y": vx / h * FLOW_DT if h > 0 else 0.0,
           "quality": 255 if valid > 0.5 else 0, "distance": -1.0}
    core.on_flow(t, msg)
    if core.move_cmd:
        core.move_odometry = False          # use the sim's offset as is
        core.move = np.array([f[22], f[23]], float)


def main():
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = DroneHoverEnvCfgDD24()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.obs_mode = ck["obs_mode"]
    cfg.control_mode = ck["control_mode"]
    cfg.ctbr_thrust_center = ck.get("ctbr_thrust_center", cfg.ctbr_thrust_center)
    cfg.ctbr_thrust_span = ck.get("ctbr_thrust_span", cfg.ctbr_thrust_span)
    cfg.thrust_curve_k_range = tuple(ck.get("thrust_curve_k_range", cfg.thrust_curve_k_range))
    cfg.obs_history = ck.get("obs_history", cfg.obs_history)
    if ck.get("real_noise"):
        cfg.gyro_noise_std, cfg.accel_noise_std = (0.06, 0.10, 0.02), (1.0, 1.0, 2.7)
    if ck.get("move_cmd"):
        cfg.move_cmd = True
        cfg.move_step_range = tuple(ck["move_step_range"])
    cfg.ground_start_prob = 1.0 if args.takeoff else 0.0
    cfg.__post_init__()
    env = DroneHoverEnv(cfg=cfg, render_mode=None)
    dev, N, K = env.device, env.num_envs, min(args.core_envs, env.num_envs)

    actor = RLHoverActor(obs_dim=ck["obs_dim"], action_dim=4, out=ck.get("actor_out", "sigmoid")).to(dev)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    norm = load_obs_norm(ck, ck["obs_dim"], dev)
    pol = Policy(args.npz)
    m = pol.meta
    tof_max, H = m["tof_max_range"], m["obs_history"]
    F = 24 if m.get("move_cmd") else 22
    for key in ("ctbr_thrust_center", "ctbr_thrust_span"):
        assert abs(m[key] - ck[key]) < 1e-9, f"npz {key} {m[key]} != checkpoint {ck[key]}"

    cores = [None] * K
    takeoff_ticks = [0] * K
    handed = [False] * K
    T = int(env.max_episode_length)
    finished = torch.zeros(N, dtype=torch.bool, device=dev)
    ended_at = torch.full((N,), T, device=dev)
    zerr_sum = torch.zeros(N, device=dev)
    zerr_n = torch.zeros(N, device=dev)
    in_err = np.zeros(F)                    # max |core - sim| per frame field (newest frame)
    hist_err = 0.0                          # max over the whole 176 vector
    out_err = 0.0                           # max |core a - torch a| on the same obs
    mode_bad = 0
    t_node = 0.0
    obs, _ = env.reset()
    with torch.inference_mode():
        for step in range(T):
            o = obs["policy"]
            a_torch = actor(norm(o)).clamp(0.0, 1.0)
            actions = a_torch.clone()
            on = o.cpu().numpy().astype(np.float64)
            at = a_torch.cpu().numpy()
            for i in range(K):
                if bool(finished[i]):
                    continue
                f = on[i, (H - 1) * F:H * F]
                if cores[i] is None:
                    cores[i] = Core(pol, target_height=float(f[21]), safety=SafetyCfg(max_tilt_deg=180.0, max_height_m=99.0),
                                    takeoff=TakeoffCfg(enabled=args.takeoff))
                c = cores[i]
                feed(c, t_node, f, tof_max)
                tk = c.tick(t_node, active=True)
                if tk.mode == "TAKEOFF":
                    span, ctr = m["ctbr_thrust_span"], m["ctbr_thrust_center"]
                    a = np.clip(np.r_[(tk.thrust - ctr) / span, tk.rates / np.array(m["ctbr_max_rates"])], -1, 1)
                    actions[i] = torch.as_tensor((a + 1) / 2, dtype=actions.dtype, device=dev)
                    takeoff_ticks[i] += 1
                    continue
                if tk.mode != "RUN":
                    mode_bad += 1
                    print(f"env {i} step {step}: {tk.mode} {tk.reason}")
                    continue
                handed[i] = True
                core_obs = np.concatenate(list(c.hist))
                if args.takeoff:      # inputs differ by design after a takeoff; fly only
                    actions[i] = torch.as_tensor((tk.cmd + 1.0) / 2.0, dtype=actions.dtype, device=dev)
                    continue
                d = np.abs(core_obs - on[i])
                hist_err = max(hist_err, float(d.max()))
                in_err = np.maximum(in_err, d[(H - 1) * F:H * F])
                a_core = (tk.cmd + 1.0) / 2.0
                out_err = max(out_err, float(np.abs(pol(on[i]) - at[i]).max()))   # export/norm, same input
                actions[i] = torch.as_tensor(a_core, dtype=actions.dtype, device=dev)  # Core flies this drone
            obs, _, term, trunc, _ = env.step(actions)
            t_node += 1.0 / m["policy_hz"]
            live = ~finished
            ended_at[term & live] = step
            finished |= term | trunc
            still = ~finished
            if step >= T // 2:
                err = (env._local_pos() - env._target)[:, 2].abs()
                zerr_sum[still] += err[still]
                zerr_n[still] += 1
            if bool(finished.all()):
                break

    surv = (ended_at >= T - 1).float()
    zmean = (zerr_sum / zerr_n.clamp(min=1)).cpu()
    core_ok, torch_ok = surv[:K].cpu(), surv[K:].cpu()
    from deploy.dd24_policy_core import FRAME_NAMES, MOVE_NAMES
    worst = sorted(zip(FRAME_NAMES + (MOVE_NAMES if F == 24 else []), in_err.tolist()), key=lambda x: -x[1])[:6]
    res = {
        "checkpoint": args.checkpoint, "npz": args.npz, "iter": ck.get("iter"),
        "ctbr_thrust_center": m["ctbr_thrust_center"],
        "input_max_abs_err": hist_err, "input_worst_fields": worst,
        "output_max_abs_err_npz_vs_torch": out_err, "non_run_ticks": mode_bad,
        "core_success_rate": core_ok.mean().item(), "torch_success_rate": torch_ok.mean().item(),
        "core_z_err_mean_m": float(zmean[:K][core_ok > 0].mean()) if core_ok.sum() else None,
        "torch_z_err_mean_m": float(zmean[K:][torch_ok > 0].mean()) if torch_ok.sum() else None,
    }
    # Flow is rebuilt from a message (division by height, validity re-decided), so allow float slop.
    res["takeoff"] = bool(args.takeoff)
    if args.takeoff:
        res["handed_over"] = sum(handed)
        res["takeoff_s_mean"] = float(np.mean([n for n in takeoff_ticks if n])) / m["policy_hz"] if any(takeoff_ticks) else None
        res["pass"] = bool(core_ok.mean().item() >= 0.9 and sum(handed) == K and mode_bad == 0)
    else:
        res["pass"] = bool(hist_err < 1e-4 and out_err < 1e-4 and mode_bad == 0)
    print(json.dumps(res, indent=2))
    out = args.out or os.path.join(os.path.dirname(args.checkpoint), "validate.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    env.close()
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)   # Kit shutdown hangs on Oscar (see oscar/README.md)
