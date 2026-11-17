"""Headless evaluation of a trained hover policy in Isaac Lab ("the test").

Runs every env for exactly ONE full episode with the deterministic policy (mean
action, no exploration noise), from the same randomised starts used in training,
and reports how often it survives and how precisely it holds the hover point.

    ./isaaclab.sh -p eval_rl_hover.py --headless --airframe dd24 \
        --checkpoint runs/dd24_s0/best.pt --num_envs 512 --out runs/dd24_s0/eval.json

Metrics (written to JSON and printed):
  success_rate      fraction of episodes that last the full 20 s
  crash/net/flip    fraction ending by hitting the ground / the net geofence / tipping over
  z_err_mean_m      mean |height error| over the last half of each surviving episode
  xy_err_mean_m     mean horizontal distance from the hover point (same window)
  pos_err_p95_m     95th percentile 3D position error (same window)
  min_net_dist_m    closest any drone came to the net (over everything it flew)
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--airframe", choices=["crazyflie", "dd24"], default=None,
                    help="Defaults to the airframe stored in the checkpoint.")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=12345, help="Different from training seeds.")
parser.add_argument("--scenario", default="random",
                    choices=["random", "net_only", "net_in_room", "room", "open", "fixed_net"],
                    help="random = training mix; fixed_net = the original arena")
parser.add_argument("--lag", action="store_true",
                    help="Worst-case laggy links: max latencies, 15%% ToF/flow drops, 10%% lost "
                         "setpoints, frequent stalls (top of every training range)")
parser.add_argument("--net_dropout", type=float, default=None,
                    help="Fix P(ToF misses the net) for the test, e.g. 0 or 0.7 (default: training range)")
parser.add_argument("--out", type=str, default=None, help="JSON path (default: next to checkpoint)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from environments.drone_hover_env import DroneHoverEnv, DroneHoverEnvCfg, DroneHoverEnvCfgDD24
from controllers.rl_hover_controller import RLHoverActor


def main():
    ck = torch.load(args.checkpoint, map_location="cpu")
    airframe = args.airframe or ck.get("airframe", "crazyflie")

    cfg = DroneHoverEnvCfgDD24() if airframe == "dd24" else DroneHoverEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    # Older checkpoints (before onboard mode) were trained on ground-truth inputs.
    cfg.obs_mode = ck.get("obs_mode", "privileged")
    # Checkpoints from before thrust/body-rate control commanded motors directly.
    cfg.control_mode = ck.get("control_mode", "motors")
    cfg.obs_history = ck.get("obs_history", cfg.obs_history)
    if args.net_dropout is not None:
        cfg.tof_net_dropout_range = (args.net_dropout, args.net_dropout)
    cfg.scenario = args.scenario
    if args.lag:
        top = lambda r: (r[1], r[1])  # noqa: E731
        for name in ["imu_latency_s_range", "tof_latency_s_range", "flow_latency_s_range", "cmd_latency_s_range",
                     "sensor_drop_prob_range", "imu_drop_prob_range", "cmd_drop_prob_range",
                     "sensor_stall_rate_range", "imu_stall_rate_range", "cmd_stall_rate_range"]:
            setattr(cfg, name, top(getattr(cfg, name)))
        cfg.tof_rate_hz_range = (cfg.tof_rate_hz_range[0],) * 2
        cfg.flow_rate_hz_range = (cfg.flow_rate_hz_range[0],) * 2
    cfg.__post_init__()
    if ck.get("obs_dim") != cfg.observation_space:
        raise SystemExit(
            f"Checkpoint expects {ck.get('obs_dim')} inputs but this environment gives "
            f"{cfg.observation_space}. It was trained before the current sensor set "
            "(optical flow / scenes); evaluate it with the code it was trained with, or retrain.")
    env = DroneHoverEnv(cfg=cfg, render_mode=None)
    dev = env.device
    N = env.num_envs

    actor = RLHoverActor(obs_dim=ck.get("obs_dim", 14), action_dim=4, out=ck.get("actor_out", "sigmoid")).to(dev)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    from controllers.obs_norm import load_obs_norm
    norm = load_obs_norm(ck, actor.net[0].in_features, dev)
    hover = ck.get("hover_motor", env.hover_motor)
    rscale = ck.get("residual_scale", 0.15)

    T = int(env.max_episode_length)
    window_start = T // 2   # "steady state" = second half of the episode

    finished = torch.zeros(N, dtype=torch.bool, device=dev)
    ended_at = torch.full((N,), T, device=dev)
    reason = torch.zeros(N, dtype=torch.long, device=dev)  # 0 survived 1 crash 2 net 3 flip
    sum_z = torch.zeros(N, device=dev)
    sum_xy = torch.zeros(N, device=dev)
    n_win = torch.zeros(N, device=dev)
    win_errs = []
    min_net = torch.full((N,), float("inf"), device=dev)

    obs, _ = env.reset()
    from environments.scenes import SCENARIOS
    scen = env._scene["scen"].clone()      # scenario of each env's first episode
    with torch.inference_mode():
        for t in range(T):
            raw = actor(norm(obs["policy"][:, : actor.net[0].in_features]))
            if cfg.control_mode == "ctbr":
                actions = raw.clamp(0.0, 1.0)
            else:
                actions = torch.clamp(hover + (raw - 0.5) * rscale, 0.0, 1.0)
            obs, _, terminated, truncated, _ = env.step(actions)

            live = ~finished
            newly_term = terminated & live
            reason[newly_term & (env.term_crashed)] = 1
            reason[newly_term & (env.term_net)] = 2
            reason[newly_term & (env.term_flipped) & ~env.term_crashed & ~env.term_net] = 3
            ended_at[newly_term] = t
            finished |= terminated | truncated

            # State after the step for envs still in their first episode (finished
            # envs were auto-reset, so their state belongs to a new episode).
            still = ~finished
            pos = env._local_pos()
            err = pos - env._target
            min_net[still] = torch.minimum(min_net[still], env._dist_to_net(pos)[still])
            if t >= window_start:
                sum_z[still] += err[still, 2].abs()
                sum_xy[still] += torch.linalg.norm(err[still, :2], dim=-1)
                n_win[still] += 1
                win_errs.append(torch.linalg.norm(err[still], dim=-1))
            if bool(finished.all()):
                break

    survived = reason == 0
    has_win = n_win > 0
    all_err = torch.cat(win_errs) if win_errs else torch.zeros(1, device=dev)
    res = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "airframe": airframe,
        "obs_mode": cfg.obs_mode,
        "net_dropout_range": list(cfg.tof_net_dropout_range),
        "episodes": N,
        "episode_s": cfg.episode_length_s,
        "success_rate": survived.float().mean().item(),
        "crash_rate": (reason == 1).float().mean().item(),
        "net_rate": (reason == 2).float().mean().item(),
        "flip_rate": (reason == 3).float().mean().item(),
        "mean_survival_s": (ended_at.float().mean() * cfg.sim.dt * cfg.decimation).item(),
        "z_err_mean_m": (sum_z[has_win] / n_win[has_win]).mean().item() if has_win.any() else None,
        "xy_err_mean_m": (sum_xy[has_win] / n_win[has_win]).mean().item() if has_win.any() else None,
        "pos_err_p95_m": torch.quantile(all_err.float(), 0.95).item(),
        "min_net_dist_m": min_net[torch.isfinite(min_net)].min().item() if torch.isfinite(min_net).any() else None,
        "train_best_reward": ck.get("best_reward"),
        "train_iter": ck.get("iter"),
    }
    # Simple pass/fail so overnight runs can be skimmed quickly.
    by = {}
    for i, name in enumerate(SCENARIOS):
        m = scen == i
        if not bool(m.any()):
            continue
        hw = m & has_win
        by[name] = {
            "episodes": int(m.sum()),
            "success_rate": survived[m].float().mean().item(),
            "net_rate": (reason[m] == 2).float().mean().item(),
            "crash_rate": ((reason[m] == 1) | (reason[m] == 3)).float().mean().item(),
            "z_err_mean_m": (sum_z[hw] / n_win[hw]).mean().item() if hw.any() else None,
            "xy_err_mean_m": (sum_xy[hw] / n_win[hw]).mean().item() if hw.any() else None,
        }
    res["scenario"] = args.scenario
    res["lag_worst_case"] = bool(args.lag)
    res["by_scenario"] = by
    res["pass"] = bool(res["success_rate"] >= 0.95 and (res["xy_err_mean_m"] or 9) < 0.15
                       and (res["z_err_mean_m"] or 9) < 0.10)

    out = args.out or os.path.join(os.path.dirname(args.checkpoint), "eval.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("EVAL", json.dumps(res, indent=2))
    print(f"wrote {out}")
    env.close()


if __name__ == "__main__":
    # Isaac Sim's simulation_app.close() can hang forever in headless containers
    # (it did on Oscar: 6+ hours after training had finished). Everything is already
    # written to disk by now, so exit the process directly instead.
    import os as _os, sys as _sys, traceback as _tb
    _code = 0
    try:
        main()
    except Exception:
        _tb.print_exc()
        _code = 1
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(_code)
