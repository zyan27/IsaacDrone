"""Flight recorder (Isaac Lab, headless): fly a policy on N drones for one episode
each, log every step, save an .npz for analyze_flights.py.

    ./isaaclab.sh -p flight_recorder.py --headless --checkpoint runs/x_s0/best.pt \
        --num_envs 512 --out runs/x_s0/flights/flights.npz [--scenario fixed_net] [--lag]

Same setup as eval_rl_hover.py (deterministic policy, test seed), so failure rates
match the tests.
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--scenario", default="random",
                    choices=["random", "net_only", "net_in_room", "room", "open", "fixed_net"])
parser.add_argument("--lag", action="store_true", help="Worst-case laggy links (as eval --lag)")
parser.add_argument("--net_dropout", type=float, default=None)
parser.add_argument("--out", default=None, help="Default: <checkpoint dir>/flights/flights.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from environments.drone_hover_env import DroneHoverEnv, DroneHoverEnvCfgDD24  # noqa: E402
from environments.scenes import SCENARIOS  # noqa: E402
from controllers.rl_hover_controller import RLHoverActor  # noqa: E402
from controllers.obs_norm import load_obs_norm  # noqa: E402
from tools.flight_log import record  # noqa: E402

LAG_FIELDS = ["imu_latency_s_range", "tof_latency_s_range", "flow_latency_s_range", "cmd_latency_s_range",
              "sensor_drop_prob_range", "imu_drop_prob_range", "cmd_drop_prob_range",
              "sensor_stall_rate_range", "imu_stall_rate_range", "cmd_stall_rate_range"]


def main():
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("control_mode") != "ctbr" or ck.get("obs_mode") != "onboard":
        raise SystemExit("flight recorder needs an onboard-sensor, thrust+rates (ctbr) checkpoint")
    cfg = DroneHoverEnvCfgDD24()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.obs_mode, cfg.control_mode = "onboard", "ctbr"
    cfg.obs_history = ck.get("obs_history", cfg.obs_history)
    cfg.scenario = args.scenario
    if args.net_dropout is not None:
        cfg.tof_net_dropout_range = (args.net_dropout, args.net_dropout)
    if args.lag:
        for name in LAG_FIELDS:
            r = getattr(cfg, name)
            setattr(cfg, name, (r[1], r[1]))
        cfg.tof_rate_hz_range = (cfg.tof_rate_hz_range[0],) * 2
        cfg.flow_rate_hz_range = (cfg.flow_rate_hz_range[0],) * 2
    cfg.__post_init__()
    if ck.get("obs_dim") != cfg.observation_space:
        raise SystemExit(f"checkpoint expects {ck.get('obs_dim')} inputs, env gives {cfg.observation_space}")

    env = DroneHoverEnv(cfg=cfg, render_mode=None)
    dev = env.device
    actor = RLHoverActor(obs_dim=ck["obs_dim"], action_dim=4, out=ck.get("actor_out", "sigmoid")).to(dev)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    norm = load_obs_norm(ck, ck["obs_dim"], dev)
    policy = lambda obs: actor(norm(obs["policy"])).clamp(0.0, 1.0)  # noqa: E731

    log = record(env, policy)
    log["scenario_names"] = np.array(SCENARIOS)
    log["tof_names"] = np.array([t[0] for t in cfg.tof_sensors])
    log["meta"] = np.array([os.path.abspath(args.checkpoint), args.scenario, str(args.lag), str(args.net_dropout)])
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "flights", "flights.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **log)
    r = log["reason"]
    print(f"wrote {out}: {len(r)} episodes, survived {np.mean(r == 0):.3f}, crash {np.mean(r == 1):.3f}, "
          f"net {np.mean(r == 2):.3f}, flip {np.mean(r == 3):.3f}")
    env.close()


if __name__ == "__main__":
    import os as _os, sys as _sys, traceback as _tb   # Kit shutdown can hang (see train_rl_hover.py)
    _code = 0
    try:
        main()
    except Exception:
        _tb.print_exc()
        _code = 1
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(_code)
