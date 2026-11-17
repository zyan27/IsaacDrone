"""Record an MP4 of a trained hover policy flying in Isaac Sim (headless).

    ./isaaclab.sh -p record_video.py --headless --enable_cameras \
        --checkpoint runs/dd24_hover_1115_s0/best.pt --view arena --seconds 20

Views:
  arena  fixed camera looking into the net enclosure of env 0 (shows drift + net)
  chase  camera that follows drone 0 (shows attitude / wobble up close)
  top    straight down onto the arena (shows horizontal drift most clearly)

The net is drawn as a grid of thin strands and the hover target as a red dot;
both are visual only. Episodes that end (crash / net) restart, as in training.
On Oscar use oscar/record_video.sbatch, which runs all of this for you.
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--airframe", choices=["crazyflie", "dd24"], default=None)
parser.add_argument("--view", choices=["arena", "chase", "top"], default="arena")
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--num_envs", type=int, default=1, help="Drones simulated (only env 0 is filmed)")
parser.add_argument("--scenario", default="fixed_net",
                    choices=["fixed_net", "random", "net_only", "net_in_room", "room", "open"],
                    help="The drawn net matches only fixed_net; other scenes aren't drawn (yet)")
parser.add_argument("--net_dropout", type=float, default=None,
                    help="Fix P(ToF misses the net), e.g. 0.7 for the worst case")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", default=None, help="MP4 path (default: <run>/videos/<view>.mp4)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True          # rendering frames headless needs cameras on

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402

from environments.drone_hover_env import DroneHoverEnv, DroneHoverEnvCfg, DroneHoverEnvCfgDD24  # noqa: E402
from controllers.rl_hover_controller import RLHoverActor  # noqa: E402

VIEWS = {
    "arena": dict(origin_type="env", eye=(3.4, -3.4, 2.6), lookat=(0.0, 0.0, 0.9)),
    "chase": dict(origin_type="asset_root", asset_name="drone", eye=(-0.9, -0.9, 0.45), lookat=(0.0, 0.0, 0.0)),
    "top":   dict(origin_type="env", eye=(0.0, -0.01, 5.5), lookat=(0.0, 0.0, 0.0)),
}


def main():
    import imageio.v2 as imageio

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    airframe = args.airframe or ck.get("airframe", "crazyflie")
    cfg = DroneHoverEnvCfgDD24() if airframe == "dd24" else DroneHoverEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.obs_mode = ck.get("obs_mode", "privileged")
    # Checkpoints from before thrust/body-rate control commanded motors directly.
    cfg.control_mode = ck.get("control_mode", "motors")
    cfg.obs_history = ck.get("obs_history", cfg.obs_history)
    if args.net_dropout is not None:
        cfg.tof_net_dropout_range = (args.net_dropout, args.net_dropout)
    cfg.scenario = args.scenario
    cfg.show_net = args.scenario == "fixed_net"
    # Space the arenas out so neighbouring nets don't overlap in the shot.
    cfg.scene.env_spacing = max(cfg.arena_size_xy) + 1.5
    cfg.viewer = ViewerCfg(env_index=0, resolution=(args.width, args.height), **VIEWS[args.view])
    cfg.__post_init__()
    if ck.get("obs_dim") != cfg.observation_space:
        raise SystemExit(f"Checkpoint expects {ck.get('obs_dim')} inputs, environment gives "
                         f"{cfg.observation_space}: it predates the current sensor set.")

    env = DroneHoverEnv(cfg=cfg, render_mode="rgb_array")
    dev = env.device
    actor = RLHoverActor(obs_dim=ck.get("obs_dim", 14), action_dim=4, out=ck.get("actor_out", "sigmoid")).to(dev)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    from controllers.obs_norm import load_obs_norm
    norm = load_obs_norm(ck, actor.net[0].in_features, dev)
    hover = ck.get("hover_motor", env.hover_motor)
    rscale = ck.get("residual_scale", 0.15)
    n_in = actor.net[0].in_features

    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    tag = f"_net{int(args.net_dropout * 100)}" if args.net_dropout is not None else ""
    out = args.out or os.path.join(run_dir, "videos", f"{args.view}{tag}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fps = round(1.0 / env.step_dt)
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=1)

    steps = int(args.seconds / env.step_dt)
    warmup = 15   # the renderer returns blank frames for the first few steps
    resets = 0
    obs, _ = env.reset()
    with torch.inference_mode():
        for t in range(steps + warmup):
            raw = actor(norm(obs["policy"][:, :n_in]))
            if cfg.control_mode == "ctbr":
                actions = raw.clamp(0.0, 1.0)
            else:
                actions = torch.clamp(hover + (raw - 0.5) * rscale, 0.0, 1.0)
            obs, _, term, trunc, _ = env.step(actions)
            if bool(term[0]):
                resets += 1
                why = "net" if bool(env.term_net[0]) else ("flip" if bool(env.term_flipped[0]) else "crash")
                print(f"  t={t * env.step_dt:5.2f}s drone 0 episode ended ({why}) - restarting")
            frame = env.render()
            if t >= warmup and frame is not None:
                writer.append_data(frame)
    writer.close()
    print(f"wrote {out}  ({args.seconds:.0f} s at {fps} fps, view={args.view}, "
          f"episode ends during clip: {resets})")
    env.close()


if __name__ == "__main__":
    import os as _os, sys as _sys, traceback as _tb   # see train_rl_hover.py: Kit shutdown can hang
    _code = 0
    try:
        main()
    except Exception:
        _tb.print_exc()
        _code = 1
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(_code)
