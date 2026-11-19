import argparse
import csv
import os
import random
import shutil
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iters", type=int, default=2000)
parser.add_argument("--steps", type=int, default=128)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--num_minibatches", type=int, default=4,
                    help="Minibatches per epoch (rsl_rl / Isaac Lab default 4). With 1024 envs x 128 "
                         "steps that's 32768 samples each and 20 updates per iteration, the regime the "
                         "adaptive-KL schedule is designed for.")
parser.add_argument("--minibatch_size", type=int, default=0,
                    help="Fixed minibatch size instead of --num_minibatches (0 = use num_minibatches). "
                         "The old default 1024 meant ~512 updates/iteration and collapsed the adaptive lr.")
parser.add_argument("--save_path", type=str, default=None,
                    help="Optional extra copy of the best checkpoint (old behaviour).")
parser.add_argument("--run_dir", type=str, default="runs/hover",
                    help="Folder for best.pt, latest.pt, progress.csv, DONE marker.")
parser.add_argument("--resume", action="store_true",
                    help="Continue from <run_dir>/latest.pt if it exists.")
parser.add_argument("--save_every", type=int, default=25, help="Write latest.pt every N iters.")
parser.add_argument("--time_limit_min", type=float, default=0.0,
                    help="Stop cleanly (saving latest.pt) after this many minutes; 0 = no limit.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (the adaptive schedule takes over)")
parser.add_argument("--lr_schedule", choices=["adaptive", "linear", "constant"], default="adaptive",
                    help="adaptive: adjust every minibatch to keep the policy step near --desired_kl "
                         "(as rsl_rl / Isaac Lab do); linear: decay to 5%% of --lr by the last iter")
parser.add_argument("--desired_kl", type=float, default=0.01)
parser.add_argument("--init_log_std", type=float, default=-1.6,
                    help="Initial exploration noise (log std of the [0,1] action). -1.6 = std 0.2 "
                         "(about +-0.4 in a: +-1 rad/s rate noise). Much smaller makes the KL "
                         "hypersensitive and the adaptive lr collapses.")
parser.add_argument("--lr_min", type=float, default=1e-5)
parser.add_argument("--lr_max", type=float, default=1e-3)
parser.add_argument("--reward", default="v1", help="Reward preset: v1 (original) or calm (see REWARD_PRESETS)")
parser.add_argument("--gamma", type=float, default=0.99,
                    help="Discount: 0.99 looks ~2 s ahead at 50 Hz, 0.995 ~4 s (use with --reward calm)")
parser.add_argument("--lr_max_final", type=float, default=None,
                    help="Decay the learning-rate ceiling from --lr_max to this over the run "
                         "(geometric). Lets the adaptive rule polish at the end instead of "
                         "pushing at full strength forever. Off by default.")
parser.add_argument("--init_from", type=str, default=None,
                    help="Fine-tune: start from this checkpoint's weights, input normalisation and "
                         "action noise (fresh optimizer, iteration 0). Ignored when --resume finds "
                         "latest.pt in run_dir, so a resubmitted fine-tune continues itself.")
parser.add_argument("--scenario", default="random",
                    choices=["random", "net_only", "net_in_room", "room", "open", "fixed_net"],
                    help="Scenes to train in (random = mix of nets, rooms, open halls)")
parser.add_argument("--control_mode", choices=["ctbr", "motors"], default="ctbr",
                    help="ctbr = thrust + body rates through a simulated PX4 (deployable via MAVROS)")
parser.add_argument("--obs_mode", choices=["onboard", "privileged"], default="onboard",
                    help="onboard = only real-drone sensors (default); privileged = ground truth")
parser.add_argument("--airframe", choices=["crazyflie", "dd24"], default="crazyflie")
parser.add_argument("--residual_frac", type=float, default=0.33,
                    help="Policy thrust authority around hover, as a fraction of the hover command.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torch.optim as optim

from environments.drone_hover_env import DroneHoverEnv, DroneHoverEnvCfg, DroneHoverEnvCfgDD24
from controllers.rl_hover_controller import RLHoverActor
from controllers.obs_norm import RunningNorm


class Critic(nn.Module):
    def __init__(self, obs_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def main():
    t_start = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)

    env_cfg = DroneHoverEnvCfgDD24() if args.airframe == "dd24" else DroneHoverEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.obs_mode = args.obs_mode
    from environments.drone_hover_env import apply_reward_preset
    apply_reward_preset(env_cfg, args.reward)
    print(f"reward preset {args.reward}: crash {env_cfg.rew_crash} hvel {env_cfg.rew_hvel} "
          f"action_rate {env_cfg.rew_action_rate} tilt {env_cfg.rew_tilt}; gamma {args.gamma}")
    env_cfg.scenario = args.scenario
    env_cfg.control_mode = args.control_mode
    env_cfg.__post_init__()   # recompute observation sizes for the chosen mode
    env_cfg.hover_height = 1.0

    env = DroneHoverEnv(cfg=env_cfg, render_mode=None)
    device = env.device
    obs_dim = env_cfg.observation_space
    # Asymmetric actor-critic: the critic sees ground truth (training only).
    critic_dim = env_cfg.state_space or obs_dim
    print(f"obs_mode={args.obs_mode} actor inputs={obs_dim} critic inputs={critic_dim}")

    def critic_in(o):
        return o["critic"] if "critic" in o else o["policy"]

    actor = RLHoverActor(obs_dim=obs_dim, action_dim=4, out="linear").to(device)
    critic = Critic(obs_dim=critic_dim).to(device)
    # Running input normalisation (saved in the checkpoint, frozen at deployment).
    actor_norm = RunningNorm(obs_dim).to(device)
    # Accelerometer inputs (first 3 of each 22-number frame): their normalisation spread
    # is logged, a healthy value is a few m/s^2 (reset spikes once pushed it to 19-87).
    acc_idx = ([k * 22 + j for k in range(env_cfg.obs_history) for j in range(3)]
               if args.obs_mode == "onboard" and obs_dim == 22 * env_cfg.obs_history else [])
    acc_std = lambda: actor_norm.var[acc_idx].sqrt().mean().item()  # noqa: E731
    critic_norm = RunningNorm(critic_dim).to(device)

    log_std = nn.Parameter(torch.ones(4, device=device) * args.init_log_std)

    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()) + [log_std],
        lr=args.lr,
    )

    gamma = args.gamma
    gae_lambda = 0.95
    clip_eps = 0.2
    value_coef = 0.5
    entropy_coef = 0.01

    # Hover command comes from the airframe's thrust-to-weight (was hard-coded to
    # the Crazyflie's 1/2.2, which makes a DD24 permanently over-thrust).
    hover_motor = env.hover_motor
    residual_scale = 2.0 * args.residual_frac * hover_motor
    print(f"hover_motor={hover_motor:.3f} residual_scale={residual_scale:.3f}")

    def checkpoint(it, best_reward):
        return {
            "actor": actor.state_dict(),
            "actor_out": "linear",
            "obs_norm": actor_norm.to_dict(),
            "critic_norm": critic_norm.to_dict(),
            "critic": critic.state_dict(),
            "log_std": log_std.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "iter": it,
            "best_reward": best_reward,
            "obs_dim": obs_dim,
            "critic_dim": critic_dim,
            "obs_mode": args.obs_mode,
            "obs_history": env_cfg.obs_history,
            "scenario": args.scenario,
            "control_mode": args.control_mode,
            "ctbr_thrust_center": env_cfg.ctbr_thrust_center,
            "accel_clip": getattr(env_cfg, "accel_clip", None),
            "reward_preset": args.reward,
            "gamma": args.gamma,
            "ctbr_thrust_span": env_cfg.ctbr_thrust_span,
            "ctbr_max_rates": list(env_cfg.ctbr_max_rates),
            "policy_hz": 1.0 / (env_cfg.sim.dt * env_cfg.decimation),
            "lr_schedule": args.lr_schedule,
            "hover_motor": hover_motor,
            "residual_scale": residual_scale,
            "airframe": args.airframe,
            "seed": args.seed,
        }

    def save_atomic(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)   # never leaves a half-written checkpoint if the job is killed

    latest_path = os.path.join(args.run_dir, "latest.pt")
    best_path = os.path.join(args.run_dir, "best.pt")
    csv_path = os.path.join(args.run_dir, "progress.csv")
    done_path = os.path.join(args.run_dir, "DONE")

    start_it = 0
    best_reward = -1e9
    if args.resume and os.path.isfile(latest_path):
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        if "obs_norm" in ck:
            actor_norm.load_dict(ck["obs_norm"])
            critic_norm.load_dict(ck["critic_norm"])
        with torch.no_grad():
            log_std.copy_(ck["log_std"].to(device))
        optimizer.load_state_dict(ck["optimizer"])
        start_it = ck["iter"] + 1
        best_reward = ck["best_reward"]
        print(f"Resumed from {latest_path} at iter {start_it} (best reward {best_reward:.4f})")
    elif args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        if ck.get("obs_dim", obs_dim) != obs_dim:
            raise SystemExit(f"--init_from {args.init_from}: obs_dim {ck.get('obs_dim')} != env {obs_dim}")
        if ck.get("actor_out", "sigmoid") != "linear":
            raise SystemExit(f"--init_from {args.init_from}: old sigmoid-output checkpoint, can't fine-tune")
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        actor_norm.load_dict(ck["obs_norm"])
        critic_norm.load_dict(ck["critic_norm"])
        with torch.no_grad():
            log_std.copy_(ck["log_std"].to(device))
        print(f"Fine-tuning from {args.init_from} (its iter {ck.get('iter')}, best reward "
              f"{ck.get('best_reward')}); log_std {log_std.data.tolist()}")

    new_csv = not os.path.isfile(csv_path) or start_it == 0
    csv_file = open(csv_path, "w" if new_csv else "a", newline="")
    writer = csv.writer(csv_file)
    if new_csv:
        writer.writerow(["iter", "reward", "ep_len", "log_std", "crash_frac", "net_frac", "minutes",
                         "lr", "kl", "acc_std"])

    obs, _ = env.reset()
    it = start_it - 1
    stopped_early = False

    for it in range(start_it, args.iters):
        obs_buf = []
        crit_buf = []
        action_raw_buf = []
        logprob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []
        ep_done_buf = []
        n_crash = torch.zeros((), device=device)
        n_net = torch.zeros((), device=device)

        for _ in range(args.steps):
            # Update the running statistics with this step's inputs, then feed the
            # networks normalised inputs (stored normalised, so PPO's old/new
            # log-probs use identical inputs).
            actor_norm.update(obs["policy"])
            critic_norm.update(critic_in(obs))
            policy_obs = actor_norm(obs["policy"])
            crit_obs = critic_norm(critic_in(obs))

            with torch.no_grad():
                mean = actor(policy_obs)
                std = torch.exp(log_std).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)

                action_raw = dist.sample()
                action_raw_clamped = torch.clamp(action_raw, 0.0, 1.0)
                logprob = dist.log_prob(action_raw).sum(dim=-1)
                value = critic(crit_obs)

                if args.control_mode == "ctbr":
                    actions = action_raw_clamped          # env maps it to thrust + body rates
                else:
                    residual = (action_raw_clamped - 0.5) * residual_scale
                    actions = torch.clamp(hover_motor + residual, 0.0, 1.0)

            obs, env_reward, terminated, truncated, info = env.step(actions)

            # Reward is computed inside the env from ground-truth state, and
            # includes the net/geofence and crash penalties.
            reward = env_reward
            done = terminated | truncated

            obs_buf.append(policy_obs)
            crit_buf.append(crit_obs)
            action_raw_buf.append(action_raw)
            logprob_buf.append(logprob)
            reward_buf.append(reward)
            done_buf.append(done.float())
            value_buf.append(value)

            # DirectRLEnv already auto-resets only the finished envs inside step().
            # (The old global env.reset() here restarted ALL envs whenever ANY
            # one crashed, so episodes almost never got long.)
            ep_done_buf.append(done.float())
            n_crash += (env.term_crashed | env.term_flipped).float().sum()
            n_net += env.term_net.float().sum()

        with torch.no_grad():
            next_value = critic(critic_norm(critic_in(obs)))

        obs_buf = torch.stack(obs_buf)
        crit_buf = torch.stack(crit_buf)
        action_raw_buf = torch.stack(action_raw_buf)
        logprob_buf = torch.stack(logprob_buf)
        reward_buf = torch.stack(reward_buf)
        done_buf = torch.stack(done_buf)
        value_buf = torch.stack(value_buf)

        advantages = torch.zeros_like(reward_buf)
        last_gae = torch.zeros(args.num_envs, device=device)

        for t in reversed(range(args.steps)):
            # done_buf[t] is the outcome of the step taken at t, so it gates the
            # bootstrap from t+1 (the old code used done_buf[t+1], off by one).
            next_nonterminal = 1.0 - done_buf[t]
            next_values = next_value if t == args.steps - 1 else value_buf[t + 1]

            delta = reward_buf[t] + gamma * next_values * next_nonterminal - value_buf[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + value_buf

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_crit = crit_buf.reshape(-1, critic_dim)
        b_actions_raw = action_raw_buf.reshape(-1, 4)
        b_old_logprob = logprob_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = value_buf.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = b_obs.shape[0]
        mb_size = args.minibatch_size or max(batch_size // args.num_minibatches, 1)
        indices = torch.arange(batch_size, device=device)

        if args.lr_max_final is not None:
            lr_cap = args.lr_max * (args.lr_max_final / args.lr_max) ** min(it / max(args.iters - 1, 1), 1.0)
        else:
            lr_cap = args.lr_max
        for g in optimizer.param_groups:
            g["lr"] = min(g["lr"], lr_cap)
        if args.lr_schedule == "linear":
            frac = max(1.0 - it / max(args.iters, 1), 0.05)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * frac
        kl_sum, kl_n = 0.0, 0

        for _ in range(args.epochs):
            perm = indices[torch.randperm(batch_size, device=device)]
            ep_kl_sum, ep_kl_n = 0.0, 0

            for start in range(0, batch_size, mb_size):
                mb_idx = perm[start:start + mb_size]

                mean = actor(b_obs[mb_idx])
                std = torch.exp(log_std).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)

                new_logprob = dist.log_prob(b_actions_raw[mb_idx]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                log_ratio = new_logprob - b_old_logprob[mb_idx]
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    # How far this update moved the policy (low-variance KL estimate).
                    kl = ((ratio - 1.0) - log_ratio).mean().item()
                kl_sum += kl
                kl_n += 1
                ep_kl_sum += kl
                ep_kl_n += 1

                pg_loss_1 = -b_advantages[mb_idx] * ratio
                pg_loss_2 = -b_advantages[mb_idx] * torch.clamp(
                    ratio,
                    1.0 - clip_eps,
                    1.0 + clip_eps,
                )

                policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                new_value = critic(b_crit[mb_idx])
                value_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()) + [log_std],
                    1.0,
                )
                optimizer.step()

            # Adapt the learning rate ONCE per epoch from this epoch's mean KL (per
            # minibatch compounded x1.5^20 per iteration and ran away in smoke test 2).
            ep_kl = ep_kl_sum / max(ep_kl_n, 1)
            if args.lr_schedule == "adaptive":
                lr_now = optimizer.param_groups[0]["lr"]
                if ep_kl > 2.0 * args.desired_kl:
                    lr_now = max(lr_now / 1.5, args.lr_min)
                elif ep_kl < 0.5 * args.desired_kl:
                    lr_now = min(lr_now * 1.5, lr_cap)
                lr_now = min(lr_now, lr_cap)
                for g in optimizer.param_groups:
                    g["lr"] = lr_now
            # Standard PPO safety: stop this iteration's updates if the policy moved too far.
            if ep_kl > 4.0 * args.desired_kl:
                break

        mean_reward = reward_buf.mean().item()
        lr_now = optimizer.param_groups[0]["lr"]
        mean_kl = kl_sum / max(kl_n, 1)
        n_eps = torch.stack(ep_done_buf).sum().item()
        avg_ep_len = args.steps * args.num_envs / max(n_eps, 1.0)

        minutes = (time.time() - t_start) / 60.0
        crash_frac = n_crash.item() / max(n_eps, 1.0)
        net_frac = n_net.item() / max(n_eps, 1.0)

        print(
            f"iter={it:04d} "
            f"reward={mean_reward:.4f} "
            f"ep_len~{avg_ep_len:.0f} "
            f"crash={crash_frac:.2f} net={net_frac:.2f} "
            f"log_std={log_std.mean().item():.3f} "
            f"lr={lr_now:.1e} kl={mean_kl:.4f} "
            f"t={minutes:.1f}min",
            flush=True,
        )
        writer.writerow([it, f"{mean_reward:.5f}", f"{avg_ep_len:.1f}",
                         f"{log_std.mean().item():.4f}", f"{crash_frac:.4f}",
                         f"{net_frac:.4f}", f"{minutes:.2f}", f"{lr_now:.3e}", f"{mean_kl:.5f}",
                         f"{acc_std():.3f}" if acc_idx else ""])
        csv_file.flush()

        if mean_reward > best_reward:
            best_reward = mean_reward
            save_atomic(checkpoint(it, best_reward), best_path)
            if args.save_path:
                os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
                shutil.copy(best_path, args.save_path)

        if (it + 1) % args.save_every == 0:
            save_atomic(checkpoint(it, best_reward), latest_path)

        if args.time_limit_min > 0 and minutes >= args.time_limit_min:
            print(f"Time limit reached ({minutes:.1f} min) - saving and stopping at iter {it}.")
            stopped_early = True
            break

    save_atomic(checkpoint(it, best_reward), latest_path)
    csv_file.close()
    if not stopped_early:
        open(done_path, "w").write(f"finished {args.iters} iters, best reward {best_reward:.4f}\n")
        print(f"Training complete -> {best_path}")
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