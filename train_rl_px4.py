"""PPO for the DD24 flown through the PX4 inner loop (DronePX4Env).

Actor: DD24 sensors only (down ToF, IMU, their sample ages, last setpoint, height command) -> CTBR setpoint
       for PX4 (same message the real drone gets via mavros). Deployable.
Critic: sensors + ground-truth position error and velocity (training only).

Unlike train_rl_hover.py, finished envs are left to the env's own auto-reset, so
one crash doesn't restart every drone.
"""
import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--iters", type=int, default=1000)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--minibatch_size", type=int, default=8192)
parser.add_argument("--save_path", type=str, default="checkpoints/px4_hover.pt")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torch.optim as optim

from controllers.px4_policy import PX4Actor, mlp
from environments.drone_px4_env import DronePX4Env, DronePX4EnvCfg, policy_to_ctbr


def main():
    cfg = DronePX4EnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = DronePX4Env(cfg=cfg, render_mode=None)
    dev = env.device
    obs_dim, state_dim = cfg.observation_space, cfg.state_space

    actor = PX4Actor(obs_dim).to(dev)
    # Start near a zero action (hover thrust, zero rates) instead of a random thrash.
    with torch.no_grad():
        actor.net[-1].weight.mul_(0.01)
        actor.net[-1].bias.zero_()
    critic = mlp(state_dim, 1).to(dev)
    log_std = nn.Parameter(torch.full((4,), -1.0, device=dev))
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [log_std], lr=3e-4)

    gamma, lam, clip_eps, value_coef, entropy_coef = 0.99, 0.95, 0.2, 0.5, 0.0

    obs, _ = env.reset()
    best = -1e9
    ep_ret = torch.zeros(args.num_envs, device=dev)
    ep_len = torch.zeros(args.num_envs, device=dev)
    t_start = time.time()

    for it in range(args.iters):
        buf = {k: [] for k in ("obs", "state", "act", "logp", "rew", "done", "val")}
        finished_ret, finished_len, crashes = [], [], 0

        for _ in range(args.steps):
            o, s = obs["policy"], obs["critic"]
            with torch.no_grad():
                mean = actor(o)
                dist = torch.distributions.Normal(mean, log_std.exp().expand_as(mean))
                a = dist.sample()
                logp = dist.log_prob(a).sum(-1)
                v = critic(s).squeeze(-1)
            obs, rew, term, trunc, _ = env.step(policy_to_ctbr(a))
            done = term | trunc

            ep_ret += rew
            ep_len += 1
            if done.any():
                finished_ret.append(ep_ret[done].clone())
                finished_len.append(ep_len[done].clone())
                crashes += int(term.sum())
                ep_ret[done] = 0.0
                ep_len[done] = 0.0

            for k, x in zip(buf, (o, s, a, logp, rew, done.float(), v)):
                buf[k].append(x)

        b = {k: torch.stack(x) for k, x in buf.items()}
        with torch.no_grad():
            next_v = critic(obs["critic"]).squeeze(-1)
        adv = torch.zeros_like(b["rew"])
        gae = torch.zeros(args.num_envs, device=dev)
        for t in reversed(range(args.steps)):
            nv = next_v if t == args.steps - 1 else b["val"][t + 1]
            nonterm = 1.0 - b["done"][t]
            delta = b["rew"][t] + gamma * nv * nonterm - b["val"][t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + b["val"]

        flat = lambda x, d=None: x.reshape(-1, d) if d else x.reshape(-1)  # noqa: E731
        f_obs, f_state, f_act = flat(b["obs"], obs_dim), flat(b["state"], state_dim), flat(b["act"], 4)
        f_logp, f_adv, f_ret = flat(b["logp"]), flat(adv), flat(ret)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        n = f_obs.shape[0]
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, args.minibatch_size):
                mb = perm[i:i + args.minibatch_size]
                mean = actor(f_obs[mb])
                dist = torch.distributions.Normal(mean, log_std.exp().expand_as(mean))
                ratio = (dist.log_prob(f_act[mb]).sum(-1) - f_logp[mb]).exp()
                pg = -torch.min(ratio * f_adv[mb], ratio.clamp(1 - clip_eps, 1 + clip_eps) * f_adv[mb]).mean()
                vl = 0.5 * (critic(f_state[mb]).squeeze(-1) - f_ret[mb]).pow(2).mean()
                loss = pg + value_coef * vl - entropy_coef * dist.entropy().sum(-1).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()) + [log_std], 1.0)
                optimizer.step()
                with torch.no_grad():
                    log_std.clamp_(-3.0, 0.0)

        step_rew = b["rew"].mean().item()
        if finished_len:
            lens = torch.cat(finished_len)
            ep_s = lens.float().mean().item() * cfg.sim.dt * cfg.decimation
            n_eps = lens.numel()
        else:
            ep_s, n_eps = float("nan"), 0
        print(f"iter={it:04d} step_reward={step_rew:.4f} episodes={n_eps} mean_ep_len={ep_s:5.2f}s "
              f"crashes={crashes} std={log_std.exp().mean().item():.3f} elapsed={time.time() - t_start:6.0f}s")

        if step_rew > best:
            best = step_rew
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                        "log_std": log_std.detach().cpu(), "best_step_reward": best, "iter": it}, args.save_path)
            print(f"saved best step_reward {best:.4f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
