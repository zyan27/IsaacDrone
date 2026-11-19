"""Flight recorder: run a policy on N drones for one episode each and log
everything per policy step, plus each episode's random conditions.

Pure torch/numpy (no Isaac imports), so it is testable without the simulator.
Used by flight_recorder.py (Isaac) and analysed by analyze_flights.py.

Per-step arrays are (T, N, k), float16, valid for t < ended_at[n] + 1:
  pos, vel_w, quat(wxyz), angvel_b, target, net_dist   true state BEFORE the step
  frame   the 22 newest policy inputs (as held/delayed: what the policy saw)
  tof_true, flow_now                                   truth / instantaneous flow
  cmd_sent, cmd_px4, motors                            what was sent / arrived / motor cmds
  stalled (T,N,4) bool                                 IMU, ToF, flow, cmd link frozen
Terminal state (the instant an episode ended, before the auto-reset) is in term_*.
"""
from __future__ import annotations

import numpy as np
import torch

REASONS = ("survived", "crash", "net", "flip")


def _np(x, dt=np.float16):
    return x.detach().float().cpu().numpy().astype(dt)


def episode_conditions(env) -> dict:
    """Each env's random conditions for its current episode (call right after reset)."""
    c = {}
    add = lambda k, v: c.__setitem__(k, _np(v, np.float32).reshape(env.num_envs, -1))  # noqa: E731
    add("tof_net_miss_p", env._tof_drop_p)
    add("tof_ghost_p", env._tof_ghost_p)
    add("flow_scale", env._flow_scale)
    add("flow_invalid_p", env._flow_drop_p)
    add("thrust_scale_min", env._thrust_scale.min(-1).values)
    add("thrust_scale_spread", env._thrust_scale.max(-1).values - env._thrust_scale.min(-1).values)
    add("accel_bias", torch.linalg.norm(env._accel_bias, dim=-1))
    add("gyro_bias", torch.linalg.norm(env._gyro_bias, dim=-1))
    add("gravity_dir_bias", torch.linalg.norm(env._grav_bias, dim=-1))
    dt = env.cfg.sim.dt * env.cfg.decimation
    for name, h in [("imu", env._imu_s), ("tof", env._tof_s), ("flow", env._flow_s), ("cmd", env._cmd_link)]:
        if name != "cmd":
            add(f"{name}_rate_hz", h.rate)
        add(f"{name}_latency_s", h.lat.float() * dt)
        add(f"{name}_drop_p", h.drop_p)
        add(f"{name}_stall_per_s", h.stall_rate)
    px4 = env._px4
    from environments.px4_ctbr import PX4_RATE_P
    base = torch.tensor(PX4_RATE_P, device=px4.P.device)
    add("px4_gain_rollpitch", (px4.P[:, :2] / base[:2]).mean(-1))
    add("px4_gain_yaw", px4.P[:, 2] / base[2])
    add("px4_mix_rollpitch", px4.mix_gain[:, :2].mean(-1))
    add("motor_tau_s", px4.tau_m)
    add("thrust_curve_k", px4.k_curve)
    add("target_height", env._target[:, 2])
    return {k: v[:, 0] for k, v in c.items()}


def record(env, policy, T: int | None = None) -> dict:
    """policy(obs_dict) -> actions. Runs until every env's FIRST episode ended."""
    N, dev = env.num_envs, env.device
    T = T or int(env.max_episode_length)
    # Capture the state at termination (step() auto-resets terminated envs).
    orig = env._get_dones
    term = {}

    def dones_hook():
        out = orig()
        d = env.drone.data
        term["pos"], term["quat"] = env._local_pos().clone(), d.root_quat_w.clone()
        term["vel"], term["net"] = d.root_lin_vel_w.clone(), env._dist_to_net(term["pos"])
        return out
    env._get_dones = dones_hook

    obs, _ = env.reset()
    cond = episode_conditions(env)
    scene = {k: v.detach().cpu().numpy() for k, v in env._scene.items()}
    finished = torch.zeros(N, dtype=torch.bool, device=dev)
    ended_at = torch.full((N,), T - 1, dtype=torch.long, device=dev)
    reason = torch.zeros(N, dtype=torch.long, device=dev)
    term_pos = torch.zeros(N, 3, device=dev)
    term_quat = torch.zeros(N, 4, device=dev)
    term_vel = torch.zeros(N, 3, device=dev)
    logs = {k: [] for k in ["pos", "vel_w", "quat", "angvel_b", "target", "net_dist", "frame", "tof_meas",
                            "tof_true", "flow_now", "cmd_sent", "cmd_px4", "motors", "stalled"]}
    with torch.inference_mode():
        for t in range(T):
            d = env.drone.data
            pos = env._local_pos()
            dbg = env._dbg
            stalled = torch.stack([h.stall_left > 0 for h in (env._imu_s, env._tof_s, env._flow_s, env._cmd_link)], -1)
            for k, v in [("pos", pos), ("vel_w", d.root_lin_vel_w), ("quat", d.root_quat_w),
                         ("angvel_b", d.root_ang_vel_b), ("target", env._target),
                         ("net_dist", env._dist_to_net(pos)[:, None]), ("frame", dbg["frame"]),
                         ("tof_meas", dbg["tof_meas"]), ("tof_true", dbg["tof_true"]), ("flow_now", dbg["flow_now"])]:
                logs[k].append(v.to(torch.float16).clone())
            logs["stalled"].append(stalled.clone())
            obs, _, terminated, truncated, _ = env.step(policy(obs))
            logs["cmd_sent"].append(env._cmd.to(torch.float16).clone())
            logs["cmd_px4"].append(env._cmd_px4.to(torch.float16).clone())
            logs["motors"].append(env._px4.u_act.to(torch.float16).clone())

            new = terminated & ~finished
            reason[new & env.term_crashed] = 1
            reason[new & env.term_net & ~env.term_crashed] = 2
            reason[new & env.term_flipped & ~env.term_crashed & ~env.term_net] = 3
            ended_at[new | (truncated & ~finished)] = t
            term_pos[new], term_quat[new], term_vel[new] = term["pos"][new], term["quat"][new], term["vel"][new]
            finished |= terminated | truncated
            if bool(finished.all()):
                break
    env._get_dones = orig
    out = {k: torch.stack(v).cpu().numpy() for k, v in logs.items()}
    out.update(ended_at=ended_at.cpu().numpy(), reason=reason.cpu().numpy(),
               term_pos=term_pos.cpu().numpy(), term_quat=term_quat.cpu().numpy(), term_vel=term_vel.cpu().numpy(),
               step_dt=np.float32(env.cfg.sim.dt * env.cfg.decimation),
               tof_pos_b=env._tof_pos_b.cpu().numpy(), tof_dir_b=env._tof_dir_b.cpu().numpy(),
               tof_max=np.float32(env.cfg.tof_max_range), max_rates=env._max_rates.cpu().numpy().reshape(-1),
               ctbr=np.array([env.cfg.ctbr_thrust_center, env.cfg.ctbr_thrust_span], np.float32))
    out.update({f"cond_{k}": v for k, v in cond.items()})
    out.update({f"scene_{k}": v for k, v in scene.items()})
    return out
