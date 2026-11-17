"""Batch evaluation of a trained RotorPy DD24 policy ("the test").

Every env flies ONE episode with the deterministic policy from a random start;
reports how many finish without leaving the arena and how tightly they hold the goal.

    python rotorpy_dd24/eval_policy.py --model runs/rotorpy_s0/final.zip --task hover --net \
        --episodes 256 --out runs/rotorpy_s0/eval.json
"""
import argparse
import json
import os
import sys

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(__file__))
from train_paths import build_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", choices=["hover", "goto", "takeoff"], default="hover")
    p.add_argument("--goal", type=float, nargs=3, default=None,
                   help="Default: origin for hover, (0,0,1) for goto/takeoff")
    p.add_argument("--pos_bound", type=float, default=1.0)
    p.add_argument("--net", action="store_true")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    goal = np.array(args.goal if args.goal is not None
                    else ([0.0, 0.0, 0.0] if args.task == "hover" else [0.0, 0.0, 1.0]))
    N = args.episodes
    env = build_env(N, goal.tolist(), task=args.task, pos_bound=args.pos_bound, net=args.net)
    model = PPO.load(args.model, device="cpu")

    obs = env.reset()
    dt = env.t_step
    T = int(round(env.max_time / dt))
    half = T // 2
    finished = np.zeros(N, bool)
    success = np.zeros(N, bool)
    ended = np.full(N, T)
    errs_win = [[] for _ in range(N)]

    for t in range(T + 5):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        dones = np.asarray(dones).ravel()
        for i in range(N):
            if finished[i]:
                continue
            if dones[i]:
                # RotorPy stores the whole batch's pre-reset obs in each info.
                o = np.asarray(infos[i]["terminal_observation"])
                o = o[i] if o.ndim == 2 else o
            else:
                o = obs[i]
            e = float(np.linalg.norm(np.asarray(o)[0:3] - goal))
            if t >= half:
                errs_win[i].append(e)
            if dones[i]:
                finished[i] = True
                success[i] = bool(infos[i].get("TimeLimit.truncated", False))
                ended[i] = t
        if finished.all():
            break

    per_ep = np.array([np.mean(e) if e else np.nan for e in errs_win])
    allw = np.concatenate([np.array(e) for e in errs_win if e]) if any(errs_win) else np.array([np.nan])
    res = {
        "model": os.path.abspath(args.model),
        "task": args.task,
        "net": args.net,
        "episodes": N,
        "episode_s": float(env.max_time),
        "success_rate": float(success.mean()),
        "out_of_arena_rate": float((finished & ~success).mean()),
        "mean_survival_s": float(ended.mean() * dt),
        "pos_err_mean_m": float(np.nanmean(per_ep[success])) if success.any() else None,
        "pos_err_p95_m": (float(np.nanpercentile(allw, 95)) if np.isfinite(allw).any() else None),
    }
    res["pass"] = bool(res["success_rate"] >= 0.95 and (res["pos_err_mean_m"] or 9) < 0.15)
    out = args.out or os.path.join(os.path.dirname(args.model), "eval.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("EVAL", json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
