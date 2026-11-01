"""Feed real flight logs through the same code policy_node.py runs (oracle_core), offline:
held sensors at 20 Hz, takeoff-frame quaternion, wall-map estimate, observation, policy,
setpoint. The drone in the logs was flown by hand, so this checks the inputs and that the
outputs are sane (no NaN, tilt and thrust in range), not closed-loop flight.

    python deploy/replay_check.py deploy/dd24_oracle_wallmap.npz LOG_CSV_DIR... [--height 0.5]

Each flight (mavros extended_state landed_state -> 2) starts the node's profile 1 s before
takeoff; the commanded height follows the node (1 s at 0, then 0.3 m/s up to --height).
Topic mapping as in the 10-25 logs: down = right_tof, front, left, right = bottom_tof.
"""
import argparse
import bisect
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_core import FRD, Policy, Scripted, WallMap, observation, qmul, tilt_deg, yaw_of  # noqa: E402

TOPICS = {"down": "right", "front": "front", "left": "left", "right": "bottom"}


def load(d):
    rows = lambda n: list(csv.DictReader(open(f"{d}/{n}.csv")))  # noqa: E731
    stamp = lambda r: float(r["header.stamp.sec"]) + 1e-9 * float(r["header.stamp.nanosec"])  # noqa: E731
    tof = []
    for k in ("down", "front", "left", "right"):
        R = [(float(r["recv_time"]), stamp(r), float(r["range"])) for r in rows(f"noducksgiven__{TOPICS[k]}_tof_driver_node__range")]
        tof.append([x for x in R if x[2] < 100.0])     # 999 = no return: held value ages
    imu = [(float(r["recv_time"]), stamp(r), np.array([float(r[f"orientation.{c}"]) for c in "wxyz"]),
            np.array([float(r[f"angular_velocity.{c}"]) for c in "xyz"]),
            np.array([float(r[f"linear_acceleration.{c}"]) for c in "xyz"])) for r in rows("mavros__imu__data")]
    es = [(float(r["recv_time"]), int(r["landed_state"])) for r in rows("mavros__extended_state")]
    return tof, imu, es


def latest(series, t):
    i = bisect.bisect_right([s[0] for s in series], t) - 1
    return series[i] if i >= 0 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("policy")
    p.add_argument("logs", nargs="+")
    p.add_argument("--height", type=float, default=0.5)
    p.add_argument("--action", choices=["rates", "attitude"])
    p.add_argument("--scripted", action="store_true", help="policy_node.py --controller scripted (the policy file is ignored)")
    args = p.parse_args()
    pol = Scripted() if args.scripted else Policy(args.policy, args.action)
    print("scripted PID" if args.scripted else f"{args.policy}: {pol.action_mode}, {pol.n_obs} inputs, hover {pol.hover}, tilt max {math.degrees(pol.tilt_max):.0f} deg")
    bad = 0
    for d in args.logs:
        tof, imu, es = load(d)
        ups = [tb for (_, a), (tb, b) in zip(es, es[1:]) if a != 2 and b == 2]
        downs = [tb for (_, a), (tb, b) in zip(es, es[1:]) if a == 2 and b != 2]
        for n, (t_up, t_dn) in enumerate(zip(ups, downs), 1):
            t0 = t_up - 1.0
            side = [[x[2] for x in tof[i] if t0 - 1.0 <= x[0] < t0] for i in (1, 2, 3)]
            walls = [float(np.median(s)) for s in side]
            yaw0 = yaw_of(latest(imu, t0)[2])
            q_inv = np.array([math.cos(yaw0 / 2), 0, 0, -math.sin(yaw0 / 2)])
            wm = WallMap(*walls)
            target, last_sp, out = 0.0, np.zeros(4, np.float32), []
            t = t0
            while t <= t_dn:
                if t - t0 >= 1.0:
                    target = min(target + 0.3 / 20.0, args.height)
                _, s_imu, q, w, a = latest(imu, t)
                held = [latest(s, t) for s in tof]
                if None in held:
                    t += 0.05
                    continue
                quat = qmul(q_inv, q)
                quat = -quat if quat[0] < 0 else quat
                gyro, acc = (w * FRD).astype(np.float32), (a * FRD).astype(np.float32)
                r = np.array([h[2] for h in held], np.float32)
                ages = np.array([t - h[1] for h in held] + [t - s_imu])
                est = wm.step(r, quat, acc, ages[0:4]) if pol.oracle else None
                obs = observation(r, quat, gyro, acc, ages, last_sp, target, est)
                sp = pol.command(est, quat, target) if args.scripted else pol.setpoint(pol(obs))
                last_sp = sp
                out.append((t - t0, obs, sp, tilt_deg(q), est))
                t += 0.05
            O = np.array([o[1] for o in out])
            S = np.array([o[2] for o in out])
            nan = int(np.isnan(O).sum() + np.isnan(S).sum())
            bad += nan
            print(f"== {os.path.basename(d)} flight {n}: {len(out)} steps, walls front {walls[0]:.2f} left {walls[1]:.2f} right {walls[2]:.2f} m, NaN {nan}")
            print("   obs min " + " ".join(f"{x:+.2f}" for x in O.min(0)))
            print("   obs max " + " ".join(f"{x:+.2f}" for x in O.max(0)))
            floor = S[[i for i, o in enumerate(out) if o[0] < 1.0]]
            print(f"   on floor, target 0 (first 1 s): thrust {floor[:, 0].min():.3f}..{floor[:, 0].max():.3f}, "
                  f"roll/pitch cmd |max| {np.degrees(np.abs(floor[:, 1:3]).max()):.1f} deg")
            print(f"   whole flight: thrust {S[:, 0].min():.3f}..{S[:, 0].max():.3f} (mean {S[:, 0].mean():.3f}), "
                  f"roll cmd {np.degrees(S[:, 1]).min():+.1f}..{np.degrees(S[:, 1]).max():+.1f} deg, "
                  f"pitch cmd {np.degrees(S[:, 2]).min():+.1f}..{np.degrees(S[:, 2]).max():+.1f} deg, "
                  f"yaw rate {S[:, 3].min():+.2f}..{S[:, 3].max():+.2f} rad/s")
            if pol.oracle:
                P = np.array([o[4][0] for o in out])
                far = np.hypot(P[:, 0], P[:, 1])
                first = next((o[0] for o, f in zip(out, far) if f > 0.8), None)
                print(f"   estimate: x {P[:, 0].min():+.2f}..{P[:, 0].max():+.2f}, y {P[:, 1].min():+.2f}..{P[:, 1].max():+.2f}, "
                      f"z {P[:, 2].min():+.2f}..{P[:, 2].max():+.2f} m; node abort (xy > 0.8 m) at "
                      + (f"{first:.2f} s" if first is not None else "never"))
    print("OK" if bad == 0 else f"FAIL: {bad} NaN")


if __name__ == "__main__":
    main()
