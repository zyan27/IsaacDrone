"""Decide whether a smoke-test run is healthy enough to launch the long run.

    python3 oscar/check_smoke.py [runs/smoke_s0]      -> exit 0 = go, 1 = stop
Plain python3 (no venv). Checks: all four tests ran, rewards are finite, the
learning rate isn't pinned at either limit, and KL isn't exploding.
"""
import csv, json, math, os, sys

run = sys.argv[1] if len(sys.argv) > 1 else "runs/smoke_s0"
LR_MIN, LR_MAX = 1e-5, 1e-3
problems = []

for f in ["eval.json", "eval_net70.json", "eval_fixednet.json", "eval_lag.json"]:
    p = os.path.join(run, f)
    if not os.path.exists(p):
        problems.append(f"{f} missing (a test crashed; see the .err log)")
    else:
        try:
            json.load(open(p))
        except Exception as e:
            problems.append(f"{f} unreadable: {e}")

rows = list(csv.DictReader(open(os.path.join(run, "progress.csv")))) if os.path.exists(os.path.join(run, "progress.csv")) else []
if len(rows) < 10:
    problems.append(f"only {len(rows)} training iterations logged")
else:
    tail = rows[len(rows) // 2:]                       # second half of the smoke run
    lr = [float(r["lr"]) for r in tail]
    kl = [float(r["kl"]) for r in tail]
    rew = [float(r["reward"]) for r in rows]
    if any(not math.isfinite(x) for x in rew + kl + lr):
        problems.append("NaN/inf in reward, kl or lr")
    if all(x <= LR_MIN * 1.01 for x in lr):
        problems.append(f"lr pinned at the minimum {LR_MIN:g} (updates too violent)")
    if all(x >= LR_MAX * 0.99 for x in lr) and max(kl) < 0.004:
        problems.append(f"lr pinned at the maximum {LR_MAX:g} with tiny KL (policy not responding)")
    if max(kl) > 0.08:
        problems.append(f"KL spiked to {max(kl):.3f}")
    acc = rows[-1].get("acc_std") or ""
    if acc and float(acc) > 10.0:
        problems.append(f"accelerometer normalisation spread {float(acc):.1f} m/s^2 (> 10): reset spikes are "
                        "still getting into the IMU input")
    print(f"second half of smoke run: lr {min(lr):.1e}..{max(lr):.1e}, kl {min(kl):.4f}..{max(kl):.4f}, "
          f"reward {rew[0]:.3f} -> {rew[-1]:.3f}, accelerometer spread {rows[-1].get('acc_std') or 'n/a'} m/s^2")

if problems:
    print("STOP - not launching the long run:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("GO - smoke test healthy")
