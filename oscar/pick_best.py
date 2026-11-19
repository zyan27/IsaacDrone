"""Compare runs on their four tests and name the best one.
    python3 oscar/pick_best.py runs/a_s0 runs/a_s1 [...]
Score = mean success rate over the four tests (ties: lower xy error).
Last line: BEST <run_dir>   (or NONE if no run has all four tests)."""
import json, os, sys
TESTS = [("eval.json", "normal"), ("eval_net70.json", "net 70% missed"),
         ("eval_fixednet.json", "fixed net"), ("eval_lag.json", "worst lag")]
rows = []
for d in sys.argv[1:]:
    res = {}
    for f, _ in TESTS:
        p = os.path.join(d, f)
        if os.path.exists(p):
            try:
                res[f] = json.load(open(p))
            except Exception:
                pass
    print(f"\n{d}")
    print(f"  {'test':16s} {'success':>8s} {'crash':>7s} {'net':>7s} {'z err':>7s} {'xy err':>7s}")
    for f, name in TESTS:
        r = res.get(f)
        if r is None:
            print(f"  {name:16s}  (missing)"); continue
        fmt = lambda v: f"{v:7.3f}" if isinstance(v, (int, float)) else "    n/a"  # noqa: E731
        print(f"  {name:16s} {r['success_rate']:8.3f} {fmt(r['crash_rate'])} {fmt(r['net_rate'])} "
              f"{fmt(r.get('z_err_mean_m'))} {fmt(r.get('xy_err_mean_m'))}")
    if len(res) == len(TESTS):
        score = sum(r["success_rate"] for r in res.values()) / len(TESTS)
        xy = res["eval.json"].get("xy_err_mean_m") or 9.0
        it = res["eval.json"].get("train_iter")
        print(f"  score {score:.3f}  (checkpoint from training iteration {it})")
        rows.append((score, -xy, d))
print()
print(f"BEST {max(rows)[2]}" if rows else "NONE")
