"""Print a table of every run's training progress + test results.

    python oscar/summarize_runs.py            # all runs
    python oscar/summarize_runs.py 1114       # only runs whose name contains "1114"
Works with plain python3 (no venv needed).
"""
import csv, glob, json, os, sys

filt = sys.argv[1] if len(sys.argv) > 1 else ""
rows = []
for d in sorted(glob.glob("runs/*")):
    if filt not in d or not os.path.isdir(d):
        continue
    r = {"run": os.path.basename(d), "status": "done" if os.path.exists(f"{d}/DONE") else "running/partial"}
    prog = f"{d}/progress.csv"
    if os.path.exists(prog):
        with open(prog) as f:
            lines = list(csv.DictReader(f))
        if lines:
            r["iter"] = lines[-1]["iter"]
            r["reward"] = lines[-1]["reward"]
            r["ep_len"] = lines[-1]["ep_len"]
    ev = f"{d}/eval.json"
    if os.path.exists(ev):
        e = json.load(open(ev))
        r["success"] = f'{e.get("success_rate", 0):.0%}'
        err = e.get("xy_err_mean_m", e.get("pos_err_mean_m"))
        r["err_m"] = f"{err:.3f}" if isinstance(err, (int, float)) else "-"
        r["net_hits"] = f'{e.get("net_rate", e.get("out_of_arena_rate", 0)):.0%}'
        r["PASS"] = "PASS" if e.get("pass") else "fail"
    rows.append(r)

cols = ["run", "status", "iter", "reward", "ep_len", "success", "err_m", "net_hits", "PASS"]
if not rows:
    print("no runs found"); sys.exit()
w = {c: max(len(c), *(len(str(r.get(c, "-"))) for r in rows)) for c in cols}
print("  ".join(c.ljust(w[c]) for c in cols))
for r in rows:
    print("  ".join(str(r.get(c, "-")).ljust(w[c]) for c in cols))
