"""Why and when does the policy fail? Analyse a flight_recorder.py log.

    python analyze_flights.py runs/x_s0/flights/flights.npz [--fails 12] [--drifts 4] [--no_video]

Writes next to the .npz:
  report.txt        outcomes, failure modes, when failures happen, which conditions
                    make failure / drift more likely
  summary.json      per-episode table (outcome, time, mode, key numbers, conditions)
  plots/*.png       flight-recorder dashboard for each selected episode
  videos/*.mp4      top + side animation of each selected episode (scene drawn from
                    the log: net, room walls, boxes; ToF beams as the policy saw them)

Needs numpy + matplotlib (+ imageio-ffmpeg for videos). No simulator.
Failure modes are HEURISTIC labels from the last 1-2 s of each flight; the
dashboards show the evidence.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

REASONS = ("survived", "crash", "net", "flip")


# ------------------------------------------------------------------ geometry
def quat_to_R(q):
    """(..., 4) wxyz -> (..., 3, 3) body->world."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-9)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def tilt_deg(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-9)
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1, 1)))


class Log:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.path = path
        self.d = {k: z[k] for k in z.files}
        d = self.d
        self.dt = float(d["step_dt"])
        self.N = d["reason"].shape[0]
        self.S = d["tof_true"].shape[-1]
        self.tof_names = [str(s) for s in d["tof_names"]] if "tof_names" in d else [f"tof{i}" for i in range(self.S)]
        self.scen_names = [str(s) for s in d["scenario_names"]] if "scenario_names" in d else []
        self.tof_max = float(d["tof_max"])
        self.cond = {k[5:]: d[k].astype(float) for k in d if k.startswith("cond_")}
        S = self.S
        # policy input frame layout: accel3 gyro3 grav3 | tof S | cmd4 | flow vx vy valid | target z
        self.i_tof, self.i_cmd, self.i_flow = 9, 9 + S, 13 + S

    def ep(self, n):
        """Arrays for episode n, truncated to its own steps (float32)."""
        L = int(self.d["ended_at"][n]) + 1
        e = {k: self.d[k][:L, n].astype(np.float32)
             for k in ["pos", "vel_w", "quat", "angvel_b", "target", "net_dist", "frame", "tof_meas", "tof_true",
                       "flow_now", "cmd_sent", "cmd_px4", "motors"]}
        e["stalled"] = self.d["stalled"][:L, n]
        e["t"] = np.arange(L) * self.dt
        R = quat_to_R(e["quat"])
        e["vel_b"] = np.einsum("tji,tj->ti", R, e["vel_w"])         # R^T v
        e["tilt"] = tilt_deg(e["quat"])
        e["xy_err"] = np.linalg.norm(e["pos"][:, :2] - e["target"][:, :2], axis=-1)
        e["R"] = R
        e["reason"] = REASONS[int(self.d["reason"][n])]
        e["term_pos"] = self.d["term_pos"][n]
        return e

    def scene(self, n):
        return {k[6:]: self.d[k][n] for k in self.d if k.startswith("scene_")}


# --------------------------------------------------------------- per episode
def features(lg: Log, n: int) -> dict:
    e = lg.ep(n)
    L, dt = len(e["t"]), lg.dt
    w1, w2, w05 = max(1, min(L, int(1.0 / dt))), max(1, min(L, int(2.0 / dt))), max(1, min(L, int(0.5 / dt)))
    fr = e["frame"]
    down = lg.tof_names.index("down") if "down" in lg.tof_names else 0
    valid = fr[:, lg.i_flow + 2] > 0.5
    # Readings vs truth over the last ~0.25 s (max latency + hold), so normal lag
    # during fast motion doesn't count as "wrong"; scale errors / glitches / net
    # misses / ghosts do. Frozen topics are reported separately.
    win = int(round(0.25 / dt))
    flow_err = np.linalg.norm(outside_recent(fr[:, lg.i_flow:lg.i_flow + 2], e["vel_b"][:, :2], win), axis=-1)
    tof_down_seen = fr[:, lg.i_tof + down] * lg.tof_max
    tof_down_err = outside_recent(tof_down_seen, np.minimum(e["tof_true"][:, down], lg.tof_max), win)
    cmd = e["cmd_sent"]
    sc = lg.scene(n)
    f = {
        "episode": n, "outcome": e["reason"], "end_s": round(L * dt, 2),
        "scenario": lg.scen_names[int(sc["scen"])] if lg.scen_names else int(sc["scen"]),
        "max_tilt_last1s_deg": float(e["tilt"][-w1:].max()),
        "max_rate_last1s": float(np.linalg.norm(e["angvel_b"][-w1:], axis=-1).max()),
        "min_z_last1s": float(e["pos"][-w1:, 2].min()),
        "mean_vz_last1s": float(e["vel_w"][-w1:, 2].mean()),
        "max_hspeed_last1s": float(np.linalg.norm(e["vel_w"][-w1:, :2], axis=-1).max()),
        "flow_valid_last2s": float(valid[-w2:].mean()),
        "flow_err_last2s": float(flow_err[-w2:][valid[-w2:]].mean()) if valid[-w2:].any() else None,
        "down_tof_bad_frac_last1s": float((tof_down_err[-w1:] > 0.15).mean()),
        "thrust_sat_low_last1s": float((cmd[-w1:, 0] <= -0.98).mean()),
        "thrust_sat_high_last1s": float((cmd[-w1:, 0] >= 0.98).mean()),
        "rate_sat_last1s": float((np.abs(cmd[-w1:, 1:]) >= 0.98).any(-1).mean()),
        "stall_last05s": {k: bool(e["stalled"][-w05:, i].any()) for i, k in enumerate(["imu", "tof", "flow", "cmd"])},
        "final_xy_err": float(e["xy_err"][-1]), "max_xy_err": float(e["xy_err"].max()),
        "mean_xy_err_2nd_half": float(e["xy_err"][L // 2:].mean()),
        "min_net_dist": float(e["net_dist"].min()),
    }
    if e["reason"] == "net":
        p = e["term_pos"]
        to_ceiling = float(sc["keep_h"]) - p[2]
        to_side = (np.asarray(sc["keep_half"]) - np.abs(p[:2])).min()
        f["hit"] = "ceiling" if to_ceiling < to_side else "wall/net"
    f["mode"], f["hints"] = classify(f)
    f["conditions"] = {k: round(float(v[n]), 4) for k, v in lg.cond.items()}
    return f


def outside_recent(seen, truth, win):
    """How far each reading is from the range the truth covered over the last `win`
    steps: 0 if a delayed/held copy of the truth explains it, >0 if it's simply wrong."""
    pad = np.concatenate([np.repeat(truth[:1], win, axis=0), truth], axis=0)
    w = np.lib.stride_tricks.sliding_window_view(pad, win + 1, axis=0)   # (L, ..., win+1)
    lo, hi = w.min(-1), w.max(-1)
    return np.maximum(lo - seen, 0) + np.maximum(seen - hi, 0)


def classify(f):
    """Heuristic failure-mode label + supporting hints."""
    hints = []
    if f["stall_last05s"]["cmd"]:
        hints.append("command link frozen in the last 0.5 s")
    for k in ["tof", "flow", "imu"]:
        if f["stall_last05s"][k]:
            hints.append(f"{k} topic frozen in the last 0.5 s")
    if f["flow_valid_last2s"] < 0.5:
        hints.append(f"flow valid only {f['flow_valid_last2s']:.0%} of the last 2 s")
    if f["flow_err_last2s"] is not None and f["flow_err_last2s"] > 0.15:
        hints.append(f"flow wrong by {f['flow_err_last2s']:.2f} m/s on average (beyond lag)")
    if f["down_tof_bad_frac_last1s"] > 0.2:     # persistent, not single-reading glitches
        hints.append(f"down ToF wrong for {f['down_tof_bad_frac_last1s']:.0%} of the last 1 s")
    if f["thrust_sat_low_last1s"] > 0.2:
        hints.append("thrust at its minimum")
    if f["thrust_sat_high_last1s"] > 0.2:
        hints.append("thrust at its maximum")
    if f["rate_sat_last1s"] > 0.3:
        hints.append("body-rate commands saturated")
    o = f["outcome"]
    if o == "survived":
        return "survived", hints
    early = f["end_s"] < 1.5
    if o == "flip" or (o == "crash" and f["max_tilt_last1s_deg"] > 45):
        mode = "lost attitude"
    elif o == "crash":
        mode = "lost height"
    elif f.get("hit") == "ceiling":
        mode = "climbed into ceiling"
    elif f["flow_valid_last2s"] < 0.5:
        mode = "drifted into net/wall without flow"
    elif f["max_hspeed_last1s"] > 0.8:
        mode = "fast excursion into net/wall"
    else:
        mode = "slow drift into net/wall"
    return (f"early: {mode}" if early else mode), hints


def risk_table(values, fail):
    """Failure rate in the lowest vs highest third of a condition."""
    ok = np.isfinite(values)
    v, y = values[ok], fail[ok]
    if len(v) < 30 or np.ptp(v) < 1e-9:
        return None
    lo, hi = np.quantile(v, [1 / 3, 2 / 3])
    a, b = y[v <= lo].mean() if (v <= lo).any() else np.nan, y[v >= hi].mean() if (v >= hi).any() else np.nan
    return a, b, lo, hi


def rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else 0.0


# ------------------------------------------------------------------- report
def report(lg: Log, feats, out_dir):
    reason = lg.d["reason"]
    N = lg.N
    lines = [f"Flight recorder: {lg.path}", f"meta: {' | '.join(str(s) for s in lg.d.get('meta', []))}",
             f"{N} episodes of up to {lg.d['frame'].shape[0] * lg.dt:.0f} s", ""]
    lines.append("OUTCOMES")
    for i, r in enumerate(REASONS):
        lines.append(f"  {r:9s} {np.mean(reason == i):6.1%}  ({int(np.sum(reason == i))})")
    if lg.scen_names:
        lines += ["", "BY SCENE                  episodes  survived  crash   net   flip"]
        scen = lg.d["scene_scen"]
        for i, s in enumerate(lg.scen_names):
            m = scen == i
            if m.any():
                lines.append(f"  {s:22s} {int(m.sum()):8d}  {np.mean(reason[m] == 0):7.1%} "
                             f"{np.mean(reason[m] == 1):6.1%} {np.mean(reason[m] == 2):5.1%} {np.mean(reason[m] == 3):5.1%}")
    fails = [f for f in feats if f["outcome"] != "survived"]
    lines += ["", f"FAILURE MODES (heuristic, {len(fails)} failures)"]
    modes = {}
    for f in fails:
        modes.setdefault(f["mode"], []).append(f)
    for m, fs in sorted(modes.items(), key=lambda kv: -len(kv[1])):
        t = np.array([f["end_s"] for f in fs])
        lines.append(f"  {m:40s} {len(fs):4d}  ({len(fs) / max(len(fails), 1):5.1%})  at t = "
                     f"{np.percentile(t, 25):4.1f}-{np.percentile(t, 75):4.1f} s (median {np.median(t):4.1f})")
        hc = {}
        for f in fs:
            for h in f["hints"]:
                key = h.split(" by ")[0].split(" only ")[0].split(" up to ")[0]
                hc[key] = hc.get(key, 0) + 1
        for h, c in sorted(hc.items(), key=lambda kv: -kv[1])[:4]:
            lines.append(f"      {c / len(fs):4.0%} of these: {h}")
    if fails:
        t = np.array([f["end_s"] for f in fails])
        edges = [0, 1, 2, 5, 10, 15, 20.01]
        lines += ["", "WHEN FAILURES HAPPEN (seconds into the flight)"]
        for a, b in zip(edges[:-1], edges[1:]):
            c = int(((t >= a) & (t < b)).sum())
            lines.append(f"  {a:4.0f}-{min(b, 20):2.0f} s  {c:4d}  {'#' * int(round(40 * c / len(t)))}")

    fail = (reason != 0).astype(float)
    lines += ["", "WHAT MAKES FAILURE MORE LIKELY  (failure rate: lowest third -> highest third of each condition)"]
    rows = []
    for k, v in lg.cond.items():
        r = risk_table(v, fail)
        if r and np.isfinite(r[0]) and np.isfinite(r[1]):
            a, b, lo, hi = r
            rows.append((abs(np.log((b + 0.005) / (a + 0.005))), k, a, b, lo, hi))
    for _, k, a, b, lo, hi in sorted(rows, reverse=True)[:10]:
        arrow = "worse when HIGH" if b > a else "worse when LOW"
        lines.append(f"  {k:24s} {a:6.1%} (<= {lo:.3g})  ->  {b:6.1%} (>= {hi:.3g})   {arrow}")

    surv = [f for f in feats if f["outcome"] == "survived"]
    if surv:
        dr = np.array([f["mean_xy_err_2nd_half"] for f in surv])
        idx = np.array([f["episode"] for f in surv])
        lines += ["", f"DRIFT (survivors, mean xy error over 2nd half): median {np.median(dr):.2f} m, "
                      f"90th pct {np.percentile(dr, 90):.2f} m, worst {dr.max():.2f} m",
                  "  conditions most linked with drift (rank correlation, +: more drift when higher):"]
        cr = sorted(((rank_corr(v[idx], dr), k) for k, v in lg.cond.items()), key=lambda x: -abs(x[0]))[:8]
        for c, k in cr:
            lines.append(f"    {k:24s} {c:+.2f}")
    lines += ["", "Selected episodes (plots/ and videos/):"]
    txt = "\n".join(lines)
    open(os.path.join(out_dir, "report.txt"), "w").write(txt + "\n")
    return txt


def select(feats, n_fail, n_drift):
    """Failures spread over modes (round robin), then the worst drifters."""
    by = {}
    for f in feats:
        if f["outcome"] != "survived":
            by.setdefault(f["mode"], []).append(f)
    for v in by.values():
        v.sort(key=lambda f: f["end_s"])
    picks, keys = [], sorted(by, key=lambda k: -len(by[k]))
    while len(picks) < n_fail and any(by[k] for k in keys):
        for k in keys:
            if by[k] and len(picks) < n_fail:
                picks.append(by[k].pop(len(by[k]) // 2))
    surv = sorted([f for f in feats if f["outcome"] == "survived"], key=lambda f: -f["mean_xy_err_2nd_half"])
    return picks + surv[:n_drift]


# --------------------------------------------------------------------- plots
def draw_scene_top(ax, sc):
    import matplotlib.patches as mp
    kh = np.asarray(sc["keep_half"])
    ax.add_patch(mp.Rectangle(-kh, *(2 * kh), fill=False, ls="--", ec="0.5", lw=1, label="keep-in limit"))
    if bool(sc["net_on"]):
        nh = np.asarray(sc["net_half"])
        ax.add_patch(mp.Rectangle(-nh, *(2 * nh), fill=False, ec="tab:green", lw=2, label="net"))
    if bool(sc["room_on"]):
        rh = np.asarray(sc["room_half"])
        ax.add_patch(mp.Rectangle(-rh, *(2 * rh), fill=False, ec="saddlebrown", lw=2.5, label="room wall"))
    for c, h, on in zip(sc["clut_c"], sc["clut_half"], sc["clut_on"]):
        if on:
            ax.add_patch(mp.Rectangle(c[:2] - h[:2], *(2 * h[:2]), fc="0.75", ec="0.4", label="box"))
    lim = max(float(np.max(kh)), float(np.max(sc["room_half"])) if bool(sc["room_on"]) else 0, 1.0) + 0.4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")


def draw_scene_side(ax, sc):
    import matplotlib.patches as mp
    kh = np.asarray(sc["keep_half"])
    ax.axhline(0, color="k", lw=1.5)
    ax.plot([-kh[0], -kh[0], kh[0], kh[0]], [0, sc["keep_h"], sc["keep_h"], 0], "--", color="0.5", lw=1)
    if bool(sc["net_on"]):
        nh, h = float(sc["net_half"][0]), float(sc["net_h"])
        ax.plot([-nh, -nh, nh, nh], [0, h, h, 0], color="tab:green", lw=2)
    if bool(sc["room_on"]):
        rh, h = float(sc["room_half"][0]), float(sc["room_h"])
        ax.plot([-rh, -rh, rh, rh], [0, h, h, 0], color="saddlebrown", lw=2.5)
    for c, hh, on in zip(sc["clut_c"], sc["clut_half"], sc["clut_on"]):
        if on:
            ax.add_patch(mp.Rectangle((c[0] - hh[0], c[2] - hh[2]), 2 * hh[0], 2 * hh[2], fc="0.75", ec="0.4"))
    lim = max(float(kh[0]), float(sc["room_half"][0]) if bool(sc["room_on"]) else 0, 1.0) + 0.4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.1, max(float(sc["keep_h"]), 1.0) + 0.4)
    ax.set_aspect("equal")


def shade(ax, t, mask, color, label=None):
    if not mask.any():
        return
    on = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for i, (a, b) in enumerate(zip(on[::2], on[1::2])):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=color, alpha=0.25, lw=0, label=label if i == 0 else None)


def dashboard(lg: Log, f, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = f["episode"]
    e, sc = lg.ep(n), lg.scene(n)
    t, fr = e["t"], e["frame"]
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    title = (f"episode {n}  |  {f['scenario']}  |  {f['outcome'].upper()} at {f['end_s']:.2f} s  |  mode: {f['mode']}"
             + (f"\nhints: {'; '.join(f['hints'])}" if f["hints"] else ""))
    fig.suptitle(title, fontsize=11)

    ax = fig.add_subplot(gs[0:2, 0])
    draw_scene_top(ax, sc)
    p = e["pos"]
    ax.plot(p[:, 0], p[:, 1], color="tab:blue", lw=1.2)
    ax.plot(*p[0, :2], "o", color="tab:blue", label="start")
    ax.plot(*e["target"][0, :2], "+", color="k", ms=12, label="target")
    end_xy = e["term_pos"][:2] if f["outcome"] != "survived" else p[-1, :2]
    ax.plot(end_xy[0], end_xy[1], "X", color="red", ms=10, label="end")
    ax.set_title("top view (x forward at start = right)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=7, loc="upper right")

    ax = fig.add_subplot(gs[2, 0])
    draw_scene_side(ax, sc)
    ax.plot(p[:, 0], p[:, 2], color="tab:blue", lw=1.2)
    ax.set_title("side view (x vs height)")

    down = lg.tof_names.index("down") if "down" in lg.tof_names else 0
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, p[:, 2], label="true height", color="k")
    ax.plot(t, fr[:, lg.i_tof + down] * lg.tof_max, label="down ToF (as seen)", color="tab:orange", lw=1)
    ax.plot(t, e["target"][:, 2], "--", color="0.5", label="target")
    shade(ax, t, e["stalled"][:, 1], "tab:orange", "ToF frozen")
    ax.set_ylabel("m")
    ax.set_title("height")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, e["tilt"], label="tilt (deg)", color="tab:purple")
    ax2 = ax.twinx()
    ax2.plot(t, np.linalg.norm(e["angvel_b"], axis=-1), color="tab:red", lw=0.8, alpha=0.7)
    ax2.set_ylabel("|body rate| rad/s", color="tab:red")
    shade(ax, t, e["stalled"][:, 0], "tab:purple", "IMU frozen")
    ax.set_title("attitude")
    ax.legend(fontsize=7, loc="upper left")

    ax = fig.add_subplot(gs[2, 1])
    valid = fr[:, lg.i_flow + 2] > 0.5
    for i, (c, nm) in enumerate([("tab:blue", "x fwd"), ("tab:green", "y left")]):
        ax.plot(t, e["vel_b"][:, i], color=c, lw=1.2, label=f"true v{nm}")
        ax.plot(t, np.where(valid, fr[:, lg.i_flow + i], np.nan), ".", color=c, ms=2, label=f"flow v{nm} (seen)")
    shade(ax, t, ~valid, "0.6", "flow invalid")
    shade(ax, t, e["stalled"][:, 2], "tab:cyan", "flow frozen")
    ax.set_ylabel("m/s")
    ax.set_title("horizontal velocity: truth vs optical flow")
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, e["xy_err"], color="tab:blue", label="xy error")
    ax.plot(t, e["net_dist"][:, 0], color="tab:green", label="distance to keep-in limit")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("m")
    ax.set_title("drift and margin")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, e["cmd_sent"][:, 0], color="k", lw=1, label="thrust sent")
    ax.plot(t, e["cmd_px4"][:, 0], color="tab:orange", lw=1, ls="--", label="thrust PX4 got")
    for i, c in zip(range(1, 4), ["tab:red", "tab:green", "tab:blue"]):
        ax.plot(t, e["cmd_sent"][:, i], color=c, lw=0.7, alpha=0.8, label=["roll", "pitch", "yaw"][i - 1] + " rate")
    shade(ax, t, e["stalled"][:, 3], "tab:red", "command link frozen")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("policy commands (normalised, +-1 = limit)")
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[2, 2])
    for i, nm in enumerate(lg.tof_names):
        if nm in ("down",):
            continue
        ax.plot(t, fr[:, lg.i_tof + i] * lg.tof_max, lw=1, label=f"{nm} seen")
        ax.plot(t, np.minimum(e["tof_true"][:, i], lg.tof_max), lw=0.6, ls=":", color=ax.lines[-1].get_color())
    ax.set_ylabel("m")
    ax.set_title("ToFs: as seen (solid) vs true distance (dotted)")
    ax.legend(fontsize=7, ncol=2)
    for a in fig.axes[2:]:            # time-series panels: mark the end of the flight
        a.axvline(t[-1], color="red", lw=0.8, alpha=0.5)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def animate(lg: Log, f, path, fps=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    n = f["episode"]
    e, sc = lg.ep(n), lg.scene(n)
    t, p, R, fr = e["t"], e["pos"], e["R"], e["frame"]
    L, dt = len(t), lg.dt
    if f["outcome"] == "survived":
        stride, start = max(1, int(round(2.0 / (dt * fps)))), 0           # whole flight at 2x
    else:
        stride, start = max(1, int(round(1.0 / (dt * fps)))), max(0, L - int(8.0 / dt))   # last 8 s, real time
    idx = list(range(start, L, stride)) + [L - 1] * int(fps * 1.0)          # hold the last frame 1 s
    fig, (axt, axs) = plt.subplots(1, 2, figsize=(12.8, 6.4), dpi=100, gridspec_kw=dict(width_ratios=[1, 1]))
    draw_scene_top(axt, sc)
    draw_scene_side(axs, sc)
    axt.plot(*e["target"][0, :2], "+", color="k", ms=12)
    axs.axhline(e["target"][0, 2], ls=":", color="0.5", lw=0.8)
    trail_t, = axt.plot([], [], color="tab:blue", lw=1)
    trail_s, = axs.plot([], [], color="tab:blue", lw=1)
    dot_t, = axt.plot([], [], "o", color="tab:blue", ms=8)
    dot_s, = axs.plot([], [], "o", color="tab:blue", ms=8)
    head, = axt.plot([], [], color="k", lw=2)
    beams = {nm: (axs if nm in ("down", "up") else axt).plot([], [], lw=1.5)[0] for nm in lg.tof_names}
    txt = fig.text(0.01, 0.01, "", family="monospace", fontsize=9, va="bottom")
    fig.suptitle(f"episode {n} | {f['scenario']} | {f['outcome'].upper()} at {f['end_s']:.2f} s | {f['mode']}",
                 fontsize=11)
    axt.set_title("top view")
    axs.set_title("side view (x, height)")
    fig.subplots_adjust(bottom=0.14, top=0.9, wspace=0.15)
    tofp, tofd = lg.d["tof_pos_b"], lg.d["tof_dir_b"]
    down = lg.tof_names.index("down") if "down" in lg.tof_names else 0
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    for k in idx:
        a = max(0, k - int(6.0 / dt))
        trail_t.set_data(p[a:k + 1, 0], p[a:k + 1, 1])
        trail_s.set_data(p[a:k + 1, 0], p[a:k + 1, 2])
        dot_t.set_data([p[k, 0]], [p[k, 1]])
        dot_s.set_data([p[k, 0]], [p[k, 2]])
        fwd = R[k] @ np.array([0.18, 0, 0])
        head.set_data([p[k, 0], p[k, 0] + fwd[0]], [p[k, 1], p[k, 1] + fwd[1]])
        for i, nm in enumerate(lg.tof_names):
            o = p[k] + R[k] @ tofp[i]
            dirw = R[k] @ tofd[i]
            seen = fr[k, lg.i_tof + i] * lg.tof_max
            miss = seen >= lg.tof_max - 1e-3
            end = o + dirw * min(seen, 1.5 if miss else seen)
            ln = beams[nm]
            if nm in ("down", "up"):
                ln.set_data([o[0], end[0]], [o[2], end[2]])
            else:
                ln.set_data([o[0], end[0]], [o[1], end[1]])
            ln.set_color("0.6" if miss else "tab:orange")
            ln.set_linestyle(":" if miss else "-")
        st = e["stalled"][k]
        frozen = [nm for nm, s in zip(["IMU", "ToF", "FLOW", "CMD"], st) if s]
        valid = fr[k, lg.i_flow + 2] > 0.5
        txt.set_text(f"t={t[k]:5.2f}s  height {p[k, 2]:.2f} m (down ToF sees {fr[k, lg.i_tof + down] * lg.tof_max:.2f})  "
                     f"tilt {e['tilt'][k]:4.1f} deg  xy err {e['xy_err'][k]:.2f} m  "
                     f"flow {'ok ' if valid else 'INVALID'}  frozen: {','.join(frozen) or '-'}\n"
                     f"orange beams = ToF readings as the policy saw them; grey dotted = 'no target'")
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    writer.close()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--fails", type=int, default=12, help="failure episodes to plot/animate")
    ap.add_argument("--drifts", type=int, default=4, help="worst-drift survivors to plot/animate")
    ap.add_argument("--no_video", action="store_true")
    a = ap.parse_args()
    lg = Log(a.npz)
    out = os.path.dirname(os.path.abspath(a.npz))
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out, "videos"), exist_ok=True)
    feats = [features(lg, n) for n in range(lg.N)]
    txt = report(lg, feats, out)
    picks = select(feats, a.fails, a.drifts)
    sel_lines = []
    for f in picks:
        tag = "drift" if f["outcome"] == "survived" else f["outcome"]
        name = f"ep{f['episode']:04d}_{tag}_{f['end_s']:05.2f}s"
        dashboard(lg, f, os.path.join(out, "plots", name + ".png"))
        if not a.no_video:
            animate(lg, f, os.path.join(out, "videos", name + ".mp4"))
        sel_lines.append(f"  {name}: {f['mode']}" + (f" | {'; '.join(f['hints'])}" if f["hints"] else ""))
    with open(os.path.join(out, "report.txt"), "a") as fh:
        fh.write("\n".join(sel_lines) + "\n")
    json.dump(feats, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(txt)
    print("\n".join(sel_lines))
    print(f"\nwrote {out}/report.txt, summary.json, plots/ ({len(picks)}), "
          f"{'videos/' if not a.no_video else 'no videos'}")


if __name__ == "__main__":
    main()
