"""Live view of PX4 PPO training, with no rendering cost.

Every few iterations it rewrites:
  <out_dir>/train_live.png   top:   height vs command for 4 of the training drones, last 20 s
                             left:  learning curves (step reward, share of drones off the floor)
                             right: the same 4 drones from above (net walls drawn at +-1.5 m)
  <out_dir>/train_live.csv   one row per iteration
Open the PNG in the Open OnDemand file browser (refresh it) or ask Claude to look at it.
"""
import csv
import os

import cv2
import numpy as np
import torch

W, H = 1200, 820
COLORS = [(214, 120, 31), (44, 160, 44), (40, 39, 214), (189, 103, 148)]  # BGR
GRID, TEXT, BG = (220, 220, 220), (40, 40, 40), (255, 255, 255)


def _polyline(img, xs, ys, color, thick=2):
    """Draw a line, broken at NaNs (episode resets)."""
    ok = np.isfinite(xs) & np.isfinite(ys)
    start = None
    for i in range(len(xs) + 1):
        if i < len(xs) and ok[i]:
            start = i if start is None else start
            continue
        if start is not None and i - start > 1:
            pts = np.stack([xs[start:i], ys[start:i]], -1).round().astype(np.int32)
            cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)
        start = None


def _text(img, s, x, y, scale=0.5, color=TEXT):
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class TrainMonitor:
    def __init__(self, out_dir: str, dt: float, n_show: int = 4, window_s: float = 20.0):
        os.makedirs(out_dir, exist_ok=True)
        self.png = os.path.join(out_dir, "train_live.png")
        self.csv_file = open(os.path.join(out_dir, "train_live.csv"), "w", newline="")
        self.rows = csv.writer(self.csv_file)
        self.rows.writerow(["iter", "elapsed_s", "step_reward", "mean_ep_len_s", "crashes", "off_floor", "z_err_m"])
        self.dt, self.n = dt, n_show
        self.trace = np.full((int(window_s / dt), n_show, 4), np.nan, np.float32)  # height, target, x, y
        self.hist = []  # (iter, step_reward, off_floor)
        self._reset_iter()

    def _reset_iter(self):
        self.up = self.off = self.zerr = 0.0

    def step(self, env, done: torch.Tensor):
        """Call after every env.step (state is already auto-reset for done envs)."""
        h, tz = env._height(), env.target_z
        up = tz > env.cfg.floor_target_m
        self.up = self.up + up.sum()
        self.off = self.off + (up & (h > env.cfg.floor_height_m)).sum()
        self.zerr = self.zerr + torch.where(up, (h - tz).abs(), torch.zeros_like(h)).sum()
        xy = env.drone.data.root_pos_w[: self.n, :2] - env.scene.env_origins[: self.n, :2]
        row = torch.cat([h[: self.n, None], tz[: self.n, None], xy], -1)
        row[done[: self.n]] = float("nan")
        self.trace = np.roll(self.trace, -1, axis=0)
        self.trace[-1] = row.cpu().numpy()

    def end_iter(self, it, elapsed, step_reward, ep_len_s, crashes):
        up = max(float(self.up), 1.0)
        off, zerr = float(self.off) / up, float(self.zerr) / up
        self._reset_iter()
        self.hist.append((it, step_reward, off))
        self.rows.writerow([it, f"{elapsed:.0f}", f"{step_reward:.4f}", f"{ep_len_s:.2f}", crashes, f"{off:.3f}", f"{zerr:.3f}"])
        self.csv_file.flush()
        return off, zerr

    def draw(self, title: str):
        img = np.full((H, W, 3), BG, np.uint8)
        _text(img, title, 20, 28, 0.7)

        # Height vs command, last window.
        x0, y0, x1, y1 = 70, 50, W - 20, 400
        zmax = 1.4
        T = len(self.trace)
        tx = x0 + (x1 - x0) * np.arange(T) / (T - 1)
        zy = lambda z: y1 - (y1 - y0) * np.clip(z, 0, zmax) / zmax  # noqa: E731
        for z in np.arange(0, zmax + 1e-6, 0.2):
            cv2.line(img, (x0, int(zy(z))), (x1, int(zy(z))), GRID, 1)
            _text(img, f"{z:.1f}", 25, zy(z) + 5, 0.45)
        _text(img, f"height (solid) vs command (thin), m   last {T * self.dt:.0f} s", x0, y0 - 6, 0.5)
        for i in range(self.n):
            _polyline(img, tx, zy(self.trace[:, i, 1]), tuple(c // 2 + 120 for c in COLORS[i]), 1)
            _polyline(img, tx, zy(self.trace[:, i, 0]), COLORS[i], 2)

        # Learning curves.
        x0, y0, x1, y1 = 70, 460, 640, 790
        _text(img, "step reward (blue, max 3.5)   off the floor when commanded (green, 0-100%)", x0 - 50, y0 - 10, 0.45)
        cv2.rectangle(img, (x0, y0), (x1, y1), GRID, 1)
        if len(self.hist) > 1:
            it = np.array([h[0] for h in self.hist], np.float32)
            ix = x0 + (x1 - x0) * (it - it[0]) / max(it[-1] - it[0], 1)
            rew = np.array([h[1] for h in self.hist], np.float32)
            off = np.array([h[2] for h in self.hist], np.float32)
            lo = min(rew.min(), 0.0)
            _polyline(img, ix, y1 - (y1 - y0) * (rew - lo) / (3.5 - lo), (200, 110, 30), 2)
            _polyline(img, ix, y1 - (y1 - y0) * off, (40, 160, 40), 2)
            _text(img, f"iter {int(it[0])}", x0, y1 + 20, 0.45)
            _text(img, f"iter {int(it[-1])}: reward {rew[-1]:.2f}, off floor {off[-1] * 100:.0f}%", x1 - 330, y1 + 20, 0.45)

        # Top-down xy.
        cx, cy, half_px, half_m = 920, 625, 165, 1.5
        cv2.rectangle(img, (cx - half_px, cy - half_px), (cx + half_px, cy + half_px), (150, 150, 150), 1)
        cv2.drawMarker(img, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 12, 1)
        _text(img, "from above (+-1.5 m, cross = takeoff spot)", cx - half_px, cy - half_px - 10, 0.45)
        for i in range(self.n):
            _polyline(img, cx + half_px * self.trace[:, i, 2] / half_m, cy - half_px * self.trace[:, i, 3] / half_m, COLORS[i], 2)

        tmp = self.png + ".tmp.png"
        cv2.imwrite(tmp, img)
        os.replace(tmp, self.png)  # never leave a half-written PNG for the viewer
