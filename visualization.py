"""Shared helpers for headless playback videos: target markers in the scene and a
HUD (altitude vs target, error, drift trace) drawn onto each rendered frame.

Import only after the Isaac Sim app is running (markers need the stage).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


def make_target_markers() -> VisualizationMarkers:
    """Red ball at the hover target, translucent green disc around it, and a thin
    pole down to the ground so sideways drift reads against a fixed reference."""
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/HoverTarget",
        markers={
            "ball": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.1, 0.1), emissive_color=(0.6, 0.0, 0.0)
                ),
            ),
            "disc": sim_utils.CylinderCfg(
                radius=0.15,
                height=0.004,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.9, 0.3), emissive_color=(0.0, 0.3, 0.1), opacity=0.35
                ),
            ),
            "pole": sim_utils.CylinderCfg(
                radius=0.004,
                height=1.0,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9), opacity=0.6),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def show_targets(markers: VisualizationMarkers, env_origins: torch.Tensor, target_z: float):
    n = env_origins.shape[0]
    target = env_origins.clone()
    target[:, 2] += target_z
    pole = env_origins.clone()
    pole[:, 2] += target_z / 2.0
    translations = torch.cat([target, target, pole])
    indices = torch.tensor([0] * n + [1] * n + [2] * n)
    scales = torch.ones(3 * n, 3)
    scales[2 * n:, 2] = target_z  # pole cylinder is 1 m tall; stretch to the target height
    markers.visualize(translations=translations, scales=scales, marker_indices=indices)


class Hud:
    """Text block, altitude strip chart and top-down drift plot for one drone."""

    def __init__(self, title: str, target_z: float, seconds: float):
        self.title = title
        self.target_z = target_z
        self.seconds = seconds
        self.t, self.z, self.xy = [], [], []

    def add(self, t: float, pos_local: torch.Tensor):
        self.t.append(t)
        self.z.append(float(pos_local[2]))
        self.xy.append((float(pos_local[0]), float(pos_local[1])))

    def draw(self, frame: np.ndarray, extra_lines: list[str] | None = None) -> np.ndarray:
        img = np.ascontiguousarray(frame[..., :3])
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        white, red, green, grey = (255, 255, 255), (230, 40, 40), (60, 220, 100), (150, 150, 150)

        # Text block, top left.
        z, (x, y) = self.z[-1], self.xy[-1]
        lines = [
            self.title,
            f"t        {self.t[-1]:5.2f} s",
            f"altitude {z:5.3f} m   target {self.target_z:.2f} m",
            f"error    {z - self.target_z:+6.3f} m",
            f"drift    {np.hypot(x, y):5.3f} m  (x {x:+.2f}, y {y:+.2f})",
        ] + (extra_lines or [])
        box_h = 14 + 26 * len(lines)
        box_w = 30 + max(cv2.getTextSize(line, font, 0.6, 1)[0][0] for line in lines)
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (20, 36 + 26 * i), font, 0.6, white, 1, cv2.LINE_AA)

        # Altitude strip chart, bottom left.
        cx, cy, cw, ch = 10, h - 190, 420, 180
        overlay = img.copy()
        cv2.rectangle(overlay, (cx, cy), (cx + cw, cy + ch), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        z_max = max(1.5 * self.target_z, max(self.z) * 1.1)
        to_px = lambda tt, zz: (  # noqa: E731
            int(cx + 40 + (cw - 50) * tt / self.seconds),
            int(cy + ch - 20 - (ch - 35) * zz / z_max),
        )
        ty = to_px(0, self.target_z)[1]
        for xs in range(cx + 40, cx + cw - 10, 12):
            cv2.line(img, (xs, ty), (xs + 6, ty), red, 1)
        cv2.putText(img, "altitude (m)", (cx + 45, cy + 16), font, 0.45, white, 1, cv2.LINE_AA)
        cv2.putText(img, f"{self.target_z:.1f}", (cx + 6, ty + 4), font, 0.4, red, 1, cv2.LINE_AA)
        cv2.putText(img, "0", (cx + 20, cy + ch - 16), font, 0.4, grey, 1, cv2.LINE_AA)
        cv2.putText(img, f"{self.seconds:.0f}s", (cx + cw - 40, cy + ch - 4), font, 0.4, grey, 1, cv2.LINE_AA)
        pts = np.array([to_px(tt, zz) for tt, zz in zip(self.t, self.z)], np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, green, 2, cv2.LINE_AA)

        # Top-down drift plot, bottom right (+-0.5 m box, target at center).
        s, half = 180, 0.5
        dx, dy = w - s - 10, h - s - 10
        overlay = img.copy()
        cv2.rectangle(overlay, (dx, dy), (dx + s, dy + s), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        to_xy = lambda px_, py_: (int(dx + s / 2 - py_ / half * s / 2), int(dy + s / 2 - px_ / half * s / 2))  # noqa: E731
        c = to_xy(0, 0)
        cv2.circle(img, c, int(0.15 / half * s / 2), (60, 160, 80), 1, cv2.LINE_AA)
        cv2.drawMarker(img, c, red, cv2.MARKER_CROSS, 12, 1)
        trail = np.array([to_xy(a, b) for a, b in self.xy], np.int32)
        if len(trail) > 1:
            cv2.polylines(img, [trail], False, green, 1, cv2.LINE_AA)
        cv2.circle(img, to_xy(*self.xy[-1]), 4, white, -1)
        cv2.putText(img, "top view +-0.5 m", (dx + 6, dy + 16), font, 0.42, white, 1, cv2.LINE_AA)
        cv2.putText(img, "x up, y left", (dx + 6, dy + s - 8), font, 0.38, grey, 1, cv2.LINE_AA)
        return img
