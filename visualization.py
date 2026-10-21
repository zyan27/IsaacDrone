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

    def __init__(self, title: str, target_z: float, seconds: float, room_half: float | None = None):
        self.title = title
        self.room_half = room_half
        self.target_z = target_z
        self.seconds = seconds
        self.t, self.z, self.xy = [], [], []

    def add(self, t: float, pos_local: torch.Tensor):
        self.t.append(t)
        self.z.append(float(pos_local[2]))
        self.xy.append((float(pos_local[0]), float(pos_local[1])))

    def draw(self, frame: np.ndarray, extra_lines: list[str] | None = None) -> np.ndarray:
        """Scene on the left, readouts on a side panel so nothing covers the drone."""
        scene = np.ascontiguousarray(frame[..., :3])
        h = scene.shape[0]
        pw = 480
        img = np.full((h, pw, 3), 24, np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        white, red, green, grey = (255, 255, 255), (230, 40, 40), (60, 220, 100), (150, 150, 150)

        # Text block.
        z, (x, y) = self.z[-1], self.xy[-1]
        lines = [
            f"t        {self.t[-1]:5.2f} s",
            f"altitude {z:5.3f} m   target {self.target_z:.2f} m",
            f"error    {z - self.target_z:+6.3f} m",
            f"off ctr  {np.hypot(x, y):5.3f} m  (x {x:+.2f}, y {y:+.2f})",
        ] + (extra_lines or [])
        cv2.putText(img, self.title, (14, 26), font, 0.52, white, 1, cv2.LINE_AA)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (14, 52 + 21 * i), font, 0.46, white, 1, cv2.LINE_AA)
        top = 52 + 21 * len(lines) + 4

        # Altitude strip chart.
        cx, cy, cw, ch = 10, top, pw - 20, 150
        cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), (60, 60, 60), 1)
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

        # Top-down plot of the room (or +-0.5 m), target at center.
        half = self.room_half or 0.5
        s = min(pw - 20, h - (cy + ch) - 20)
        dx, dy = (pw - s) // 2, cy + ch + 10
        cv2.rectangle(img, (dx, dy), (dx + s, dy + s), (60, 60, 60), 1)
        to_xy = lambda px_, py_: (int(dx + s / 2 - py_ / half * s / 2), int(dy + s / 2 - px_ / half * s / 2))  # noqa: E731
        c = to_xy(0, 0)
        cv2.circle(img, c, max(2, int(0.15 / half * s / 2)), (60, 160, 80), 1, cv2.LINE_AA)
        cv2.drawMarker(img, c, red, cv2.MARKER_CROSS, 12, 1)
        trail = np.array([to_xy(a, b) for a, b in self.xy], np.int32)
        if len(trail) > 1:
            cv2.polylines(img, [trail], False, green, 1, cv2.LINE_AA)
        cv2.circle(img, to_xy(*self.xy[-1]), 4, white, -1)
        label = f"top view, walls at +-{half:.1f} m" if self.room_half else "top view +-0.5 m"
        cv2.putText(img, label, (dx + 6, dy + 16), font, 0.42, white, 1, cv2.LINE_AA)
        cv2.putText(img, "x up, y left", (dx + 6, dy + s - 8), font, 0.38, grey, 1, cv2.LINE_AA)
        return np.concatenate([scene, img], axis=1)
