"""Randomised flight scenes for sensor simulation (pure torch, no Isaac imports).

Every episode each env draws one scenario, so the policy sees many contexts:

  net_only     the safety net with nothing within ToF range behind it
  net_in_room  the net inside a room: beams that miss the net hit the room's
               walls/ceiling or clutter (people, furniture) between net and walls
  room         a room with solid walls and furniture along them; no net
  open         a big hall: no walls within sensor range, only far clutter
  fixed_net    the original fixed arena from environments/arena.py (for
               comparisons, the old tests and videos)

Geometry, all relative to the env origin (floor at z = 0):
  keep-in box  where the drone must stay (net / furniture line / virtual fence);
               leaving it (minus net_margin) ends the episode, as before
  net          axis-aligned box of netting: ToF beams may pass through it
  room         axis-aligned box of solid walls (+ ceiling)
  clutter      K axis-aligned boxes outside the keep-in box (pure sensor clutter;
               the drone can't reach them without leaving the keep-in box first)

Nothing here is simulated by PhysX: the scene only affects the ToF readings and
the keep-in boundary, so changing it every episode costs nothing.
"""
from __future__ import annotations

import torch

SCENARIOS = ("net_only", "net_in_room", "room", "open", "fixed_net")
INF = float("inf")


def _u(lo, hi, *shape, device):
    return torch.empty(*shape, device=device).uniform_(lo, hi)


def empty_scenes(n: int, k: int, device) -> dict:
    z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
    return {
        "scen": torch.zeros(n, dtype=torch.long, device=device),
        "keep_half": z(n, 2), "keep_h": z(n),
        "net_on": torch.zeros(n, dtype=torch.bool, device=device), "net_half": z(n, 2), "net_h": z(n),
        "room_on": torch.zeros(n, dtype=torch.bool, device=device), "room_half": z(n, 2), "room_h": z(n),
        "clut_c": z(n, k, 3), "clut_half": z(n, k, 3),
        "clut_on": torch.zeros(n, k, dtype=torch.bool, device=device),
    }


def sample_scenes(n: int, cfg, device, scenario: str = "random") -> dict:
    """Draw n scenes. `cfg` supplies the ranges (see DroneHoverEnvCfg 'scene_*')."""
    k = cfg.scene_clutter_max
    s = empty_scenes(n, k, device)
    u = lambda lo, hi, *shape: _u(lo, hi, *shape, device=device)  # noqa: E731

    if scenario == "random":
        probs = torch.tensor(cfg.scene_probs, device=device, dtype=torch.float)
        s["scen"] = torch.multinomial(probs / probs.sum(), n, replacement=True)
    else:
        s["scen"][:] = SCENARIOS.index(scenario)
    sc = s["scen"]
    is_ = {name: sc == i for i, name in enumerate(SCENARIOS)}

    # ---- nets (net_only, net_in_room): random size, ceiling sometimes open --------
    has_net = is_["net_only"] | is_["net_in_room"]
    net_half = torch.stack([u(*cfg.scene_net_side, n), u(*cfg.scene_net_side, n)], -1) / 2
    net_h = u(*cfg.scene_net_height, n) * (torch.rand(n, device=device) < cfg.scene_net_ceiling_prob)
    s["net_on"] = has_net
    s["net_half"] = torch.where(has_net[:, None], net_half, s["net_half"])
    s["net_h"] = torch.where(has_net, net_h, s["net_h"])

    # ---- fixed_net: the original arena ------------------------------------------
    fx = is_["fixed_net"]
    s["net_on"] |= fx
    s["net_half"][fx] = torch.tensor([cfg.arena_size_xy[0] / 2, cfg.arena_size_xy[1] / 2], device=device)
    s["net_h"][fx] = max(cfg.arena_height, 0.0)

    # ---- rooms ----------------------------------------------------------------
    # net_in_room: room walls a random gap beyond the net
    nir = is_["net_in_room"]
    gap = torch.stack([u(*cfg.scene_room_gap, n), u(*cfg.scene_room_gap, n)], -1)
    room_h_nir = torch.maximum(s["net_h"], u(*cfg.scene_room_height, n))
    # room: solid walls, furniture band along them; keep-in = inside the band
    rm = is_["room"]
    room_half_rm = torch.stack([u(*cfg.scene_room_side, n), u(*cfg.scene_room_side, n)], -1) / 2
    s["room_on"] = nir | rm
    s["room_half"] = torch.where(nir[:, None], s["net_half"] + gap,
                                 torch.where(rm[:, None], room_half_rm, s["room_half"]))
    s["room_h"] = torch.where(nir, room_h_nir, torch.where(rm, u(*cfg.scene_room_height, n), s["room_h"]))

    # ---- keep-in box -------------------------------------------------------------
    band = u(*cfg.scene_furniture_band, n)
    open_half = torch.stack([u(*cfg.scene_open_side, n), u(*cfg.scene_open_side, n)], -1) / 2
    keep_half = torch.where(s["net_on"][:, None], s["net_half"],
                            torch.where(rm[:, None], (s["room_half"] - band[:, None]).clamp(min=0.9),
                                        open_half))
    # Height of the keep-in box: net ceiling, else room ceiling, else a virtual 4 m fence.
    keep_h = torch.where(s["net_on"] & (s["net_h"] > 0), s["net_h"],
                         torch.where(s["room_on"], s["room_h"], torch.full_like(s["net_h"], cfg.scene_virtual_ceiling)))
    s["keep_half"], s["keep_h"] = keep_half, keep_h

    # ---- clutter: boxes outside the keep-in box, within `scene_clutter_reach` of it ------
    clut = ~fx[:, None] & (torch.rand(n, k, device=device) < cfg.scene_clutter_prob)
    side = torch.randint(0, 4, (n, k), device=device)              # +x, -x, +y, -y
    half = torch.stack([u(*cfg.scene_clutter_half_xy, n, k), u(*cfg.scene_clutter_half_xy, n, k),
                        u(*cfg.scene_clutter_half_z, n, k)], -1)
    out = u(0.05, cfg.scene_clutter_reach, n, k)                    # distance beyond keep-in
    kx, ky = keep_half[:, None, 0], keep_half[:, None, 1]
    along_x = u(-1, 1, n, k) * kx
    along_y = u(-1, 1, n, k) * ky
    sx = torch.where(side == 0, 1.0, torch.where(side == 1, -1.0, 0.0))
    sy = torch.where(side == 2, 1.0, torch.where(side == 3, -1.0, 0.0))
    cx = torch.where(sx != 0, sx * (kx + out + half[..., 0]), along_x)
    cy = torch.where(sy != 0, sy * (ky + out + half[..., 1]), along_y)
    # stay inside the room if there is one (else the clutter would be behind a wall)
    rx, ry = s["room_half"][:, None, 0], s["room_half"][:, None, 1]
    in_room = s["room_on"][:, None]
    cx = torch.where(in_room, cx.clamp(-(rx - half[..., 0]), rx - half[..., 0]), cx)
    cy = torch.where(in_room, cy.clamp(-(ry - half[..., 1]), ry - half[..., 1]), cy)
    s["clut_c"] = torch.stack([cx, cy, half[..., 2]], -1)            # standing on the floor
    s["clut_half"] = half
    # drop clutter that would poke into the keep-in box after clamping
    inside = ((cx.abs() - half[..., 0]) < kx) & ((cy.abs() - half[..., 1]) < ky)
    s["clut_on"] = clut & ~inside
    return s


def assign(dst: dict, src: dict, ids: torch.Tensor):
    for key, v in src.items():
        dst[key][ids] = v


def _box_exit(o, d, half_xy, h, on):
    """Distance along each ray to the inside faces of an axis-aligned box centred on
    the origin (walls at +-half, ceiling at h if h > 0). o, d: (N, S, 3)."""
    N, S = o.shape[:2]
    t = torch.full((N, S), INF, device=o.device)
    dd = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    for axis in (0, 1):
        hb = half_xy[:, None, axis]
        for sign in (1.0, -1.0):
            ti = (sign * hb - o[..., axis]) / dd[..., axis]
            t = torch.minimum(t, torch.where(ti > 0, ti, torch.full_like(ti, INF)))
    tc = (h[:, None] - o[..., 2]) / dd[..., 2]
    t = torch.minimum(t, torch.where((tc > 0) & (h[:, None] > 0), tc, torch.full_like(tc, INF)))
    return torch.where(on[:, None], t, torch.full_like(t, INF))


def scene_ranges(o: torch.Tensor, d: torch.Tensor, s: dict):
    """Ray distances from origins o (N,S,3) along unit directions d (N,S,3), env-local.
    Returns dict of (N,S) tensors: floor, net, room, clutter (inf = no hit)."""
    dd = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    tf = -o[..., 2] / dd[..., 2]
    floor = torch.where(tf > 0, tf, torch.full_like(tf, INF))
    net = _box_exit(o, d, s["net_half"], s["net_h"], s["net_on"])
    room = _box_exit(o, d, s["room_half"], s["room_h"], s["room_on"])
    # clutter: ray / AABB slab test, (N,S,K)
    c = s["clut_c"][:, None]          # (N,1,K,3)
    e = s["clut_half"][:, None]
    oo, di = o[:, :, None], dd[:, :, None]
    t1, t2 = (c - e - oo) / di, (c + e - oo) / di
    tmin = torch.minimum(t1, t2).amax(-1)
    tmax = torch.maximum(t1, t2).amin(-1)
    hit = (tmax >= tmin) & (tmin > 0) & s["clut_on"][:, None]
    clutter = torch.where(hit, tmin, torch.full_like(tmin, INF)).amin(-1) if c.shape[2] else torch.full_like(floor, INF)
    return {"floor": floor, "net": net, "room": room, "clutter": clutter}
