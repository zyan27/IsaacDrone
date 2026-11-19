"""DD24 policy core: everything the MAVROS node does except ROS (numpy only, testable).

Rebuilds the policy's 176 inputs exactly as environments/drone_hover_env.py does
(see deploy/INTERFACE.md), runs the exported actor, maps its output to a PX4
thrust + body-rate setpoint, and decides whether it is safe to send.

All times are seconds on ONE clock (the node's), taken when messages ARRIVE.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

FRAME_NAMES = (["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "grav_x", "grav_y", "grav_z"]
               + [f"tof_{n}" for n in ("down", "up", "front", "left", "right")]
               + ["prev_thrust", "prev_roll_rate", "prev_pitch_rate", "prev_yaw_rate",
                  "flow_vx", "flow_vy", "flow_valid", "target_height"])

# Values sensor drivers use for "nothing measured". The Duckietown VL53L1X driver
# publishes 999 when nothing is in range (seen on the top ToF facing the ceiling).
TOF_SENTINEL_MIN = 900.0


# ------------------------------------------------------------------ policy
class Policy:
    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta_json"]))
        self.W = [z[f"W{i}"].astype(np.float64) for i in range(3)]
        self.b = [z[f"b{i}"].astype(np.float64) for i in range(3)]
        self.mean = z["norm_mean"].astype(np.float64)
        self.std = np.maximum(np.sqrt(z["norm_var"].astype(np.float64)), self.meta["norm_min_std"])
        self.clip = self.meta["norm_clip"]

    def normalise(self, obs):
        return np.clip((obs - self.mean) / self.std, -self.clip, self.clip)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Raw 176 inputs -> action in [0, 1]^4 (same as training's clamp)."""
        h = self.normalise(obs)
        for i in range(2):
            h = self.W[i] @ h + self.b[i]
            h = np.where(h > 0, h, np.expm1(np.minimum(h, 0)))          # ELU
        out = self.W[2] @ h + self.b[2]
        if self.meta["actor_out"] == "sigmoid":
            out = 1.0 / (1.0 + np.exp(-out))
        return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------- conversions
def clean_tof(r: float, msg_max_range: float | None, tof_min: float, tof_max: float) -> float:
    """Driver range -> metres as in training: 'no target' -> tof_max, clamp to [tof_min, tof_max].
    No target = NaN / inf / negative / driver sentinel (999) / beyond the driver's own max_range."""
    if r is None or not math.isfinite(r) or r < 0 or r >= TOF_SENTINEL_MIN:
        return tof_max
    if msg_max_range is not None and math.isfinite(msg_max_range) and msg_max_range > 0 and r > msg_max_range + 0.05:
        return tof_max
    return min(max(r, tof_min), tof_max)


def roll_pitch_deg(g):
    """Signed roll / pitch (deg, FLU) from the body gravity direction.
    roll > 0: right side down; pitch > 0: nose down."""
    return math.degrees(math.atan2(-g[1], -g[2])), math.degrees(math.atan2(g[0], -g[2]))


def gravity_dir_body(q_xyzw) -> np.ndarray:
    """MAVROS orientation (ENU world -> FLU body, x y z w) -> gravity direction in body
    frame, R^T (0,0,-1); (0,0,-1) when level. Matches projected_gravity_b in the sim."""
    x, y, z, w = q_xyzw
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    # third row of R (body->world) = world z-axis expressed in body
    up_b = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
    return -up_b


@dataclass
class FlowCfg:
    height_range: tuple = (0.08, 4.0)
    max_rate: float = 2.0             # rad/s image rate the camera can track
    quality_min: int = 1
    use_gyro_fields: bool = True      # subtract integrated_*gyro (PX4 convention)
    sign_x: float = 1.0               # flip if the bench check shows forward = negative
    sign_y: float = 1.0
    swap_xy: bool = False


def flow_velocity(msg: dict, height: float | None, cfg: FlowCfg):
    """mavros_msgs/OpticalFlowRad -> (vx, vy, valid) in body FLU m/s, like the sim input.

    MAVLink OPTICAL_FLOW_RAD: +x body motion gives +integrated_y; +y (FRD right)
    motion gives -integrated_x; *_gyro fields are the rotation over the same window.
    So FLU: vx = rate_y * h, vy = rate_x * h (rate = (flow - gyro) / dt)."""
    dt = msg.get("integration_time_us", 0) * 1e-6
    if height is None or dt <= 0:
        return 0.0, 0.0, 0.0
    fx, fy = msg["integrated_x"], msg["integrated_y"]
    if cfg.use_gyro_fields:
        fx, fy = fx - msg["integrated_xgyro"], fy - msg["integrated_ygyro"]
    rx, ry = fx / dt, fy / dt
    vx, vy = ry * height, rx * height
    if cfg.swap_xy:
        vx, vy = vy, vx
    vx, vy = cfg.sign_x * vx, cfg.sign_y * vy
    lo, hi = cfg.height_range
    ok = (msg.get("quality", 0) >= cfg.quality_min and lo < height < hi
          and math.hypot(rx, ry) < cfg.max_rate and math.isfinite(vx) and math.isfinite(vy))
    return (vx, vy, 1.0) if ok else (0.0, 0.0, 0.0)


# --------------------------------------------------------------------- core
@dataclass
class SafetyCfg:
    imu_neutral_s: float = 0.10       # IMU older than this -> neutral setpoint (zero rates, centre thrust)
    tof_down_neutral_s: float = 0.20
    stop_s: float = 0.50              # any of the above older than this -> stop sending (PX4 failsafe)
    flow_invalid_s: float = 0.30      # flow older than this -> treated as invalid (trained case)
    tof_other_stale_s: float = 0.50   # side/up ToF older than this -> "no target" (trained case)
    max_tilt_deg: float = 60.0        # tilt beyond this -> stop sending
    max_height_m: float = 2.2         # down ToF above this -> stop sending
    thrust_gain: float = 1.0          # multiply the thrust setpoint (only if the real hover throttle needs it)


@dataclass
class Tick:
    mode: str                         # RUN | NEUTRAL | STOP | WAIT
    reason: str
    cmd: np.ndarray                   # policy command a in [-1,1]^4 (what the next frame sees)
    thrust: float | None = None       # setpoint to send (None when STOP)
    rates: np.ndarray | None = None
    frame: np.ndarray | None = None
    warnings: list = field(default_factory=list)


class Core:
    def __init__(self, policy: Policy, target_height: float, flow_cfg: FlowCfg | None = None,
                 safety: SafetyCfg | None = None):
        m = policy.meta
        self.p, self.m = policy, m
        self.target_height = float(target_height)
        self.flow_cfg = flow_cfg or FlowCfg(height_range=tuple(m["flow_height_range"]), max_rate=m["flow_max_rate"])
        self.safety = safety or SafetyCfg()
        self.tof_names = list(m["tof_names"])
        self.tof_min, self.tof_max = m["tof_min_range"], m["tof_max_range"]
        self.max_rates = np.array(m["ctbr_max_rates"])
        self.H = m["obs_history"]
        self.hist: deque = deque(maxlen=self.H)
        self.cmd = np.zeros(4)
        self.imu = None                     # (t, accel3, gyro3, quat_xyzw)
        self.tof = {n: None for n in self.tof_names}   # name -> (t, metres)
        self.flow = None                    # (t, vx, vy, valid)
        self.active = False
        self.disabled = set()               # ToFs switched off (e.g. view blocked): always "no target"

    # ---- message inputs
    def on_imu(self, t, accel, gyro, quat_xyzw):
        acc = np.asarray(accel, float)
        clip = self.m.get("accel_clip")
        if clip:                            # same clip as training (newer policies)
            acc = np.clip(acc, -clip, clip)
        self.imu = (t, acc, np.asarray(gyro, float), tuple(quat_xyzw))

    def on_tof(self, t, name, r, msg_max_range=None):
        self.tof[name] = (t, clean_tof(r, msg_max_range, self.tof_min, self.tof_max))

    def down_height(self, t_now=None):
        d = self.tof.get("down")
        if d is None or (t_now is not None and t_now - d[0] > self.safety.stop_s):
            return None
        return d[1]

    def on_flow(self, t, msg: dict):
        h = self.down_height()
        h = None if h is None or h >= self.tof_max else h
        self.flow = (t,) + flow_velocity(msg, h, self.flow_cfg)

    # ---- one 50 Hz tick
    def neutral(self):
        return float(np.clip(self.m["ctbr_thrust_center"] * self.safety.thrust_gain, 0, 1)), np.zeros(3)

    def build_frame(self, t):
        _, acc, gyro, q = self.imu
        g = gravity_dir_body(q)
        tofs = []
        for n in self.tof_names:
            v = self.tof[n]
            stale = v is None or n in self.disabled or (n != "down" and t - v[0] > self.safety.tof_other_stale_s)
            tofs.append((self.tof_max if stale else v[1]) / self.tof_max)
        if self.flow is None or t - self.flow[0] > self.safety.flow_invalid_s:
            flow = (0.0, 0.0, 0.0)
        else:
            flow = self.flow[1:]
        return np.concatenate([acc, gyro, g, tofs, self.cmd, flow, [self.target_height]]).astype(np.float64)

    def tick(self, t, active: bool) -> Tick:
        s = self.safety
        warn = []
        # Freshness of what the policy can't fly without.
        if self.imu is None or self.tof.get("down") is None:
            self.active = False
            return Tick("STOP", "waiting for IMU / down ToF", self.cmd.copy())
        age_imu, age_down = t - self.imu[0], t - self.tof["down"][0]
        if max(age_imu, age_down) > s.stop_s:
            self.active = False
            return Tick("STOP", f"stale: IMU {age_imu * 1e3:.0f} ms, down ToF {age_down * 1e3:.0f} ms", self.cmd.copy())
        g = gravity_dir_body(self.imu[3])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))
        h = self.tof["down"][1]
        if tilt > s.max_tilt_deg:
            self.active = False
            return Tick("STOP", f"tilt {tilt:.0f} deg > {s.max_tilt_deg:.0f}", self.cmd.copy())
        if active and h > s.max_height_m and h < self.tof_max:
            self.active = False
            return Tick("STOP", f"height {h:.2f} m > {s.max_height_m:.2f}", self.cmd.copy())
        for n in self.tof_names:
            if n != "down" and n not in self.disabled and (self.tof[n] is None or t - self.tof[n][0] > s.tof_other_stale_s):
                warn.append(f"ToF {n} missing/stale -> no target")
        if not active:
            # Not flying the policy (not OFFBOARD+armed): keep PX4 fed with a calm setpoint
            # and restart cleanly when it begins (like a fresh training episode).
            self.active = False
            self.cmd = np.zeros(4)
            thr, rates = self.neutral()
            return Tick("WAIT", "not in OFFBOARD + armed", self.cmd.copy(), thr, rates, warnings=warn)
        if not self.active:                 # episode start: history = copies of the first frame, prev cmd = 0
            self.cmd = np.zeros(4)
            self.hist.clear()
            self.active = True
        if age_imu > s.imu_neutral_s or age_down > s.tof_down_neutral_s:
            self.cmd = np.zeros(4)
            thr, rates = self.neutral()
            return Tick("NEUTRAL", f"stale: IMU {age_imu * 1e3:.0f} ms, down ToF {age_down * 1e3:.0f} ms",
                        self.cmd.copy(), thr, rates, warnings=warn)
        frame = self.build_frame(t)
        if not self.hist:
            for _ in range(self.H):
                self.hist.append(frame)
        else:
            self.hist.append(frame)
        obs = np.concatenate(list(self.hist))
        a = self.p(obs)
        self.cmd = 2.0 * a - 1.0
        c, span = self.m["ctbr_thrust_center"], self.m["ctbr_thrust_span"]
        thrust = float(np.clip((c + self.cmd[0] * span) * s.thrust_gain, 0.0, 1.0))
        rates = self.cmd[1:4] * self.max_rates
        return Tick("RUN", "", self.cmd.copy(), thrust, rates, frame, warn)

    def out_of_training_range(self, frame, z_lim=6.0):
        """Inputs far outside what training saw (|z| > z_lim on the newest frame):
        usually a sign, unit or axis mismatch."""
        H, F = self.H, len(frame)
        sl = slice((H - 1) * F, H * F)
        z = (frame - self.p.mean[sl]) / self.p.std[sl]
        return [(FRAME_NAMES[i], float(frame[i]), float(z[i])) for i in np.argsort(-np.abs(z)) if abs(z[i]) > z_lim]
