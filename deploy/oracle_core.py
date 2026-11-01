"""Numpy-only pieces of the DD24 policy that policy_node.py and replay_check.py share:
the MLP, the observation, the wall-map estimator and the action -> setpoint mapping.
Each mirrors environments/drone_px4_env.py (cited per function) so the Pi feeds the
policy what it saw in training.

Frames: the sim's world is the takeoff frame (heading 0 at spawn), so the IMU quaternion
is rotated by the takeoff heading before it reaches the policy or the estimator.
mavros gives FLU body / ENU world; the policy uses FRD gyro / accel and wxyz quaternions.
"""
import math

import numpy as np

FRD = np.array([1.0, -1.0, -1.0], np.float32)
G = 9.81
AGE_CLIP_S = 1.0           # DronePX4EnvCfg.age_clip_s
SIDE_DEFAULT_M = 3.5       # held side ToF before its first return (side_tof_range_m[1])
DIRS_B = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0], [0, -1, 0]], np.float64)   # down, front, left, right (FLU)


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def qrot(q, v):
    return qmul(qmul(q, np.r_[0.0, v]), q * np.array([1, -1, -1, -1]))[1:]


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_deg(q):
    w, x, y, z = q
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


class Policy:
    """MLP from deploy/export_policy.py (ELU hidden layers, tanh output)."""

    def __init__(self, path, action_mode=None):
        z = np.load(path)
        self.layers = []
        i = 0
        while f"W{i}" in z:
            self.layers.append((z[f"W{i}"], z[f"b{i}"]))
            i += 1
        self.hover, self.thr_range, self.rate_scale = float(z["hover_thrust"]), float(z["thrust_range"]), z["rate_scale"]
        self.action_mode = action_mode or (str(z["action_mode"]) if "action_mode" in z else "rates")
        self.tilt_max = float(z["att_tilt_max_rad"]) if "att_tilt_max_rad" in z else 0.35
        self.n_obs = self.layers[0][0].shape[1]
        self.oracle = self.n_obs == 30          # 24 base + 6 oracle slots (set_policy_obs)
        assert self.n_obs in (24, 30), f"policy expects {self.n_obs} inputs: history / estimator obs not supported here"

    def __call__(self, obs):
        x = obs
        for k, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if k < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(x))  # ELU
        return np.clip(np.tanh(x), -1.0, 1.0)

    def setpoint(self, a):
        """policy_to_ctbr + _pre_physics_step: [thrust, roll, pitch, yaw] where roll / pitch are
        FRD angles (rad) in attitude mode and FRD rates in rates mode; yaw is always a rate."""
        thrust = min(max(self.hover + self.thr_range * a[0], 0.0), 1.0)
        rates = a[1:4] * self.rate_scale
        if self.action_mode == "attitude":
            rates[0:2] = a[1:3] * self.tilt_max
        return np.r_[thrust, rates].astype(np.float32)


class WallMap:
    """DronePX4Env._estimate_wallmap for one drone. Walls: x+ at the front reading, y+ at the
    left, y- at the right, all taken on the floor before takeoff; no back wall (never seen).
    Rays start floor_h above the floor: on the real floor the drone sits ~0.3 deg off level and,
    from height 0, every side ray tilted down would count as a floor hit (the sim sits level)."""

    def __init__(self, front, left, right, dt=0.05, alpha=0.5, beta=0.15, floor_h=0.05):
        self.hi = (front, left)
        self.lo = (-1e3, -right)
        self.dt, self.alpha, self.beta, self.floor_h = dt, alpha, beta, floor_h
        self.p, self.v, self.fresh = np.zeros(3), np.zeros(3), True

    def step(self, tof, quat, accel_frd, tof_age):
        dt = self.dt
        a_w = qrot(quat, accel_frd * FRD) + np.array([0.0, 0.0, -G])
        if self.fresh:
            p, v = np.zeros(3), np.zeros(3)
        else:
            p, v = self.p + self.v * dt + 0.5 * a_w * dt * dt, self.v + a_w * dt
        dirs = [qrot(quat, d) for d in DIRS_B]
        ok = tof_age < 0.2
        meas = [None, None, None]
        dz = -dirs[0][2]
        if ok[0] and dz > 0.5:
            meas[2] = tof[0] * dz
        sx, sy = [], []
        for i in (1, 2, 3):
            if not ok[i]:
                continue
            d = dirs[i]
            wx = self.hi[0] if d[0] > 0 else self.lo[0]
            wy = self.hi[1] if d[1] > 0 else self.lo[1]
            tx = (wx - p[0]) / d[0] if abs(d[0]) > 1e-3 else 1e6
            ty = (wy - p[1]) / d[1] if abs(d[1]) > 1e-3 else 1e6
            tf = -(max(p[2], 0.0) + self.floor_h) / d[2] if d[2] < -1e-3 else 1e6
            if tx <= ty and tx < tf:
                sx.append(wx - tof[i] * d[0])
            elif ty < tx and ty < tf:
                sy.append(wy - tof[i] * d[1])
        if sx:
            meas[0] = sum(sx) / len(sx)
        if sy:
            meas[1] = sum(sy) / len(sy)
        al, be = (1.0, 0.0) if self.fresh else (self.alpha, self.beta)
        for i in range(3):
            if meas[i] is not None:
                r = meas[i] - p[i]
                p[i] += al * r
                v[i] += be / dt * r
        self.p, self.v, self.fresh = p, v, False
        return p, v


def observation(tof, quat, gyro_frd, accel_frd, ages, last_sp, target, est=None):
    """DronePX4Env._sensors (+ the oracle slots of _get_observations when est = (p, v))."""
    ages = np.clip(ages, 0.0, AGE_CLIP_S)
    parts = [tof, quat, gyro_frd, accel_frd, ages, last_sp, [target]]
    if est is not None:
        p, v = est
        parts += [[-p[0], -p[1], target - p[2]], v]
    return np.concatenate(parts).astype(np.float32)


def attitude_quat_enu(roll_frd, pitch_frd, yaw_enu):
    """DronePX4Env._attitude_to_rates' setpoint: yaw (current heading) * pitch * roll, as an
    ENU / FLU quaternion for mavros (FRD pitch about +y right = FLU pitch about -y)."""
    h = lambda a: (math.cos(a / 2), math.sin(a / 2))  # noqa: E731
    cz, sz = h(yaw_enu)
    cy, sy = h(-pitch_frd)
    cx, sx = h(roll_frd)
    return qmul(qmul(np.array([cz, 0, 0, sz]), np.array([cy, 0, sy, 0])), np.array([cx, sx, 0, 0]))


class Scripted:
    """play_px4.py's reference commander on the wall-map estimate instead of ground truth:
    height PID on the down ToF, position PD -> tilt (clipped to max_tilt), attitude setpoints."""
    action_mode, oracle, n_obs = "attitude", True, 0

    def __init__(self, hover=0.47, max_tilt_rad=math.radians(5.0)):
        self.hover, self.max_tilt, self.z_int = hover, max_tilt_rad, 0.0

    def command(self, est, quat, target):
        p, v = est
        if target <= 0.0:
            self.z_int = 0.0
            return np.zeros(4, np.float32)
        z_err = target - p[2]
        if abs(z_err) < 0.15:            # no windup on the climb
            self.z_int += z_err * 0.02
        thrust = min(max(self.hover + 0.12 * z_err - 0.16 * v[2] + 0.08 * self.z_int, 0.0), 1.0)
        ax, ay = 1.5 * -p[0] - 2.0 * v[0], 1.5 * -p[1] - 2.0 * v[1]
        yaw = yaw_of(quat)
        c, s = math.cos(yaw), math.sin(yaw)
        fwd, left = c * ax + s * ay, -s * ax + c * ay
        pitch_flu = min(max(fwd / G, -self.max_tilt), self.max_tilt)    # +pitch tilts thrust forward
        roll_flu = min(max(-left / G, -self.max_tilt), self.max_tilt)   # -roll tilts thrust left
        return np.array([thrust, roll_flu, -pitch_flu, 0.0], np.float32)  # FRD roll, pitch
