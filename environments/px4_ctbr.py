"""PX4 multicopter rate controller + control allocation + motors, batched in torch.

This is what sits between the policy and the motors on the real DD24 when the
policy runs as a MAVROS node in OFFBOARD mode and publishes
/mavros/setpoint_raw/attitude (collective thrust + body rates):

  body-rate setpoint --> PX4 rate PID (MC_*RATE_P/I/D/K) --> normalised torque
  collective thrust  --------------------------------------> control allocation
                                                            (quad X, CA_ROTOR*)
                     --> per-motor command 0..1 --> motor lag --> thrust curve

Gains are PX4's defaults (your duckiedrone-px4-v4.params doesn't override them);
D-term filter is IMU_DGYRO_CUTOFF = 20 Hz from that file. Every episode the gains,
mixing strength, motor time constant and thrust curve are randomised so the
policy doesn't depend on this model being exact.

Frames: everything here is FLU body frame (x fwd, y left, z up), like ROS/MAVROS.
PX4 works in FRD internally; MAVROS converts, and a PID is symmetric under that
sign change, so the behaviour is the same.

Motor order matches the env: m1 front-right, m2 back-right, m3 back-left,
m4 front-left. Props: m1/m3 CCW, m2/m4 CW (seen from above), as on the DD24.
PX4 numbers them differently: PX4 motor 1 = m1 (FR), 2 = m3 (BL), 3 = m4 (FL),
4 = m2 (BR). Only matters if you ever send per-motor commands; with
thrust + body-rate setpoints PX4 does its own allocation.
"""
from __future__ import annotations

import math

import torch

# PX4 defaults (MC_ROLLRATE_*, MC_PITCHRATE_*, MC_YAWRATE_*), parallel form with K = 1.
PX4_RATE_P = (0.15, 0.15, 0.20)
PX4_RATE_I = (0.20, 0.20, 0.10)
PX4_RATE_D = (0.003, 0.003, 0.0)
PX4_RATE_K = (1.0, 1.0, 1.0)
PX4_INT_LIM = (0.30, 0.30, 0.30)
PX4_GYRO_CUTOFF_HZ = 40.0      # IMU_GYRO_CUTOFF default
PX4_DGYRO_CUTOFF_HZ = 20.0     # IMU_DGYRO_CUTOFF in duckiedrone-px4-v4.params

# Quad-X allocation, FLU torques, motor order m1 FR, m2 BR, m3 BL, m4 FL.
# (sign of each motor's contribution to +roll / +pitch / +yaw torque)
# Yaw: m1/m3 are CCW props (CA_ROTOR0/1_KM = +0.05) whose reaction turns the body
# clockwise (-z FLU), so +yaw comes from speeding up the CW props m2/m4.
ALLOC_ROLL = (-1.0, -1.0, 1.0, 1.0)
ALLOC_PITCH = (-1.0, 1.0, 1.0, -1.0)
ALLOC_YAW = (-1.0, 1.0, -1.0, 1.0)
MIX_RP, MIX_Y = 0.7071, 1.0    # PX4 quad-X normalisation (roll/pitch 1/sqrt2, yaw 1)


def _lpf_alpha(dt: float, cutoff_hz: float) -> float:
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    return dt / (dt + rc)


class Px4RateController:
    def __init__(self, n: int, device, idle: float = 0.04):
        self.n, self.device, self.idle = n, device, idle
        t = lambda v: torch.tensor(v, device=device).repeat(n, 1)   # noqa: E731
        self.P, self.I, self.D, self.K = t(PX4_RATE_P), t(PX4_RATE_I), t(PX4_RATE_D), t(PX4_RATE_K)
        self.int_lim = torch.tensor(PX4_INT_LIM, device=device)
        self.mix_gain = torch.ones(n, 3, device=device)       # per-axis allocation strength
        self.tau_m = torch.full((n, 1), 0.03, device=device)  # motor time constant (s)
        self.k_curve = torch.full((n, 1), 0.4, device=device)  # thrust = (1-k) u + k u^2
        self.alloc = torch.tensor([ALLOC_ROLL, ALLOC_PITCH, ALLOC_YAW], device=device)  # (3, 4)
        self.integ = torch.zeros(n, 3, device=device)
        self.gyro_f = torch.zeros(n, 3, device=device)
        self.prev_gyro_f = torch.zeros(n, 3, device=device)
        self.dgyro_f = torch.zeros(n, 3, device=device)
        self.u_act = torch.zeros(n, 4, device=device)

    # ------------------------------------------------------------------ reset
    def reset(self, ids, gain_scale, mix_scale, tau_m, k_curve, u0):
        """gain_scale (n,3) multiplies P, I, D; mix_scale (n,3); tau_m (n,1);
        k_curve (n,1); u0 (n,1) initial motor command (e.g. hover)."""
        base = lambda v: torch.tensor(v, device=self.device)  # noqa: E731
        self.P[ids] = base(PX4_RATE_P) * gain_scale
        self.I[ids] = base(PX4_RATE_I) * gain_scale
        self.D[ids] = base(PX4_RATE_D) * gain_scale
        self.mix_gain[ids] = mix_scale
        self.tau_m[ids] = tau_m
        self.k_curve[ids] = k_curve
        self.integ[ids] = 0.0
        self.gyro_f[ids] = 0.0
        self.prev_gyro_f[ids] = 0.0
        self.dgyro_f[ids] = 0.0
        self.u_act[ids] = u0.expand(-1, 4)

    def hover_command(self, thrust_frac):
        """Normalised collective command c giving per-motor thrust fraction
        `thrust_frac` of max on this env's thrust curve: solve (1-k)c + k c^2 = f."""
        k = self.k_curve
        a, b = k.clamp(min=1e-6), 1.0 - k
        return torch.where(k > 1e-6, (-b + torch.sqrt(b * b + 4 * a * thrust_frac)) / (2 * a), thrust_frac)

    # ------------------------------------------------------------------- step
    def step(self, rate_sp, thrust_sp, gyro_meas, dt):
        """One PX4 control tick. rate_sp, gyro_meas: (n,3) rad/s FLU; thrust_sp: (n,1)
        in [0,1]. Returns per-motor thrust as a fraction of max thrust (n,4)."""
        # gyro low-pass (IMU_GYRO_CUTOFF), then D on the filtered measurement (PX4 uses
        # angular acceleration, not error derivative), low-passed at IMU_DGYRO_CUTOFF.
        self.gyro_f = self.gyro_f + _lpf_alpha(dt, PX4_GYRO_CUTOFF_HZ) * (gyro_meas - self.gyro_f)
        dgyro = (self.gyro_f - self.prev_gyro_f) / dt
        self.prev_gyro_f = self.gyro_f.clone()
        self.dgyro_f = self.dgyro_f + _lpf_alpha(dt, PX4_DGYRO_CUTOFF_HZ) * (dgyro - self.dgyro_f)

        err = rate_sp - self.gyro_f
        # PX4 shrinks the integrator gain for large errors (i_factor), then clamps.
        i_factor = (1.0 - (err / math.radians(400.0)) ** 2).clamp(0.0, 1.0)
        self.integ = (self.integ + i_factor * self.I * err * dt).clamp(-self.int_lim, self.int_lim)
        torque = self.K * (self.P * err + self.integ - self.D * self.dgyro_f)     # normalised

        mix = torch.tensor([MIX_RP, MIX_RP, MIX_Y], device=self.device) * self.mix_gain
        u = thrust_sp + (torque * mix) @ self.alloc                               # (n,4)
        # Desaturate like PX4 without airmode: keep the torque differences, move the
        # collective down if a motor exceeds 1 (then up if one drops below idle), clip.
        u = u - (u.amax(dim=-1, keepdim=True) - 1.0).clamp(min=0.0)
        u = u + (self.idle - u.amin(dim=-1, keepdim=True)).clamp(min=0.0)
        u = u.clamp(self.idle, 1.0)

        # motors: first-order lag, then thrust curve
        self.u_act = self.u_act + (u - self.u_act) * (dt / (self.tau_m + dt))
        k = self.k_curve
        return (1.0 - k) * self.u_act + k * self.u_act ** 2
