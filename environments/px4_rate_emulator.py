"""Batched torch emulation of PX4's multicopter inner loop (Part 3a).

Takes the same setpoint the policy will send PX4 over MAVLink
(SET_ATTITUDE_TARGET: collective thrust in [0, 1] + FRD body rates in rad/s)
and returns per-motor commands in [0, 1], like PX4's actuator outputs.

Ported from PX4 v1.15.4 (what the DD24 flies):
  gyro filter       src/modules/sensors/vehicle_angular_velocity (2nd-order LPF on
                    rates at IMU_GYRO_CUTOFF, D-term on LPF'd derivative at IMU_DGYRO_CUTOFF)
  rate PID          src/lib/rate_control/rate_control.cpp (incl. i_factor + anti-windup)
  allocation        control_allocator ControlAllocationPseudoInverse normalization +
                    ControlAllocationSequentialDesaturation::mixAirmodeDisabled (MC_AIRMODE 0)
THR_MDL_FAC = 0, so motor thrust is linear in the command.

Not modeled: ESC/motor lag, idle throttle, battery sag. All tensors are (N, ...).
"""
from __future__ import annotations

import math

import torch

from sim_px4.px4_defaults import ATT, FILTERS, QUAD_X, RATE


class _Biquad:
    """PX4 math::LowPassFilter2p (2nd-order Butterworth), batched over envs."""

    def __init__(self, sample_hz: float, cutoff_hz: float, shape, device):
        fr = sample_hz / cutoff_hz
        ohm = math.tan(math.pi / fr)
        c = 1.0 + 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm
        self.b0 = ohm * ohm / c
        self.b1 = 2.0 * self.b0
        self.b2 = self.b0
        self.a1 = 2.0 * (ohm * ohm - 1.0) / c
        self.a2 = (1.0 - 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm) / c
        self.d1 = torch.zeros(shape, device=device)
        self.d2 = torch.zeros(shape, device=device)

    def reset(self, env_ids, value):
        # Settle at a constant input, as PX4's reset(sample) does.
        d = value / (self.b0 + self.b1 + self.b2)
        self.d1[env_ids] = d
        self.d2[env_ids] = d

    def apply(self, x):
        d0 = x - self.d1 * self.a1 - self.d2 * self.a2
        y = d0 * self.b0 + self.d1 * self.b1 + self.d2 * self.b2
        self.d2 = self.d1
        self.d1 = d0
        return y


class PX4RateEmulator:
    def __init__(self, num_envs: int, dt: float, device):
        self.dt = dt
        self.device = device
        f = lambda v: torch.tensor(v, dtype=torch.float32, device=device)  # noqa: E731

        # MC_*RATE_K scales P, I and D (PX4 "ideal" form).
        k = f(RATE.K)
        self.kp, self.ki, self.kd = f(RATE.P) * k, f(RATE.I) * k, f(RATE.D) * k
        self.kff = f(RATE.FF)
        self.int_lim = f(RATE.int_lim)
        self.rate_max = f(ATT.rate_max_deg_s) * math.pi / 180.0

        self.gyro_lpf = _Biquad(1.0 / dt, FILTERS.gyro_cutoff_hz, (num_envs, 3), device)
        self.dgyro_lpf = _Biquad(1.0 / dt, FILTERS.dgyro_cutoff_hz, (num_envs, 3), device)
        self.prev_rate = torch.zeros(num_envs, 3, device=device)
        self.rate_int = torch.zeros(num_envs, 3, device=device)
        self.sat_pos = torch.zeros(num_envs, 3, dtype=torch.bool, device=device)
        self.sat_neg = torch.zeros(num_envs, 3, dtype=torch.bool, device=device)

        # Effectiveness (FRD, rotor axis -z): roll -py*ct, pitch px*ct, yaw km*ct, thrust_z -ct.
        px, py, km, ct = f(QUAD_X.px), f(QUAD_X.py), f(QUAD_X.km), QUAD_X.ct
        B = torch.stack([-py * ct, px * ct, km * ct, -ct * torch.ones(4, device=device)])
        mix = torch.linalg.pinv(B)  # (4 motors, [roll, pitch, yaw, thrust_z])

        # ControlAllocationPseudoInverse::updateControlAllocationMatrixScale.
        nz = lambda col: (col.abs() > 1e-3).sum()  # noqa: E731
        rp_scale = torch.maximum(
            torch.sqrt(mix[:, 0].square().sum() / (nz(mix[:, 0]) / 2.0)),
            torch.sqrt(mix[:, 1].square().sum() / (nz(mix[:, 1]) / 2.0)),
        )
        mix[:, 0] /= rp_scale
        mix[:, 1] /= rp_scale
        mix[:, 2] /= mix[:, 2].max()
        mix[:, 3] /= mix[:, 3].abs().sum() / nz(mix[:, 3])
        mix[mix.abs() < 1e-3] = 0.0
        self.mix = mix                          # thrust_z column is -1 per motor
        self.unmix = torch.linalg.pinv(mix)     # motor cmds -> achieved [r, p, y, thrust_z]

    def reset(self, env_ids, rate_frd=None):
        rate = torch.zeros(len(env_ids), 3, device=self.device) if rate_frd is None else rate_frd
        self.gyro_lpf.reset(env_ids, rate)
        self.dgyro_lpf.reset(env_ids, torch.zeros_like(rate))
        self.prev_rate[env_ids] = rate
        self.rate_int[env_ids] = 0.0
        self.sat_pos[env_ids] = False
        self.sat_neg[env_ids] = False

    def step(self, thrust: torch.Tensor, rate_sp_frd: torch.Tensor, gyro_frd: torch.Tensor,
             landed: torch.Tensor | None = None) -> torch.Tensor:
        """thrust (N,) in [0,1]; rate_sp_frd, gyro_frd (N,3) rad/s. Returns motors (N,4) in [0,1]."""
        rate_sp = torch.clamp(rate_sp_frd, -self.rate_max, self.rate_max)

        rate = self.gyro_lpf.apply(gyro_frd)
        ang_acc = self.dgyro_lpf.apply((rate - self.prev_rate) / self.dt)
        self.prev_rate = rate

        # RateControl::update
        err = rate_sp - rate
        torque = self.kp * err + self.rate_int - self.kd * ang_acc + self.kff * rate_sp

        # RateControl::updateIntegral (skipped while landed)
        e = torch.where(self.sat_pos, torch.clamp(err, max=0.0), err)
        e = torch.where(self.sat_neg, torch.clamp(e, min=0.0), e)
        i_factor = torch.clamp(1.0 - (e / math.radians(400.0)).square(), min=0.0)
        new_int = torch.clamp(self.rate_int + i_factor * self.ki * e * self.dt, -self.int_lim, self.int_lim)
        if landed is not None:
            new_int = torch.where(landed.unsqueeze(-1), self.rate_int, new_int)
        self.rate_int = new_int

        torque = torch.clamp(torque, -1.0, 1.0)
        motors = self._allocate(torque, torch.clamp(thrust, 0.0, 1.0))

        # Allocator saturation feedback for anti-windup (unallocated torque).
        achieved = (motors @ self.unmix.T)[:, 0:3]
        self.sat_pos = (torque - achieved) > 1e-4
        self.sat_neg = (torque - achieved) < -1e-4
        return motors

    # --- ControlAllocationSequentialDesaturation, airmode disabled ---------------
    @staticmethod
    def _desat_gain(vec, sp, lo, hi):
        use = vec.abs() >= 0.2
        k_lo = torch.where(use & (sp < lo), (lo - sp) / vec, torch.zeros_like(sp))
        k_hi = torch.where(use & (sp > hi), (hi - sp) / vec, torch.zeros_like(sp))
        k = torch.cat([k_lo, k_hi], dim=-1)
        return k.clamp(max=0.0).min(dim=-1).values + k.clamp(min=0.0).max(dim=-1).values

    def _desaturate(self, sp, vec, increase_only=False, hi=1.0):
        gain = self._desat_gain(vec, sp, 0.0, hi)
        # PX4 returns before both passes when increase_only and the first gain is negative.
        run = (gain >= 0.0) if increase_only else torch.ones_like(gain, dtype=torch.bool)
        sp = sp + torch.where(run, gain, 0.0).unsqueeze(-1) * vec
        gain = 0.5 * self._desat_gain(vec, sp, 0.0, hi)
        return sp + torch.where(run, gain, 0.0).unsqueeze(-1) * vec

    def _allocate(self, torque, thrust):
        m = self.mix
        n = torque.shape[0]
        roll, pitch, yaw, tz = (m[:, i].expand(n, 4) for i in range(4))
        sp = torque[:, 0:1] * roll + torque[:, 1:2] * pitch + (-thrust).unsqueeze(-1) * tz
        sp = self._desaturate(sp, tz, increase_only=True)
        sp = self._desaturate(sp, roll)
        sp = self._desaturate(sp, pitch)
        # mixYaw: allow 15% headroom above max for yaw, then only reduce thrust.
        sp = sp + torque[:, 2:3] * yaw
        sp = self._desaturate(sp, yaw, hi=1.15)
        sp = self._desaturate(sp, tz, increase_only=True)
        return torch.clamp(sp, 0.0, 1.0)
