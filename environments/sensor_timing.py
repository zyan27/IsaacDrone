"""Message timing as the policy / PX4 will see it on the drone (pure torch).

The policy runs at a fixed rate (50 Hz, a ROS timer in the MAVROS node) and always
uses the LATEST message of each topic. HeldSensor reproduces what that looks like:

    raw (true value now) --> delay line (latency) --> sampled at the topic's own
    rate (phase clock, random start) --> held until the next delivered sample

and the ways a real link gets laggy:
  * drops   : an individual update never arrives (prob per update) -> the old
              value is held one period longer
  * stalls  : the topic freezes for 0.1-0.4 s (I2C hiccup, Pi CPU spike, busy
              serial link), starting at random (rate per second)

The same class models the command link node -> MAVROS -> PX4: latency, dropped
setpoints (PX4 keeps flying the last one) and stalls.

Rates, latencies, drop probabilities and stall rates are drawn per env per
episode from ranges, so the policy can't rely on exact timing.
"""
from __future__ import annotations

import torch


class HeldSensor:
    def __init__(self, n: int, dim: int, max_delay_steps: int, device):
        self.n, self.dim, self.L, self.device = n, dim, max_delay_steps + 1, device
        self.buf = torch.zeros(n, self.L, dim, device=device)
        self.hold = torch.zeros(n, dim, device=device)
        self.phase = torch.zeros(n, device=device)
        self.rate = torch.ones(n, device=device)
        self.lat = torch.zeros(n, dtype=torch.long, device=device)
        self.drop_p = torch.zeros(n, device=device)
        self.stall_rate = torch.zeros(n, device=device)      # stalls per second
        self.stall_dur = (0.1, 0.4)                          # s
        self.stall_left = torch.zeros(n, device=device)      # s remaining
        self.fill = torch.ones(n, dtype=torch.bool, device=device)

    def reset(self, ids, rate_hz, latency_steps, drop_p=None, stall_rate=None):
        k = len(ids)
        self.rate[ids] = rate_hz
        self.lat[ids] = latency_steps.clamp(0, self.L - 1)
        self.phase[ids] = torch.rand(k, device=self.device)
        self.drop_p[ids] = 0.0 if drop_p is None else drop_p
        self.stall_rate[ids] = 0.0 if stall_rate is None else stall_rate
        self.stall_left[ids] = 0.0
        self.fill[ids] = True

    def __call__(self, raw: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance one policy step with this step's true value; return what the
        receiver has now."""
        if bool(self.fill.any()):          # fresh episode: pretend it's been steady
            f = self.fill
            self.buf[f] = raw[f][:, None].expand(-1, self.L, -1)
            self.hold[f] = raw[f]
            self.fill[f] = False
        self.buf = torch.cat([self.buf[:, 1:], raw[:, None]], dim=1)
        delayed = self.buf[torch.arange(self.n, device=self.device), self.L - 1 - self.lat]

        self.phase = self.phase + self.rate * dt
        fresh = self.phase >= 1.0
        self.phase = torch.where(fresh, self.phase - 1.0, self.phase)

        # stalls: start at random, block delivery while running
        start = (self.stall_left <= 0) & (torch.rand(self.n, device=self.device) < self.stall_rate * dt)
        lo, hi = self.stall_dur
        self.stall_left = torch.where(start, lo + (hi - lo) * torch.rand(self.n, device=self.device),
                                      self.stall_left)
        stalled = self.stall_left > 0
        self.stall_left = (self.stall_left - dt).clamp(min=0.0)
        # drops: an individual update is lost
        dropped = torch.rand(self.n, device=self.device) < self.drop_p

        deliver = fresh & ~stalled & ~dropped
        self.hold = torch.where(deliver[:, None], delayed, self.hold)
        return self.hold.clone()
