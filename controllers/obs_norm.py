"""Running observation normalisation (like rsl_rl's EmpiricalNormalization).

During training the mean/variance of every input is tracked over all rollouts and
inputs are fed to the networks as (x - mean) / std. The statistics are saved in
the checkpoint ("obs_norm") and frozen afterwards: evaluation, videos and the
deployed MAVROS node must apply exactly the same transform (load_obs_norm does).

Why: raw inputs mix scales (accelerometer ~9.8 m/s^2 on z in every history step,
ToFs 0-1, flow ~0.1 m/s). Unnormalised, a tiny learning rate still moved the
policy a lot and a moderate one blew up, so the adaptive-KL schedule couldn't
control the step size (smoke test: lr pinned at 1e-5 with KL ~0.011).
"""
from __future__ import annotations

import torch
import torch.nn as nn

MIN_STD = 1e-2      # inputs that barely vary (e.g. "flow valid") aren't blown up
CLIP = 10.0


class RunningNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """Merge a batch (N, dim) into the running statistics (Chan et al. parallel update)."""
        x = x.reshape(-1, x.shape[-1]).float()
        n = x.shape[0]
        if n == 0:
            return
        b_mean = x.mean(0)
        b_var = x.var(0, unbiased=False)
        tot = self.count + n
        delta = b_mean - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + b_var * n + delta ** 2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.var).clamp(min=MIN_STD)
        return ((x - self.mean) / std).clamp(-CLIP, CLIP)

    def to_dict(self):
        return {"mean": self.mean.detach().cpu(), "var": self.var.detach().cpu(), "count": self.count.detach().cpu()}

    def load_dict(self, d):
        self.mean.copy_(d["mean"]); self.var.copy_(d["var"]); self.count.copy_(d["count"])


def load_obs_norm(ck: dict, dim: int, device):
    """Frozen normaliser from a checkpoint, or identity for checkpoints made before
    normalisation existed."""
    if "obs_norm" not in ck:
        return lambda x: x
    norm = RunningNorm(dim).to(device)
    norm.load_dict(ck["obs_norm"])
    norm.eval()
    return norm
