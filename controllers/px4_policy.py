import torch
import torch.nn as nn


def mlp(inp, out, hidden=256):
    return nn.Sequential(nn.Linear(inp, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, out))


class PX4Actor(nn.Module):
    """DD24 sensors (DronePX4Env 'policy' obs) -> action in [-1, 1]^4, mapped to a
    CTBR setpoint for PX4 by environments.drone_px4_env.policy_to_ctbr."""

    def __init__(self, obs_dim=20, act_dim=4):
        super().__init__()
        self.net = mlp(obs_dim, act_dim)

    def forward(self, obs):
        return torch.tanh(self.net(obs))
