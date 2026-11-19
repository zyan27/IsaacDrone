import torch
import torch.nn as nn


class RLHoverActor(nn.Module):
    """MLP policy mean in [0, 1]-action space.

    out="linear" (new runs): plain linear output, initialised at 0.5 with tiny weights;
        the env clamps to [0, 1]. Can't saturate, so gradients never vanish.
    out="sigmoid" (old checkpoints): sigmoid output. A large learning rate can push it
        to 0/1, where it stops learning (seen in the 2nd smoke test).
    """
    def __init__(self, obs_dim=14, action_dim=4, out="sigmoid"):
        super().__init__()
        layers = [nn.Linear(obs_dim, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, action_dim)]
        if out == "sigmoid":
            layers.append(nn.Sigmoid())
        else:
            nn.init.orthogonal_(layers[-1].weight, gain=0.01)
            nn.init.constant_(layers[-1].bias, 0.5)
        self.out = out
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


RLHoverPolicy = RLHoverActor


class RLHoverController:
    def __init__(self, device="cuda:0", checkpoint=None):
        self.device = device
        # Defaults match the old Crazyflie setup; new checkpoints carry their own.
        self.hover_motor = 1.0 / 2.2
        self.residual_scale = 0.15
        obs_dim = 16
        self.control_mode = "motors"
        state = None
        ck = {}

        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            ck = state if isinstance(state, dict) else {}
            if "actor" in state:
                self.control_mode = state.get("control_mode", "motors")
                self.hover_motor = state.get("hover_motor", self.hover_motor)
                self.residual_scale = state.get("residual_scale", self.residual_scale)
                obs_dim = state.get("obs_dim", 14)
                state = state["actor"]

        self.obs_dim = obs_dim
        self.policy = RLHoverActor(obs_dim=obs_dim, out=ck.get("actor_out", "sigmoid")).to(device)
        self.policy.eval()
        if state is not None:
            self.policy.load_state_dict(state)
        from controllers.obs_norm import load_obs_norm
        self.norm = load_obs_norm(ck, obs_dim, device)

    def reset(self, num_envs):
        pass

    def act(self, obs):
        with torch.no_grad():
            residual_raw = self.policy(self.norm(obs[:, : self.obs_dim]))

        if self.control_mode == "ctbr":
            return torch.clamp(residual_raw, 0.0, 1.0)   # env maps to thrust + body rates
        residual = (residual_raw - 0.5) * self.residual_scale
        actions = self.hover_motor + residual

        return torch.clamp(actions, 0.0, 1.0)