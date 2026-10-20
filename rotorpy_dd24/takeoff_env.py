"""TakeoffHoverEnv: start the DD24 at rest ON THE GROUND and require it to climb
to a fixed goal and hold it precisely.

rotorpy's QuadrotorEnv spawns the drone already airborne (random point, rotors at
hover speed). That is not the real task: the DD24 sits on the pad, motors off,
then takes off. This subclass overrides reset_idx so every episode begins:
    position = [small random x, small random y, 0]   (on the pad)
    velocity = 0, attitude = level, body rates = 0
    rotor_speeds = 0                                   (motors off)
and the policy must spin up, climb, and hold the goal.

There is no ground-contact model in rotorpy (free space), so "on the ground"
just means starting at rest at z=0; the drone will not rest on a surface, it
simply begins from standstill.
"""
import numpy as np
import torch

from rotorpy.learning.quadrotor_environments import QuadrotorEnv


class TakeoffHoverEnv(QuadrotorEnv):
    def __init__(self, *args, goal=(0.0, 0.0, 1.0), xy_spread=0.3, **kwargs):
        # goal/spread must exist before super().__init__ (reset_idx reads them).
        self.goal = np.asarray(goal, dtype=np.float64)
        self.xy_spread = float(xy_spread)
        super().__init__(*args, **kwargs)

    def reset_idx(self, env_idx, options):
        """Place env `env_idx` at rest on the pad, motors off. `options` ignored
        on purpose — the start distribution is defined entirely here."""
        dev = self.device
        xy = (torch.rand(2, device=dev).double() * 2 - 1) * self.xy_spread
        self.vehicle_states['x'][env_idx] = torch.tensor(
            [xy[0].item(), xy[1].item(), 0.0], device=dev).double()
        self.vehicle_states['v'][env_idx] = torch.zeros(3, device=dev).double()
        self.vehicle_states['q'][env_idx] = torch.tensor([0., 0., 0., 1.], device=dev).double()
        self.vehicle_states['w'][env_idx] = torch.zeros(3, device=dev).double()
        self.vehicle_states['wind'][env_idx] = torch.zeros(3, device=dev).double()
        self.vehicle_states['rotor_speeds'][env_idx] = torch.zeros(4, device=dev).double()

        self.t[env_idx] = 0.0
        self.reward[env_idx] = 0.0
        self.cum_rewards[env_idx] = 0.0
