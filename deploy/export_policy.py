"""Export a train_rl_px4.py checkpoint to a numpy .npz the Pi can run without torch.

    python deploy/export_policy.py checkpoints/px4_hover.pt deploy/policy.npz

Plain torch only (no Isaac), so it runs on the login node or a laptop.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

POLICY_HOVER_THRUST = 0.47   # keep in sync with environments/drone_px4_env.py
POLICY_THRUST_RANGE = 0.25
POLICY_RATE_SCALE = (3.0, 3.0, 2.0)


def main(src: str, dst: str):
    ckpt = torch.load(src, map_location="cpu")
    sd = ckpt["actor"]
    layers = sorted({k.split(".")[1] for k in sd if k.startswith("net.")}, key=int)
    arrays = {}
    for i, name in enumerate(layers):
        arrays[f"W{i}"] = sd[f"net.{name}.weight"].numpy().astype(np.float32)
        arrays[f"b{i}"] = sd[f"net.{name}.bias"].numpy().astype(np.float32)
    arrays["hover_thrust"] = np.float32(POLICY_HOVER_THRUST)
    arrays["thrust_range"] = np.float32(POLICY_THRUST_RANGE)
    arrays["rate_scale"] = np.array(POLICY_RATE_SCALE, np.float32)
    np.savez(dst, **arrays)

    # Check the numpy forward pass (same math as deploy/policy_node.py) against torch.
    from controllers.px4_policy import PX4Actor
    actor = PX4Actor(arrays["W0"].shape[1])
    actor.load_state_dict(sd)
    obs = np.random.randn(64, arrays["W0"].shape[1]).astype(np.float32)
    x = obs.T
    for i in range(len(layers)):
        x = arrays[f"W{i}"] @ x + arrays[f"b{i}"][:, None]
        if i < len(layers) - 1:
            x = np.where(x > 0, x, np.expm1(x))
    err = np.abs(np.tanh(x).T - actor(torch.from_numpy(obs)).detach().numpy()).max()
    print(f"numpy vs torch max abs diff {err:.2e}")
    assert err < 1e-4
    print(f"{src} (iter {ckpt.get('iter')}) -> {dst}: "
          + ", ".join(f"{k}{arrays[k].shape}" for k in arrays if k.startswith("W")))


if __name__ == "__main__":
    main(*sys.argv[1:3])
