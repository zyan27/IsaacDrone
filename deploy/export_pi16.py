"""Export a --obs_pi16 --action velocity checkpoint for the Duckiedrone Pi node's .npz loader:
W0, b0, W1, b1, ... with each W shaped [in, out], tanh after every layer (the actor is trained
with tanh hidden layers for this), spec_version 1, no obs_mean / obs_std (inputs are unscaled).

    python deploy/export_pi16.py bitwise_improvement/checkpoints/X.pt deploy/X_pi16.npz

Plain torch only (no Isaac), so it runs in the container or on a laptop.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main(src: str, dst: str):
    ckpt = torch.load(src, map_location="cpu")
    assert ckpt.get("act") == "tanh", f"{src}: actor activation {ckpt.get('act')!r}, the Pi loader needs tanh"
    sd = ckpt["actor"]
    layers = sorted({k.split(".")[1] for k in sd if k.startswith("net.")}, key=int)
    arrays = {}
    for i, name in enumerate(layers):
        arrays[f"W{i}"] = sd[f"net.{name}.weight"].numpy().T.astype(np.float32).copy()   # [in, out]
        arrays[f"b{i}"] = sd[f"net.{name}.bias"].numpy().astype(np.float32)
    assert arrays["W0"].shape[0] == 16 and arrays[f"W{len(layers) - 1}"].shape[1] == 4
    arrays["spec_version"] = np.int32(1)
    np.savez(dst, **arrays)

    # The Pi's forward pass (x @ W + b, tanh every layer) against torch.
    from controllers.px4_policy import PX4Actor
    actor = PX4Actor(16, act="tanh")
    actor.load_state_dict(sd)
    obs = np.random.randn(64, 16).astype(np.float32)
    x = obs
    for i in range(len(layers)):
        x = np.tanh(x @ arrays[f"W{i}"] + arrays[f"b{i}"])
    err = np.abs(x - actor(torch.from_numpy(obs)).detach().numpy()).max()
    print(f"numpy (Pi loader) vs torch max abs diff {err:.2e}")
    assert err < 1e-4
    print(f"{src} (iter {ckpt.get('iter')}) -> {dst}: " + ", ".join(f"W{i}{arrays[f'W{i}'].shape}" for i in range(len(layers))))


if __name__ == "__main__":
    main(*sys.argv[1:3])
