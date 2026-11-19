"""Export a trained checkpoint to policy.npz for the drone (numpy only, no PyTorch there).

    python deploy/export_policy.py runs/dd24_hover_1115_1834_s0/best.pt deploy/policy.npz

The .npz holds the actor weights, the frozen input normalisation and every number
the node needs to rebuild the inputs and map the outputs exactly as in training.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from environments.dd24_params import DD24_TOF_SENSORS, TOF_MAX_RANGE_M, TOF_MIN_RANGE_M  # noqa: E402


def main(ckpt, out):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if ck.get("control_mode") != "ctbr" or ck.get("obs_mode") != "onboard":
        raise SystemExit("needs an onboard-sensor thrust+body-rate (ctbr) checkpoint")
    sd = ck["actor"]
    lin = sorted({k.split(".")[1] for k in sd if k.startswith("net.") and k.endswith(".weight")}, key=int)
    if len(lin) != 3:
        raise SystemExit(f"unexpected actor layout {lin}")
    arr = {}
    for i, j in enumerate(lin):
        arr[f"W{i}"] = sd[f"net.{j}.weight"].float().numpy()
        arr[f"b{i}"] = sd[f"net.{j}.bias"].float().numpy()
    norm = ck["obs_norm"]
    arr["norm_mean"] = norm["mean"].float().numpy()
    arr["norm_var"] = norm["var"].float().numpy()
    meta = {
        "source_checkpoint": os.path.abspath(ckpt),
        "train_iter": int(ck.get("iter", -1)),
        "actor_out": ck.get("actor_out", "sigmoid"),
        "obs_dim": int(ck["obs_dim"]),
        "obs_history": int(ck["obs_history"]),
        "policy_hz": float(ck.get("policy_hz", 50.0)),
        "ctbr_thrust_center": float(ck["ctbr_thrust_center"]),
        "ctbr_thrust_span": float(ck["ctbr_thrust_span"]),
        "ctbr_max_rates": [float(x) for x in ck["ctbr_max_rates"]],
        "tof_names": [s[0] for s in DD24_TOF_SENSORS],      # order of the 5 ToF inputs
        "tof_max_range": float(TOF_MAX_RANGE_M),
        "tof_min_range": float(TOF_MIN_RANGE_M),
        "flow_height_range": [0.08, 4.0],
        "flow_max_rate": 2.0,
        "norm_min_std": 0.01,
        "norm_clip": 10.0,
        "accel_clip": ck.get("accel_clip"),     # None for policies trained before the IMU fix
    }
    if meta["obs_dim"] != 22 * meta["obs_history"]:
        raise SystemExit(f"obs_dim {meta['obs_dim']} != 22 x {meta['obs_history']}: unexpected input layout")
    arr["meta_json"] = np.array(json.dumps(meta))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, **arr)
    print(f"wrote {out}: {meta['obs_dim']} inputs, actor {[arr[f'W{i}'].shape for i in range(3)]}, "
          f"out={meta['actor_out']}, from iter {meta['train_iter']} of {ckpt}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
