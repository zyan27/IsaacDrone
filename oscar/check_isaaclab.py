"""Smoke test for the Isaac Lab install: starts Isaac Sim headless (needed before
isaaclab_assets / omni.* can be imported), then checks imports and the GPU."""
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import torch  # noqa: E402
import isaaclab  # noqa: E402
import isaaclab_assets  # noqa: E402,F401

print("Isaac Lab OK:", isaaclab.__file__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
app.close()
