"""Path B starter: DD24 with REAL PX4 (SITL) in the loop, via Pegasus Simulator.

This is the "simulated PX4 flight controller" port. PX4 runs as SITL and flies
the drone; Isaac Sim provides the physics. This is a SEPARATE programming model
from the Isaac Lab RL env in ../environments (which is path A, batched, no PX4).

!!! NOT RUNNABLE ON THE MAC AND NOT YET TESTED. Requires, on a Linux + NVIDIA
    GPU machine:
      - Isaac Sim (match the Pegasus release: official v4.5.1 -> Isaac Sim 4.5;
        community forks claim 5.1 / 6.0 but are unverified)
      - PegasusSimulator installed as an Isaac Sim extension
      - PX4-Autopilot checked out and built for SITL (px4_sitl)
    Launch with Pegasus's runner, e.g.:  ISAACSIM_PYTHON pegasus_dd24_px4.py

API names below follow the Pegasus "Your First Simulation" tutorial:
  https://pegasussimulator.github.io/PegasusSimulator/

TODO before this is a real DD24:
  1. DD24 USD mesh + mass/inertia  (no public DD24 USD; using Pegasus Iris now)
  2. A PX4 airframe for the DD24 (mamba-f405-mk2 params) instead of 'iris'
  3. Thrust curve / rotor config from dd24_params (measure first)
"""
from isaacsim import SimulationApp

# SimulationApp MUST be created before any other omni / pegasus import.
simulation_app = SimulationApp({"headless": False})

import omni.timeline
from omni.isaac.core.world import World
from scipy.spatial.transform import Rotation

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)

# DD24 physical params (single source of truth, shared with the Isaac Lab env).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environments.dd24_params import DD24


class DD24PX4Sim:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        # Pegasus singleton + world
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Box Room"])

        # --- PX4 SITL backend ------------------------------------------------
        # px4_vehicle_model selects the PX4 AIRFRAME. 'iris' is the stock quad.
        # TODO: replace with a DD24 airframe (mamba-f405-mk2 geometry/mixer) so
        # PX4's controller is tuned to the real board.
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,              # auto-starts px4_sitl
            "px4_dir": self.pg.px4_path,         # set via PegasusInterface config
            "px4_vehicle_model": self.pg.px4_default_airframe,  # TODO: 'dd24'
        })

        # --- Vehicle ---------------------------------------------------------
        # MultirotorConfig defaults to a quadratic thrust curve + linear drag +
        # IMU/GPS/baro/mag. Mass/inertia come from the USD rigid body.
        # TODO: swap ROBOTS["Iris"] for a DD24 USD, and set the thrust curve
        # from DD24 (mass={DD24.mass_kg} kg, thrust/motor={DD24.max_thrust_per_motor_N} N).
        multirotor_config = MultirotorConfig()
        multirotor_config.backends = [PX4MavlinkBackend(mavlink_config)]

        Multirotor(
            "/World/dd24",
            ROBOTS["Iris"],                      # TODO: DD24 USD
            0,
            [0.0, 0.0, 0.10],                    # spawn 10 cm up
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=multirotor_config,
        )

        self.world.reset()
        self.stop_sim = False

    def run(self):
        self.timeline.play()
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
        self.timeline.stop()
        simulation_app.close()


def main():
    print(f"[DD24/PX4] mass={DD24.mass_kg} kg  TWR={DD24.thrust_to_weight:.2f}  "
          f"FC={DD24.fc_firmware} target={DD24.fc_board}")
    print("[DD24/PX4] NOTE: using Pegasus Iris USD + stock PX4 airframe until the "
          "DD24 USD and DD24 PX4 airframe are built (see TODOs).")
    DD24PX4Sim().run()


if __name__ == "__main__":
    main()
