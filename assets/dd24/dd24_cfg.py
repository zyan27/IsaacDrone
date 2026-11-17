"""Isaac Lab ArticulationCfg for the DD24, mirroring isaaclab_assets' CRAZYFLIE_CFG.

The USD is produced by:
    python assets/dd24/build_dd24_urdf.py ...      # mesh + mass + inertia -> URDF
    bash   assets/dd24/convert_to_usd.sh           # URDF -> USD (needs Isaac Lab)
Mass and inertia live in the USD; the env reads them back from PhysX at startup
and prints them, so check that printout against your measured values.
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

DD24_USD_PATH = os.environ.get(
    "DD24_USD_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dd24.usd"),
)

if not os.path.isfile(DD24_USD_PATH):
    raise FileNotFoundError(
        f"DD24 USD not found at {DD24_USD_PATH}. Build it with "
        "assets/dd24/build_dd24_urdf.py + assets/dd24/convert_to_usd.sh "
        "(or set DD24_USD_PATH)."
    )

DD24_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Drone",
    spawn=sim_utils.UsdFileCfg(
        usd_path=DD24_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={".*": 0.0},
        joint_vel={"m1_joint": 200.0, "m2_joint": -200.0, "m3_joint": 200.0, "m4_joint": -200.0},
    ),
    # Props are free-spinning and purely visual; thrust is applied as an external
    # force/torque on `body` by the env.
    actuators={
        "dummy": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0),
    },
)
