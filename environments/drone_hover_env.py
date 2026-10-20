from __future__ import annotations

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.sensors.imu import Imu, ImuCfg

import math
from pxr import UsdGeom

from .dd24_params import DD24


@configclass
class DroneHoverEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 2
    action_space = 4
    observation_space = 14
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
    )
    robot = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Drone"
    )
    hover_height = 1.0
    crash_height = 0.1
    arm_length = 0.046
    max_thrust_to_weight = 2.2
    yaw_coeff = 0.00001
    max_prop_visual_speed = 50000.0


@configclass
class DroneHoverEnvCfgDD24(DroneHoverEnvCfg):
    """DD24 airframe config. Overrides the Crazyflie dynamics scalars with the
    Duckietown DD24 numbers from `dd24_params.py`.

    STILL A PARTIAL PORT — two things remain before dynamics truly match a DD24:
      1. `robot` below still points at the Crazyflie USD mesh (there is no
         public DD24 USD yet). Until it is swapped, the body MASS and INERTIA
         read from PhysX are the Crazyflie's, not the DD24's. Build/convert a
         DD24 URDF->USD (motors at arm_length, hub at measured mass) and set
         `robot = DD24_CFG.replace(...)` here.
      2. Several DD24 params are [EST]/[TODO] — measure them (see dd24_params).
    """
    # Geometry + propulsion that ARE used by _pre_physics_step today:
    arm_length = DD24.arm_length_m              # [EST] measure wheelbase/2
    max_thrust_to_weight = DD24.thrust_to_weight  # [EST] from thrust stand
    yaw_coeff = DD24.yaw_coeff                  # [EST] identify from logs
    # TODO: robot = DD24_CFG.replace(prim_path="/World/envs/env_.*/Drone")


class DroneHoverEnv(DirectRLEnv):
    cfg: DroneHoverEnvCfg
    def __init__(self, cfg: DroneHoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros(self.num_envs, 4, device=self.device)

        self.force_b = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.torque_b = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self.body_id = self.drone.find_bodies("body")[0]

        self.prop_joint_names = [
            "m1_joint",
            "m2_joint",
            "m3_joint",
            "m4_joint",
        ]

        self.prop_joint_ids = [
            self.drone.find_joints(name)[0][0]
            for name in self.prop_joint_names
        ]

        masses = self.drone.root_physx_view.get_masses()
        self.mass = masses[0].sum().item()
        self.g = abs(self.cfg.sim.gravity[2])
        self.hover_thrust = self.mass * self.g

        self.max_total_thrust = self.hover_thrust * self.cfg.max_thrust_to_weight
        self.max_motor_thrust = self.max_total_thrust / 4.0

        print("Drone mass:", self.mass)
        print("Hover thrust:", self.hover_thrust)
        print("Max motor thrust:", self.max_motor_thrust)
        print("Joint names:", self.drone.joint_names)
        print("Prop joint ids:", self.prop_joint_ids)
        stage = self.sim.stage
        self.imu_sensors = None



    def _setup_scene(self):
        self.drone = Articulation(self.cfg.robot)
        self.scene.articulations["drone"] = self.drone

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)
    
        self._setup_imu_sensors()
    
    def _setup_imu_sensors(self):
        self.imu_sensors = []
        for i in range(self.scene.cfg.num_envs):
            imu_cfg = ImuCfg(
                prim_path=f"/World/envs/env_{i}/Drone/body",
            )
            imu = Imu(cfg=imu_cfg)
            self.imu_sensors.append(imu)
        print(f"Created {len(self.imu_sensors)} IMU sensors")

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = torch.clamp(actions.clone(), 0.0, 1.0)

        f1 = self.actions[:, 0] * self.max_motor_thrust
        f2 = self.actions[:, 1] * self.max_motor_thrust
        f3 = self.actions[:, 2] * self.max_motor_thrust
        f4 = self.actions[:, 3] * self.max_motor_thrust

        total_thrust = f1 + f2 + f3 + f4

        tau_x = self.cfg.arm_length * ((f3 + f4) - (f1 + f2))
        tau_y = self.cfg.arm_length * ((f2 + f3) - (f1 + f4))
        tau_z = self.cfg.yaw_coeff * ((f1 + f3) - (f2 + f4))

        self.force_b.zero_()
        self.torque_b.zero_()

        self.force_b[:, 0, 2] = total_thrust
        self.torque_b[:, 0, 0] = tau_x
        self.torque_b[:, 0, 1] = tau_y
        self.torque_b[:, 0, 2] = tau_z

    def _apply_action(self):
        self.drone.set_external_force_and_torque(
            self.force_b,
            self.torque_b,
            body_ids=self.body_id,
        )

        if not hasattr(self, "prop_xforms"):
            stage = self.sim.stage

            prop_paths = [
                "/World/envs/env_0/Drone/m1_prop/ccw_prop",
                "/World/envs/env_0/Drone/m2_prop/cw_prop",
                "/World/envs/env_0/Drone/m3_prop/ccw_prop",
                "/World/envs/env_0/Drone/m4_prop/cw_prop",
            ]

            self.prop_angles = [0.0, 0.0, 0.0, 0.0]
            self.prop_xforms = []

            for path in prop_paths:
                prim = stage.GetPrimAtPath(path)

                if not prim.IsValid():
                    print("BAD PROP PATH:", path)

                xform = UsdGeom.XformCommonAPI(prim)
                self.prop_xforms.append(xform)

        for i in range(4):
            if self.actions[0, i] > 0.05:
                speed = self.cfg.max_prop_visual_speed

                if i in [1, 3]:
                    speed *= -1.0

                self.prop_angles[i] += speed * self.cfg.sim.dt

            self.prop_xforms[i].SetRotate(
                (0.0, 0.0, math.degrees(self.prop_angles[i])),
                UsdGeom.XformCommonAPI.RotationOrderXYZ,
            )
    def _get_observations(self):
        root_state = self.drone.data.root_state_w

        pos_w = root_state[:, 0:3]
        quat_w = root_state[:, 3:7]
        lin_vel_w = root_state[:, 7:10]
        ang_vel_w = root_state[:, 10:13]
        
        lin_acc_b = torch.zeros(self.num_envs, 3, device=self.device)
        if self.imu_sensors is not None:
            for i, imu in enumerate(self.imu_sensors):
                try:
                    if imu.data.lin_acc_b is not None:
                        lin_acc_b[i] = imu.data.lin_acc_b[i]
                except Exception as e:
                    pass

        z_error = self.cfg.hover_height - pos_w[:, 2:3]

        obs = torch.cat(
            [
                lin_acc_b,
                z_error,
                quat_w,
                lin_vel_w,
                ang_vel_w,
            ],
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self):
        z = self.drone.data.root_state_w[:, 2]
        z_error = torch.abs(self.cfg.hover_height - z)
        return -z_error

    def _get_dones(self):
        z = self.drone.data.root_state_w[:, 2]

        crashed = z < self.cfg.crash_height
        timed_out = self.episode_length_buf >= self.max_episode_length - 1

        return crashed, timed_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.drone._ALL_INDICES

        super()._reset_idx(env_ids)

        root_state = self.drone.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] += self.scene.env_origins[env_ids]

        # Keep each env's x/y origin (zeroing them stacks every drone on env_0).
        root_state[:, 2] = self.scene.env_origins[env_ids, 2] + 0.5

        root_state[:, 7:13] = 0.0

        self.drone.write_root_pose_to_sim(root_state[:, 0:7], env_ids)
        self.drone.write_root_velocity_to_sim(root_state[:, 7:13], env_ids)
        if hasattr(self, "prop_joint_pos"):
            self.prop_joint_pos[env_ids] = 0.0
        self.drone.reset(env_ids)