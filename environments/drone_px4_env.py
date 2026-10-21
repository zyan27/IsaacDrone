"""DD24 with PX4 in the loop (Part 3).

    commander (policy / script)
        | CTBR setpoint = SET_ATTITUDE_TARGET: thrust [0,1] + FRD body rates (rad/s), 50 Hz
        v
    PX4 inner loop (controller="emulated": px4_rate_emulator, v1.15.4 gains)
        | 4 motor commands [0,1] = PX4 actuator outputs (PWM to the ESCs), 250 Hz
        v
    ESC + motor model: first-order speed lag, thrust = Fmax * speed**2
        | per-motor thrust + prop reaction torque at the DD24 motor positions
        v
    Isaac Sim rigid body with DD24 mass / inertia

The ESC's 3-phase commutation is not simulated; a first-order lag from command to
rotor speed stands in for ESC + motor dynamics. controller="sitl" (dt-px4 over
MAVLink) is the next step, see sim_px4/ARCHITECTURE.md; it plugs in where the
emulator is called and keeps the same action spec.

Visual mesh is the Crazyflie scaled to the DD24's 0.23 m motor diagonal; mass and
inertia are overridden to the DD24 values, so the CF mesh's own are irrelevant.
"""
from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul
from isaaclab_assets import CRAZYFLIE_CFG

from sim_px4.px4_defaults import ATT, QUAD_X
from .dd24_params import DD24
from .px4_rate_emulator import PX4RateEmulator

CF_MOTOR_DIAGONAL_M = 0.092  # Crazyflie 2.x motor-to-motor diagonal

# Policy output a in [-1, 1]^4 -> CTBR setpoint. Shared by training and playback.
POLICY_THRUST_RANGE = 0.25                 # +- around the hover command
POLICY_RATE_SCALE = (3.0, 3.0, 2.0)        # rad/s at |a| = 1 (roll, pitch, yaw, FRD)


def policy_to_ctbr(a: torch.Tensor, hover_cmd: float) -> torch.Tensor:
    a = torch.clamp(a, -1.0, 1.0)
    thrust = hover_cmd + POLICY_THRUST_RANGE * a[:, 0:1]
    rates = a[:, 1:4] * torch.tensor(POLICY_RATE_SCALE, device=a.device)
    return torch.cat([thrust, rates], dim=-1)


@configclass
class DronePX4EnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 5  # 250 Hz physics + PX4 loop, 50 Hz commander
    # action = [thrust 0..1, roll rate, pitch rate, yaw rate (FRD, rad/s)]
    action_space = 4
    # Policy sees only what the real DD24 measures (no GPS / optical flow):
    #   5 ToF ranges [down, front, left, right, up] (5), vertical speed (1),
    #   attitude quat (4), gyro FRD (3), accelerometer FRD (3), last CTBR setpoint (4)
    observation_space = 20
    # Training-only critic also gets ground truth: pos error (3), world lin vel (3)
    state_space = 26
    sim: SimulationCfg = SimulationCfg(dt=1 / 250, render_interval=decimation, gravity=(0.0, 0.0, -9.81))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    robot = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Drone",
        spawn=CRAZYFLIE_CFG.spawn.replace(scale=(DD24.arm_length_m * 2 / CF_MOTOR_DIAGONAL_M,) * 3),
    )

    controller = "emulated"  # "sitl" -> dt-px4 over MAVLink (not wired yet)
    hover_height = 1.0
    spawn_height = 0.3
    crash_height = 0.08
    max_tilt_rad = 1.2

    # Indoor room around each drone: the side/top ToFs range off its walls and
    # ceiling, which is what makes position in the room observable.
    room_half_width_m = 1.5
    room_height_m = 2.5
    wall_clearance_m = 0.17   # arm 0.115 + prop radius 0.05
    spawn_xy_range_m = 0.8

    # Sensor model (1-sigma). ToF: 5x VL53L1X (bottom, front, left, right, top)
    # on the DD24-B, read by the Pi through the HUT.
    tof_range_max_m = 4.0
    tof_noise_m = 0.01
    vz_noise_mps = 0.03      # PX4 EKF vertical speed from ToF + accel
    att_noise_rad = 0.01
    gyro_noise_rps = 0.01
    accel_noise_mps2 = 0.05

    mass_kg = DD24.mass_kg
    inertia_kgm2 = DD24.inertia_kgm2
    lever_arm_m = DD24.lever_arm_m
    max_thrust_per_motor_N = DD24.max_thrust_per_motor_N
    thrust_curve_exponent = DD24.thrust_curve_exponent
    motor_time_constant_s = DD24.motor_time_constant_s
    yaw_coeff = DD24.yaw_coeff


class DronePX4Env(DirectRLEnv):
    cfg: DronePX4EnvCfg

    def __init__(self, cfg: DronePX4EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if cfg.controller != "emulated":
            raise NotImplementedError("controller='sitl' needs sim_px4/mavlink_bridge.py (next step)")

        n, dev = self.num_envs, self.device
        self.body_id = self.drone.find_bodies("body")[0]
        self._set_dd24_mass_properties()

        self.px4 = PX4RateEmulator(n, self.physics_dt, dev)
        self.ctbr = torch.zeros(n, 4, device=dev)       # last setpoint sent to PX4
        self.motor_cmd = torch.zeros(n, 4, device=dev)  # PX4 actuator outputs
        self.motor_speed = torch.zeros(n, 4, device=dev)  # normalized rotor speed
        self.rate_max = torch.tensor(ATT.rate_max_deg_s, device=dev) * math.pi / 180.0

        # Motor geometry in Isaac's FLU body frame (PX4 lists FRD: flip y).
        px, py = torch.tensor(QUAD_X.px), torch.tensor(QUAD_X.py)
        self.motor_x = (px * cfg.lever_arm_m).to(dev)
        self.motor_y = (-py * cfg.lever_arm_m).to(dev)
        # km > 0 = CCW prop = clockwise reaction torque seen from above = -z in FLU.
        self.motor_yaw_sign = (-torch.sign(torch.tensor(QUAD_X.km))).to(dev)

        self.hover_cmd = (cfg.mass_kg * 9.81 / (4 * cfg.max_thrust_per_motor_N)) ** (1 / cfg.thrust_curve_exponent)
        self.force_b = torch.zeros(n, 1, 3, device=dev)
        self.torque_b = torch.zeros(n, 1, 3, device=dev)
        print(f"[DronePX4Env] mass {cfg.mass_kg:.3f} kg, inertia {cfg.inertia_kgm2}, "
              f"hover motor cmd {self.hover_cmd:.3f}, PX4 loop {1 / self.physics_dt:.0f} Hz")

    def _setup_scene(self):
        self.drone = Articulation(self.cfg.robot)
        self.scene.articulations["drone"] = self.drone
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._spawn_room("/World/envs/env_0/Room")
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_room(self, root: str):
        """Visual-only walls (ToF ranges are computed analytically against the same
        planes). The ceiling is left invisible so the camera can see in."""
        a, h, th = self.cfg.room_half_width_m, self.cfg.room_height_m, 0.04
        mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.72, 0.65), opacity=0.35)
        walls = {
            "wall_front": ((a + th / 2, 0.0, h / 2), (th, 2 * a, h)),
            "wall_back": ((-a - th / 2, 0.0, h / 2), (th, 2 * a, h)),
            "wall_left": ((0.0, a + th / 2, h / 2), (2 * a + 2 * th, th, h)),
            "wall_right": ((0.0, -a - th / 2, h / 2), (2 * a + 2 * th, th, h)),
        }
        for name, (pos, size) in walls.items():
            wall = sim_utils.CuboidCfg(size=size, visual_material=mat)
            wall.func(f"{root}/{name}", wall, translation=pos)

    def _set_dd24_mass_properties(self):
        # Put all DD24 mass/inertia on the hub; the CF prop links become near-massless.
        view = self.drone.root_physx_view
        ids = torch.arange(self.num_envs, dtype=torch.int32)
        masses = view.get_masses()
        inertias = view.get_inertias()
        prop_mass = 1e-3
        masses[:] = prop_mass
        masses[:, self.body_id] = self.cfg.mass_kg - prop_mass * (masses.shape[1] - 1)
        inertias[:] = torch.tensor([1e-6, 0, 0, 0, 1e-6, 0, 0, 0, 1e-6])
        ixx, iyy, izz = self.cfg.inertia_kgm2
        inertias[:, self.body_id] = torch.tensor([ixx, 0, 0, 0, iyy, 0, 0, 0, izz])
        view.set_masses(masses, ids)
        view.set_inertias(inertias, ids)

    # --- commander -> PX4 ------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        thrust = torch.clamp(actions[:, 0], 0.0, 1.0)
        rates = torch.clamp(actions[:, 1:4], -self.rate_max, self.rate_max)
        self.ctbr = torch.cat([thrust.unsqueeze(-1), rates], dim=-1)

    # --- PX4 -> motors -> body, every physics step -------------------------------
    def _apply_action(self):
        gyro_flu = self.drone.data.root_ang_vel_b
        gyro_frd = gyro_flu * torch.tensor([1.0, -1.0, -1.0], device=self.device)
        landed = self.drone.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2] < self.cfg.crash_height + 0.02
        self.motor_cmd = self.px4.step(self.ctbr[:, 0], self.ctbr[:, 1:4], gyro_frd, landed)

        alpha = self.physics_dt / (self.cfg.motor_time_constant_s + self.physics_dt)
        self.motor_speed += alpha * (self.motor_cmd - self.motor_speed)
        thrust = self.cfg.max_thrust_per_motor_N * self.motor_speed.pow(self.cfg.thrust_curve_exponent)

        self.force_b.zero_()
        self.force_b[:, 0, 2] = thrust.sum(-1)
        self.torque_b[:, 0, 0] = (self.motor_y * thrust).sum(-1)
        self.torque_b[:, 0, 1] = (-self.motor_x * thrust).sum(-1)
        self.torque_b[:, 0, 2] = (self.motor_yaw_sign * self.cfg.yaw_coeff * thrust).sum(-1)
        self.drone.set_external_force_and_torque(self.force_b, self.torque_b, body_ids=self.body_id)

    # --- RL interface ------------------------------------------------------------
    def _sensors(self):
        """What the DD24 can measure: ToF range, EKF vertical speed and attitude, IMU."""
        d, n, dev = self.drone.data, self.num_envs, self.device
        noise = lambda k, sigma: sigma * torch.randn(n, k, device=dev)  # noqa: E731
        tof = (self._tof_ranges() + noise(5, self.cfg.tof_noise_m)).clamp(0.0, self.cfg.tof_range_max_m)
        vz = d.root_lin_vel_w[:, 2:3] + noise(1, self.cfg.vz_noise_mps)
        dq = torch.cat([torch.ones(n, 1, device=dev), 0.5 * noise(3, self.cfg.att_noise_rad)], dim=-1)
        quat = quat_mul(d.root_quat_w, dq / dq.norm(dim=-1, keepdim=True))
        frd = torch.tensor([1.0, -1.0, -1.0], device=dev)
        gyro = d.root_ang_vel_b * frd + noise(3, self.cfg.gyro_noise_rps)
        # Specific force (what an accelerometer reads) = body accel minus gravity, in body frame.
        acc_w = d.body_lin_acc_w[:, self.body_id[0]] - torch.tensor(self.cfg.sim.gravity, device=dev)
        accel = quat_apply_inverse(d.root_quat_w, acc_w) * frd + noise(3, self.cfg.accel_noise_mps2)
        return torch.cat([tof, vz, quat, gyro, accel, self.ctbr], dim=-1)

    def _tof_ranges(self):
        """Ray from the body along each ToF axis to the room's floor, walls and ceiling.
        Order: down, front, left, right, up (FLU body axes). Beyond max range reads max."""
        d = self.drone.data
        dirs_b = torch.tensor([[0, 0, -1], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
                              dtype=torch.float32, device=self.device)
        n = self.num_envs
        dirs_w = quat_apply(d.root_quat_w.repeat_interleave(5, 0), dirs_b.repeat(n, 1)).view(n, 5, 3)
        p = (d.root_pos_w - self.scene.env_origins).unsqueeze(1)  # (n, 1, 3)
        a, h = self.cfg.room_half_width_m, self.cfg.room_height_m
        lo = torch.tensor([-a, -a, 0.0], device=self.device)
        hi = torch.tensor([a, a, h], device=self.device)
        safe = torch.where(dirs_w.abs() < 1e-6, torch.full_like(dirs_w, 1e-6), dirs_w)
        t = torch.where(dirs_w > 0, (hi - p) / safe, (lo - p) / safe)
        t = torch.where(dirs_w.abs() < 1e-6, torch.full_like(t, 1e6), t).clamp(min=0.0)
        return t.min(dim=-1).values.clamp(max=self.cfg.tof_range_max_m)

    def _get_observations(self):
        d = self.drone.data
        sensors = self._sensors()
        target = self.scene.env_origins.clone()
        target[:, 2] += self.cfg.hover_height
        critic = torch.cat([sensors, target - d.root_pos_w, d.root_lin_vel_w], dim=-1)
        return {"policy": sensors, "critic": critic}

    def _get_rewards(self):
        # Ground truth is fine here: the reward is only used in training.
        d = self.drone.data
        z = d.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        xy = d.root_pos_w[:, 0:2] - self.scene.env_origins[:, 0:2]
        return (
            1.0
            - 2.0 * (z - self.cfg.hover_height).abs()
            - 1.0 * torch.linalg.norm(xy, dim=-1)
            - 0.3 * torch.linalg.norm(d.root_lin_vel_w[:, 0:2], dim=-1)
            - 0.3 * d.root_lin_vel_w[:, 2].abs()
            - 0.3 * self._tilt()
            - 0.05 * torch.linalg.norm(d.root_ang_vel_b, dim=-1)
        )

    def _tilt(self):
        up_b = quat_apply_inverse(self.drone.data.root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))
        return torch.acos(torch.clamp(up_b[:, 2], -1.0, 1.0))

    def _get_dones(self):
        z = self.drone.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        flying = self.episode_length_buf > 25  # allow liftoff from the spawn height
        crashed = (z < self.cfg.crash_height) & flying
        crashed |= self._tilt() > self.cfg.max_tilt_rad
        pos = self.drone.data.root_pos_w - self.scene.env_origins
        wall = self.cfg.room_half_width_m - self.cfg.wall_clearance_m
        crashed |= (pos[:, 0:2].abs() > wall).any(-1) | (z > self.cfg.room_height_m - 0.1)
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return crashed, timed_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.drone._ALL_INDICES
        super()._reset_idx(env_ids)
        root_state = self.drone.data.default_root_state[env_ids].clone()
        k = len(env_ids)
        root_state[:, 0:3] = self.scene.env_origins[env_ids]
        r = self.cfg.spawn_xy_range_m
        root_state[:, 0:2] += (torch.rand(k, 2, device=self.device) * 2 - 1) * r
        root_state[:, 2] += self.cfg.spawn_height
        yaw = (torch.rand(k, device=self.device) * 2 - 1) * math.pi
        root_state[:, 3:7] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
        root_state[:, 7:13] = 0.0
        self.drone.write_root_pose_to_sim(root_state[:, 0:7], env_ids)
        self.drone.write_root_velocity_to_sim(root_state[:, 7:13], env_ids)
        self.drone.reset(env_ids)
        if hasattr(self, "px4"):
            self.px4.reset(env_ids)
            self.ctbr[env_ids] = 0.0
            self.motor_cmd[env_ids] = 0.0
            self.motor_speed[env_ids] = 0.0
