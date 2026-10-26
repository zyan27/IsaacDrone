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

The task is the real-drone test: start on the floor, climb straight up to a
commanded height, hold, come straight down, land. The policy sees only what the
DD24 reliably gives the Pi (flight logs 2026-10-25, data/flight_logs/):
  - the downward ToF (on the drone: /noducksgiven/right_tof_driver_node/range,
    the only one that reads ~0 on the floor and tracks the hops)
  - /mavros/imu/data: attitude quaternion, gyro, accelerometer
PX4's local position / vertical velocity is not used: in the logs it drifts by
metres while the drone sits still. Side / top ToFs are not used either (net walls).

Every sensor stream is sampled-and-held at its own rate, arrives late, and drops
out (the logs: ToF 17-20 Hz, IMU 50 Hz with gaps up to 1 s); the policy also sees
each stream's sample age, which the Pi computes as now - header.stamp.

Mass, inertia, motor lag and thrust (hover thrust 0.42-0.52 around the logged
0.47) are randomised so the policy doesn't depend on the exact estimates.

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

# Policy output a in [-1, 1]^4 -> CTBR setpoint. Shared by training, playback and
# the Pi node (deploy/policy_node.py), so all three send PX4 the same thing.
POLICY_HOVER_THRUST = 0.47                 # logged hover thrust setpoint, 2026-10-25
POLICY_THRUST_RANGE = 0.25                 # +- around the hover thrust
POLICY_RATE_SCALE = (3.0, 3.0, 2.0)        # rad/s at |a| = 1 (roll, pitch, yaw, FRD)


def policy_to_ctbr(a: torch.Tensor, hover_cmd: float = POLICY_HOVER_THRUST) -> torch.Tensor:
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
    # Policy observation (each sensor value is the latest sample of its stream):
    #   down ToF (1), attitude quat wxyz (4), gyro FRD (3), accel FRD (3)   = 11 held values
    #   sample age, s (2: ToF, IMU), last CTBR setpoint (4), commanded height (1)
    observation_space = 18
    # Training-only critic also gets ground truth: pos error (3), world lin vel (3)
    state_space = 24
    sim: SimulationCfg = SimulationCfg(dt=1 / 250, render_interval=decimation, gravity=(0.0, 0.0, -9.81))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    robot = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Drone",
        spawn=CRAZYFLIE_CFG.spawn.replace(scale=(DD24.arm_length_m * 2 / CF_MOTOR_DIAGONAL_M,) * 3),
    )

    controller = "emulated"  # "sitl" -> dt-px4 over MAVLink (not wired yet)

    # Height command (ToF reading to hold): wait on the floor, ramp straight up,
    # hold, sometimes come back down and land, repeat.
    hover_height = 0.5            # play_px4.py scripted default
    target_z_range_m = (0.3, 1.0)
    target_hold_s = (2.0, 6.0)
    target_start_s = (0.5, 2.0)   # idle on the floor before the first climb
    target_land_prob = 0.3
    target_speed_mps = 0.3
    max_tilt_rad = 1.0

    # Net enclosure around each drone (visual walls; touching one ends the episode).
    room_half_width_m = 1.5
    room_height_m = 2.5
    wall_clearance_m = 0.17   # arm 0.115 + prop radius 0.05
    spawn_xy_range_m = 0.5
    crash_penalty = -10.0

    # Sensor model (1-sigma). Down ToF: VL53L1X, reads ~0 on the floor, ~1.5 mm
    # noise at rest; the logs show ~1 cm in flight.
    tof_range_max_m = 2.0
    tof_noise_m = 0.01
    att_noise_rad = 0.01
    gyro_noise_rps = 0.003
    accel_noise_mps2 = 0.05

    # Sensor timing, per stream, drawn per episode (logs: ToF 17-20 Hz, IMU ~50 Hz,
    # ~0.2 IMU gaps/s of up to 1 s).
    tof_rate_hz = (12.0, 25.0)
    imu_rate_hz = (35.0, 60.0)
    tof_latency_s = (0.010, 0.060)
    imu_latency_s = (0.002, 0.025)
    dropout_per_s = (0.1, 0.3)    # ToF, IMU
    dropout_s = (0.05, 1.0)
    age_clip_s = 1.0
    cmd_latency_s = (0.010, 0.050)  # Pi -> mavros -> PX4 setpoint delay

    # Physical parameters and their per-env randomisation (multiplicative unless noted).
    mass_kg = DD24.mass_kg
    inertia_kgm2 = DD24.inertia_kgm2
    lever_arm_m = DD24.lever_arm_m
    thrust_curve_exponent = DD24.thrust_curve_exponent
    motor_time_constant_s = DD24.motor_time_constant_s
    yaw_coeff = DD24.yaw_coeff
    hover_thrust_range = (0.42, 0.52)   # absolute; sets Fmax per episode
    mass_scale = (0.9, 1.1)
    inertia_scale = (0.7, 1.3)
    motor_tau_s = (0.01, 0.04)          # absolute


class DronePX4Env(DirectRLEnv):
    cfg: DronePX4EnvCfg

    def __init__(self, cfg: DronePX4EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if cfg.controller != "emulated":
            raise NotImplementedError("controller='sitl' needs sim_px4/mavlink_bridge.py (next step)")

        n, dev = self.num_envs, self.device
        self.body_id = self.drone.find_bodies("body")[0]
        u = lambda r, k=n: r[0] + (r[1] - r[0]) * torch.rand(k, device=dev)  # noqa: E731
        self.mass = cfg.mass_kg * u(cfg.mass_scale)
        self.motor_tau = u(cfg.motor_tau_s)
        self.fmax = torch.zeros(n, device=dev)
        self._set_mass_properties(u(cfg.inertia_scale))

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

        self.force_b = torch.zeros(n, 1, 3, device=dev)
        self.torque_b = torch.zeros(n, 1, 3, device=dev)

        # Sensor streams: raw vector = [tof 1, quat 4, gyro 3, accel 3];
        # stream 0 = down ToF, 1 = IMU (quat + gyro + accel, one mavros topic).
        self.col_stream = torch.tensor([0] + [1] * 10, device=dev)
        self.stream_rate = [cfg.tof_rate_hz, cfg.imu_rate_hz]
        self.stream_lat = [cfg.tof_latency_s, cfg.imu_latency_s]
        max_lat = max(hi for _, hi in self.stream_lat)
        self.hist_len = int(math.ceil(max_lat / self.physics_dt)) + 2
        self.hist = torch.zeros(self.hist_len, n, 11, device=dev)
        self.hist_ptr = 0
        self.held = torch.zeros(n, 11, device=dev)
        self.age = torch.zeros(n, 2, device=dev)
        self.period = torch.ones(n, 2, device=dev)
        self.timer = torch.zeros(n, 2, device=dev)
        self.lat_steps = torch.zeros(n, 2, dtype=torch.long, device=dev)
        self.drop_rate = torch.zeros(n, 2, device=dev)
        self.drop_left = torch.zeros(n, 2, device=dev)
        self.cmd_hist_len = int(math.ceil(cfg.cmd_latency_s[1] / self.physics_dt)) + 1
        self.cmd_hist = torch.zeros(self.cmd_hist_len, n, 4, device=dev)
        self.cmd_delay = torch.zeros(n, dtype=torch.long, device=dev)

        self.target_z = torch.zeros(n, device=dev)
        self.target_goal = torch.zeros(n, device=dev)
        self.target_timer = torch.zeros(n, device=dev)

        self.rest_z = self._measure_rest_height()
        print(f"[DronePX4Env] mass {cfg.mass_kg:.3f} kg, inertia {cfg.inertia_kgm2}, "
              f"rest height {self.rest_z:.3f} m, PX4 loop {1 / self.physics_dt:.0f} Hz")

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
        """Visual-only walls. The ceiling is left invisible so the camera can see in."""
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

    def _set_mass_properties(self, inertia_scale: torch.Tensor):
        # Put all DD24 mass/inertia on the hub; the CF prop links become near-massless.
        view = self.drone.root_physx_view
        ids = torch.arange(self.num_envs, dtype=torch.int32)
        masses = view.get_masses()
        inertias = view.get_inertias()
        prop_mass = 1e-3
        masses[:] = prop_mass
        masses[:, self.body_id] = (self.mass.cpu() - prop_mass * (masses.shape[1] - 1)).unsqueeze(-1)
        inertias[:] = torch.tensor([1e-6, 0, 0, 0, 1e-6, 0, 0, 0, 1e-6])
        diag = torch.tensor(self.cfg.inertia_kgm2) * inertia_scale.cpu().unsqueeze(-1)  # (n, 3)
        hub = torch.zeros(self.num_envs, 9)
        hub[:, 0], hub[:, 4], hub[:, 8] = diag[:, 0], diag[:, 1], diag[:, 2]
        inertias[:, self.body_id] = hub.unsqueeze(1)
        view.set_masses(masses, ids)
        view.set_inertias(inertias, ids)

    def _measure_rest_height(self) -> float:
        """Drop every drone from 0.15 m with motors off and read where the body
        settles: that height is ToF = 0, as on the real drone."""
        state = self.drone.data.default_root_state.clone()
        state[:, 0:3] = self.scene.env_origins
        state[:, 2] += 0.15
        self.drone.write_root_pose_to_sim(state[:, 0:7])
        self.drone.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:13]))
        for _ in range(250):
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        z = self.drone.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        return float(z.median())

    # --- commander -> PX4 ------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        thrust = torch.clamp(actions[:, 0], 0.0, 1.0)
        rates = torch.clamp(actions[:, 1:4], -self.rate_max, self.rate_max)
        self.ctbr = torch.cat([thrust.unsqueeze(-1), rates], dim=-1)
        self._step_target(self.step_dt)

    def _step_target(self, dt: float):
        """Commanded height: wait, ramp straight up/down to a new height, hold; some
        holds are at 0 (landed)."""
        lo, hi = self.cfg.target_z_range_m
        self.target_timer -= dt
        new = self.target_timer <= 0
        k = int(new.sum())
        if k:
            goal = lo + (hi - lo) * torch.rand(k, device=self.device)
            land = (torch.rand(k, device=self.device) < self.cfg.target_land_prob) & (self.target_goal[new] > 0)
            self.target_goal[new] = torch.where(land, torch.zeros_like(goal), goal)
            h0, h1 = self.cfg.target_hold_s
            ramp = (self.target_goal[new] - self.target_z[new]).abs() / self.cfg.target_speed_mps
            self.target_timer[new] = ramp + h0 + (h1 - h0) * torch.rand(k, device=self.device)
        step = self.cfg.target_speed_mps * dt
        self.target_z += (self.target_goal - self.target_z).clamp(-step, step)

    # --- PX4 -> motors -> body, every physics step -------------------------------
    def _apply_action(self):
        self._sample_sensors()
        # Setpoint reaches PX4 after a per-episode delay.
        p = self.hist_ptr % self.cmd_hist_len
        self.cmd_hist[p] = self.ctbr
        idx = (p - self.cmd_delay) % self.cmd_hist_len
        sp = self.cmd_hist[idx, torch.arange(self.num_envs, device=self.device)]

        gyro_flu = self.drone.data.root_ang_vel_b
        gyro_frd = gyro_flu * torch.tensor([1.0, -1.0, -1.0], device=self.device)
        landed = self._height() < 0.02
        self.motor_cmd = self.px4.step(sp[:, 0], sp[:, 1:4], gyro_frd, landed)

        alpha = (self.physics_dt / (self.motor_tau + self.physics_dt)).unsqueeze(-1)
        self.motor_speed += alpha * (self.motor_cmd - self.motor_speed)
        thrust = self.fmax.unsqueeze(-1) * self.motor_speed.pow(self.cfg.thrust_curve_exponent)

        self.force_b.zero_()
        self.force_b[:, 0, 2] = thrust.sum(-1)
        self.torque_b[:, 0, 0] = (self.motor_y * thrust).sum(-1)
        self.torque_b[:, 0, 1] = (-self.motor_x * thrust).sum(-1)
        self.torque_b[:, 0, 2] = (self.motor_yaw_sign * self.cfg.yaw_coeff * thrust).sum(-1)
        self.drone.set_external_force_and_torque(self.force_b, self.torque_b, body_ids=self.body_id)

    # --- sensors -------------------------------------------------------------------
    def _height(self):
        """Body height above its resting height (0 when sitting on the floor)."""
        return self.drone.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2] - self.rest_z

    def _raw_sensors(self):
        """What the DD24 measures, every physics step: down ToF, IMU attitude/gyro/accel."""
        d, n, dev = self.drone.data, self.num_envs, self.device
        noise = lambda k, sigma: sigma * torch.randn(n, k, device=dev)  # noqa: E731
        # Range along the body's down axis to the floor.
        down_w = quat_apply(d.root_quat_w, torch.tensor([0.0, 0.0, -1.0], device=dev).expand(n, 3))
        cos = (-down_w[:, 2]).clamp(min=0.2)
        tof = (self._height().clamp(min=0.0) / cos).unsqueeze(-1) + noise(1, self.cfg.tof_noise_m)
        tof = tof.clamp(0.0, self.cfg.tof_range_max_m)
        dq = torch.cat([torch.ones(n, 1, device=dev), 0.5 * noise(3, self.cfg.att_noise_rad)], dim=-1)
        quat = quat_mul(d.root_quat_w, dq / dq.norm(dim=-1, keepdim=True))
        frd = torch.tensor([1.0, -1.0, -1.0], device=dev)
        gyro = d.root_ang_vel_b * frd + noise(3, self.cfg.gyro_noise_rps)
        # Specific force (what an accelerometer reads) = body accel minus gravity, in body frame.
        acc_w = d.body_lin_acc_w[:, self.body_id[0]] - torch.tensor(self.cfg.sim.gravity, device=dev)
        accel = quat_apply_inverse(d.root_quat_w, acc_w) * frd + noise(3, self.cfg.accel_noise_mps2)
        return torch.cat([tof, quat, gyro, accel], dim=-1)

    def _sample_sensors(self):
        """Record the noisy readings, then let each stream that is due and not
        dropped out publish its reading from lat_steps ago."""
        dt, n, dev = self.physics_dt, self.num_envs, self.device
        self.hist_ptr += 1
        p = self.hist_ptr % self.hist_len
        self.hist[p] = self._raw_sensors()

        self.timer -= dt
        self.drop_left -= dt
        start = torch.rand(n, 2, device=dev) < self.drop_rate * dt
        lo, hi = self.cfg.dropout_s
        self.drop_left = torch.where(start & (self.drop_left <= 0), lo + (hi - lo) * torch.rand(n, 2, device=dev), self.drop_left)
        due = self.timer <= 0
        self.timer = torch.where(due, self.timer + self.period, self.timer)
        publish = due & (self.drop_left <= 0)

        lat_cols = self.lat_steps[:, self.col_stream]                     # (n, 11)
        idx = ((p - lat_cols) % self.hist_len).unsqueeze(0)               # (1, n, 11)
        delayed = self.hist.gather(0, idx).squeeze(0)
        self.held = torch.where(publish[:, self.col_stream], delayed, self.held)
        self.age = torch.where(publish, self.lat_steps.float() * dt, self.age + dt)

    def _reset_streams(self, env_ids: torch.Tensor):
        k, dev, dt = len(env_ids), self.device, self.physics_dt
        u = lambda r: r[0] + (r[1] - r[0]) * torch.rand(k, device=dev)  # noqa: E731
        self.period[env_ids] = torch.stack([1.0 / u(r) for r in self.stream_rate], -1)
        self.timer[env_ids] = torch.rand(k, 2, device=dev) * self.period[env_ids]
        self.lat_steps[env_ids] = torch.stack([(u(r) / dt).round().long() for r in self.stream_lat], -1)
        self.drop_rate[env_ids] = u(self.cfg.dropout_per_s).unsqueeze(-1).expand(k, 2)
        self.drop_left[env_ids] = 0.0
        self.cmd_delay[env_ids] = (u(self.cfg.cmd_latency_s) / dt).round().long().clamp(max=self.cmd_hist_len - 1)
        self.cmd_hist[:, env_ids] = 0.0
        raw = self._raw_sensors()[env_ids]
        self.hist[:, env_ids] = raw
        self.held[env_ids] = raw
        self.age[env_ids] = 0.0

    # --- RL interface ------------------------------------------------------------
    def _sensors(self):
        """Policy observation: held sensor values, their ages, last setpoint, height command."""
        age = self.age.clamp(max=self.cfg.age_clip_s)
        return torch.cat([self.held, age, self.ctbr, self.target_z.unsqueeze(-1)], dim=-1)

    def _get_observations(self):
        d = self.drone.data
        sensors = self._sensors()
        pos_err = torch.zeros(self.num_envs, 3, device=self.device)
        pos_err[:, 0:2] = -(d.root_pos_w[:, 0:2] - self.scene.env_origins[:, 0:2])
        pos_err[:, 2] = self.target_z - self._height()
        critic = torch.cat([sensors, pos_err, d.root_lin_vel_w], dim=-1)
        return {"policy": sensors, "critic": critic}

    def _get_rewards(self):
        # Ground truth is fine here: the reward is only used in training.
        # Every term is >= 0 while flying, so staying airborne always beats crashing.
        # No xy-position term: nothing the policy sees can tell it where it is
        # sideways, so it is rewarded for not moving sideways instead.
        d = self.drone.data
        z_err = (self._height() - self.target_z).abs()
        h_speed = torch.linalg.norm(d.root_lin_vel_w[:, 0:2], dim=-1)
        v_err = d.root_lin_vel_w[:, 2].abs()
        spin = torch.linalg.norm(d.root_ang_vel_b, dim=-1)
        r = (
            0.25                                 # alive
            + 1.0 * torch.exp(-5.0 * z_err)
            + 0.5 * torch.exp(-3.0 * h_speed)
            + 0.25 * torch.exp(-4.0 * self._tilt())
            + 0.25 * torch.exp(-2.0 * v_err)
            + 0.25 * torch.exp(-0.5 * spin)
        )
        return torch.where(self.reset_terminated, torch.full_like(r, self.cfg.crash_penalty), r)

    def _tilt(self):
        up_b = quat_apply_inverse(self.drone.data.root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))
        return torch.acos(torch.clamp(up_b[:, 2], -1.0, 1.0))

    def _get_dones(self):
        crashed = self._tilt() > self.cfg.max_tilt_rad
        pos = self.drone.data.root_pos_w - self.scene.env_origins
        wall = self.cfg.room_half_width_m - self.cfg.wall_clearance_m
        crashed |= (pos[:, 0:2].abs() > wall).any(-1) | (pos[:, 2] > self.cfg.room_height_m - 0.1)
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
        # On the floor (rest_z doesn't exist yet during the first reset in __init__).
        root_state[:, 2] += getattr(self, "rest_z", 0.15) + 0.002
        yaw = (torch.rand(k, device=self.device) * 2 - 1) * math.pi
        root_state[:, 3:7] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
        root_state[:, 7:13] = 0.0
        self.drone.write_root_pose_to_sim(root_state[:, 0:7], env_ids)
        self.drone.write_root_velocity_to_sim(root_state[:, 7:13], env_ids)
        self.drone.reset(env_ids)
        if hasattr(self, "rest_z"):
            dev = self.device
            self.px4.reset(env_ids)
            self.ctbr[env_ids] = 0.0
            self.motor_cmd[env_ids] = 0.0
            self.motor_speed[env_ids] = 0.0
            h = self.cfg.hover_thrust_range
            hover = h[0] + (h[1] - h[0]) * torch.rand(k, device=dev)
            self.fmax[env_ids] = self.mass[env_ids] * 9.81 / (4 * hover.pow(self.cfg.thrust_curve_exponent))
            self.target_z[env_ids] = 0.0
            self.target_goal[env_ids] = 0.0
            s0, s1 = self.cfg.target_start_s
            self.target_timer[env_ids] = s0 + (s1 - s0) * torch.rand(k, device=dev)
            self._reset_streams(env_ids)
