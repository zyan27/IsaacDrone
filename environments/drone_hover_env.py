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
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.sensors.imu import Imu, ImuCfg

from pxr import UsdGeom

from .dd24_params import (DD24, DD24_TOF_SENSORS, TOF_MAX_RANGE_M, TOF_MIN_RANGE_M,
                          TOF_NOISE_STD_M, TOF_NOISE_STD_FRAC)
from . import arena as A
from .scenes import SCENARIOS, assign, empty_scenes, sample_scenes, scene_ranges
from .px4_ctbr import Px4RateController
from .sensor_timing import HeldSensor


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4) in (w, x, y, z) order."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def tof_ranges(pos_local, quat, sensor_pos_b, sensor_dir_b, half_xy, height):
    """True range from each ToF sensor to the floor / net walls / net ceiling.

    pos_local: (N, 3) drone CG position relative to the arena centre (floor at z=0)
    quat:      (N, 4) drone attitude (w, x, y, z)
    sensor_pos_b, sensor_dir_b: (S, 3) mounting point and pointing direction in the body frame
    half_xy:   (2,) half-size of the net; height: ceiling height (<= 0 = no ceiling)
    Returns (range (N, S), hit_net (N, S) bool). From inside the box the nearest
    positive plane crossing is where the ray exits, i.e. what the sensor sees.
    """
    N, S = pos_local.shape[0], sensor_pos_b.shape[0]
    q = quat[:, None, :].expand(N, S, 4)
    origin = pos_local[:, None, :] + quat_rotate(q, sensor_pos_b[None].expand(N, S, 3))
    d = quat_rotate(q, sensor_dir_b[None].expand(N, S, 3))
    inf = torch.full((N, S), float("inf"), device=pos_local.device)

    def plane(axis, value):
        dk = d[..., axis]
        t = (value - origin[..., axis]) / torch.where(dk.abs() < 1e-9, torch.full_like(dk, 1e-9), dk)
        return torch.where(t > 0, t, inf)

    t_floor = plane(2, 0.0)
    t_net = torch.minimum(torch.minimum(plane(0, half_xy[0]), plane(0, -half_xy[0])),
                          torch.minimum(plane(1, half_xy[1]), plane(1, -half_xy[1])))
    if height > 0:
        t_net = torch.minimum(t_net, plane(2, height))
    rng = torch.minimum(t_floor, t_net)
    return rng, t_net < t_floor


@configclass
class DroneHoverEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    # 200 Hz physics (the simulated PX4 rate loop runs every physics step),
    # 50 Hz policy = the MAVROS node's timer rate.
    decimation = 4
    action_space = 4
    # ---- What the policy observes ----------------------------------------------
    # "onboard" (default): ONLY what the real drone measures, with its imperfections:
    #     per step [accel_b(3), gyro_b(3), gravity_dir_b(3), tof(S), prev_cmd(4),
    #               flow_vel_b(2), flow_valid(1), target_height(1)]
    #     stacked over the last `obs_history` policy steps, so velocity has to be
    #     inferred from how the readings change. No ground-truth height / position /
    #     velocity. tof = range / tof_max_range in [0, 1].
    # "privileged": the old ground-truth layout, kept for the PID demo:
    #     [imu_acc(3), z_err(1), quat(4), lin_vel(3), gyro_b(3), xy_err(2), tof(S)]
    # The CRITIC always gets ground truth ("critic" obs). It only exists during
    # training, so this costs nothing on the real drone and makes learning far easier.
    obs_mode = "onboard"
    obs_history = 8          # 160 ms: spans 3+ ToF and 2+ flow updates
    observation_space = 0   # set in __post_init__ from obs_mode / sensors / history
    state_space = 0         # set in __post_init__
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
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
    imu: ImuCfg = ImuCfg(prim_path="/World/envs/env_.*/Drone/body")

    # ---- What the policy commands --------------------------------------------
    # "ctbr" (default): collective thrust + body rates, i.e. what a MAVROS node sends
    #   on /mavros/setpoint_raw/attitude in OFFBOARD; a simulated PX4 rate controller
    #   (environments/px4_ctbr.py) turns it into motor commands at 200 Hz.
    # "motors": the old direct per-motor commands (kept for old checkpoints / PID).
    # Policy action a in [-1, 1]^4 (from the sigmoid output s: a = 2s - 1):
    #   thrust = ctbr_thrust_center + a0 * ctbr_thrust_span  (normalised 0..1, PX4 style)
    #   body rates (FLU, rad/s) = a1..3 * ctbr_max_rates
    control_mode = "ctbr"
    ctbr_thrust_center = 0.27
    ctbr_thrust_span = 0.20
    ctbr_max_rates = (2.5, 2.5, 1.5)
    # Per-episode mismatch between this PX4 model and the real one:
    px4_gain_scale_range = (0.8, 1.2)     # rate P/I/D
    px4_mix_scale_range = (0.75, 1.25)    # allocation / torque effectiveness
    motor_tau_range = (0.02, 0.05)        # s, ESC + motor spin-up
    thrust_curve_k_range = (0.2, 0.6)     # thrust = (1-k) u + k u^2 (THR_MDL_FAC = 0 on the drone)

    # ---- Sensor timing (what the MAVROS node actually receives) -------------------
    imu_rate_hz_range = (40.0, 60.0)       # /mavros/imu/data
    imu_latency_s_range = (0.0, 0.03)
    tof_rate_hz_range = (18.0, 22.0)       # 20 Hz ToFs
    tof_latency_s_range = (0.03, 0.06)

    # ---- Laggy links (per episode). Drops: P(an update is lost); stalls: rate of
    # 0.1-0.4 s freezes per second. Ranges start at 0, so calm episodes stay common.
    sensor_drop_prob_range = (0.0, 0.15)   # ToF / flow updates lost
    imu_drop_prob_range = (0.0, 0.05)      # /mavros/imu/data messages lost
    sensor_stall_rate_range = (0.0, 0.2)   # ToF / flow freezes per second
    imu_stall_rate_range = (0.0, 0.05)
    # Command link: node -> MAVROS -> serial -> PX4 (+ Pi compute time).
    cmd_latency_s_range = (0.0, 0.04)
    cmd_drop_prob_range = (0.0, 0.10)      # setpoint lost: PX4 keeps flying the last one
    cmd_stall_rate_range = (0.0, 0.05)

    hover_height = 1.0
    crash_height = A.CRASH_HEIGHT
    # Perpendicular lever arm used by the mixer below (NOT center-to-motor for an
    # X frame: lever = center_to_motor / sqrt(2)).
    arm_length = 0.046
    max_thrust_to_weight = 2.2
    yaw_coeff = 0.00001
    max_prop_visual_speed = 50000.0
    # ---- Time-of-flight range sensors -----------------------------------------
    # (name, position in body frame (m), pointing direction). Each is ray-cast from
    # its own mounting point, so tilting the drone changes the readings as it would
    # for real. The Crazyflie default is a single downward sensor at the centre.
    tof_sensors = (("down", (0.0, 0.0, -0.01), (0.0, 0.0, -1.0)),)
    tof_max_range = TOF_MAX_RANGE_M
    tof_min_range = TOF_MIN_RANGE_M
    tof_noise_std = TOF_NOISE_STD_M
    tof_noise_frac = TOF_NOISE_STD_FRAC

    # ---- Domain randomisation: re-drawn every episode -------------------------
    # The net is a sparse mesh, so an IR ToF may or may not see it. Each episode
    # draws P(net reading -> "no target") from this range, so the policy learns to
    # fly whether the net shows up on the ToFs always, sometimes, or rarely.
    tof_net_dropout_range = (0.0, 0.7)
    tof_glitch_prob = 0.01                 # random "no target" on any ToF reading
    accel_noise_std = 0.05                 # m/s^2, white noise per reading
    accel_bias_range = 0.10                # m/s^2, constant offset per episode
    # Isaac's IMU differentiates velocity, so the reset teleport shows up as an
    # acceleration of hundreds of m/s^2. Those spikes inflated the input normalisation
    # (accel std 19-87 m/s^2) and the policy learned to ignore its accelerometer.
    # For the first steps of an episode report what an IMU at that attitude reads
    # (gravity only), and clip every reading like a real IMU's range. The deployed node
    # applies the same clip.
    imu_settle_steps = 3                   # policy steps (60 ms)
    accel_clip = 30.0                      # m/s^2, per axis
    gyro_noise_std = 0.01                  # rad/s
    gyro_bias_range = 0.01                 # rad/s (gyros drift; this is the slow part)
    gravity_dir_noise_std = 0.01           # attitude estimate (roll/pitch) error...
    gravity_dir_bias_range = 0.03          # ...incl. ~2 deg IMU mounting misalignment
    thrust_scale_range = (0.90, 1.10)      # per motor: motor/prop/battery spread, mass error
    # Spurious short ToF returns (people walking past, multipath, dust): per episode
    # P(reading replaced by a random shorter distance).
    tof_ghost_range = (0.0, 0.03)

    # ---- Optical flow: Pi camera -> Raspberry Pi (flow) -> PX4 ---------------------
    # Horizontal velocity in the body frame = flow rate x height from the DOWN ToF,
    # so its error scales with the ToF's error. The camera pipeline adds: a frame
    # rate below the 50 Hz policy (last value held), latency (capture + Pi + MAVLink),
    # a max trackable image speed, imperfect gyro de-rotation, and the camera's
    # lever arm (it sits at the nose, so yaw/pitch rates move it sideways/vertically).
    flow_noise_std = 0.05                  # m/s
    flow_noise_per_m = 0.03                # extra m/s of noise per metre of height
    flow_scale_range = (0.90, 1.10)        # per-episode calibration (focal length) error
    flow_dropout_range = (0.0, 0.3)        # per-episode P(no valid flow): floor texture/light
    flow_height_range = (0.08, 4.0)        # m; outside this the sensor reports invalid
    flow_cam_pos_b = (0.114, -0.004, 0.011)  # m, Pi camera position from the CAD (nose)
    flow_rate_hz_range = (13.0, 17.0)      # 15 Hz camera -> Pi flow
    flow_latency_s_range = (0.06, 0.12)    # exposure + Pi processing + transport
    flow_max_rate = 2.0                    # rad/s image speed (|v|/h) trackable at 15 fps
    flow_rot_residual = 0.05               # fraction of body rate left after gyro de-rotation

    # ---- Scenes (re-drawn every episode; see environments/scenes.py) ------------
    # "random" draws from scene_probs; or force one of scenes.SCENARIOS, e.g.
    # "fixed_net" = the original fixed arena (used for videos / old comparisons).
    scenario = "random"
    #             net_only net_in_room room  open  fixed_net
    scene_probs = (0.20,   0.35,       0.30, 0.15, 0.0)
    scene_net_side = (2.5, 5.0)            # m, inner side length of the net (each axis)
    scene_net_height = (2.0, 3.5)
    scene_net_ceiling_prob = 0.7           # else the net is open at the top
    scene_room_gap = (0.3, 3.0)            # net -> room wall distance
    scene_room_side = (3.0, 9.0)           # rooms without a net
    scene_room_height = (2.4, 4.0)
    scene_furniture_band = (0.4, 1.0)      # furniture zone along room walls (no-fly)
    scene_open_side = (3.0, 6.0)           # virtual keep-in box in open halls
    scene_virtual_ceiling = 4.0
    scene_clutter_max = 6                  # boxes per scene (furniture, people, poles)
    scene_clutter_prob = 0.5               # each one present with this probability
    scene_clutter_half_xy = (0.1, 0.5)
    scene_clutter_half_z = (0.2, 1.0)
    scene_clutter_reach = 2.5              # how far outside the keep-in box they can be
    # Hover task in random scenes: hold the SPAWN position at a random target height
    # (given to the policy as an input). fixed_net keeps the old centre @ hover_height.
    target_height_range = (0.6, 1.5)

    # Prop prims to spin for visuals. Missing prims just disable the effect.
    prop_visual_paths = (
        "m1_prop/ccw_prop",
        "m2_prop/cw_prop",
        "m3_prop/ccw_prop",
        "m4_prop/cw_prop",
    )

    # ---- Flight arena (the safety net) --------------------------------------
    # The net is modelled as a GEOFENCE, not as a physical/cloth object:
    #   * get within `net_margin` of any net face -> episode terminates + penalty
    #   * within `net_warn_band` of the margin   -> smooth shaping penalty
    # The arena is centred on each env origin. [TODO] measure the INSIDE of the net.
    arena_size_xy = A.ARENA_SIZE_XY   # set in environments/arena.py (shared with RotorPy)
    arena_height = A.ARENA_HEIGHT
    net_margin = A.NET_MARGIN
    net_warn_band = A.NET_WARN_BAND
    max_tilt_rad = 1.2           # terminate if tilted beyond this (~70 deg)
    # Draw the net (0.5 m strand grid) and the hover target in the scene. Purely
    # visual, no collisions; only used for videos (record_video.py), off in training.
    show_net = False

    # Spawn distribution (relative to env origin). Randomised so the policy has to
    # actively correct xy drift instead of only ever seeing a perfect start.
    spawn_xy_range = 0.5
    spawn_z_range = (0.3, 1.2)
    spawn_lin_vel_range = 0.3
    spawn_ang_vel_range = 0.5

    # ---- Reward weights ------------------------------------------------------
    # Computed in the env from ground-truth state (privileged); the policy itself
    # only sees the observation vector.
    rew_alive = 1.0
    rew_z = 2.0
    rew_xy = 1.0
    rew_lin_vel = 0.10
    rew_ang_vel = 0.05
    rew_action = 0.02          # magnitude of body-rate commands
    rew_action_rate = 0.05     # change in command between steps (smooth = deployable)
    rew_net = 2.0
    rew_crash = 10.0
    rew_hvel = 0.0             # extra penalty on horizontal speed (calm drift damping)
    rew_tilt = 0.0             # penalty on tilt^2 (rad^2): stay near level, far from flips
    reward_preset = "v1"       # see REWARD_PRESETS / apply_reward_preset


    def __post_init__(self):
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        S = len(self.tof_sensors)
        if self.obs_mode == "onboard":
            self.observation_space = (17 + S) * self.obs_history
        elif self.obs_mode == "privileged":
            self.observation_space = 16 + S
        else:
            raise ValueError(f"obs_mode must be 'onboard' or 'privileged', not {self.obs_mode!r}")
        # critic: pos - target(3), velocity(3), quat(4), body rates(3), true ToF(S),
        # net-dropout prob(1), motor thrust scales(4), keep-in box(3), scenario
        # one-hot(len(SCENARIOS)), ghost prob(1), flow dropout prob(1)
        self.state_space = 18 + S + 3 + len(SCENARIOS) + 2


@configclass
class DroneHoverEnvCfgDD24(DroneHoverEnvCfg):
    """DD24 airframe config.

    `robot` points at the DD24 USD produced by assets/dd24/build_dd24_asset.py.
    Mass and inertia come from that USD (PhysX), so they are only correct once the
    asset has been built from YOUR measured values (see dd24_params.py).
    """
    # Lever arm for an X frame = center-to-motor / sqrt(2).
    arm_length = DD24.arm_length_m / math.sqrt(2.0)
    max_thrust_to_weight = DD24.thrust_to_weight
    yaw_coeff = DD24.yaw_coeff
    # URDF props are articulation links that already spin on their free joints, so
    # no xform-based spinning (editing a physics link's transform fights PhysX).
    prop_visual_paths = ()
    # Five ToFs at their CAD positions (see dd24_params.DD24_TOF_SENSORS).
    tof_sensors = DD24_TOF_SENSORS

    def __post_init__(self):
        super().__post_init__()
        # Imported lazily so the Crazyflie config works before the DD24 USD exists.
        from assets.dd24.dd24_cfg import DD24_CFG
        self.robot = DD24_CFG.replace(prim_path="/World/envs/env_.*/Drone")


# Reward variants (train_rl_hover.py --reward). "v1" = the original weights.
# "calm": the fine-tune showed v1 rates "drift less, crash more" as an improvement:
# crashes were cheap (10) next to the immediate position reward, and nothing
# discouraged the aggressive, saturating control that turns a short link freeze into
# a crash. calm makes losing the drone expensive, damps drift through horizontal
# velocity instead of chasing position, and rewards smooth, near-level flight.
# Use it with --gamma 0.995 (look ~4 s ahead instead of ~2 s).
REWARD_PRESETS = {
    "v1": {},
    "calm": dict(rew_crash=40.0, rew_hvel=0.4, rew_action_rate=0.2, rew_action=0.05, rew_tilt=0.5),
}


def apply_reward_preset(cfg, name: str):
    if name not in REWARD_PRESETS:
        raise ValueError(f"unknown reward preset {name!r}; choose from {list(REWARD_PRESETS)}")
    for k, v in REWARD_PRESETS[name].items():
        if not hasattr(cfg, k):
            raise AttributeError(k)
        setattr(cfg, k, v)
    cfg.reward_preset = name
    return cfg


class DroneHoverEnv(DirectRLEnv):
    cfg: DroneHoverEnvCfg

    def __init__(self, cfg: DroneHoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros(self.num_envs, 4, device=self.device)
        self.force_b = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.torque_b = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self.body_id = self.drone.find_bodies("body")[0]
        self.prop_joint_ids = [
            self.drone.find_joints(f"m{i}_joint")[0][0] for i in range(1, 5)
        ]

        masses = self.drone.root_physx_view.get_masses()
        self.mass = masses[0].sum().item()
        self.g = abs(self.cfg.sim.gravity[2])
        self.hover_thrust = self.mass * self.g
        self.max_total_thrust = self.hover_thrust * self.cfg.max_thrust_to_weight
        self.max_motor_thrust = self.max_total_thrust / 4.0
        # Per-motor command (0..1) that exactly cancels gravity.
        self.hover_motor = 1.0 / self.cfg.max_thrust_to_weight

        self._alloc_buffers()

        inertia = self.drone.root_physx_view.get_inertias()[0, self.body_id[0]]
        print(f"Drone mass: {self.mass:.4f} kg  | body inertia diag: "
              f"{inertia.view(3, 3).diagonal().tolist()}")
        print(f"Hover thrust: {self.hover_thrust:.3f} N | max motor thrust: "
              f"{self.max_motor_thrust:.3f} N | hover cmd: {self.hover_motor:.3f}")
        print("Joint names:", self.drone.joint_names)
        print(f"Arena (net) inner size: {self.cfg.arena_size_xy} x {self.cfg.arena_height} m, "
              f"margin {self.cfg.net_margin} m")
        self._prop_xforms = None

    def _alloc_buffers(self):
        """Per-env buffers (sensor layout, randomisation draws, scenes, targets,
        observation history). Separate from __init__ so tests can build them too."""
        self.actions = torch.zeros(self.num_envs, 4, device=self.device)
        self.force_b = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.torque_b = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._target_local = torch.tensor(
            [0.0, 0.0, self.cfg.hover_height], device=self.device
        )
        self._tof_pos_b = torch.tensor([t[1] for t in self.cfg.tof_sensors], device=self.device)
        self._tof_dir_b = torch.tensor([t[2] for t in self.cfg.tof_sensors], device=self.device)
        self._tof_dir_b = self._tof_dir_b / torch.linalg.norm(self._tof_dir_b, dim=-1, keepdim=True)
        N, S = self.num_envs, len(self.cfg.tof_sensors)
        # Per-episode randomisation draws (filled in _reset_idx).
        self._tof_drop_p = torch.zeros(N, 1, device=self.device)
        self._accel_bias = torch.zeros(N, 3, device=self.device)
        self._gyro_bias = torch.zeros(N, 3, device=self.device)
        self._grav_bias = torch.zeros(N, 3, device=self.device)
        self._thrust_scale = torch.ones(N, 4, device=self.device)
        self._tof_ghost_p = torch.zeros(N, 1, device=self.device)
        self._flow_scale = torch.ones(N, 1, device=self.device)
        self._flow_drop_p = torch.zeros(N, 1, device=self.device)
        self._flow_cam_r = torch.tensor(self.cfg.flow_cam_pos_b, device=self.device)
        # What the MAVROS node sees: each topic at its own rate and latency.
        self._imu_s = HeldSensor(N, 9, 3, self.device)        # accel, gyro, gravity dir
        self._tof_s = HeldSensor(N, S, 6, self.device)
        self._flow_s = HeldSensor(N, 3, 8, self.device)
        # Command link to PX4: what PX4 actually flies (delayed / dropped / stalled).
        self._cmd_link = HeldSensor(N, 4, 3, self.device)
        self._cmd_px4 = torch.zeros(N, 4, device=self.device)
        # Thrust/body-rate command (a in [-1,1]^4) and the simulated PX4 below it.
        self._cmd = torch.zeros(N, 4, device=self.device)
        self._cmd_prev = torch.zeros(N, 4, device=self.device)
        self._px4 = Px4RateController(N, self.device)
        self._max_rates = torch.tensor(self.cfg.ctbr_max_rates, device=self.device)
        names = [t[0] for t in self.cfg.tof_sensors]
        self._down_idx = names.index("down") if "down" in names else None
        self._scene = empty_scenes(N, self.cfg.scene_clutter_max, self.device)
        self._target = self._target_local.repeat(N, 1)      # per-env hover target
        # Observation history for "onboard" mode.
        self._frame_dim = 17 + S
        self._hist = torch.zeros(N, self.cfg.obs_history, self._frame_dim, device=self.device)
        self._hist_fill = torch.ones(N, dtype=torch.bool, device=self.device)
        self._half_xy = torch.tensor(
            [self.cfg.arena_size_xy[0] / 2.0, self.cfg.arena_size_xy[1] / 2.0],
            device=self.device,
        )

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.drone = Articulation(self.cfg.robot)
        self.scene.articulations["drone"] = self.drone

        # One IMU sensor over all envs, registered with the scene so it is
        # actually updated every step (the old per-env sensors never were, so
        # the accel obs was always zero).
        self.imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self.imu

        if self.cfg.show_net:
            self._spawn_net_visual("/World/envs/env_0")

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        # Drones in different envs must not collide with each other.
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_net_visual(self, env_prim: str, spacing: float = 0.5, strand: float = 0.008):
        """Net as a grid of thin dark strands on the 4 walls (+ ceiling) and a red
        marker at the hover point. Visual only (no collision props), spawned in
        env_0 before cloning so every env gets a copy."""
        hx, hy = self.cfg.arena_size_xy[0] / 2.0, self.cfg.arena_size_xy[1] / 2.0
        H = self.cfg.arena_height if self.cfg.arena_height > 0 else 2.5
        strand_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.12))
        bars = []

        def ticks(lo, hi):
            n = max(int(round((hi - lo) / spacing)), 1)
            return [lo + (hi - lo) * k / n for k in range(n + 1)]

        for sx in (-1, 1):   # walls at x = +-hx
            bars += [((strand, strand, H), (sx * hx, y, H / 2)) for y in ticks(-hy, hy)]
            bars += [((strand, 2 * hy, strand), (sx * hx, 0.0, z)) for z in ticks(0.0, H)]
        for sy in (-1, 1):   # walls at y = +-hy
            bars += [((strand, strand, H), (x, sy * hy, H / 2)) for x in ticks(-hx, hx)]
            bars += [((2 * hx, strand, strand), (0.0, sy * hy, z)) for z in ticks(0.0, H)]
        if self.cfg.arena_height > 0:   # ceiling grid
            bars += [((strand, 2 * hy, strand), (x, 0.0, H)) for x in ticks(-hx, hx)]
            bars += [((2 * hx, strand, strand), (0.0, y, H)) for y in ticks(-hy, hy)]
        for i, (size, pos) in enumerate(bars):
            cfg = sim_utils.CuboidCfg(size=size, visual_material=strand_mat)
            cfg.func(f"{env_prim}/Net/strand_{i}", cfg, translation=pos)
        target = sim_utils.SphereCfg(
            radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)))
        target.func(f"{env_prim}/Net/hover_target", target, translation=(0.0, 0.0, self.cfg.hover_height))

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor):
        if self.cfg.control_mode == "ctbr":
            # Policy output in [0,1] (sigmoid) -> a in [-1,1]: the setpoint the MAVROS
            # node would publish. Held for the 4 physics steps until the next policy step.
            self._cmd_prev = self._cmd.clone()
            self._cmd = 2.0 * torch.clamp(actions.clone(), 0.0, 1.0) - 1.0
            # The policy (and its 'previous command' input) knows what it SENT; PX4
            # flies what ARRIVES through the link.
            self._cmd_px4 = self._cmd_link(self._cmd, self.cfg.sim.dt * self.cfg.decimation)
            return
        self.actions = torch.clamp(actions.clone(), 0.0, 1.0)
        self._set_wrench(self.actions * self.max_motor_thrust * self._thrust_scale)

    def _set_wrench(self, f: torch.Tensor):
        """Body force/torque from per-motor thrusts f (N,4) in newtons."""
        f1, f2, f3, f4 = f[:, 0], f[:, 1], f[:, 2], f[:, 3]

        self.force_b.zero_()
        self.torque_b.zero_()
        self.force_b[:, 0, 2] = f.sum(dim=-1)
        self.torque_b[:, 0, 0] = self.cfg.arm_length * ((f3 + f4) - (f1 + f2))
        self.torque_b[:, 0, 1] = self.cfg.arm_length * ((f2 + f3) - (f1 + f4))
        # Yaw: m1 (front-right) and m3 (back-left) carry CCW props (PX4 quad X, your
        # CA_ROTOR0/1_KM = +0.05). A CCW prop's drag turns the BODY clockwise seen
        # from above, i.e. -z in FLU, so speeding up m1+m3 gives NEGATIVE yaw torque.
        # (This was sign-flipped before; RotorPy already had it right.)
        self.torque_b[:, 0, 2] = self.cfg.yaw_coeff * ((f2 + f4) - (f1 + f3))

    def ctbr_setpoint(self, cmd: torch.Tensor):
        """a in [-1,1]^4 -> (normalised collective thrust (N,1), body rates FLU (N,3)).
        Exactly what the deployed node writes into mavros_msgs/AttitudeTarget."""
        thrust = (self.cfg.ctbr_thrust_center + cmd[:, 0:1] * self.cfg.ctbr_thrust_span).clamp(0.0, 1.0)
        return thrust, cmd[:, 1:4] * self._max_rates

    def _apply_action(self):
        if self.cfg.control_mode == "ctbr":
            # One tick of the simulated PX4 rate controller (runs every physics step).
            thrust, rates = self.ctbr_setpoint(self._cmd_px4)
            gyro = self.drone.data.root_ang_vel_b + self._gyro_bias \
                + torch.randn_like(self._gyro_bias) * self.cfg.gyro_noise_std
            frac = self._px4.step(rates, thrust, gyro, self.cfg.sim.dt)
            self.actions = self._px4.u_act          # motor commands, for logging / prop visuals
            self._set_wrench(frac * self.max_motor_thrust * self._thrust_scale)
        self.drone.set_external_force_and_torque(
            self.force_b, self.torque_b, body_ids=self.body_id
        )
        if self.sim.has_gui():
            self._spin_props_visual()

    def _spin_props_visual(self):
        if self._prop_xforms is None:
            stage = self.sim.stage
            self._prop_xforms = []
            for rel in self.cfg.prop_visual_paths:
                prim = stage.GetPrimAtPath(f"/World/envs/env_0/Drone/{rel}")
                if not prim.IsValid():
                    print(f"[viz] prop prim not found ({rel}); prop spin disabled")
                    self._prop_xforms = []
                    break
                self._prop_xforms.append(UsdGeom.XformCommonAPI(prim))
            self._prop_angles = [0.0] * len(self._prop_xforms)
        for i, xform in enumerate(self._prop_xforms):
            if self.actions[0, i] > 0.05:
                speed = self.cfg.max_prop_visual_speed * (-1.0 if i in (1, 3) else 1.0)
                self._prop_angles[i] += speed * self.cfg.sim.dt
            xform.SetRotate(
                (0.0, 0.0, math.degrees(self._prop_angles[i])),
                UsdGeom.XformCommonAPI.RotationOrderXYZ,
            )

    # ---------------------------------------------------------------- helpers
    def _local_pos(self) -> torch.Tensor:
        """Position relative to this env's arena centre (env origin)."""
        return self.drone.data.root_pos_w - self.scene.env_origins

    def _dist_to_net(self, pos_local: torch.Tensor) -> torch.Tensor:
        """Distance to the nearest face of this env's keep-in box (net, furniture line
        or virtual fence; walls and ceiling)."""
        d = (self._scene["keep_half"] - pos_local[:, 0:2].abs()).min(dim=-1).values
        return torch.minimum(d, self._scene["keep_h"] - pos_local[:, 2])

    def _tof_readings(self, pos_local: torch.Tensor):
        """Simulated ToF measurements and true ranges (m), each (N, S).

        Beams are cast from each sensor's mounting point against this env's scene.
        A beam that reaches the NET is caught with prob 1 - drop_p; otherwise it
        passes through to whatever is behind (room wall, clutter, or nothing).
        Then: distance-dependent noise, "no target" beyond max range, random
        glitches, and occasional spurious short "ghost" returns."""
        N, S = pos_local.shape[0], self._tof_pos_b.shape[0]
        q = self.drone.data.root_quat_w[:, None, :].expand(N, S, 4)
        o = pos_local[:, None, :] + quat_rotate(q, self._tof_pos_b[None].expand(N, S, 3))
        d = quat_rotate(q, self._tof_dir_b[None].expand(N, S, 3))
        r = scene_ranges(o, d, self._scene)
        solid = torch.minimum(torch.minimum(r["floor"], r["room"]), r["clutter"])
        true = torch.minimum(solid, r["net"])
        net_caught = torch.rand_like(true) >= self._tof_drop_p
        seen = torch.minimum(solid, torch.where(net_caught, r["net"], torch.full_like(true, float("inf"))))

        mx, mn = self.cfg.tof_max_range, self.cfg.tof_min_range
        finite = torch.where(torch.isfinite(seen), seen, torch.full_like(seen, 2 * mx))
        meas = finite + torch.randn_like(finite) * (self.cfg.tof_noise_std + self.cfg.tof_noise_frac * finite.clamp(max=mx))
        ghost = torch.rand_like(meas) < self._tof_ghost_p
        meas = torch.where(ghost, mn + torch.rand_like(meas) * (meas.clamp(max=mx) - mn), meas)
        no_target = (finite > mx) & ~ghost
        no_target |= torch.rand_like(meas) < self.cfg.tof_glitch_prob
        meas = torch.where(no_target, torch.full_like(meas, mx), meas)
        return meas.clamp(mn, mx), true.clamp(max=mx)

    def _optical_flow(self, tof_meas: torch.Tensor, tof_true: torch.Tensor) -> torch.Tensor:
        """Pi-camera optical flow as PX4 receives it: body-frame horizontal velocity
        (N, 2) + valid flag (N, 1) at this instant; its 15 Hz rate and latency
        are applied by a HeldSensor in _get_observations.

        velocity = (camera's ground velocity) x (measured / true height) x calibration
                   + noise + leftover rotation, invalid when too low/high, too fast to
                   track, on a poor floor (per-episode dropout) or with no down ToF.
        The camera is at flow_cam_pos_b, so rotation moves it: v_cam = v + w x r."""
        N = tof_meas.shape[0]
        if self._down_idx is None:
            return torch.zeros(N, 3, device=self.device)
        data = self.drone.data
        q_inv = data.root_quat_w * torch.tensor([1.0, -1.0, -1.0, -1.0], device=self.device)
        w_b = data.root_ang_vel_b
        v_cam = quat_rotate(q_inv, data.root_lin_vel_w) + torch.cross(w_b, self._flow_cam_r.expand(N, 3), dim=-1)
        v_xy = v_cam[:, 0:2]
        h_true = tof_true[:, self._down_idx:self._down_idx + 1]
        h_meas = tof_meas[:, self._down_idx:self._down_idx + 1]
        ratio = (h_meas / h_true.clamp(min=1e-3)).clamp(0.25, 4.0)
        # imperfect de-rotation: a pitch rate looks like x motion, a roll rate like -y motion
        rot = self.cfg.flow_rot_residual * h_true * torch.stack([w_b[:, 1], -w_b[:, 0]], dim=-1)
        noise = torch.randn_like(v_xy) * (self.cfg.flow_noise_std + self.cfg.flow_noise_per_m * h_true)
        v = v_xy * ratio * self._flow_scale + rot + noise
        lo, hi = self.cfg.flow_height_range
        trackable = (torch.linalg.norm(v_xy, dim=-1, keepdim=True) / h_true.clamp(min=1e-3)) < self.cfg.flow_max_rate
        valid = (h_true > lo) & (h_true < hi) & (h_meas < self.cfg.tof_max_range) & trackable \
            & (torch.rand_like(h_true) >= self._flow_drop_p)
        return torch.cat([torch.where(valid, v, torch.zeros_like(v)), valid.float()], dim=-1)

    # ------------------------------------------------------------ observations
    def _get_observations(self):
        pos = self._local_pos()
        data = self.drone.data
        tof_m, tof_true = self._tof_readings(pos)
        mx = self.cfg.tof_max_range

        # Accelerometer without reset spikes (see accel_clip / imu_settle_steps).
        settle = (self.episode_length_buf < self.cfg.imu_settle_steps)[:, None]
        lin_acc = torch.where(settle, -data.projected_gravity_b * 9.81, self.imu.data.lin_acc_b)

        if self.cfg.obs_mode == "privileged":
            policy = torch.cat([
                lin_acc.clamp(-self.cfg.accel_clip, self.cfg.accel_clip),
                self._target[:, 2:3] - pos[:, 2:3],
                data.root_quat_w,
                data.root_lin_vel_w,
                self.imu.data.ang_vel_b,
                self._target[:, 0:2] - pos[:, 0:2],
                tof_m / mx,
            ], dim=-1)
        else:
            # What the flight controller actually has, with realistic errors.
            accel = (lin_acc + self._accel_bias
                     + torch.randn_like(self._accel_bias) * self.cfg.accel_noise_std   # incl. gravity, like a real IMU
                     ).clamp(-self.cfg.accel_clip, self.cfg.accel_clip)
            gyro = self.imu.data.ang_vel_b + self._gyro_bias \
                + torch.randn_like(self._gyro_bias) * self.cfg.gyro_noise_std        # body axes
            # Roll/pitch from the attitude estimate as the direction of gravity in the
            # body frame. Drift-free on a real drone (accelerometer-corrected), unlike
            # yaw or integrated position, so it's safe to give the policy.
            grav = data.projected_gravity_b + self._grav_bias \
                + torch.randn_like(self._grav_bias) * self.cfg.gravity_dir_noise_std
            grav = grav / torch.linalg.norm(grav, dim=-1, keepdim=True)
            if self.cfg.control_mode == "ctbr":
                prev_cmd = self._cmd
            else:
                prev_cmd = (self.actions - self.hover_motor) / self.hover_motor
            flow = self._optical_flow(tof_m, tof_true)
            # Each topic at its own rate / latency (latest message, like the ROS node).
            dt = self.cfg.sim.dt * self.cfg.decimation
            imu = self._imu_s(torch.cat([accel, gyro, grav], dim=-1), dt)
            tof_h = self._tof_s(tof_m / mx, dt)
            flow_h = self._flow_s(flow, dt)
            frame = torch.cat([imu, tof_h, prev_cmd, flow_h, self._target[:, 2:3]], dim=-1)
            # For tools/flight_log.py (references only, no copies): what the policy saw.
            self._dbg = {"frame": frame, "tof_meas": tof_m, "tof_true": tof_true, "flow_now": flow}

            self._hist = torch.cat([self._hist[:, 1:], frame[:, None]], dim=1)
            fill = self._hist_fill
            if bool(fill.any()):   # fresh episode: history = copies of the first reading
                self._hist[fill] = frame[fill][:, None].expand(-1, self.cfg.obs_history, -1)
                self._hist_fill[fill] = False
            policy = self._hist.reshape(self.num_envs, -1)

        sc = self._scene
        critic = torch.cat([
            pos - self._target,
            data.root_lin_vel_w,
            data.root_quat_w,
            data.root_ang_vel_b,
            tof_true / mx,
            self._tof_drop_p,
            self._thrust_scale,
            sc["keep_half"], sc["keep_h"][:, None],
            torch.nn.functional.one_hot(sc["scen"], len(SCENARIOS)).float(),
            self._tof_ghost_p,
            self._flow_drop_p,
        ], dim=-1)
        return {"policy": policy, "critic": critic}

    # ----------------------------------------------------------------- reward
    def _get_rewards(self):
        pos = self._local_pos()
        data = self.drone.data
        err = pos - self._target

        # Soft "don't go near the net" term: 0 far away, ramps to 1 at the hard margin.
        dist = self._dist_to_net(pos)
        net_pen = torch.clamp(
            (self.cfg.net_margin + self.cfg.net_warn_band - dist) / self.cfg.net_warn_band,
            0.0,
            1.0,
        ) ** 2

        tilt = torch.acos(torch.clamp(-data.projected_gravity_b[:, 2], -1.0, 1.0))
        reward = (
            self.cfg.rew_alive
            - self.cfg.rew_z * err[:, 2].abs()
            - self.cfg.rew_xy * torch.linalg.norm(err[:, 0:2], dim=-1)
            - self.cfg.rew_lin_vel * torch.linalg.norm(data.root_lin_vel_w, dim=-1)
            - self.cfg.rew_hvel * torch.linalg.norm(data.root_lin_vel_w[:, 0:2], dim=-1)
            - self.cfg.rew_tilt * tilt ** 2
            - self.cfg.rew_ang_vel * torch.linalg.norm(data.root_ang_vel_b, dim=-1)
            - self._action_penalty()
            - self.cfg.rew_net * net_pen
            # reset_terminated is filled by _get_dones(), which DirectRLEnv calls first.
            - self.cfg.rew_crash * self.reset_terminated.float()
        )
        return reward

    def _action_penalty(self):
        if self.cfg.control_mode == "ctbr":
            return (self.cfg.rew_action * torch.linalg.norm(self._cmd[:, 1:4], dim=-1)
                    + self.cfg.rew_action_rate * torch.linalg.norm(self._cmd - self._cmd_prev, dim=-1))
        return self.cfg.rew_action * torch.linalg.norm(self.actions - self.hover_motor, dim=-1)

    # ------------------------------------------------------------------ dones
    def _get_dones(self):
        pos = self._local_pos()
        crashed = pos[:, 2] < self.cfg.crash_height
        hit_net = self._dist_to_net(pos) < self.cfg.net_margin
        # projected_gravity_b z is -1 when level, +1 upside down.
        tilt = torch.acos(torch.clamp(-self.drone.data.projected_gravity_b[:, 2], -1.0, 1.0))
        flipped = tilt > self.cfg.max_tilt_rad
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        # Kept for logging/evaluation (why did each episode end?).
        self.term_crashed, self.term_net, self.term_flipped = crashed, hit_net, flipped
        return crashed | hit_net | flipped, timed_out

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.drone._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)
        dev = self.device
        root_state = self.drone.data.default_root_state[env_ids].clone()

        def u(lo, hi, *shape):
            return torch.empty(*shape, device=dev).uniform_(lo, hi)

        # New scene for these envs (or the fixed arena).
        c = self.cfg
        scene = sample_scenes(n, c, dev, c.scenario)
        assign(self._scene, scene, env_ids)
        fixed = scene["scen"] == SCENARIOS.index("fixed_net")

        # Spawn well inside the keep-in box; random scenes hold the spawn xy at a
        # random target height, fixed_net keeps the old centre @ hover_height.
        room_xy = (scene["keep_half"] - (c.net_margin + c.net_warn_band + 0.2)).clamp(min=0.2)
        r = c.spawn_xy_range
        xy_rand = (torch.rand(n, 2, device=dev) * 2 - 1) * room_xy
        xy = torch.where(fixed[:, None], (torch.rand(n, 2, device=dev) * 2 - 1) * r, xy_rand)
        z_hi = torch.minimum(torch.full((n,), c.spawn_z_range[1], device=dev), scene["keep_h"] - 0.7)
        z = c.spawn_z_range[0] + torch.rand(n, device=dev) * (z_hi - c.spawn_z_range[0]).clamp(min=0.05)
        local = torch.cat([xy, z[:, None]], dim=-1)
        th_hi = torch.minimum(torch.full((n,), c.target_height_range[1], device=dev), scene["keep_h"] - 0.8)
        th = c.target_height_range[0] + torch.rand(n, device=dev) * (th_hi - c.target_height_range[0]).clamp(min=0.0)
        target = torch.cat([xy, th[:, None]], dim=-1)
        target[fixed] = self._target_local
        self._target[env_ids] = target
        # Spawn relative to each env's own origin (previously every drone was
        # placed at world (0, 0, 0.5), i.e. all stacked on top of each other).
        root_state[:, 0:3] = self.scene.env_origins[env_ids] + local
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        root_state[:, 7:10] = u(-self.cfg.spawn_lin_vel_range, self.cfg.spawn_lin_vel_range, n, 3)
        root_state[:, 10:13] = u(-self.cfg.spawn_ang_vel_range, self.cfg.spawn_ang_vel_range, n, 3)

        self.drone.write_root_pose_to_sim(root_state[:, 0:7], env_ids)
        self.drone.write_root_velocity_to_sim(root_state[:, 7:13], env_ids)
        self.drone.reset(env_ids)
        self.actions[env_ids] = self.hover_motor if hasattr(self, "hover_motor") else 0.0

        # New random sensor/motor imperfections for this episode.
        self._tof_ghost_p[env_ids] = u(*c.tof_ghost_range, n, 1)
        self._flow_scale[env_ids] = u(*c.flow_scale_range, n, 1)
        self._flow_drop_p[env_ids] = u(*c.flow_dropout_range, n, 1)
        step_dt = c.sim.dt * c.decimation
        steps = lambda rng: torch.round(u(*rng, n) / step_dt).long()   # noqa: E731
        self._imu_s.reset(env_ids, u(*c.imu_rate_hz_range, n), steps(c.imu_latency_s_range),
                          u(*c.imu_drop_prob_range, n), u(*c.imu_stall_rate_range, n))
        self._tof_s.reset(env_ids, u(*c.tof_rate_hz_range, n), steps(c.tof_latency_s_range),
                          u(*c.sensor_drop_prob_range, n), u(*c.sensor_stall_rate_range, n))
        self._flow_s.reset(env_ids, u(*c.flow_rate_hz_range, n), steps(c.flow_latency_s_range),
                           u(*c.sensor_drop_prob_range, n), u(*c.sensor_stall_rate_range, n))
        # rate 1000 Hz: a setpoint is due every policy step (unless dropped / stalled)
        self._cmd_link.reset(env_ids, torch.full((n,), 1000.0, device=dev), steps(c.cmd_latency_s_range),
                             u(*c.cmd_drop_prob_range, n), u(*c.cmd_stall_rate_range, n))
        self._cmd_px4[env_ids] = 0.0
        # PX4 model mismatch for this episode; start the motors near hover, and the
        # held command at "hover thrust, zero rates" so the first steps are calm.
        k = u(*c.thrust_curve_k_range, n, 1)
        self._px4.reset(env_ids, u(*c.px4_gain_scale_range, n, 3), u(*c.px4_mix_scale_range, n, 3),
                        u(*c.motor_tau_range, n, 1), k, torch.full((n, 1), c.ctbr_thrust_center, device=dev))
        self._cmd[env_ids] = 0.0
        self._cmd_prev[env_ids] = 0.0
        self._tof_drop_p[env_ids] = u(*c.tof_net_dropout_range, n, 1)
        self._accel_bias[env_ids] = u(-c.accel_bias_range, c.accel_bias_range, n, 3)
        self._gyro_bias[env_ids] = u(-c.gyro_bias_range, c.gyro_bias_range, n, 3)
        self._grav_bias[env_ids] = u(-c.gravity_dir_bias_range, c.gravity_dir_bias_range, n, 3)
        self._thrust_scale[env_ids] = u(*c.thrust_scale_range, n, 4)
        self._hist_fill[env_ids] = True
