"""Every randomisation in DronePX4EnvCfg switched off: one drone, one room, one command.

Spawn at the box centre with heading 0, climb to 0.5 m one second after the start and
hold it for the rest of the episode. No sensor noise, misses or dropouts; fixed sensor
rates and latencies (the log means); nominal mass, inertia, motors, CG and IMU mounting.
The randomisations come back one at a time once this flies (see DronePX4EnvCfg).
"""


def make_deterministic(cfg):
    # Spawn and command
    cfg.spawn_xy_range_m = 0.0
    cfg.spawn_yaw_rad = 0.0
    cfg.spawn_air_prob = 0.0
    cfg.target_start_s = (1.0, 1.0)
    cfg.target_z_range_m = (0.5, 0.5)
    cfg.target_land_prob = 0.0
    # Room
    cfg.wall_dist_range_m = (cfg.room_half_width_m, cfg.room_half_width_m)
    cfg.side_tof_range_m = (3.5, 3.5)
    # Sensors
    cfg.tof_invalid_prob = 0.0
    cfg.tof_noise_m = 0.0
    cfg.att_noise_rad = 0.0
    cfg.gyro_noise_rps = 0.0
    cfg.accel_noise_mps2 = 0.0
    cfg.tof_rate_hz = (20.0, 20.0)
    cfg.imu_rate_hz = (50.0, 50.0)
    cfg.tof_latency_s = (0.03, 0.03)
    cfg.imu_latency_s = (0.01, 0.01)
    cfg.dropout_per_s = (0.0, 0.0)
    cfg.cmd_latency_s = (0.04, 0.04)
    cfg.stream_phase_random = False
    # Body
    cfg.hover_thrust_range = (0.47, 0.47)
    cfg.mass_scale = (1.0, 1.0)
    cfg.inertia_scale = (1.0, 1.0)
    cfg.motor_tau_s = (0.025, 0.025)
    cfg.cg_offset_m = 0.0
    cfg.imu_tilt_rad = 0.0
    return cfg
