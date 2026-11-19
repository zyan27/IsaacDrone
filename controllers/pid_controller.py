import torch


def quat_to_euler_xyz(q):
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return roll, pitch, yaw


class DroneWaypointPID:
    def __init__(self, dt, device="cuda:0"):
        self.dt = dt
        self.device = device

        self.target = torch.tensor([[0.0, 0.0, 1.0]], device=device)

        self.hover_motor = 1.0 / 2.2

        self.kp_z = 0.55
        self.kd_z = 0.35

        self.kp_xy = 0.10
        self.kd_xy = 0.16

        self.max_target_angle = 0.20

        self.kp_angle = 0.35
        self.kd_angle = 0.16

        self.yaw_rate_gain = 0.08

    def reset(self, num_envs):
        pass

    def set_target(self, target_xyz):
        self.target = target_xyz.to(self.device)

    def act(self, obs):
        # Note: With acceleration-based observations, position tracking is not available.
        # This controller requires position for waypoint following.
        # Using acceleration and z_error to estimate altitude control only.
        accel = obs[:, 0:3]
        z_error = obs[:, 3:4]
        quat = obs[:, 4:8]
        lin_vel = obs[:, 8:11]
        ang_vel = obs[:, 11:14]

        roll, pitch, yaw = quat_to_euler_xyz(quat)

        # Extract z_error and use lateral acceleration for stability
        z_error_val = z_error[:, 0]
        ax = accel[:, 0]
        ay = accel[:, 1]
        az = accel[:, 2]

        vx = lin_vel[:, 0]
        vy = lin_vel[:, 1]
        vz = lin_vel[:, 2]

        roll_rate = ang_vel[:, 0]
        pitch_rate = ang_vel[:, 1]
        yaw_rate = ang_vel[:, 2]

        collective = self.kp_z * z_error_val - self.kd_z * vz
        collective = torch.clamp(collective, -0.18, 0.18)

        # Use acceleration damping instead of position tracking
        desired_pitch = torch.clamp(
            -0.05 * ax - 0.12 * vx,
            -0.03,
            0.03,
        )

        desired_roll = torch.clamp(
            -0.05 * ay + 0.12 * vy,
            -0.03,
            0.03,
        )

        desired_pitch = torch.clamp(
            desired_pitch,
            -self.max_target_angle,
            self.max_target_angle,
        )

        desired_roll = torch.clamp(
            desired_roll,
            -self.max_target_angle,
            self.max_target_angle,
        )

        roll_cmd = self.kp_angle * (desired_roll - roll) - self.kd_angle * roll_rate
        pitch_cmd = self.kp_angle * (desired_pitch - pitch) - self.kd_angle * pitch_rate
        yaw_cmd = -self.yaw_rate_gain * yaw_rate

        roll_cmd = torch.clamp(roll_cmd, -0.08, 0.08)
        pitch_cmd = torch.clamp(pitch_cmd, -0.08, 0.08)
        yaw_cmd = torch.clamp(yaw_cmd, -0.03, 0.03)

        base = torch.ones_like(z_error) * self.hover_motor
        base = base + collective

        # +yaw torque (counter-clockwise from above) comes from the CW props m2/m4;
        # m1/m3 are CCW props and turn the body clockwise (see drone_hover_env._set_wrench).
        m1 = base - roll_cmd - pitch_cmd - yaw_cmd
        m2 = base - roll_cmd + pitch_cmd + yaw_cmd
        m3 = base + roll_cmd + pitch_cmd - yaw_cmd
        m4 = base + roll_cmd - pitch_cmd + yaw_cmd
        

        actions = torch.stack([m1, m2, m3, m4], dim=-1)

        return torch.clamp(actions, 0.0, 1.0)