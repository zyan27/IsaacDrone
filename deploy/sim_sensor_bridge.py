#!/usr/bin/env python3
"""Stand-in ToF + optical-flow sensors for a VIRTUAL Duckiedrone (Duckiematrix).
NEVER run this on the real drone: it would publish fake sensor data next to the real ones.

The virtual DD24 has PX4 + MAVROS but (as far as we know) no ToF drivers or flow
pipeline. This node makes them from the simulator's ground truth, in exactly the
formats the real drone uses, so dd24_policy_node.py runs unchanged:

  /{robot}/{bottom,top,front,left,right}_tof_driver_node/range   sensor_msgs/Range, 30 Hz
      ray cast from each sensor (positions/directions from the DD24 CAD) against the
      floor and an optional box room around the takeoff point; nothing in range -> 999
      (like the Duckietown VL53L1X driver), max_range 3.6 m
  /{robot}/optical_flow/raw   mavros_msgs/OpticalFlowRad, 15 Hz
      PX4 OPTICAL_FLOW_RAD convention, already de-rotated (gyro fields 0):
      integrated_x = vy_body / h * dt,  integrated_y = vx_body / h * dt  (body FLU)

Inputs (ground truth from PX4 SITL via MAVROS): /mavros/local_position/pose (ENU, z =
height above the takeoff point), /mavros/local_position/velocity_body (FLU).

    python3 sim_sensor_bridge.py --ros-args -p robot:=<virtual robot name> \
        -p room_half:=1.5 -p ceiling:=2.5        # walls +-1.5 m, ceiling 2.5 m
        -p room_half:=0.0 -p ceiling:=0.0        # open space (floor only)
"""
import math

import numpy as np

# DD24 ToF sensors (body FLU, metres) -- same as environments/dd24_params.py
TOF_SENSORS = {   # driver name: (position, direction)
    "bottom": ((-0.067, 0.0, -0.008), (0.0, 0.0, -1.0)),
    "top":    ((-0.067, 0.0, 0.037), (0.0, 0.0, 1.0)),
    "front":  ((0.103, 0.0, -0.010), (1.0, 0.0, 0.0)),
    "left":   ((0.0, 0.044, 0.015), (0.0, 1.0, 0.0)),
    "right":  ((0.0, -0.044, 0.015), (0.0, -1.0, 0.0)),
}
MAX_RANGE, NO_TARGET = 3.6, 999.0


def quat_to_R(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def ray_box(o, d, room_half, ceiling):
    """Distance along d from o to the floor (z=0), walls (|x|,|y| = room_half) and
    ceiling (z = ceiling); 0 disables walls / ceiling. Returns inf if nothing hit."""
    best = math.inf
    for axis, planes in ((2, [0.0] + ([ceiling] if ceiling > 0 else [])),
                         (0, [-room_half, room_half] if room_half > 0 else []),
                         (1, [-room_half, room_half] if room_half > 0 else [])):
        for p in planes:
            if abs(d[axis]) < 1e-9:
                continue
            t = (p - o[axis]) / d[axis]
            if t > 1e-6:
                best = min(best, t)
    return best


def tof_ranges(pos_enu, q_xyzw, room_half=1.5, ceiling=2.5, noise=0.005, rng=None):
    """Driver-style readings (metres, 999 = no target) for each ToF."""
    R = quat_to_R(q_xyzw)
    out = {}
    for name, (p, d) in TOF_SENSORS.items():
        r = ray_box(np.asarray(pos_enu) + R @ np.array(p), R @ np.array(d), room_half, ceiling)
        if rng is not None and math.isfinite(r):
            r += rng.normal(0, noise + 0.01 * r)
        out[name] = r if r <= MAX_RANGE else NO_TARGET
    return out


def flow_message(v_body_flu, height, dt=0.0667, quality=200):
    """OpticalFlowRad fields for body velocity at a height (de-rotated). Inverse of
    dd24_policy_core.flow_velocity with default signs."""
    h = max(height, 1e-3)
    return dict(integration_time_us=int(dt * 1e6), integrated_x=v_body_flu[1] / h * dt,
                integrated_y=v_body_flu[0] / h * dt, integrated_xgyro=0.0, integrated_ygyro=0.0,
                integrated_zgyro=0.0, quality=quality if height > 0.08 else 0, distance=-1.0)


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, qos_profile_sensor_data
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from mavros_msgs.msg import OpticalFlowRad
    from sensor_msgs.msg import Range

    class Bridge(Node):
        def __init__(self):
            super().__init__("sim_sensor_bridge")
            P = self.declare_parameter
            robot = P("robot", "virtualdrone").value
            self.room_half = float(P("room_half", 1.5).value)
            self.ceiling = float(P("ceiling", 2.5).value)
            self.rng = np.random.default_rng(0)
            self.pose = self.vel = None
            sd = qos_profile_sensor_data
            self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.on_pose, sd)
            self.create_subscription(TwistStamped, "/mavros/local_position/velocity_body", self.on_vel, sd)
            self.tof_pubs = {n: self.create_publisher(Range, f"/{robot}/{n}_tof_driver_node/range", QoSProfile(depth=1))
                             for n in TOF_SENSORS}
            self.flow_pub = self.create_publisher(OpticalFlowRad, f"/{robot}/optical_flow/raw", QoSProfile(depth=10))
            self.create_timer(1 / 30.0, self.tick_tof)
            self.create_timer(1 / 15.0, self.tick_flow)
            self.get_logger().warn(f"SIMULATED sensors for /{robot}: room +-{self.room_half} m, ceiling {self.ceiling} m "
                                   "(virtual drone only!)")

        def on_pose(self, m):
            p, q = m.pose.position, m.pose.orientation
            self.pose = ((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))

        def on_vel(self, m):
            v = m.twist.linear
            self.vel = (v.x, v.y, v.z)

        def tick_tof(self):
            if self.pose is None:
                return
            for n, r in tof_ranges(*self.pose, self.room_half, self.ceiling, rng=self.rng).items():
                msg = Range()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = f"tof/{n}"
                msg.radiation_type, msg.field_of_view = 1, 0.47
                msg.min_range, msg.max_range, msg.range = 0.04, MAX_RANGE, float(r)
                self.tof_pubs[n].publish(msg)

        def tick_flow(self):
            if self.pose is None or self.vel is None:
                return
            h = tof_ranges(*self.pose, self.room_half, self.ceiling)["bottom"]
            f = flow_message(self.vel, h if h < NO_TARGET else 0.0)
            msg = OpticalFlowRad()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "flow_floor_view"
            for k, v in f.items():
                setattr(msg, k, type(getattr(msg, k))(v))
            self.flow_pub.publish(msg)

    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.try_shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
