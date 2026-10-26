"""Run the trained DD24 policy on the Pi: ROS 2 node, numpy only (no torch).

Plain script, run inside the container that has rclpy + mavros_msgs:
    python3 policy_node.py --policy policy.npz --height 0.5 --hold 6

Inputs (latest message of each, held, with its age = now - header.stamp):
  /mavros/imu/data                           orientation, angular_velocity, linear_acceleration
  --tof_topic (default right_tof = the one that reads ~0 on the floor in the logs)
Output, 50 Hz:
  /mavros/setpoint_raw/attitude              thrust + body rates (attitude ignored)

The node always streams setpoints (PX4 needs them before it accepts OFFBOARD), but
they are thrust 0 until the drone is armed AND in OFFBOARD. Then it flies the same
profile as play_px4.py: 1 s on the floor, climb at 0.3 m/s to --height, hold
--hold s, come down, and send thrust 0 once it reads the floor again.
To take over at any time: switch the dashboard back to STABILIZED (or disarm).

It also cuts to thrust 0 (drone drops) if tilt > 45 deg, IMU older than 0.5 s,
or ToF older than 1 s. From <1 m in the net that is the safe failure.

Frames: mavros gives ROS FLU body / ENU world; the policy was trained on FRD
gyro/accel and Isaac's wxyz quaternion (z up), and outputs FRD rates. mavros
expects FLU body rates on setpoint_raw/attitude and converts to PX4's FRD.
"""
import argparse
import math
import time

import numpy as np
import rclpy
from mavros_msgs.msg import AttitudeTarget, State
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, Range

FRD = np.array([1.0, -1.0, -1.0], np.float32)
AGE_CLIP_S = 1.0
TOF_MAX_M = 2.0
TARGET_SPEED_MPS = 0.3


class Policy:
    def __init__(self, path):
        z = np.load(path)
        self.layers = []
        i = 0
        while f"W{i}" in z:
            self.layers.append((z[f"W{i}"], z[f"b{i}"]))
            i += 1
        self.hover, self.thr_range, self.rate_scale = float(z["hover_thrust"]), float(z["thrust_range"]), z["rate_scale"]

    def __call__(self, obs):
        x = obs
        for k, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if k < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(x))  # ELU
        a = np.clip(np.tanh(x), -1.0, 1.0)
        return np.concatenate([[self.hover + self.thr_range * a[0]], a[1:4] * self.rate_scale]).astype(np.float32)


def stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class PolicyNode(Node):
    def __init__(self, args):
        super().__init__("dd24_policy")
        self.args = args
        self.policy = Policy(args.policy)
        self.imu = None
        self.tof = None
        self.armed_offboard = False
        self.t_start = None
        self.target = 0.0
        self.ctbr = np.zeros(4, np.float32)
        self.done = False
        self.create_subscription(Imu, "/mavros/imu/data", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Range, args.tof_topic, self.on_tof, qos_profile_sensor_data)
        self.create_subscription(State, "/mavros/state", self.on_state, 10)
        self.pub = self.create_publisher(AttitudeTarget, "/mavros/setpoint_raw/attitude", 10)
        self.log = open(args.log, "w") if args.log else None
        if self.log:
            self.log.write("t,active,target,tof,tof_age,imu_age,tilt_deg,thrust,p,q,r\n")
        self.create_timer(1.0 / 50.0, self.tick)
        self.get_logger().info(f"policy {args.policy}, ToF {args.tof_topic}, height {args.height} m, hold {args.hold} s")

    def on_imu(self, m):
        self.imu = m

    def on_tof(self, m):
        if 0.0 <= m.range <= max(m.max_range, TOF_MAX_M) and m.range < 100.0:  # drop 999 = no return
            self.tof = m

    def on_state(self, m):
        active = m.armed and m.mode == "OFFBOARD"
        if active and not self.armed_offboard:
            self.t_start, self.done = time.monotonic(), False
            self.get_logger().info("OFFBOARD + armed: flying profile")
        if not active and self.armed_offboard:
            self.get_logger().info("left OFFBOARD / disarmed: thrust 0")
        self.armed_offboard = active

    def goal(self, t):
        a = self.args
        if t < 1.0:
            return 0.0
        if t < 1.0 + a.height / TARGET_SPEED_MPS + a.hold:
            return a.height
        return 0.0

    def obs(self, now):
        q = self.imu.orientation
        quat = np.array([q.w, q.x, q.y, q.z], np.float32)              # Isaac order wxyz
        w = self.imu.angular_velocity
        a = self.imu.linear_acceleration
        gyro = np.array([w.x, w.y, w.z], np.float32) * FRD
        acc = np.array([a.x, a.y, a.z], np.float32) * FRD
        tof = min(max(self.tof.range, 0.0), TOF_MAX_M)
        ages = np.clip([now - stamp_s(self.tof), now - stamp_s(self.imu)], 0.0, AGE_CLIP_S)
        return np.concatenate([[tof], quat, gyro, acc, ages, self.ctbr, [self.target]]).astype(np.float32), ages

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        sp = np.zeros(4, np.float32)
        info = (0.0, 0.0, 0.0, 0.0)
        if self.armed_offboard and not self.done and self.imu is not None and self.tof is not None:
            t = time.monotonic() - self.t_start
            step = TARGET_SPEED_MPS / 50.0
            self.target += float(np.clip(self.goal(t) - self.target, -step, step))
            obs, ages = self.obs(now)
            q = self.imu.orientation
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (q.x * q.x + q.y * q.y)))))
            info = (obs[0], ages[0], ages[1], tilt)
            if tilt > 45.0 or ages[1] > 0.5 or ages[0] > 1.0:
                self.get_logger().error(f"abort: tilt {tilt:.0f} deg, ToF age {ages[0]:.2f} s, IMU age {ages[1]:.2f} s")
                self.done = True
            elif t > 1.0 + self.args.height / TARGET_SPEED_MPS + self.args.hold and self.target <= 0.0 and obs[0] < 0.05:
                self.get_logger().info("landed: thrust 0 (disarm from the dashboard)")
                self.done = True
            else:
                sp = self.policy(obs)
                sp[0] = min(max(sp[0], 0.0), self.args.max_thrust)
        if not self.armed_offboard:
            self.target = 0.0
        self.ctbr = sp
        msg = AttitudeTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        msg.body_rate.x, msg.body_rate.y, msg.body_rate.z = float(sp[1]), float(-sp[2]), float(-sp[3])  # FRD -> FLU
        msg.thrust = float(sp[0])
        self.pub.publish(msg)
        if self.log:
            active = int(self.armed_offboard and not self.done)
            self.log.write(f"{now:.3f},{active},{self.target:.3f},{info[0]:.3f},{info[1]:.3f},{info[2]:.3f},{info[3]:.1f},"
                           f"{sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f},{sp[3]:.3f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="policy.npz")
    p.add_argument("--tof_topic", default="/noducksgiven/right_tof_driver_node/range")
    p.add_argument("--height", type=float, default=0.5)
    p.add_argument("--hold", type=float, default=6.0)
    p.add_argument("--max_thrust", type=float, default=0.65, help="hard cap on the thrust setpoint")
    p.add_argument("--log", default="policy_log.csv")
    args = p.parse_args()
    rclpy.init()
    node = PolicyNode(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
