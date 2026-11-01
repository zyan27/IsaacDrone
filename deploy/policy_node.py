"""Run the trained DD24 policy on the Pi: ROS 2 node, numpy only (no torch).

Plain script, run inside the container that has rclpy + mavros_msgs:
    python3 policy_node.py --policy dd24_oracle_wallmap.npz --height 0.5 \\
        --down_topic ... --front_topic ... --left_topic ... --right_topic ...

Inputs (latest message of each, held, with its age = now - header.stamp):
  /mavros/imu/data                           orientation, angular_velocity, linear_acceleration
  --down_topic / --front_topic / --left_topic / --right_topic: the four ToFs. In the 10-25
                 logs the driver topic names do not match the sensors: right_tof reads the
                 floor (down) and bottom_tof the right side. Cover each sensor by hand
                 (props off) and check before flying. Readings of 999 (no return) are
                 ignored, as in training.
Output, 20 Hz (the policy's training rate), on /mavros/setpoint_raw/attitude:
  attitude policies (action_mode in the .npz): roll / pitch attitude quaternion with the
    current heading + yaw rate + thrust (roll / pitch rates ignored);
  rates policies: thrust + body rates (attitude ignored).

Policies with 30 inputs also get position error and velocity from the wall-map estimator
(oracle_core.WallMap): walls are the front / left / right readings of the last second
before OFFBOARD, origin = takeoff spot, x = the heading at takeoff.

The node always streams setpoints (PX4 needs them before it accepts OFFBOARD), but
they are thrust 0 until the drone is armed AND in OFFBOARD. Then it waits 1 s on
the floor, climbs at 0.3 m/s to --height and holds it until you take over.
Landing is not trained: switch the dashboard back to STABILIZED and land by hand
(or disarm).

It also cuts to thrust 0 (drone drops) if tilt > 45 deg, IMU older than 0.5 s,
down ToF older than 1 s, or the position estimate leaves --max_xy of the takeoff spot.
From <1 m in the net that is the safe failure.
"""
import argparse
import collections
import math
import time

import numpy as np
import rclpy
from mavros_msgs.msg import AttitudeTarget, State
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, Range

from oracle_core import FRD, Policy, WallMap, attitude_quat_enu, observation, qmul, tilt_deg, yaw_of

RATE_HZ = 20.0
TARGET_SPEED_MPS = 0.3


def stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class PolicyNode(Node):
    def __init__(self, args):
        super().__init__("dd24_policy")
        self.args = args
        self.policy = Policy(args.policy)
        self.imu = None
        self.tofs = [None] * 4    # down, front, left, right
        self.side_hist = collections.deque()   # (time, front, left, right) readings, last 1 s
        self.armed_offboard = False
        self.t_start = None
        self.target = 0.0
        self.last_sp = np.zeros(4, np.float32)
        self.done = False
        self.wallmap = None
        self.q_yaw0_inv = np.array([1.0, 0, 0, 0])
        self.create_subscription(Imu, "/mavros/imu/data", self.on_imu, qos_profile_sensor_data)
        topics = [args.down_topic, args.front_topic, args.left_topic, args.right_topic]
        for i, topic in enumerate(topics):
            self.create_subscription(Range, topic, lambda m, i=i: self.on_tof(i, m), qos_profile_sensor_data)
        self.create_subscription(State, "/mavros/state", self.on_state, 10)
        self.pub = self.create_publisher(AttitudeTarget, "/mavros/setpoint_raw/attitude", 10)
        self.log = open(args.log, "w") if args.log else None
        if self.log:
            self.log.write("t,active,target,tof_down,tof_front,tof_left,tof_right,tof_age,imu_age,tilt_deg,"
                           "est_x,est_y,est_z,est_vx,est_vy,est_vz,thrust,roll,pitch,yaw_rate\n")
        self.create_timer(1.0 / RATE_HZ, self.tick)
        self.get_logger().info(f"policy {args.policy} ({self.policy.action_mode}, {self.policy.n_obs} inputs"
                               f"{', wall-map estimator' if self.policy.oracle else ''}), "
                               f"ToF down/front/left/right {topics}, height {args.height} m (land by hand)")

    def on_imu(self, m):
        self.imu = m

    def on_tof(self, i, m):
        if 0.0 <= m.range < 100.0:  # 999 = no return: keep the previous reading, its age grows
            self.tofs[i] = m
            if i > 0 and None not in self.tofs[1:]:
                now = stamp_s(m)
                self.side_hist.append((now, *[t.range for t in self.tofs[1:]]))
                while self.side_hist and self.side_hist[0][0] < now - 1.0:
                    self.side_hist.popleft()

    def on_state(self, m):
        active = m.armed and m.mode == "OFFBOARD"
        if active and not self.armed_offboard:
            self.t_start, self.done = time.monotonic(), False
            if self.imu is not None:
                yaw0 = yaw_of(self.quat_enu())
                self.q_yaw0_inv = np.array([math.cos(yaw0 / 2), 0, 0, -math.sin(yaw0 / 2)])
            if self.policy.oracle:
                if self.side_hist:
                    walls = np.median(np.array([h[1:] for h in self.side_hist]), axis=0)
                    self.wallmap = WallMap(*walls, dt=1.0 / RATE_HZ, floor_h=self.args.floor_h)
                    self.get_logger().info(f"walls front {walls[0]:.2f} left {walls[1]:.2f} right {walls[2]:.2f} m")
                else:
                    self.wallmap = None
                    self.get_logger().error("no side ToF readings in the last second: not flying")
                    self.done = True
            self.get_logger().info("OFFBOARD + armed: flying profile")
        if not active and self.armed_offboard:
            self.get_logger().info("left OFFBOARD / disarmed: thrust 0")
        self.armed_offboard = active

    def quat_enu(self):
        q = self.imu.orientation
        return np.array([q.w, q.x, q.y, q.z])

    def goal(self, t):
        return 0.0 if t < 1.0 else self.args.height

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        sp = np.zeros(4, np.float32)
        tof, ages, tilt, est = [math.nan] * 4, [0.0] * 5, 0.0, (np.full(3, math.nan), np.full(3, math.nan))
        if self.armed_offboard and not self.done and self.imu is not None and None not in self.tofs:
            t = time.monotonic() - self.t_start
            step = TARGET_SPEED_MPS / RATE_HZ
            self.target += float(np.clip(self.goal(t) - self.target, -step, step))
            q_enu = self.quat_enu()
            quat = qmul(self.q_yaw0_inv, q_enu)            # takeoff frame, as in training
            w, a = self.imu.angular_velocity, self.imu.linear_acceleration
            gyro = np.array([w.x, w.y, w.z], np.float32) * FRD
            acc = np.array([a.x, a.y, a.z], np.float32) * FRD
            tof = np.array([m.range for m in self.tofs], np.float32)
            ages = np.array([now - stamp_s(m) for m in self.tofs] + [now - stamp_s(self.imu)])
            tilt = tilt_deg(q_enu)
            if self.policy.oracle:
                est = self.wallmap.step(tof, quat, acc, ages[0:4])
            obs = observation(tof, quat, gyro, acc, ages, self.last_sp, self.target, est if self.policy.oracle else None)
            far = self.policy.oracle and math.hypot(est[0][0], est[0][1]) > self.args.max_xy
            if tilt > 45.0 or ages[4] > 0.5 or ages[0] > 1.0 or far:
                self.get_logger().error(f"abort: tilt {tilt:.0f} deg, down ToF age {ages[0]:.2f} s, IMU age {ages[4]:.2f} s, "
                                        f"estimate xy ({est[0][0]:+.2f}, {est[0][1]:+.2f}) m")
                self.done = True
            else:
                sp = self.policy.setpoint(self.policy(obs))
                sp[0] = min(sp[0], self.args.max_thrust)
        if not self.armed_offboard:
            self.target = 0.0
        self.last_sp = sp
        msg = AttitudeTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.thrust = float(sp[0])
        if self.policy.action_mode == "attitude" and self.imu is not None:
            # Heading setpoint = current heading (as in training); the policy steers yaw by rate.
            q = attitude_quat_enu(float(sp[1]), float(sp[2]), yaw_of(self.quat_enu()))
            msg.type_mask = AttitudeTarget.IGNORE_ROLL_RATE | AttitudeTarget.IGNORE_PITCH_RATE
            msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = (float(c) for c in q)
            msg.body_rate.z = float(-sp[3])                                            # FRD -> FLU
        else:
            msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE
            msg.body_rate.x, msg.body_rate.y, msg.body_rate.z = float(sp[1]), float(-sp[2]), float(-sp[3])  # FRD -> FLU
        self.pub.publish(msg)
        if self.log:
            active = int(self.armed_offboard and not self.done)
            p, v = est
            self.log.write(f"{now:.3f},{active},{self.target:.3f}," + ",".join(f"{x:.3f}" for x in tof) +
                           f",{ages[0]:.3f},{ages[4]:.3f},{tilt:.1f}," + ",".join(f"{x:.3f}" for x in (*p, *v)) +
                           "," + ",".join(f"{x:.3f}" for x in sp) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="dd24_oracle_wallmap.npz")
    p.add_argument("--down_topic", default="/noducksgiven/right_tof_driver_node/range")
    p.add_argument("--front_topic", required=True)
    p.add_argument("--left_topic", required=True)
    p.add_argument("--right_topic", required=True)
    p.add_argument("--height", type=float, default=0.5)
    p.add_argument("--max_thrust", type=float, default=0.65, help="hard cap on the thrust setpoint")
    p.add_argument("--max_xy", type=float, default=0.8, help="abort (thrust 0) when the xy estimate is this far from takeoff, m")
    p.add_argument("--floor_h", type=float, default=0.05, help="side ToF height above the floor when landed, m")
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
