#!/usr/bin/env python3
"""DD24 policy node (ROS 2 Jazzy): runs the trained hover policy through MAVROS.

    python3 dd24_policy_node.py --ros-args -p policy:=/root/dd24/policy.npz -p dry_run:=true

dry_run (DEFAULT true): runs the policy on live sensors and logs everything, but
publishes NOTHING to PX4. Use it on the bench to check every input before flying.
dry_run:=false: streams setpoints to /mavros/setpoint_raw/attitude:
  * not OFFBOARD+armed -> calm setpoint (zero rates, centre thrust), so PX4 can switch;
  * OFFBOARD+armed     -> the policy (history restarts at the switch, like an episode);
  * IMU/down-ToF late  -> calm setpoint; very late, tilt > limit or too high -> STOP
    sending (PX4's offboard-loss failsafe, COM_OF_LOSS_T, takes over).
Debug topics: ~/frame (22 newest inputs), ~/command (thrust, 3 rates, mode code).
"""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from mavros_msgs.msg import AttitudeTarget, OpticalFlowRad, State
from sensor_msgs.msg import Imu, Range
from std_msgs.msg import Float32MultiArray

from dd24_policy_core import Core, FlowCfg, Policy, SafetyCfg, roll_pitch_deg

MODES = {"WAIT": 0.0, "RUN": 1.0, "NEUTRAL": 2.0, "STOP": 3.0}


class DD24PolicyNode(Node):
    def __init__(self):
        super().__init__("dd24_policy")
        P = self.declare_parameter
        policy_path = P("policy", "/root/dd24/policy.npz").value
        self.dry_run = bool(P("dry_run", True).value)
        robot = P("robot", "noducksgiven").value
        tof_topic = P("tof_topic", "/{robot}/{name}_tof_driver_node/range").value
        # policy ToF name -> driver sensor name (driver calls the up sensor "top", down "bottom")
        tof_driver = {"down": P("tof_down", "bottom").value, "up": P("tof_up", "top").value,
                      "front": P("tof_front", "front").value, "left": P("tof_left", "left").value,
                      "right": P("tof_right", "right").value}
        flow = FlowCfg(quality_min=int(P("flow_quality_min", 1).value),
                       use_gyro_fields=bool(P("flow_use_gyro_fields", True).value),
                       sign_x=float(P("flow_sign_x", 1.0).value), sign_y=float(P("flow_sign_y", 1.0).value),
                       swap_xy=bool(P("flow_swap_xy", False).value))
        safety = SafetyCfg(max_tilt_deg=float(P("max_tilt_deg", 60.0).value),
                           max_height_m=float(P("max_height_m", 2.2).value),
                           thrust_gain=float(P("thrust_gain", 1.0).value))
        self.core = Core(Policy(policy_path), float(P("target_height", 0.8).value), flow, safety)
        flow.height_range, flow.max_rate = tuple(self.core.m["flow_height_range"]), self.core.m["flow_max_rate"]
        self.log_period = 1.0 / float(P("log_hz", 2.0).value)
        rate = float(P("rate_hz", self.core.m["policy_hz"]).value)

        self.armed, self.mode = False, ""
        sd = qos_profile_sensor_data
        self.create_subscription(Imu, P("imu_topic", "/mavros/imu/data").value, self.on_imu, sd)
        for name, drv in tof_driver.items():
            if drv.lower() in ("", "none", "off"):
                self.core.disabled.add(name)
                self.get_logger().warn(f"ToF {name:5s} DISABLED -> always 'no target'")
                continue
            topic = tof_topic.format(robot=robot, name=drv)
            self.create_subscription(Range, topic, lambda m, n=name: self.on_tof(n, m), sd)
            self.get_logger().info(f"ToF {name:5s} <- {topic}")
        self.create_subscription(OpticalFlowRad, P("flow_topic", f"/{robot}/optical_flow/raw").value, self.on_flow, sd)
        self.create_subscription(State, "/mavros/state", self.on_state, QoSProfile(depth=10))
        self.pub = self.create_publisher(AttitudeTarget, P("setpoint_topic", "/mavros/setpoint_raw/attitude").value,
                                         QoSProfile(depth=10))
        self.pub_frame = self.create_publisher(Float32MultiArray, "~/frame", QoSProfile(depth=10))
        self.pub_cmd = self.create_publisher(Float32MultiArray, "~/command", QoSProfile(depth=10))
        self.last_log, self.last_mode = 0.0, None
        self.timer = self.create_timer(1.0 / rate, self.tick)
        m = self.core.m
        self.get_logger().info(
            f"policy {policy_path} (iter {m['train_iter']}), {rate:.0f} Hz, target height "
            f"{self.core.target_height:.2f} m, DRY RUN={self.dry_run}"
            + ("  -> nothing is sent to PX4" if self.dry_run else "  -> SENDING SETPOINTS"))

    def now(self):
        return time.monotonic()

    # ---- inputs
    def on_imu(self, m):
        a, w, q = m.linear_acceleration, m.angular_velocity, m.orientation
        self.core.on_imu(self.now(), (a.x, a.y, a.z), (w.x, w.y, w.z), (q.x, q.y, q.z, q.w))

    def on_tof(self, name, m):
        self.core.on_tof(self.now(), name, float(m.range), float(m.max_range))

    def on_flow(self, m):
        self.core.on_flow(self.now(), {
            "integration_time_us": m.integration_time_us, "integrated_x": m.integrated_x,
            "integrated_y": m.integrated_y, "integrated_xgyro": m.integrated_xgyro,
            "integrated_ygyro": m.integrated_ygyro, "quality": m.quality})

    def on_state(self, m):
        self.armed, self.mode = bool(m.armed), str(m.mode)

    # ---- 50 Hz
    def tick(self):
        t = self.now()
        flying = self.armed and self.mode == "OFFBOARD"
        r = self.core.tick(t, active=True if self.dry_run else flying)
        if not self.dry_run and r.mode != "STOP":
            sp = AttitudeTarget()
            sp.header.stamp = self.get_clock().now().to_msg()
            sp.header.frame_id = "base_link"
            sp.type_mask = AttitudeTarget.IGNORE_ATTITUDE
            sp.orientation.w = 1.0
            sp.body_rate.x, sp.body_rate.y, sp.body_rate.z = (float(v) for v in r.rates)
            sp.thrust = float(r.thrust)
            self.pub.publish(sp)
        if r.frame is not None:
            self.pub_frame.publish(Float32MultiArray(data=[float(v) for v in r.frame]))
        c = [float(r.thrust) if r.thrust is not None else -1.0] + \
            ([float(v) for v in r.rates] if r.rates is not None else [0.0] * 3) + [MODES[r.mode]]
        self.pub_cmd.publish(Float32MultiArray(data=c))
        if r.mode != self.last_mode:
            self.get_logger().warn(f"mode {self.last_mode} -> {r.mode} {r.reason}")
            self.last_mode = r.mode
        if t - self.last_log > self.log_period:
            self.last_log = t
            self.log_status(r, flying)

    def log_status(self, r, flying):
        core = self.core
        tofs = " ".join(f"{n[0].upper()}" + ("off" if n in core.disabled else
                        f"{(core.tof[n][1] if core.tof[n] else float('nan')):.2f}") for n in core.tof_names)
        line = f"[{r.mode}{' DRY' if self.dry_run else ''}] {'OFFBOARD+armed' if flying else self.mode or '?'} | ToF {tofs}"
        if core.flow:
            line += f" | flow vx {core.flow[1]:+.2f} vy {core.flow[2]:+.2f} {'ok' if core.flow[3] else 'INVALID'}"
        if core.imu:
            roll, pitch = roll_pitch_deg(core.build_frame(self.now())[6:9])
            line += f" | roll {roll:+5.1f} pitch {pitch:+5.1f} deg"
        if r.thrust is not None:
            line += (f" | thrust {r.thrust:.3f} rates roll {r.rates[0]:+.2f} pitch {r.rates[1]:+.2f} "
                     f"yaw {r.rates[2]:+.2f}")
        if r.reason:
            line += f" | {r.reason}"
        self.get_logger().info(line)
        if r.frame is not None:
            odd = core.out_of_training_range(r.frame)
            if odd:
                self.get_logger().warn("inputs far outside training: " + ", ".join(
                    f"{n}={v:.3f} (z={z:+.1f})" for n, v, z in odd[:4]))
        for w in r.warnings:
            self.get_logger().warn(w)


def main():
    rclpy.init()
    node = DD24PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:                      # after Ctrl-C ROS may already be shut down: nothing to do
            node.destroy_node()
            rclpy.try_shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
