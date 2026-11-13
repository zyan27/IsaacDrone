"""Press "up": with PX4 already holding xy (optical flow), climb to --height on the down ToF and hold.

    python3 climb_node.py [--height 0.5] [--down_topic /noducksgiven/bottom_tof_driver_node/range]

Streams /mavros/setpoint_raw/local velocity setpoints at 20 Hz (PX4 needs them before it accepts
OFFBOARD): vx = vy = 0 (PX4's flow position loop holds), vz = 1.0 * (height - ToF) clipped to
+-0.3 m/s, yaw rate 0. Zero until armed and in OFFBOARD. Land: switch back to POSCTL / STABILIZED.
"""
import argparse

import rclpy
from mavros_msgs.msg import PositionTarget, State
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Range

IGNORE = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
          PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ | PositionTarget.IGNORE_YAW)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=float, default=0.5)
    p.add_argument("--down_topic", default="/noducksgiven/bottom_tof_driver_node/range")
    p.add_argument("--max_vz", type=float, default=0.3)
    args = p.parse_args()
    rclpy.init()
    node = Node("dd24_climb")
    s = {"h": None, "active": False}
    node.create_subscription(Range, args.down_topic, lambda m: s.update(h=m.range) if 0.0 <= m.range < 100.0 else None,
                             qos_profile_sensor_data)
    node.create_subscription(State, "/mavros/state", lambda m: s.update(active=m.armed and m.mode == "OFFBOARD"),
                             qos_profile_sensor_data)
    pub = node.create_publisher(PositionTarget, "/mavros/setpoint_raw/local", 10)

    def tick():
        vz = 0.0
        if s["active"] and s["h"] is not None:
            vz = max(-args.max_vz, min(args.max_vz, 1.0 * (args.height - s["h"])))
        m = PositionTarget()
        m.header.stamp = node.get_clock().now().to_msg()
        m.coordinate_frame = PositionTarget.FRAME_LOCAL_NED   # mavros takes ENU here: z up
        m.type_mask = IGNORE
        m.velocity.z = vz
        pub.publish(m)
        node.get_logger().info(f"{'OFFBOARD' if s['active'] else 'waiting'} tof {s['h']} vz_cmd {vz:+.2f}",
                               throttle_duration_sec=1.0)

    node.create_timer(0.05, tick)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
