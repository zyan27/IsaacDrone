"""Stand-in ToFs for a bench test with only the flight controller attached: publishes
sensor_msgs/Range on the four topics policy_node.py reads, 20 Hz, fixed ranges.

    python3 fake_tof.py [--down 0.0] [--side 1.0]

--down 0.0 = sitting on the floor (the policy should command a climb once armed in OFFBOARD);
--down 0.5 = already at the hold height.
"""
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Range


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--down", type=float, default=0.0)
    p.add_argument("--side", type=float, default=1.0)
    p.add_argument("--prefix", default="/noducksgiven")
    args = p.parse_args()
    rclpy.init()
    node = Node("dd24_fake_tof")
    pubs = {k: node.create_publisher(Range, f"{args.prefix}/{k}_tof_driver_node/range", qos_profile_sensor_data)
            for k in ("bottom", "front", "left", "right")}
    ranges = {"bottom": args.down, "front": args.side, "left": args.side, "right": args.side}

    def tick():
        stamp = node.get_clock().now().to_msg()
        for k, pub in pubs.items():
            m = Range()
            m.header.stamp, m.header.frame_id = stamp, f"fake/{k}"
            m.min_range, m.max_range, m.range = 0.04, 3.6, ranges[k]
            pub.publish(m)

    node.create_timer(0.05, tick)
    node.get_logger().info(f"fake ToFs: down {args.down} m, sides {args.side} m on {args.prefix}/*_tof_driver_node/range")
    rclpy.spin(node)


if __name__ == "__main__":
    main()
