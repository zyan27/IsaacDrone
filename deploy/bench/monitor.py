"""Evidence that the policy is driving PX4: once a second, print and log what the node sends,
what PX4 says its attitude setpoint is, the motor outputs and the IMU.

    python3 monitor.py [--csv evidence.csv]

  sent   = /mavros/setpoint_raw/attitude         (policy_node.py -> mavros), rate and last value
  px4    = /mavros/setpoint_raw/target_attitude  (PX4's own attitude setpoint, echoed back):
           in OFFBOARD it should follow "sent"
  motors = /mavros/rc/out                        (PWM of the first 4 outputs)
"""
import argparse
import math
import time

import rclpy
from mavros_msgs.msg import AttitudeTarget, RCOut, State
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def rp_deg(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    return (math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
            math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="evidence.csv")
    args = p.parse_args()
    rclpy.init()
    node = Node("dd24_monitor")
    last = {"state": None, "sent": None, "px4": None, "rc": None, "imu": None}
    count = {k: 0 for k in last}

    def keep(k):
        def cb(m):
            last[k] = m
            count[k] += 1
        return cb

    node.create_subscription(State, "/mavros/state", keep("state"), qos_profile_sensor_data)
    node.create_subscription(AttitudeTarget, "/mavros/setpoint_raw/attitude", keep("sent"), qos_profile_sensor_data)
    node.create_subscription(AttitudeTarget, "/mavros/setpoint_raw/target_attitude", keep("px4"), qos_profile_sensor_data)
    node.create_subscription(RCOut, "/mavros/rc/out", keep("rc"), qos_profile_sensor_data)
    node.create_subscription(Imu, "/mavros/imu/data", keep("imu"), qos_profile_sensor_data)
    f = open(args.csv, "w")
    f.write("t,mode,armed,sent_hz,sent_thrust,sent_roll,sent_pitch,sent_yaw_rate,px4_hz,px4_thrust,px4_roll,px4_pitch,"
            "m1,m2,m3,m4,imu_hz,imu_roll,imu_pitch\n")
    t0 = time.monotonic()

    def report():
        dt = 1.0
        hz = {k: count[k] / dt for k in count}
        for k in count:
            count[k] = 0
        s, sent, px4, rc, imu = (last[k] for k in ("state", "sent", "px4", "rc", "imu"))
        mode, armed = (s.mode, int(s.armed)) if s else ("?", 0)
        st = (sent.thrust, *rp_deg(sent.orientation), -sent.body_rate.z) if sent else (math.nan,) * 4
        pt = (px4.thrust, *rp_deg(px4.orientation)) if px4 else (math.nan,) * 3
        m = list(rc.channels[:4]) if rc and len(rc.channels) >= 4 else [0] * 4
        ir = rp_deg(imu.orientation) if imu else (math.nan, math.nan)
        t = time.monotonic() - t0
        print(f"t={t:6.1f}s {mode:>10s} armed={armed} | sent {hz['sent']:4.1f}Hz thr {st[0]:.3f} roll {st[1]:+5.1f} pitch {st[2]:+5.1f} "
              f"yaw_rate {st[3]:+.2f} | px4 {hz['px4']:4.1f}Hz thr {pt[0]:.3f} roll {pt[1]:+5.1f} pitch {pt[2]:+5.1f} "
              f"| motors {m} | imu {hz['imu']:4.0f}Hz roll {ir[0]:+5.1f} pitch {ir[1]:+5.1f}", flush=True)
        f.write(f"{t:.1f},{mode},{armed},{hz['sent']:.1f}," + ",".join(f"{x:.3f}" for x in st) + f",{hz['px4']:.1f}," +
                ",".join(f"{x:.3f}" for x in pt) + "," + ",".join(str(x) for x in m) + f",{hz['imu']:.0f},{ir[0]:.2f},{ir[1]:.2f}\n")
        f.flush()

    node.create_timer(1.0, report)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
