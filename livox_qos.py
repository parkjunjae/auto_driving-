#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import copy

class ImuFix(Node):
    def __init__(self):
        super().__init__('imu_fix_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=200
        )

        self.sub = self.create_subscription(Imu, '/livox/imu', self.cb, qos)
        self.pub = self.create_publisher(Imu, '/livox/imu_fixed', qos)

    def cb(self, msg: Imu):
        m = copy.deepcopy(msg)

        # orientation은 제공 안 하면 -1 권장
        m.orientation_covariance[0] = -1.0

        # gyro cov
        m.angular_velocity_covariance = tuple(map(float, [
            0.0004, 0, 0,
            0, 0.0004, 0,
            0, 0, 0.0004
        ]))

        # accel cov
        m.linear_acceleration_covariance = tuple(map(float, [
            0.04, 0, 0,
            0, 0.04, 0,
            0, 0, 0.04
        ]))

        self.pub.publish(m)

def main():
    rclpy.init()
    rclpy.spin(ImuFix())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
