#!/usr/bin/env python3
"""
Livox 타임스탬프 수정 노드
Livox 라이다의 미래 타임스탬프를 현재 시스템 시간으로 변경
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import time

class LivoxTimestampFixer(Node):
    def __init__(self):
        super().__init__('livox_timestamp_fixer')

        # Livox 원본 토픽 구독
        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.callback,
            10)

        # 수정된 타임스탬프로 재발행
        self.publisher = self.create_publisher(
            PointCloud2,
            '/livox/lidar_fixed',
            10)

        self.get_logger().info('Livox Timestamp Fixer started')
        self.get_logger().info('Subscribing: /livox/lidar')
        self.get_logger().info('Publishing: /livox/lidar_fixed')

    def callback(self, msg):
        # Python time.time()과 동일한 wall clock 사용
        now = time.time()
        msg.header.stamp.sec = int(now)
        msg.header.stamp.nanosec = int((now - int(now)) * 1e9)

        # 수정된 메시지 발행
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LivoxTimestampFixer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()