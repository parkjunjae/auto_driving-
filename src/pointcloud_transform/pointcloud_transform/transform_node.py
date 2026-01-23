#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_sensor_msgs
from rclpy.qos import qos_profile_sensor_data


class PointCloudTransformNode(Node):
    def __init__(self):
        super().__init__('pointcloud_transform_node')

        # Parameters
        self.declare_parameter('input_topic', '/rtabmap/local_grid_obstacle')
        self.declare_parameter('output_topic', '/rtabmap/local_grid_obstacle_odom')
        self.declare_parameter('target_frame', 'odom')
        self.declare_parameter('tf_timeout', 0.5)
        self.declare_parameter('tf_cache_time', 5.0)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.target_frame = self.get_parameter('target_frame').value
        tf_timeout = float(self.get_parameter('tf_timeout').value)
        tf_cache_time = float(self.get_parameter('tf_cache_time').value)

        # TF2
        self.tf_buffer = Buffer(cache_time=Duration(seconds=tf_cache_time))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_timeout = Duration(seconds=tf_timeout)

        # Subscriber & Publisher
        self.subscription = self.create_subscription(
            PointCloud2, input_topic, self.callback, qos_profile_sensor_data
        )
        self.publisher = self.create_publisher(PointCloud2, output_topic, qos_profile_sensor_data)

        self.get_logger().info(
            f'Transform {input_topic} → {output_topic} (frame: {self.target_frame})'
        )

    def callback(self, msg):
        try:
            if msg.header.frame_id == self.target_frame:
                self.publisher.publish(msg)
                return

            stamp = rclpy.time.Time.from_msg(msg.header.stamp)

            if not self.tf_buffer.can_transform(
                self.target_frame,
                msg.header.frame_id,
                stamp,
                timeout=self.tf_timeout
            ):
                # ← 여기에 추가!
                self.get_logger().warn(
                    f'Transform not available: {msg.header.frame_id} → {self.target_frame}',
                    throttle_duration_sec=5.0
                )
                return

            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                stamp
            )

            transformed_cloud = tf2_sensor_msgs.do_transform_cloud(msg, transform)
            transformed_cloud.header.stamp = msg.header.stamp
            transformed_cloud.header.frame_id = self.target_frame

            self.publisher.publish(transformed_cloud)

        except TransformException as ex:
            self.get_logger().warn(f'Transform failed: {ex}')


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudTransformNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
