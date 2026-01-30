import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2


class LivoxTimestampOffset(Node):
    def __init__(self):
        super().__init__("livox_timestamp_offset")

        self.declare_parameter("input_topic", "/livox/lidar")
        self.declare_parameter("output_topic", "/livox/lidar/offset")
        self.declare_parameter("offset_sec", -1.2)

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        offset_sec = self.get_parameter("offset_sec").get_parameter_value().double_value

        self._offset = Duration(nanoseconds=int(offset_sec * 1e9))
        self._pub = self.create_publisher(PointCloud2, output_topic, qos_profile_sensor_data)
        self._sub = self.create_subscription(
            PointCloud2, input_topic, self._cb, qos_profile_sensor_data
        )

        self.get_logger().info(
            f"input_topic={input_topic} output_topic={output_topic} offset_sec={offset_sec}"
        )

    def _cb(self, msg: PointCloud2) -> None:
        # Update stamp only; keep data unchanged.
        stamp = Time.from_msg(msg.header.stamp)
        new_stamp = stamp + self._offset

        if new_stamp.nanoseconds < 0:
            new_stamp = Time(nanoseconds=0)

        msg.header.stamp = new_stamp.to_msg()
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = LivoxTimestampOffset()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
