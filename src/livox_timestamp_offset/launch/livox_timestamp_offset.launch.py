from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("input_topic", default_value="/livox/lidar"),
            DeclareLaunchArgument("output_topic", default_value="/livox/lidar/offset"),
            DeclareLaunchArgument("offset_sec", default_value="-1.2"),
            Node(
                package="livox_timestamp_offset",
                executable="livox_timestamp_offset_node",
                name="livox_timestamp_offset",
                output="screen",
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("input_topic"),
                        "output_topic": LaunchConfiguration("output_topic"),
                        "offset_sec": LaunchConfiguration("offset_sec"),
                    }
                ],
            ),
        ]
    )
