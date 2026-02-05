from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    input_topic = LaunchConfiguration("input_topic")
    output_topic = LaunchConfiguration("output_topic")
    voxel_size = LaunchConfiguration("voxel_size")
    min_hits = LaunchConfiguration("min_hits")
    hit_window_sec = LaunchConfiguration("hit_window_sec")
    max_stale_sec = LaunchConfiguration("max_stale_sec")
    z_min = LaunchConfiguration("z_min")
    z_max = LaunchConfiguration("z_max")
    min_range = LaunchConfiguration("min_range")

    node = Node(
        package="livox_pointcloud_filter",
        executable="dynamic_object_filter_node",
        name="dynamic_object_filter",
        output="screen",
        parameters=[
            {
                "input_topic": input_topic,
                "output_topic": output_topic,
                "voxel_size": voxel_size,
                "min_hits": min_hits,
                "hit_window_sec": hit_window_sec,
                "max_stale_sec": max_stale_sec,
                "z_min": z_min,
                "z_max": z_max,
                "min_range": min_range,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("input_topic", default_value="/livox/lidar/filtered"),
            DeclareLaunchArgument("output_topic", default_value="/livox/lidar/static_filtered"),
            DeclareLaunchArgument("voxel_size", default_value="0.10"),
            DeclareLaunchArgument("min_hits", default_value="3"),
            DeclareLaunchArgument("hit_window_sec", default_value="3.0"),
            DeclareLaunchArgument("max_stale_sec", default_value="8.0"),
            DeclareLaunchArgument("z_min", default_value="0.03"),
            DeclareLaunchArgument("z_max", default_value="1.8"),
            DeclareLaunchArgument("min_range", default_value="0.2"),
            node,
        ]
    )
