from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('obs_mode', default_value='scan'),
        DeclareLaunchArgument('scan_in', default_value='/scan'),
        DeclareLaunchArgument('costmap_in', default_value='/local_costmap/costmap'),
        DeclareLaunchArgument('odom_in', default_value='/odometry/filtered'),
        DeclareLaunchArgument('goal_in', default_value='/goal_pose'),
        DeclareLaunchArgument('cmd_out', default_value='/cmd_vel_raw'),
        DeclareLaunchArgument('policy_type', default_value='rule'),
        DeclareLaunchArgument('policy_path', default_value=''),
        DeclareLaunchArgument('range_max', default_value='6.0'),
        DeclareLaunchArgument('front_angle', default_value='0.35'),
        DeclareLaunchArgument('side_angle', default_value='1.2'),
        DeclareLaunchArgument('obstacle_threshold', default_value='80'),
        DeclareLaunchArgument('max_lin', default_value='0.5'),
        DeclareLaunchArgument('max_ang', default_value='1.0'),
        DeclareLaunchArgument('stop_dist', default_value='0.6'),
        DeclareLaunchArgument('slow_dist', default_value='1.5'),
        DeclareLaunchArgument('turn_gain', default_value='1.0'),
        DeclareLaunchArgument('goal_dist_scale', default_value='5.0'),
        DeclareLaunchArgument('loop_hz', default_value='20.0'),
        DeclareLaunchArgument('sensor_timeout', default_value='0.5'),
        Node(
            package='rl_local_policy',
            executable='rl_local_policy_node',
            name='rl_local_policy',
            output='screen',
            parameters=[{
                'obs_mode': LaunchConfiguration('obs_mode'),
                'scan_in': LaunchConfiguration('scan_in'),
                'costmap_in': LaunchConfiguration('costmap_in'),
                'odom_in': LaunchConfiguration('odom_in'),
                'goal_in': LaunchConfiguration('goal_in'),
                'cmd_out': LaunchConfiguration('cmd_out'),
                'policy_type': LaunchConfiguration('policy_type'),
                'policy_path': LaunchConfiguration('policy_path'),
                'range_max': LaunchConfiguration('range_max'),
                'front_angle': LaunchConfiguration('front_angle'),
                'side_angle': LaunchConfiguration('side_angle'),
                'obstacle_threshold': LaunchConfiguration('obstacle_threshold'),
                'max_lin': LaunchConfiguration('max_lin'),
                'max_ang': LaunchConfiguration('max_ang'),
                'stop_dist': LaunchConfiguration('stop_dist'),
                'slow_dist': LaunchConfiguration('slow_dist'),
                'turn_gain': LaunchConfiguration('turn_gain'),
                'goal_dist_scale': LaunchConfiguration('goal_dist_scale'),
                'loop_hz': LaunchConfiguration('loop_hz'),
                'sensor_timeout': LaunchConfiguration('sensor_timeout'),
            }],
        ),
    ])
