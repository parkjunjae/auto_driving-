import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import SetRemap


def generate_launch_description():
    rtabmap_pkg = get_package_share_directory('rtabmap_launch')
    pid_pkg = get_package_share_directory('cmd_vel_pid')

    use_sim_time = LaunchConfiguration('use_sim_time')
    nav2_params = LaunchConfiguration('nav2_params')
    rtabmap_viz = LaunchConfiguration('rtabmap_viz')
    rviz = LaunchConfiguration('rviz')
    log_level = LaunchConfiguration('log_level')
    rtabmap_args = LaunchConfiguration('rtabmap_args')
    database_path = LaunchConfiguration('database_path')

    cmd_in = LaunchConfiguration('cmd_in')
    cmd_out = LaunchConfiguration('cmd_out')
    odom_in = LaunchConfiguration('odom_in')
    imu_in = LaunchConfiguration('imu_in')
    use_imu = LaunchConfiguration('use_imu')
    kp = LaunchConfiguration('kp')
    ki = LaunchConfiguration('ki')
    kd = LaunchConfiguration('kd')
    i_min = LaunchConfiguration('i_min')
    i_max = LaunchConfiguration('i_max')
    w_max = LaunchConfiguration('w_max')
    w_acc_max = LaunchConfiguration('w_acc_max')
    loop_hz = LaunchConfiguration('loop_hz')
    cmd_timeout = LaunchConfiguration('cmd_timeout')
    meas_timeout = LaunchConfiguration('meas_timeout')
    reset_i_on_zero_ref = LaunchConfiguration('reset_i_on_zero_ref')
    stop_threshold = LaunchConfiguration('stop_threshold')

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([rtabmap_pkg, 'launch', 'rtabmap_nav2.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'nav2_params': nav2_params,
            'rtabmap_viz': rtabmap_viz,
            'rviz': rviz,
            'log_level': log_level,
            'rtabmap_args': rtabmap_args,
            'database_path': database_path,
        }.items(),
    )

    pid_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pid_pkg, 'launch', 'angular_pid_cmdvel.launch.py'])
        ),
        launch_arguments={
            'cmd_in': cmd_in,
            'cmd_out': cmd_out,
            'odom_in': odom_in,
            'imu_in': imu_in,
            'use_imu': use_imu,
            'kp': kp,
            'ki': ki,
            'kd': kd,
            'i_min': i_min,
            'i_max': i_max,
            'w_max': w_max,
            'w_acc_max': w_acc_max,
            'loop_hz': loop_hz,
            'cmd_timeout': cmd_timeout,
            'meas_timeout': meas_timeout,
            'reset_i_on_zero_ref': reset_i_on_zero_ref,
            'stop_threshold': stop_threshold,
        }.items(),
    )

    nav2_group = GroupAction(
        actions=[
            SetRemap(src='/cmd_vel', dst='/cmd_vel_raw'),
            nav2_launch,
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'nav2_params',
            default_value=PathJoinSubstitution(
                [rtabmap_pkg, 'launch', 'config', 'nav2_rtabmap_params.yaml']
            ),
        ),
        DeclareLaunchArgument('rtabmap_viz', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('log_level', default_value='warn'),
        DeclareLaunchArgument('rtabmap_args', default_value='--delete_db_on_start'),
        DeclareLaunchArgument(
            'database_path',
            default_value=os.path.expanduser('~/.ros/rtabmap_nav2.db')
        ),
        DeclareLaunchArgument('cmd_in', default_value='/cmd_vel_raw'),
        DeclareLaunchArgument('cmd_out', default_value='/cmd_vel'),
        DeclareLaunchArgument('odom_in', default_value='/odometry/filtered'),
        DeclareLaunchArgument('imu_in', default_value='/camera/camera/imu_fixed'),
        DeclareLaunchArgument('use_imu', default_value='false'),
        DeclareLaunchArgument('kp', default_value='2.0'),
        DeclareLaunchArgument('ki', default_value='0.0'),
        DeclareLaunchArgument('kd', default_value='0.1'),
        DeclareLaunchArgument('i_min', default_value='-0.5'),
        DeclareLaunchArgument('i_max', default_value='0.5'),
        DeclareLaunchArgument('w_max', default_value='1.5'),
        DeclareLaunchArgument('w_acc_max', default_value='3.0'),
        DeclareLaunchArgument('loop_hz', default_value='50.0'),
        DeclareLaunchArgument('cmd_timeout', default_value='0.5'),
        DeclareLaunchArgument('meas_timeout', default_value='0.5'),
        DeclareLaunchArgument('reset_i_on_zero_ref', default_value='true'),
        DeclareLaunchArgument('stop_threshold', default_value='1e-3'),
        nav2_group,
        pid_launch,
    ])
