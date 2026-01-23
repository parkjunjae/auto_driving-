from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
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

    return LaunchDescription([
        DeclareLaunchArgument('cmd_in', default_value='/cmd_vel_raw',
                              description='Input cmd_vel topic'),
        DeclareLaunchArgument('cmd_out', default_value='/cmd_vel',
                              description='Output cmd_vel topic'),
        DeclareLaunchArgument('odom_in', default_value='/odometry/filtered',
                              description='Odometry topic'),
        DeclareLaunchArgument('imu_in', default_value='/camera/camera/imu_fixed',
                              description='IMU topic'),
        DeclareLaunchArgument('use_imu', default_value='false',
                              description='Use IMU for angular velocity'),
        DeclareLaunchArgument('kp', default_value='2.0',
                              description='PID kp'),
        DeclareLaunchArgument('ki', default_value='0.0',
                              description='PID ki'),
        DeclareLaunchArgument('kd', default_value='0.1',
                              description='PID kd'),
        DeclareLaunchArgument('i_min', default_value='-0.5',
                              description='Integral min'),
        DeclareLaunchArgument('i_max', default_value='0.5',
                              description='Integral max'),
        DeclareLaunchArgument('w_max', default_value='1.5',
                              description='Angular speed limit'),
        DeclareLaunchArgument('w_acc_max', default_value='3.0',
                              description='Angular accel limit'),
        DeclareLaunchArgument('loop_hz', default_value='50.0',
                              description='Control loop rate'),
        DeclareLaunchArgument('cmd_timeout', default_value='0.5',
                              description='Cmd timeout'),
        DeclareLaunchArgument('meas_timeout', default_value='0.5',
                              description='Measurement timeout'),
        DeclareLaunchArgument('reset_i_on_zero_ref', default_value='true',
                              description='Reset integrator at stop'),
        DeclareLaunchArgument('stop_threshold', default_value='1e-3',
                              description='Zero threshold'),
        Node(
            package='cmd_vel_pid',
            executable='angular_pid_cmdvel',
            name='angular_pid_cmdvel',
            output='screen',
            parameters=[{
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
            }],
        ),
    ])
