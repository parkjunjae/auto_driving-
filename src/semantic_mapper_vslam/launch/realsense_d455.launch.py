#!/usr/bin/env python3
"""
RealSense D455 Launch File
Optimized for RTAB-Map VSLAM
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import SetParameter


def generate_launch_description():
    return LaunchDescription([
        # Global parameters
        SetParameter(name='use_sim_time', value=False),

        # Launch arguments
        DeclareLaunchArgument(
            'enable_gyro',
            default_value='true',
            description='Enable gyroscope'
        ),
        DeclareLaunchArgument(
            'enable_accel',
            default_value='true',
            description='Enable accelerometer'
        ),
        DeclareLaunchArgument(
            'unite_imu_method',
            default_value='2',
            description='0=None, 1=copy, 2=linear_interpolation'
        ),
        DeclareLaunchArgument(
            'depth_profile',
            default_value='848x480x30',
            description='Depth stream profile (WxHxFPS)'
        ),
        DeclareLaunchArgument(
            'color_profile',
            default_value='640x480x30',
            description='Color stream profile (WxHxFPS)'
        ),

        # Launch RealSense camera driver
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(
                    get_package_share_directory('realsense2_camera'),
                    'launch', 'rs_launch.py'
                )
            ]),
            launch_arguments={
                'camera_namespace': '',
                'enable_color': 'true',
                'enable_depth': 'true',
                'enable_gyro': LaunchConfiguration('enable_gyro'),
                'enable_accel': LaunchConfiguration('enable_accel'),
                'unite_imu_method': LaunchConfiguration('unite_imu_method'),
                'align_depth.enable': 'true',
                'enable_sync': 'true',
                'initial_reset': 'true',
                'depth_module.depth_profile': LaunchConfiguration('depth_profile'),
                'rgb_camera.color_profile': LaunchConfiguration('color_profile'),
            }.items(),
        ),
    ])
