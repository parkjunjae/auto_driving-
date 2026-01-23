#!/usr/bin/env python3
"""
RTAB-Map VSLAM Launch File
RealSense D455 RGB-D SLAM for Agile Tracer
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    # RTAB-Map parameters
    parameters = [{
        'frame_id': 'camera_link',
        'subscribe_depth': True,
        'subscribe_rgb': True,
        'subscribe_odom_info': True,
        'approx_sync': False,
        # D455는 IMU가 없는 경우가 많아 초기화 대기 없이 바로 시작
        'wait_imu_to_init': False,
        'always_check_imu_tf': False,

        # RTAB-Map specific
        'RGBD/NeighborLinkRefining': 'true',
        'RGBD/ProximityBySpace': 'true',
        'RGBD/AngularUpdate': '0.01',
        'RGBD/LinearUpdate': '0.01',
        'RGBD/OptimizeFromGraphEnd': 'false',

        # Loop closure
        'Mem/IncrementalMemory': 'true',
        'Mem/InitWMWithAllNodes': 'false',

        # Point cloud
        'Grid/Sensor': '0',  # 0=RGB-D, 1=2D laser, 2=3D laser
        'Grid/RangeMax': '5.0',
        'Grid/CellSize': '0.05',
    }]

    remappings = [
        ('imu', '/camera/imu/data'),
        ('rgb/image', '/camera/color/image_raw'),
        ('rgb/camera_info', '/camera/color/camera_info'),
        ('depth/image', '/camera/aligned_depth_to_color/image_raw'),
    ]

    return LaunchDescription([
        # Global parameters
        SetParameter(name='use_sim_time', value=False),

        # Launch arguments
        DeclareLaunchArgument(
            'localization',
            default_value='false',
            description='Set to true for localization mode'
        ),
        DeclareLaunchArgument(
            'rtabmap_args',
            default_value='',
            description='Extra arguments for rtabmap node'
        ),
        DeclareLaunchArgument(
            'rtabmap_viz',
            default_value='false',
            description='Launch rtabmap_viz for visualization'
        ),

        # IMU filter (Madgwick)
        Node(
            package='imu_filter_madgwick',
            executable='imu_filter_madgwick_node',
            name='imu_filter',
            output='screen',
            parameters=[{
                'use_mag': False,
                'world_frame': 'enu',
                'publish_tf': False,
            }],
            remappings=[
                ('imu/data_raw', '/camera/camera/imu'),
            ],
        ),

        # RTAB-Map RGB-D Odometry
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='rgbd_odometry',
            output='screen',
            parameters=parameters,
            remappings=remappings,
        ),

        # RTAB-Map SLAM
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=parameters,
            remappings=remappings,
            arguments=['-d', LaunchConfiguration('rtabmap_args')],
        ),
    ])
