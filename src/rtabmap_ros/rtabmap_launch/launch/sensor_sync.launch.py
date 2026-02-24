"""
센서 시간동기화 + Deskew + Static TF 통합 런치 파일

파이프라인:
1. LiDAR 타임스탬프 오프셋 보정
2. IMU 바이어스 보정 + Madgwick 필터
3. LiDAR Deskew (base_link 기준)
4. Static TF (센서 마운트 위치)

사용법:
  ros2 launch rtabmap_launch sensor_sync.launch.py

  # 타임스탬프 오프셋 측정 후 조정:
  ros2 launch rtabmap_launch sensor_sync.launch.py lidar_offset_sec:=-0.42
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ==================== Launch Arguments ====================
    use_sim_time = LaunchConfiguration('use_sim_time')

    # LiDAR 타임스탬프 오프셋 (ros2 topic echo로 측정 후 조정)
    lidar_offset_sec = LaunchConfiguration('lidar_offset_sec')

    # 센서 마운트 위치 (실측값으로 조정)
    livox_x = LaunchConfiguration('livox_x')
    livox_y = LaunchConfiguration('livox_y')
    livox_z = LaunchConfiguration('livox_z')
    livox_roll = LaunchConfiguration('livox_roll')
    livox_pitch = LaunchConfiguration('livox_pitch')
    livox_yaw = LaunchConfiguration('livox_yaw')

    camera_x = LaunchConfiguration('camera_x')
    camera_y = LaunchConfiguration('camera_y')
    camera_z = LaunchConfiguration('camera_z')
    camera_roll = LaunchConfiguration('camera_roll')
    camera_pitch = LaunchConfiguration('camera_pitch')
    camera_yaw = LaunchConfiguration('camera_yaw')


    # ==================== 1. LiDAR 타임스탬프 오프셋 ====================
    # Livox 타임스탬프가 시스템 시간과 다를 경우 보정
    # 측정 방법: ros2 topic echo /livox/lidar --field header.stamp 와 현재 시간 비교
    lidar_timestamp_offset_node = Node(
        package='livox_timestamp_offset',
        executable='livox_timestamp_offset_node',
        name='livox_timestamp_offset',
        output='screen',
        parameters=[{
            'input_topic': '/livox/lidar',
            'output_topic': '/livox/lidar/synced',
            'offset_sec': lidar_offset_sec,
        }],
    )

    # ==================== 2. IMU 파이프라인 ====================
    # 바이어스 보정 (정지 상태에서 캘리브레이션)
    imu_bias_corrector_node = Node(
        package='camera_imu_pipeline_cpp',
        executable='camera_imu_bias_corrector_cpp',
        name='camera_imu_bias_corrector',
        output='screen',
        parameters=[{
            'input_topic': '/camera/camera/imu',
            'output_topic': '/camera/camera/imu_bias_corrected',
            'target_frame': '',  # 프레임 변환 안 함
            'tf_timeout_sec': 0.05,
            'calib_samples': 1000,
            'stationary_threshold': 0.01,
            'gyro_cov': 0.01,   # 공분산 낮춤 (더 신뢰)
            'accel_cov': 0.01,
            'publish_during_calib': True,
        }],
    )

    # Madgwick 필터 (roll/pitch 추정)
    imu_madgwick_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[{
            'use_mag': False,
            'publish_tf': False,
            'world_frame': 'enu',
            'gain': 0.1,  # 낮을수록 안정적, 높을수록 반응 빠름
        }],
        remappings=[
            ('imu/data_raw', '/camera/camera/imu_bias_corrected'),
            ('imu/data', '/camera/camera/imu_fixed'),
        ],
    )

    # ==================== 3. LiDAR Deskew ====================
    # base_link 기준으로 deskew (odom 드리프트 영향 최소화)
    lidar_deskew_node = Node(
        package='rtabmap_util',
        executable='lidar_deskewing',
        name='lidar_deskewing',
        output='screen',
        parameters=[{
            'fixed_frame_id': 'base_link',  # odom 대신 base_link 사용
            'queue_size': 5,
            'qos': 2,
            'wait_for_transform': 0.2,  # 0.5 → 0.2 (더 엄격)
            'slerp': True,
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('input_cloud', '/livox/lidar/synced'),  # 타임스탬프 보정된 것 사용
            ('output_cloud', '/livox/lidar/synced/deskewed'),
        ],
    )

    # ==================== 4. Static TF (센서 마운트) ====================
    # Livox LiDAR → base_link
    livox_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='livox_tf',
        arguments=[
            '--x', livox_x,
            '--y', livox_y,
            '--z', livox_z,
            '--roll', livox_roll,
            '--pitch', livox_pitch,
            '--yaw', livox_yaw,
            '--frame-id', 'base_link',
            '--child-frame-id', 'livox_frame',
        ],
    )

    # RealSense Camera → base_link
    camera_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf',
        arguments=[
            '--x', camera_x,
            '--y', camera_y,
            '--z', camera_z,
            '--roll', camera_roll,
            '--pitch', camera_pitch,
            '--yaw', camera_yaw,
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_link',
        ],
    )

    # ==================== Launch Description ====================
    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        # LiDAR 타임스탬프 오프셋 (측정 후 조정 필요)
        DeclareLaunchArgument('lidar_offset_sec', default_value='0.018',
            description='LiDAR timestamp offset in seconds. Measure: now - lidar_stamp'),

        # Livox 마운트 위치 (base_link 기준)
        DeclareLaunchArgument('livox_x', default_value='0.0'),
        DeclareLaunchArgument('livox_y', default_value='0.0'),
        DeclareLaunchArgument('livox_z', default_value='0.83'),  # 로봇 위 83cm (12cm 낮춤)
        DeclareLaunchArgument('livox_roll', default_value='0.0'),
        DeclareLaunchArgument('livox_pitch', default_value='0.0'),
        DeclareLaunchArgument('livox_yaw', default_value='0.0'),

        # Camera 마운트 위치 (base_link 기준)
        DeclareLaunchArgument('camera_x', default_value='0.2'),  # 전방 20cm
        DeclareLaunchArgument('camera_y', default_value='0.0'),
        DeclareLaunchArgument('camera_z', default_value='0.58'),  # 로봇 위 58cm (12cm 낮춤)
        DeclareLaunchArgument('camera_roll', default_value='0.0'),
        DeclareLaunchArgument('camera_pitch', default_value='0.0'),
        DeclareLaunchArgument('camera_yaw', default_value='0.0'),
        # Nodes
        lidar_timestamp_offset_node,
        imu_bias_corrector_node,
        imu_madgwick_node,
        lidar_deskew_node,
        livox_tf_node,
        camera_tf_node,
    ])
