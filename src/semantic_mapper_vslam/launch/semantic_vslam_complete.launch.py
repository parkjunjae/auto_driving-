#!/usr/bin/env python3
"""
Complete Semantic VSLAM System
RealSense D455 + RTAB-Map + YOLO Object Detection
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    # Package directories
    pkg_semantic_mapper = get_package_share_directory('semantic_mapper_vslam')

    # YOLO model path (사용자가 지정해야 함)
    default_model_path = os.path.expanduser('~/models/yolov8l.engine')

    # 허용할 클래스 13개
    allow_classes = [
        'tv', 'cup', 'monitor', 'laptop', 'chair', 'couch', 'book',
        'keyboard', 'mouse', 'potted plant', 'bottle', 'cell phone',
        'dining table'
    ]

    return LaunchDescription([
        # Global parameters
        SetParameter(name='use_sim_time', value=False),

        # ===== Launch Arguments =====
        DeclareLaunchArgument(
            'model_path',
            default_value=default_model_path,
            description='Path to YOLO TensorRT engine (.engine) or PyTorch model (.pt)'
        ),
        DeclareLaunchArgument(
            'yolo_conf',
            default_value='0.35',
            description='YOLO confidence threshold'
        ),
        DeclareLaunchArgument(
            'yolo_iou',
            default_value='0.50',
            description='YOLO IoU threshold for NMS'
        ),
        DeclareLaunchArgument(
            'yolo_input_size',
            default_value='640',
            description='YOLO input size (640, 896, etc.)'
        ),
        DeclareLaunchArgument(
            'launch_rtabmap_viz',
            default_value='false',
            description='Launch RTAB-Map visualization'
        ),

        # ===== RealSense D455 =====
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(pkg_semantic_mapper, 'launch', 'realsense_d455.launch.py')
            ]),
            launch_arguments={
                'enable_gyro': 'true',
                'enable_accel': 'true',
                'unite_imu_method': '2',
                'depth_profile': '848x480x30',
                'color_profile': '640x480x30',
            }.items(),
        ),

        # ===== RTAB-Map VSLAM =====
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(pkg_semantic_mapper, 'launch', 'rtabmap_vslam.launch.py')
            ]),
            launch_arguments={
                'localization': 'false',
                'rtabmap_viz': LaunchConfiguration('launch_rtabmap_viz'),
            }.items(),
        ),

        # ===== YOLO Depth Mapper =====
        Node(
            package='semantic_mapper_vslam',
            executable='yolo_depth_mapper',
            name='yolo_depth_mapper',
            output='screen',
            parameters=[{
                # Topics
                'rgb_topic': '/camera/camera/color/image_raw',
                'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
                'camera_info_topic': '/camera/camera/color/camera_info',
                'frame_id': 'camera_color_optical_frame',
                'world_frame': 'map',

                # YOLO settings
                'model_path': LaunchConfiguration('model_path'),
                'conf_threshold': LaunchConfiguration('yolo_conf'),
                'iou_threshold': LaunchConfiguration('yolo_iou'),
                'input_size': LaunchConfiguration('yolo_input_size'),
                'device': 'cuda:0',

                # Class filtering (15개)
                'allow_classes': allow_classes,

                # Output topics
                'debug_topic': '/yolo/debug_image',
                'marker_topic': '/yolo/marker_array',
                'detection_topic': '/semantic_mapper/detections',

                # Tuning
                'min_box_px': 30,
                'core_roi': 0.60,
                'max_fps': 10.0,

                # Persistence
                'persist_k': 3,
                'persist_t': 10,
                'ema_alpha': 0.40,
                'match_dist_m': 0.40,

                # TF
                'tf_use_latest': True,
                'tf_timeout': 0.25,
            }],
        ),

        # ===== RTAB-Map Fusion =====
        Node(
            package='semantic_mapper_vslam',
            executable='yolo_rtabmap_fusion',
            name='yolo_rtabmap_fusion',
            output='screen',
            parameters=[{
                'input_dets_topic': '/semantic_mapper/detections',
                'output_dets_topic': '/semantic_mapper/detections_fused',
                'map_cloud_topic': '/rtabmap/cloud_map',

                # Fusion settings
                'search_radius': 0.7,
                'downsample_stride': 4,
                'min_points': 50,
                'percentile': 0.95,
                'adjust_position': True,
                'max_shift_m': 0.5,
                'estimate_orientation': True,
                'max_fps': 10.0,
            }],
        ),

        # ===== Object Deduplicator (중복 제거) =====
        Node(
            package='semantic_mapper_vslam',
            executable='object_deduplicator',
            name='object_deduplicator',
            output='screen',
            parameters=[{
                # 토픽
                'input_topic': '/semantic_mapper/detections_fused',
                'output_topic': '/semantic_mapper/objects',
                'marker_topic': '/semantic_mapper/objects_marker',

                # 매칭 파라미터 (SLAM drift 고려)
                'match_distance': 0.8,  # 0.8m 이내면 같은 객체 (drift 고려)
                'match_size_threshold': 0.3,  # 크기 유사도 30% 이상 (관대하게)
                'match_label_required': True,  # 라벨 일치 필수

                # 객체 관리
                'min_observations': 3,  # 3번 관측되어야 확정
                'ema_alpha': 0.3,  # 위치 업데이트 계수
                'max_objects': 500,  # 최대 객체 수
                'stale_timeout': 300.0,  # 5분 후 미관측 객체 제거

                # 출력
                'publish_all': False,  # 확정 객체만 발행
                'max_fps': 10.0,
            }],
        ),
    ])
