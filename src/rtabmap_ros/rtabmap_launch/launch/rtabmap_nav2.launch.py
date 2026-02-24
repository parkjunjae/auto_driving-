import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('rtabmap_launch')

    use_sim_time = LaunchConfiguration('use_sim_time')
    nav2_params = LaunchConfiguration('nav2_params')
    rtabmap_viz = LaunchConfiguration('rtabmap_viz')
    rviz = LaunchConfiguration('rviz')
    log_level = LaunchConfiguration('log_level')
    rtabmap_args = LaunchConfiguration('rtabmap_args')
    localization = LaunchConfiguration('localization')
    database_path = LaunchConfiguration('database_path')
    delete_db_on_start = LaunchConfiguration('delete_db_on_start')
    livox_deskewed_topic = LaunchConfiguration('livox_deskewed_topic')
    livox_filtered_topic = LaunchConfiguration('livox_filtered_topic')
    use_dynamic_filter = LaunchConfiguration('use_dynamic_filter')
    dynamic_filter_output = LaunchConfiguration('dynamic_filter_output')
    dynamic_voxel_size = LaunchConfiguration('dynamic_voxel_size')
    dynamic_min_hits = LaunchConfiguration('dynamic_min_hits')
    dynamic_hit_window_sec = LaunchConfiguration('dynamic_hit_window_sec')
    dynamic_max_stale_sec = LaunchConfiguration('dynamic_max_stale_sec')
    dynamic_min_static_sec = LaunchConfiguration('dynamic_min_static_sec')
    dynamic_z_min = LaunchConfiguration('dynamic_z_min')
    dynamic_z_max = LaunchConfiguration('dynamic_z_max')
    dynamic_min_range = LaunchConfiguration('dynamic_min_range')
    dynamic_target_frame = LaunchConfiguration('dynamic_target_frame')
    dynamic_tf_timeout_sec = LaunchConfiguration('dynamic_tf_timeout_sec')

    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, 'launch', 'rtabmap.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'rtabmap_viz': rtabmap_viz,
            'rviz': rviz,
            'localization': localization,
            # EKF(odometry/filtered) + RTAB-Map 구조를 고정해 TF 체인을 일관되게 유지
            'odom_topic': '/odometry/filtered',
            'odom_frame_id': 'odom',
            'map_frame_id': 'map',
            'publish_tf_map': 'true',
            'visual_odometry': 'false',
            'icp_odometry': 'false',
            'log_level': 'error',  
            'odom_log_level': 'error',  # Odometry 로그도 error 레벨로
            'qos': '2',  # BEST_EFFORT QoS for Nav2 compatibility
            'latch': 'false',  # VOLATILE durability for Nav2 compatibility
            'topic_queue_size': '10',  # RGB-D subscriber queue (for delayed camera streams)
            'sync_queue_size': '10',   # Approx sync queue size
            # Let RTAB-Map sync RGB + Depth directly (more reliable than /camera/camera/rgbd)
            'approx_rgbd_sync': 'true',
            'approx_sync_max_interval': '0.08',
            'rgbd_sync': 'true',
            'subscribe_rgbd': 'false',
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'rtabmap_args': rtabmap_args,  # Prefer ROS params in rtabmap.launch.py
            'database_path': database_path,
            'delete_db_on_start': delete_db_on_start,  # DB reset via ROS param
            # RTAB-Map에는 동적 필터 전 단계인 filtered만 입력 (정적맵 품질 유지)
            'scan_cloud_topic': livox_filtered_topic,
        }.items()
    )
    livox_filter_node = Node(
        package='livox_pointcloud_filter',          # 필터 노드가 들어있는 패키지
        executable='livox_pointcloud_filter_node',  # 실행할 노드 이름
        name='livox_pointcloud_filter',             # 노드 이름(ros2 node list에 표시됨)
        output='screen',                            # 로그를 터미널에 출력
        parameters=[{
            'input_topic': livox_deskewed_topic,      # 입력 포인트클라우드(데스큐 완료)
            'output_topic': livox_filtered_topic,     # 필터링 후 출력 토픽
            'leaf_size': 0.05,                       # VoxelGrid 다운샘플 크기(해상도)
            'ror_radius': 0.15,                      # ROR 반경(이웃 탐색 거리)
            'ror_min_neighbors': 2,                  # ROR 이웃 최소 개수
            'use_voxel': True,                       # VoxelGrid 다운샘플 사용 여부
            'use_ror': True,                         # ROR 노이즈 제거 사용 여부
        }],
    )
    dynamic_filter_node = Node(
        package='livox_pointcloud_filter',
        executable='dynamic_object_filter_node',
        name='dynamic_object_filter',
        output='screen',
        condition=IfCondition(use_dynamic_filter),
        parameters=[{
            'input_topic': livox_filtered_topic,          # 입력 포인트클라우드(전처리 LiDAR)
            'output_topic': dynamic_filter_output,        # 출력 포인트클라우드(정적 위주)
            'voxel_size': dynamic_voxel_size,             # 보셀 크기[m], 클수록 거칠고 빠름
            'min_hits': dynamic_min_hits,                 # 정적으로 인정할 최소 관측 횟수
            'hit_window_sec': dynamic_hit_window_sec,     # 관측 누적 시간 창[s]
            'max_stale_sec': dynamic_max_stale_sec,       # 미관측 보셀 상태 제거 시간[s]
            'min_static_sec': dynamic_min_static_sec,     # 정적으로 인정할 최소 유지시간[s]
            'z_min': dynamic_z_min,                       # 필터링 하한 높이[m]
            'z_max': dynamic_z_max,                       # 필터링 상한 높이[m]
            'min_range': dynamic_min_range,               # 센서 근접 노이즈 제거 거리[m]
            'target_frame': dynamic_target_frame,         # 누적 기준 고정 프레임(odom/map)
            'tf_timeout_sec': dynamic_tf_timeout_sec,     # TF 조회 타임아웃[s]
        }],
    )
    nav2_server_nodes = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='nav2_controller',
                executable='controller_server',
                name='controller_server',
                output='screen',
                parameters=[nav2_params, {'use_sim_time': use_sim_time}],
                arguments=['--ros-args', '--log-level', log_level],
            ),
            Node(
                package='nav2_planner',
                executable='planner_server',
                name='planner_server',
                output='screen',
                parameters=[nav2_params, {'use_sim_time': use_sim_time}],
                arguments=['--ros-args', '--log-level', log_level]
            ),
            Node(
                package='nav2_smoother',
                executable='smoother_server',
                name='smoother_server',
                output='screen',
                parameters=[nav2_params, {'use_sim_time': use_sim_time}],
                arguments=['--ros-args', '--log-level', log_level]
            ),
            Node(
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                output='screen',
                parameters=[nav2_params, {'use_sim_time': use_sim_time}],
                arguments=['--ros-args', '--log-level', 'info']
            ),
            Node(
                package='nav2_behaviors',
                executable='behavior_server',
                name='behavior_server',
                output='screen',
                parameters=[nav2_params, {'use_sim_time': use_sim_time}],
                arguments=['--ros-args', '--log-level', 'info']
            ),
            Node(
                package='nav2_waypoint_follower',
                executable='waypoint_follower',
                name='waypoint_follower',
                output='screen',
                parameters=[nav2_params, {'use_sim_time': use_sim_time}],
                arguments=['--ros-args', '--log-level', log_level]
            ),
            Node(
                package='nav2_velocity_smoother',
                executable='velocity_smoother',
                name='velocity_smoother',
                output='screen',
                parameters=[nav2_params, {'use_sim_time': use_sim_time}],
                arguments=['--ros-args', '--log-level', log_level]
            ),
        ]
    )

    # Lifecycle manager - launched with delay to ensure RTABMAP is ready and publishing obstacles
    lifecycle_manager_node = TimerAction(
        period=15.0,  # RTABMAP이 맵을 초기화할 시간 확보
        actions=[
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager',
                output='screen',
                parameters=[nav2_params, {'use_sim_time': use_sim_time}],
                arguments=['--ros-args', '--log-level', 'info']  # lifecycle_manager는 info 레벨로
            )
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation time'),
        DeclareLaunchArgument(
            'nav2_params',
            default_value=PathJoinSubstitution([pkg_share, 'launch', 'config', 'nav2_rtabmap_params.yaml']),
            description='Nav2 parameters file'
        ),
        DeclareLaunchArgument('rtabmap_viz', default_value='false', description='Launch rtabmap_viz GUI'),
        DeclareLaunchArgument('rviz', default_value='false', description='Launch RVIZ from rtabmap launch'),
        DeclareLaunchArgument('log_level', default_value='warn', description='Nav2 log level'),
        DeclareLaunchArgument('localization', default_value='false', description='Enable localization-only mode'),
        DeclareLaunchArgument('livox_deskewed_topic', default_value='/livox/lidar/synced/deskewed',
                             description='Deskewed LiDAR topic used by RTAB-Map and Livox filter'),
        DeclareLaunchArgument('livox_filtered_topic', default_value='/livox/lidar/filtered',
                             description='Filtered LiDAR topic for Nav2 costmap marking'),
        DeclareLaunchArgument('use_dynamic_filter', default_value='true',
                             description='Enable dynamic object filter for global-map-friendly static cloud'),
        DeclareLaunchArgument('dynamic_filter_output', default_value='/livox/lidar/static_filtered',
                             description='Output topic of dynamic object filter'),
        DeclareLaunchArgument('dynamic_voxel_size', default_value='0.15',
                             description='[m] voxel size for temporal static/dynamic filtering'),
        DeclareLaunchArgument('dynamic_min_hits', default_value='4',
                             description='Minimum hits in window to classify voxel as static'),
        DeclareLaunchArgument('dynamic_hit_window_sec', default_value='0.5',
                             description='[s] temporal window for hit accumulation'),
        DeclareLaunchArgument('dynamic_max_stale_sec', default_value='0.5',
                             description='[s] remove stale voxel states not seen recently'),
        DeclareLaunchArgument('dynamic_min_static_sec', default_value='0.5',
                             description='[s] minimum continuous time to accept voxel as static'),
        DeclareLaunchArgument('dynamic_z_min', default_value='0.05',
                             description='[m] minimum obstacle height to include'),
        DeclareLaunchArgument('dynamic_z_max', default_value='1.2',
                             description='[m] maximum obstacle height to include'),
        DeclareLaunchArgument('dynamic_min_range', default_value='0.8',
                             description='[m] remove near-range noisy returns'),
        DeclareLaunchArgument('dynamic_target_frame', default_value='odom',
                             description='Accumulation frame for dynamic filter (odom or map)'),
        DeclareLaunchArgument('dynamic_tf_timeout_sec', default_value='0.05',
                             description='TF lookup timeout for dynamic filter'),
        # Keep args empty to avoid overriding ROS params.
        DeclareLaunchArgument('rtabmap_args', default_value='', description='Extra CLI flags for rtabmap'),
        # Use ROS param to control DB reset from this launch file.
        DeclareLaunchArgument('delete_db_on_start', default_value='true', description='Delete RTAB-Map database at startup'),
        DeclareLaunchArgument('database_path', default_value=os.path.expanduser('~/.ros/rtabmap_nav2.db'), description=''),
        rtabmap_launch,
        livox_filter_node,
        dynamic_filter_node,
        nav2_server_nodes,
        lifecycle_manager_node
    ])
