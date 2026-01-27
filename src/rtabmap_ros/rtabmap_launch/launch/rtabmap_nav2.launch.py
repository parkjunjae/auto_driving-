import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
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
    database_path = LaunchConfiguration('database_path')
    delete_db_on_start = LaunchConfiguration('delete_db_on_start')

    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, 'launch', 'rtabmap.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'rtabmap_viz': rtabmap_viz,
            'rviz': rviz,
            'localization': 'false',  
            'log_level': 'error',  
            'odom_log_level': 'error',  # Odometry 로그도 error 레벨로
            'qos': '2',  # BEST_EFFORT QoS for Nav2 compatibility
            'latch': 'false',  # VOLATILE durability for Nav2 compatibility
            'rtabmap_args': rtabmap_args,  # Prefer ROS params in rtabmap.launch.py
            'database_path': database_path,
            'delete_db_on_start': delete_db_on_start,  # DB reset via ROS param
        }.items()
    )
    # Livox deskewing using rtabmap_util (C++), faster than Python deskew
    lidar_deskew_node = Node(
        package='rtabmap_util',
        executable='lidar_deskewing',
        name='rtabmap_lidar_deskewing',
        output='screen',
        parameters=[{
            'fixed_frame_id': 'odom',
            'queue_size': 5,
            'qos': 2,  # BEST_EFFORT
            'wait_for_transform': 0.5,
            'slerp': True,
            'use_sim_time': use_sim_time,
        }],
        remappings=[('input_cloud', '/livox/lidar')],
    )
    # Livox 포인트클라우드 필터(다운샘플 + ROR)
    # - 데스큐된 포인트를 입력으로 받아 다운샘플링 + ROR 노이즈 제거 수행
    # - 필터링 결과를 /livox/lidar/filtered로 출력하여 코스트맵에 사용
    livox_filter_node = Node(
        package='livox_pointcloud_filter',          # 필터 노드가 들어있는 패키지
        executable='livox_pointcloud_filter_node',  # 실행할 노드 이름
        name='livox_pointcloud_filter',             # 노드 이름(ros2 node list에 표시됨)
        output='screen',                            # 로그를 터미널에 출력
        parameters=[{
            'input_topic': '/livox/lidar/deskewed',  # 입력 포인트클라우드(데스큐 완료)
            'output_topic': '/livox/lidar/filtered', # 필터링 후 출력 토픽
            'leaf_size': 0.05,                       # VoxelGrid 다운샘플 크기(해상도)
            'ror_radius': 0.15,                      # ROR 반경(이웃 탐색 거리)
            'ror_min_neighbors': 2,                  # ROR 이웃 최소 개수
            'use_voxel': True,                       # VoxelGrid 다운샘플 사용 여부
            'use_ror': True,                         # ROR 노이즈 제거 사용 여부
        }],
    )
    # # ICP odometry (LiDAR) for EKF fusion
    # icp_odometry_node = Node(
    #     package='rtabmap_odom',
    #     executable='icp_odometry',
    #     name='icp_odometry',
    #     output='screen',
    #     parameters=[{
    #         'frame_id': 'base_link',
    #         'odom_frame_id': 'odom',
    #         'publish_tf': False,
    #         'wait_for_transform': 0.2,
    #         'approx_sync': False,
    #         'qos': 2,  # BEST_EFFORT
    #         'topic_queue_size': 10,
    #         'sync_queue_size': 10,
    #         'deskewing': False,
    #         'deskewing_slerp': False,
    #         'guess_frame_id': 'odom',
    #         'guess_min_translation': 0.0,
    #         'guess_min_rotation': 0.0,
    #         'always_process_most_recent_frame': True,
    #         'use_sim_time': use_sim_time,
    #     }],
    #     remappings=[
    #         ('scan_cloud', '/livox/lidar/deskewed'),
    #         ('scan', '/scan_dummy'),
    #         ('odom', '/icp_odom'),
    #         ('imu', '/livox/imu_fixed'),
    #     ],
    # )
    # PointCloud2 frame transform node (base_link → odom)
    # pointcloud_transform_node = Node(
    #     package='pointcloud_transform',
    #     executable='transform_node',
    #     name='pointcloud_transform_node',
    #     output='screen',
    #     parameters=[{
    #         'input_topic': '/rtabmap/local_grid_obstacle',
    #         'output_topic': '/rtabmap/local_grid_obstacle_odom',
    #         'target_frame': 'odom'
    #     }]
    # )

    # Nav2 server nodes (without lifecycle manager)
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
        # Keep args empty to avoid overriding ROS params.
        DeclareLaunchArgument('rtabmap_args', default_value='', description='Extra CLI flags for rtabmap'),
        # Use ROS param to control DB reset from this launch file.
        DeclareLaunchArgument('delete_db_on_start', default_value='true', description='Delete RTAB-Map database at startup'),
        DeclareLaunchArgument('database_path', default_value=os.path.expanduser('~/.ros/rtabmap_nav2.db'), description=''),
        rtabmap_launch,
        lidar_deskew_node,
        livox_filter_node,
        # icp_odometry_node,
        # pointcloud_transform_node,
        nav2_server_nodes,
        lifecycle_manager_node
    ])
