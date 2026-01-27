import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('tracer_description')
    ros_gz_share = get_package_share_directory('ros_gz_sim')
    gz_resource_root = os.path.dirname(pkg_share)

    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    gz_args = LaunchConfiguration('gz_args')
    model = LaunchConfiguration('model')
    controllers = LaunchConfiguration('controllers')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([ros_gz_share, 'launch', 'gz_sim.launch.py'])
        ),
        launch_arguments={
            'gz_args': gz_args,
        }.items(),
    )

    # Gazebo Sim 리소스 경로에 패키지 경로 추가
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[
            EnvironmentVariable('GZ_SIM_RESOURCE_PATH', default_value=''),
            TextSubstitution(text=':'),
            gz_resource_root,
        ],
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': ParameterValue(
                Command(['xacro ', model]),
                value_type=str,
            ),
        }],
    )

    # Gazebo Sim에 URDF를 스폰
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'tracer', '-topic', 'robot_description'],
        output='screen',
    )

    # /clock 브리지(ROS time 사용)
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    # ros2_control 컨트롤러 스포너
    joint_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
            '--param-file', controllers,
        ],
        output='screen',
    )

    diff_drive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'diff_drive_controller',
            '--controller-manager', '/controller_manager',
            '--param-file', controllers,
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'world',
            default_value='empty.sdf',
        ),
        DeclareLaunchArgument(
            'gz_args',
            default_value=[
                TextSubstitution(text='-r -v 2 '),
                world,
            ],
        ),
        DeclareLaunchArgument(
            'model',
            default_value=os.path.join(pkg_share, 'urdf', 'tracer_v1.xacro'),
        ),
        DeclareLaunchArgument(
            'controllers',
            default_value=os.path.join(pkg_share, 'config', 'tracer_controllers.yaml'),
        ),
        set_gz_resource_path,
        gz_sim,
        robot_state_publisher,
        clock_bridge,
        spawn_entity,
        joint_state_spawner,
        diff_drive_spawner,
    ])
