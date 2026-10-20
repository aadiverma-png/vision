import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            ExecuteProcess, GroupAction, SetEnvironmentVariable,
                            Shutdown)
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('amr_sim')
    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui')
    use_nvidia = LaunchConfiguration('use_nvidia')
    rviz = LaunchConfiguration('rviz')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    yaw = LaunchConfiguration('yaw')

    robot_description = ParameterValue(
        Command(['xacro ', os.path.join(pkg_share, 'urdf', 'amr.urdf.xacro')]),
        value_type=str,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(pkg_share, 'worlds', 'amr_world.sdf'),
            description='Full path to the Gazebo world SDF'),
        DeclareLaunchArgument('rviz', default_value='true', description='Start RViz2'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),

        DeclareLaunchArgument('gui', default_value='true', description='Start the Gazebo GUI'),
        DeclareLaunchArgument(
            'use_nvidia', default_value='true',
            description='Render on the NVIDIA dGPU (PRIME offload) instead of the iGPU'),

        # Let Gazebo find models/meshes installed in this package
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH', os.path.dirname(pkg_share)),

        # On hybrid-GPU laptops (prime-select on-demand) Gazebo otherwise renders through
        # Mesa on the iGPU, fails to create an EGL context on the NVIDIA card, and the GUI
        # becomes unresponsive and gets closed.
        GroupAction(
            condition=IfCondition(use_nvidia),
            actions=[
                SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1'),
                SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
                SetEnvironmentVariable(
                    '__EGL_VENDOR_LIBRARY_FILENAMES',
                    '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'),
            ],
            scoped=False,
        ),

        # 1a. Gazebo Harmonic server (-r: start running immediately). Only the server
        #     is required: if it exits, the whole launch shuts down.
        ExecuteProcess(
            cmd=['gz', 'sim', '-s', '-r', '-v', '2', world],
            output='screen',
            on_exit=Shutdown(reason='Gazebo server exited'),
        ),

        # 1b. Gazebo GUI as a separate process, so closing (or crashing) the window
        #     no longer kills the simulation and ROS nodes. Reopen it with: gz sim -g
        ExecuteProcess(
            cmd=['gz', 'sim', '-g', '-v', '2'],
            output='screen',
            condition=IfCondition(gui),
        ),

        # 2. Robot state publisher (static TFs from URDF + wheel TFs from /joint_states)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description,
                         'use_sim_time': True}],
        ),

        # 3. Spawn the robot from the /robot_description topic
        Node(
            package='ros_gz_sim',
            executable='create',
            output='screen',
            arguments=['-topic', 'robot_description',
                       '-name', 'amr',
                       '-x', x, '-y', y, '-z', '0.05', '-Y', yaw],
        ),

        # 4. ROS <-> Gazebo topic bridge
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            output='screen',
            parameters=[{
                'config_file': os.path.join(pkg_share, 'config', 'ros_gz_bridge.yaml'),
                'use_sim_time': True,
            }],
        ),

        # 5. Camera image bridge (image_transport-aware, faster than parameter_bridge)
        Node(
            package='ros_gz_image',
            executable='image_bridge',
            output='screen',
            arguments=['/camera/image'],
            parameters=[{'use_sim_time': True}],
        ),

        # 6. RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', PathJoinSubstitution([pkg_share, 'rviz', 'amr.rviz'])],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(rviz),
        ),
    ])
