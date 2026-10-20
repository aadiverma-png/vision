"""Every Visison case, one after another, in Gazebo + RViz, at 2x speed.

    ros2 launch visison_gazebo demo_all.launch.py              # all cases
    ros2 launch visison_gazebo demo_all.launch.py speed:=1.0   # real time
    ros2 launch visison_gazebo demo_all.launch.py cases:=3,4,9 # a selection

The demo director (visison_gazebo/demo_director.py) removes and spawns the
robots of each case in the same Gazebo session, runs the case, shows its title
and result in RViz, and moves on. No terminal command is needed between cases.
"""

import os
import tempfile

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction, OpaqueFunction,
                            SetEnvironmentVariable, Shutdown)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORLD_NAME = 'aisle_world'
MAX_ROBOTS = 8  # must match demo_director.MAX_ROBOTS


def _setup(context):
    pkg = get_package_share_directory('visison_gazebo')
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    speed = float(arg('speed'))

    # the aisle world, running at `speed` x real time
    with open(os.path.join(pkg, 'worlds', 'aisle_world.sdf')) as f:
        sdf = f.read().replace('<real_time_factor>1.0</real_time_factor>',
                               f'<real_time_factor>{speed}</real_time_factor>', 1)
    world = os.path.join(tempfile.gettempdir(), f'visison_demo_world_{speed:g}x.sdf')
    with open(world, 'w') as f:
        f.write(sdf)

    actions = [
        ExecuteProcess(cmd=['gz', 'sim', '-s', '-r', '-v', '1', world], output='log',
                       on_exit=Shutdown(reason='Gazebo server exited')),
        ExecuteProcess(cmd=['gz', 'sim', '-g', '-v', '1'], output='log',
                       condition=IfCondition(LaunchConfiguration('gui'))),
    ]

    # one bridge for every robot the director may spawn (r1..r8)
    topics = ['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
    for i in range(1, MAX_ROBOTS + 1):
        ns = f'r{i}'
        topics += [f'/{ns}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                   f'/{ns}/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                   f'/{ns}/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                   f'/{ns}/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model']
    actions.append(Node(package='ros_gz_bridge', executable='parameter_bridge', output='log',
                        arguments=topics, parameters=[{'use_sim_time': True}]))

    # URDF frames of every robot, so RViz can place each lidar scan
    model = os.path.join(pkg, 'urdf', 'amr.urdf.xacro')
    for i in range(1, MAX_ROBOTS + 1):
        ns = f'r{i}'
        urdf = xacro.process_file(model, mappings={'ns': ns, 'lidar': 'true'}).toxml()
        actions.append(Node(package='robot_state_publisher', executable='robot_state_publisher',
                            namespace=ns, output='log',
                            parameters=[{'robot_description': urdf, 'frame_prefix': f'{ns}/',
                                         'use_sim_time': True}]))

    # Open-RMF core for the RMF case (if the visison_rmf package is built)
    try:
        get_package_share_directory('visison_rmf')
        actions += [
            Node(package='rmf_traffic_ros2', executable='rmf_traffic_schedule', output='log',
                 parameters=[{'use_sim_time': True}]),
            Node(package='rmf_task_ros2', executable='rmf_task_dispatcher', output='log',
                 parameters=[{'use_sim_time': True}]),
        ]
    except Exception:
        pass

    actions.append(Node(package='visison_gazebo', executable='demo_director', output='screen', emulate_tty=True,
                        ros_arguments=['-p', 'use_sim_time:=true'],
                        parameters=[{'use_sim_time': True, 'cases': arg('cases'),
                                     'pause': float(arg('pause'))}],
                        on_exit=Shutdown(reason='demo director exited')))
    actions.append(Node(package='rviz2', executable='rviz2', output='log',
                        arguments=['-d', os.path.join(pkg, 'rviz', 'demo.rviz')],
                        parameters=[{'use_sim_time': True}],
                        condition=IfCondition(LaunchConfiguration('rviz'))))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('speed', default_value='2.0', description='Simulation speed (real-time factor)'),
        DeclareLaunchArgument('cases', default_value='all', description="'all' or case numbers, e.g. '1,3,9'"),
        DeclareLaunchArgument('pause', default_value='4.0', description='Seconds (wall time) between cases'),
        DeclareLaunchArgument('gui', default_value='true', description='Start the Gazebo GUI'),
        DeclareLaunchArgument('rviz', default_value='true', description='Start RViz'),
        DeclareLaunchArgument('use_nvidia', default_value='true', description='Render on the NVIDIA dGPU'),
        GroupAction(
            condition=IfCondition(LaunchConfiguration('use_nvidia')),
            actions=[SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1'),
                     SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
                     SetEnvironmentVariable('__EGL_VENDOR_LIBRARY_FILENAMES',
                                            '/usr/share/glvnd/egl_vendor.d/10_nvidia.json')],
            scoped=False,
        ),
        # The markers live in "world" (= aisle frame); give RViz that frame.
        Node(package='tf2_ros', executable='static_transform_publisher', output='log',
             arguments=['--frame-id', 'world', '--child-frame-id', 'aisle']),
        OpaqueFunction(function=_setup),
    ])
