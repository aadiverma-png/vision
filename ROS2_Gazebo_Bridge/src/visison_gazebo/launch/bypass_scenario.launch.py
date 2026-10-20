"""Scenario 1 in Gazebo Harmonic: lane A blocked, R2 (and more robots) oncoming in lane B.

All in one terminal:

    ros2 launch visison_gazebo bypass_scenario.launch.py r2_arrival:=50.0

Emergency braking and the reverse algorithm (lane-B robots run 12 s ahead of
their reservations, so they meet R1 in the middle of its bypass):

    ros2 launch visison_gazebo bypass_scenario.launch.py r2_early:=12 lane_b_robots:=3

Or with Gazebo kept open in terminal 1, so the scenario can be reset quickly:

    ros2 launch visison_gazebo gazebo_world.launch.py                          # terminal 1
    ros2 launch visison_gazebo bypass_scenario.launch.py start_gazebo:=false   # terminal 2

Restarting terminal 2 removes the old robots, respawns them and restarts the
scenario at t = 0. ``r2_arrival`` is when R2 passes the fallen box, in seconds
after both robots are up. Try 50 (R1 bypasses at once) or 30 (R1 waits for
R2, then bypasses).
"""

import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from visison_gazebo import scenario as sc

WORLD_NAME = 'aisle_world'  # <world name> in worlds/aisle_world.sdf
MAX_ROBOTS = 12  # robots r1..r12 left over from an earlier run are removed on restart


def _remove_robot(name):
    return ExecuteProcess(
        cmd=['gz', 'service', '-s', f'/world/{WORLD_NAME}/remove',
             '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '3000', '--req', f'name: "{name}" type: MODEL'],
        output='log',
    )


def _robot_nodes(context):
    pkg = get_package_share_directory('visison_gazebo')
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    arrival, r2_speed, early = float(arg('r2_arrival')), float(arg('r2_speed')), float(arg('r2_early'))
    n_b, n_a = int(arg('lane_b_robots')), int(arg('lane_a_followers'))
    reuse_gazebo = arg('start_gazebo').lower() in ('false', '0', 'no')
    layout = sc.fleet_layout(n_b, n_a, arrival, r2_speed, early)  # also validates the arguments
    if len(layout) > MAX_ROBOTS:
        raise RuntimeError(f'at most {MAX_ROBOTS} robots are supported by this launch file')
    model = os.path.join(pkg, 'urdf', 'amr.urdf.xacro')

    def robot(name, x, y, yaw, **mappings):
        """Spawn a robot in Gazebo and publish its URDF frames (<name>/...) for RViz."""
        urdf = xacro.process_file(model, mappings={'ns': name, **mappings}).toxml()
        return [
            Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-world', WORLD_NAME, '-string', urdf, '-name', name,
                            '-x', str(x), '-y', str(y), '-z', '0.05', '-Y', str(yaw)]),
            Node(package='robot_state_publisher', executable='robot_state_publisher',
                 namespace=name, output='log',
                 parameters=[{'robot_description': urdf, 'frame_prefix': f'{name}/',
                              'use_sim_time': True}]),
        ]

    bridge_topics = ['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
    robots = []
    for plan in layout:
        ns = plan.ns
        bridge_topics += [
            f'/{ns}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            f'/{ns}/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            f'/{ns}/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            f'/{ns}/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
        ]
        robots += robot(ns, *plan.spawn, lidar='true', color=plan.color)

    rmf = arg('rmf').lower() in ('true', '1', 'yes')
    rmf_nodes, rmf_share = [], ''
    if rmf:  # Plan B: RMF core (traffic schedule + task dispatcher) on simulation time
        rmf_share = get_package_share_directory('visison_rmf')
        rmf_nodes = [
            Node(package='rmf_traffic_ros2', executable='rmf_traffic_schedule', output='log',
                 parameters=[{'use_sim_time': True}]),
            Node(package='rmf_task_ros2', executable='rmf_task_dispatcher', output='log',
                 parameters=[{'use_sim_time': True}]),
        ]

    scenario_actions = [
        *rmf_nodes,
        *robots,
        Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
             arguments=bridge_topics, parameters=[{'use_sim_time': True}]),
        Node(package='visison_gazebo', executable='bypass_scenario', output='screen',
             emulate_tty=True,
             ros_arguments=['-p', 'use_sim_time:=true'],  # also for the RMF fleet adapters in this process
             parameters=[{
                 'use_sim_time': True,
                 'r2_arrival': arrival,
                 'r2_speed': r2_speed,
                 'r2_early': early,
                 'lane_b_robots': n_b,
                 'lane_a_followers': n_a,
                 'perception': arg('perception'),
                 'orca': arg('orca').lower() in ('true', '1', 'yes'),
                 'rmf': rmf,
                 'rmf_share': rmf_share,
                 'other_speed': float(arg('other_speed')),
                 'coop': arg('coop').lower() in ('true', '1', 'yes'),
                 'sensing_range': float(arg('sensing_range')),
             }]),
    ]
    if not reuse_gazebo:
        return scenario_actions
    # Gazebo is already running (terminal 1): remove robots left from the last
    # run, then spawn fresh ones once the removal has gone through.
    old = [_remove_robot(f'r{i}') for i in range(1, MAX_ROBOTS + 1)]
    return [*old, TimerAction(period=3.0, actions=scenario_actions)]


def generate_launch_description():
    pkg = get_package_share_directory('visison_gazebo')

    return LaunchDescription([
        DeclareLaunchArgument('r2_arrival', default_value='50.0',
                              description='Seconds after start at which R2 passes the fallen box'),
        DeclareLaunchArgument('r2_speed', default_value='0.8', description='R2 speed in lane B [m/s]'),
        DeclareLaunchArgument('r2_early', default_value='0.0',
                              description='Seconds the lane-B robots run AHEAD of their reservations '
                                          '(a timing mistake the emergency layer must catch)'),
        DeclareLaunchArgument('lane_b_robots', default_value='1',
                              description='Robots in the lane-B convoy (R2, R3, ...)'),
        DeclareLaunchArgument('rmf', default_value='false',
                              description='Plan B: tasks and traffic through Open-RMF (R1 = Visison fleet, '
                                          'lane B = another vendor fleet)'),
        DeclareLaunchArgument('other_speed', default_value='0.0',
                              description='RMF mode: the other fleet really drives this fast (RMF plans 0.8 m/s)'),
        DeclareLaunchArgument('orca', default_value='false',
                              description='Plan A: ORCA-DD collision avoidance in every robot (MotionArbiter)'),
        DeclareLaunchArgument('coop', default_value='false',
                              description='Plan A: reciprocal bypass windows (oncoming robots slow a little)'),
        DeclareLaunchArgument('lane_a_followers', default_value='0',
                              description='Extra robots following R1 in lane A'),
        DeclareLaunchArgument('perception', default_value='lidar',
                              description="'lidar' (detect the box in /r1/scan) or 'ground_truth'"),
        DeclareLaunchArgument('sensing_range', default_value='8.0',
                              description='How far ahead R1 reacts to an obstacle [m]'),
        DeclareLaunchArgument('start_gazebo', default_value='true',
                              description="'false' when gazebo_world.launch.py already runs in another terminal"),
        DeclareLaunchArgument('gui', default_value='true', description='Start the Gazebo GUI'),
        DeclareLaunchArgument('rviz', default_value='true', description='Start RViz2 with the Visison markers'),
        DeclareLaunchArgument('use_nvidia', default_value='true',
                              description='Render on the NVIDIA dGPU (PRIME offload)'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'gazebo_world.launch.py')),
            launch_arguments={'gui': LaunchConfiguration('gui'),
                              'use_nvidia': LaunchConfiguration('use_nvidia')}.items(),
            condition=IfCondition(LaunchConfiguration('start_gazebo')),
        ),

        OpaqueFunction(function=_robot_nodes),

        # The markers live in "world" (= aisle frame); give RViz that frame.
        Node(package='tf2_ros', executable='static_transform_publisher', output='log',
             arguments=['--frame-id', 'world', '--child-frame-id', 'aisle']),
        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', os.path.join(pkg, 'rviz', 'bypass.rviz')],
             parameters=[{'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
