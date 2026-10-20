"""Gazebo Harmonic with the aisle world only (terminal 1).

    ros2 launch visison_gazebo gazebo_world.launch.py

Leave it running. Start the robots and the Visison node in a second terminal
with ``bypass_scenario.launch.py start_gazebo:=false``. Restarting that
launch removes the old robots and respawns them, which resets the scenario
without restarting Gazebo.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction,
                            SetEnvironmentVariable, Shutdown)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    world = os.path.join(get_package_share_directory('visison_gazebo'), 'worlds', 'aisle_world.sdf')
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='Start the Gazebo GUI'),
        DeclareLaunchArgument('use_nvidia', default_value='true',
                              description='Render on the NVIDIA dGPU (PRIME offload)'),
        GroupAction(
            condition=IfCondition(LaunchConfiguration('use_nvidia')),
            actions=[
                SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1'),
                SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
                SetEnvironmentVariable('__EGL_VENDOR_LIBRARY_FILENAMES',
                                       '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'),
            ],
            scoped=False,
        ),
        ExecuteProcess(cmd=['gz', 'sim', '-s', '-r', '-v', '2', world], output='screen',
                       on_exit=Shutdown(reason='Gazebo server exited')),
        ExecuteProcess(cmd=['gz', 'sim', '-g', '-v', '2'], output='screen',
                       condition=IfCondition(LaunchConfiguration('gui'))),
    ])
