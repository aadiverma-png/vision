#!/usr/bin/env bash
# The complete Visison demo: every case, one after another, in Gazebo + RViz at 2x speed.
#
#   ./run_demo.sh                 # all cases, 2x speed
#   ./run_demo.sh speed:=1.0      # real time
#   ./run_demo.sh cases:=3,4,9    # a selection (see visison_gazebo/demo_director.py: CASES)
#
# Extra arguments are passed straight to `ros2 launch`.

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ROS2_Gazebo_Bridge"
source /opt/ros/jazzy/setup.bash
cd "$WS" || exit 1
colcon build --packages-select visison_gazebo visison_rmf || exit 1
source install/setup.bash 2>/dev/null
exec ros2 launch visison_gazebo demo_all.launch.py "$@"
