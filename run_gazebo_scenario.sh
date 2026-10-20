#!/usr/bin/env bash
# Build and launch Visison scenario 1 in Gazebo Harmonic with RViz (ROS 2 Jazzy).
#
#   ./run_gazebo_scenario.sh              # R2 passes the fallen box at t = 50 s
#   ./run_gazebo_scenario.sh 30           # R2 arrives earlier: R1 waits, then bypasses
#   ./run_gazebo_scenario.sh 45 rviz:=false perception:=ground_truth
#
# Any extra arguments are passed straight to `ros2 launch`.

ARRIVAL="${1:-50.0}"
shift || true
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ROS2_Gazebo_Bridge"

# ROS setup scripts are not `set -e`/`set -u` safe, so error checks are explicit.
source /opt/ros/jazzy/setup.bash
cd "$WS" || exit 1
colcon build --packages-select visison_gazebo || exit 1
source install/setup.bash 2>/dev/null

echo "Launching scenario 1 with r2_arrival=${ARRIVAL} s"
exec ros2 launch visison_gazebo bypass_scenario.launch.py r2_arrival:="${ARRIVAL}" "$@"
