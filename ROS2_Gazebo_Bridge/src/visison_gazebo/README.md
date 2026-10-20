# visison_gazebo: scenario 1 in Gazebo Harmonic

This package runs the Visison bypass in Gazebo Harmonic (ROS 2 Jazzy). The
setup: a two-lane aisle between shelves, with a fallen box blocking lane A.
R1 (blue) drives lane A and sees the box with its lidar. R2 (orange) comes
down lane B on its reserved schedule and passes the box at `r2_arrival`
seconds.

## Build and run

```bash
cd ROS2_Gazebo_Bridge
colcon build --packages-select visison_gazebo
source install/setup.bash

ros2 launch visison_gazebo bypass_scenario.launch.py r2_arrival:=50.0
```

To reset quickly, run Gazebo and the scenario in separate terminals. Gazebo stays open:

```bash
ros2 launch visison_gazebo gazebo_world.launch.py                                          # terminal 1
ros2 launch visison_gazebo bypass_scenario.launch.py start_gazebo:=false r2_arrival:=50.0  # terminal 2
```

To reset, press Ctrl+C in terminal 2 and run the same command again, with any
`r2_arrival`. It removes the old robots, respawns them and restarts the
scenario at t = 0.

The `visison` library is packaged from the repository root through the
`visison` symlink, so rebuild after you change it. Alternatively, build once
with `colcon build --symlink-install`.

| argument | default | meaning |
|---|---|---|
| `r2_arrival` | `50.0` | seconds after start at which R2 passes the fallen box |
| `r2_speed` | `0.8` | R2's speed in lane B (m/s) |
| `perception` | `lidar` | `lidar` finds the box in `/r1/scan`; `ground_truth` uses the known box |
| `sensing_range` | `8.0` | how far ahead R1 reacts (m) |
| `gui` / `rviz` | `true` | Gazebo GUI and RViz with the Visison markers |
| `use_nvidia` | `true` | render on the NVIDIA GPU (PRIME offload) |

### What to expect

R1 sees the box at about t = 13.6 s. R2's arrival time then decides what R1 does:

| `r2_arrival` | decision | what happens |
|---|---|---|
| ≳ 41 s | `BYPASS_NOW` | R1 borrows lane B from about 19–32 s and is back before R2 needs the space |
| ≲ 40 s | `WAIT_THEN_BYPASS` | R1 stops at s ≈ 16.2 m, lets R2 pass, then bypasses at `t*` |

Headless Gazebo runs (all without collisions):

| `r2_arrival` | decision | R1 in lane B | closest R1–R2 gap |
|---|---|---|---|
| 50 s | BYPASS_NOW | 21.1–30.6 s | 12.2 m |
| 40 s | WAIT_THEN_BYPASS (27.5 s wait) | 50.1–59.6 s | 4.8 m, after R2 has passed |
| 30 s | WAIT_THEN_BYPASS (17.5 s wait) | 40.1–49.6 s | 4.8 m, after R2 has passed |

The switch point comes from the exact interval search, not a fixed threshold.
It moves if you change the speeds, the margins (`scenario.py`) or
`sensing_range`.

When both robots have finished, the node logs a result block with:
- the decision
- when R1 was planned to be in lane B and when it actually was
- the closest R1–R2 gap while R1 was in lane B
- the obstacle reports
- the collisions found by the oriented-box monitor on the real Gazebo poses

## What runs

| piece | where |
|---|---|
| world: aisle, shelves, lane markings, fallen box | `worlds/aisle_world.sdf` |
| robots: the `amr_sim` AMR, namespaced (`/r1/*`, `/r2/*`; lidar only on R1) | `urdf/amr.urdf.xacro` |
| scenario constants, R2 schedule, reservations | `visison_gazebo/scenario.py` |
| lidar → obstacle box in R1's lane | `visison_gazebo/perception.py` |
| Kanayama trajectory tracker → `cmd_vel` | `visison_gazebo/tracking.py` |
| fleet table + R1 `BypassController` + R2 driver + safety monitor | `visison_gazebo/bypass_scenario_node.py` |

Topics:
- `/r1/cmd_vel`, `/r1/odom`, `/r1/scan`, `/r2/cmd_vel`, `/r2/odom`
- `/visison/status` (decision and result text)
- `/visison/obstacle_reports` (JSON)
- `/visison/markers` (lanes, lane-B reservations: R2 orange, R1's claim blue, R1's planned path)

`world` is the aisle frame: x runs along the aisle and y across it, with
lane A at y ∈ [0, 1.2] and lane B at y ∈ [1.2, 2.4].

## Tests without ROS

```bash
python3 -m pytest src/visison_gazebo/test
```

These check:
- R2's schedule meets `r2_arrival`
- 50 s gives BYPASS_NOW and 30 s gives WAIT_THEN_BYPASS
- the tracker follows the bypass to within 8 cm with the diff-drive acceleration limits
- the lidar detector finds the box's front face
