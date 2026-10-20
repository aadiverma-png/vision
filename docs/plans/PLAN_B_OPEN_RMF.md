# Plan B: Open-RMF integration (to implement after Plan A)

## Context
Open-RMF is the ROS 2 standard for traffic between fleets. It keeps a shared **traffic schedule** (every robot's planned itinerary over time), detects conflicts between itineraries, runs **negotiation** between fleets, and assigns **tasks** to robots.

Visison and RMF overlap in a useful way:
- RMF plans *routes* on a navigation graph and keeps fleets out of each other's way *as planned*.
- It has no answer for a lane that becomes **blocked by an obstacle**, for a robot that is **off its planned timing**, or for **emergency recovery**.

Visison adds exactly those, *inside* an aisle. Combined:

```
RMF task dispatcher -> RMF route planning + traffic schedule + negotiation     (whole warehouse, all fleets)
      |                                     ^
      v  navigate(goal)                     |  override_schedule(bypass path) / delays / states
Visison fleet adapter (EasyFullControl)
      -> Visison controller (bypass) + ORCA (Plan A) + emergency / standoff    (inside the aisle)
      -> robots
```

On Jazzy, RMF is packaged as `ros-jazzy-rmf-dev` and includes:
- `rmf_fleet_adapter_python`, which provides the **EasyFullControl** API;
- `rmf_traffic_editor` and `rmf_building_map_tools` for the nav graph;
- `rmf_demos_*` for task dispatch and visualisation.

## Design: new package `ROS2_Gazebo_Bridge/src/visison_rmf` (ament_python)

1. **Map / nav graph.** Draw an `aisle.building.yaml` (traffic-editor format) for the existing aisle world:
   - waypoints at both aisle ends plus parking spots;
   - directed lanes: lane A goes +x at y = 0.6, lane B goes −x at y = 1.8.

   `building_map_generator nav` produces `nav_graphs/0.yaml`. The Gazebo world stays our `aisle_world.sdf`, with the same coordinates.
2. **Visison fleet adapter** (`visison_rmf/fleet_adapter.py`), an EasyFullControl adapter. Fleet `visison_fleet` has robots r1..rN, and its config YAML holds the footprint and vicinity radii and the limits from `scenario.SPEC`. The callbacks work like this:
   - `navigate(destination, execution)`: turn the RMF goal into a lane-following `Trajectory` along the aisle in the robot's lane (from `r1_nominal_profile`-style profiles). It goes to the robot's **controller as its mission**, and the controller issues it as a PLANNER intent. **RMF never commands the robot directly.** The existing obstacle perception, `BypassController`, ORCA-DD and emergency layers keep running underneath, all through the robot's `MotionArbiter`.
   - `stop(activity)` / an RMF pause becomes a `FLEET` HOLD intent. It ranks above PLANNER and below EMERGENCY.
   - When a **bypass is committed**, `execution.override_schedule(map, points, hold)` publishes the real path *including the excursion into lane B* to RMF. The other fleets' planners see it and negotiate around it.
   - During an **emergency standoff**, override with a *hold* at the current pose until it is resolved. Delays are reported, and RMF re-plans.
   - On arrival, `execution.finished()`.
   - Robot states (position, battery) go to RMF at 2 Hz.
3. **RMF-aware reservations** (`visison_rmf/rmf_reservations.py`): `RmfReservations` implements `ReservationProvider`.
   - Our own fleet stays in the `InMemoryReservationTable`. It is atomic and fast, and the controller and coordinator need it.
   - **Other fleets' robots** come from RMF's `/fleet_states` (`rmf_fleet_msgs/FleetState`, which has each robot's location and planned path). Path points inside the aisle become read-only Visison reservations with times, refreshed at 2 Hz, and `query` returns both.
   - That lets the Visison planner choose bypass windows that respect *other vendors'* robots.
   - The exact time fields that `/fleet_states` paths carry on Jazzy get checked after install. If the paths carry no times, we fall back to position + velocity projection, as the lidar tracks do.
4. **Other fleets in the emergency layer.** Robots of other fleets are registered as **non-cooperative** members: they have a `RobotInterface` whose `state()` comes from `/fleet_states`, and their braking/follow calls are not available.
   - ORCA gives them responsibility 1.
   - `plan_chain_reverse` never asks them to move. If they would have to, the standoff escalates through RMF: our robot holds via override_schedule, a `report_emergency` goes out, and optionally RMF's emergency-alarm topic is used.
   - That limitation gets documented honestly, since cross-vendor motion commands are not possible through RMF.
5. **Demo: a mixed fleet.**
   - Fleet `visison_fleet` has R1 in lane A (Visison + ORCA + emergency).
   - Fleet `other_fleet` has R2..Rk in lane B. It is a plain EasyFullControl adapter with simple path following and no Visison, standing in for another vendor.
   - Tasks are dispatched with `rmf_demos_tasks` (`dispatch_go_to_place`): R1 is sent to the far end via lane A, and R2 the other way via lane B.
   - The box blocks lane A. R1's bypass window is chosen against R2's RMF itinerary, and the bypass appears in the RMF schedule.
   - Launch: `rmf_scenario.launch.py` starts the RMF core (traffic schedule, task dispatcher, visualisation), both fleet adapters, Gazebo, the robots and the Visison node in "RMF mode". In RMF mode, goals come from RMF instead of the scripted schedules.
6. **Visualisation:** RMF's `rmf_visualization` (schedule and fleet-state markers) next to the Visison markers in RViz.

## Files
- **New:**
  - `ROS2_Gazebo_Bridge/src/visison_rmf/`: `package.xml`, `setup.py`, `visison_rmf/{fleet_adapter.py, other_fleet_adapter.py, rmf_reservations.py}`, `config/{visison_fleet.yaml, other_fleet.yaml}`, `maps/aisle.building.yaml`, `launch/rmf_scenario.launch.py`, `test/test_rmf_offline.py`;
  - `run_rmf_scenario.sh`.
- **Changed:**
  - `visison_gazebo/bypass_scenario_node.py`: factor out the per-robot runtime (`GazeboRobot` + controller + perception) so the RMF adapter can reuse it, and add an RMF mode where goals come from `navigate()`;
  - the README (an RMF section).

## Tests and verification
1. **Offline**, with no RMF install:
   - the `/fleet_states` path → reservations conversion (aisle projection, times, and the fallback without times);
   - an RMF goal → lane trajectory conversion;
   - the bypass → `override_schedule` point list (lane-B excursion included);
   - non-cooperative handling in `plan_chain_reverse` (escalates rather than commanding another fleet).
2. After `sudo apt install ros-jazzy-rmf-dev`, confirm the EasyFullControl Python API signatures on this machine and adjust the adapter to them.
3. Gazebo + RMF, headless: `./run_rmf_scenario.sh` dispatches two go-to-place tasks.
   - Expected: R1 bypasses in a window that respects R2's RMF itinerary, and the bypass shows up in the RMF schedule.
   - Both tasks complete, with no collisions.
   - With `other_early`, where the other fleet deviates from its itinerary, the emergency layer holds R1 and the standoff resolves or escalates *without* commanding the other fleet.
4. All earlier tests, sim scenarios and Gazebo runs still pass. The RMF layer is optional, so the non-RMF launch stays unchanged.
