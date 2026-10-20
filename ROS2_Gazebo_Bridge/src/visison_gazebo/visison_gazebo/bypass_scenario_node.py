"""Runs the Visison scenario in Gazebo with any number of robots.

A fallen box blocks lane A. R1 (plus ``lane_a_followers`` robots behind it)
drives lane A, and a convoy R2, R3, ... (``lane_b_robots``) comes down lane B.
One node plays every role:

* **Fleet manager**: owns the shared space-time reservation table and the
  ``EmergencyCoordinator``.
* **Every robot**: its own lidar (``/<ns>/scan``) feeds obstacle detection in its
  lane (for its ``BypassController``) and an ``ObjectTracker`` whose tracks
  feed the collision prediction. A Kanayama controller tracks the planned
  trajectory on ``/<ns>/cmd_vel``.
* **Lane-B robots** drive their schedule so that R2 passes the box at
  ``t = r2_arrival``. With ``r2_early > 0`` they really run that much ahead of
  what they reserved. This is the timing mistake that makes the emergency layer
  brake, back the convoy off, and let R1 rejoin its lane.

A safety monitor checks the real Gazebo poses of all robots for overlaps.
Everything is also published as RViz markers on ``/visison/markers`` and as
text on ``/visison/status``.

Scenario time ``t`` starts (``t = 0``) when odometry from every robot has arrived.
"""

from __future__ import annotations

import bisect
import json
import math
import signal
import time
from collections import deque

import rclpy
from geometry_msgs.msg import Point, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from visison import (
    Action,
    BypassController,
    Command,
    Kind,
    MotionArbiter,
    MotionIntent,
    Source,
    BypassPlanner,
    ControllerState,
    EmergencyConfig,
    EmergencyCoordinator,
    FleetMember,
    InMemoryReservationTable,
    ObjectTracker,
    Obstacle,
    OrientedBox,
    Profile,
    RerouteEstimate,
    RobotState,
    Trajectory,
    boxes_overlap,
)

from . import scenario as sc
from .perception import detect_in_lane, detect_objects, drop_near, drop_walls, lane_points, scan_to_world
from .tracking import UnicycleTracker, wrap

BODY_LENGTH, BODY_WIDTH = 0.50, 0.49  # real amr_sim body incl. wheels, for the safety monitor
LIDAR_OFFSET = (0.1, 0.0, 0.225)  # lidar_link in base_footprint, from urdf/amr.urdf.xacro
LIDAR_EDGE_MARGIN = 0.1  # ignore returns within 10 cm of the lane edges (shelf faces)
MAX_DETECT_HEADING = math.radians(10)  # obstacle detection pauses while a robot turns more than this
# compute_latency covers the whole ROS pipeline here: odometry -> speed estimate from the
# pose history -> 20 Hz control tick -> cmd_vel bridge -> diff-drive plugin
EMERGENCY = EmergencyConfig(sigma_pos=0.08, sigma_vel=0.15, min_gap=0.2, compute_latency=0.15)


def transform(stamp, parent: str, child: str, x: float, y: float, z: float, yaw: float) -> TransformStamped:
    tf = TransformStamped()
    tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, parent, child
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = x, y, z
    tf.transform.rotation.z, tf.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return tf


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class GazeboRobot:
    """visison ``RobotInterface`` backed by Gazebo diff-drive odometry, cmd_vel and a lidar.

    The diff-drive odometry frame starts at the spawn pose, so the world
    (= aisle) pose is the spawn pose composed with the odometry pose.
    """

    def __init__(self, node: Node, plan: sc.RobotPlan, tf: TransformBroadcaster, dt: float = 0.05,
                 orca: bool = False):
        self.node, self.plan, self.tf = node, plan, tf
        self.robot_id, self.ns, self.lane, self.direction = plan.robot_id, plan.ns, plan.lane, plan.direction
        self.spawn = plan.spawn
        self.pose: tuple[float, float, float, float] | None = None  # x, y, yaw, v (signed)
        self.history: deque[tuple[float, tuple]] = deque(maxlen=250)  # (stamp, pose), ~5 s at 50 Hz
        self.parked = False
        self.w = 0.0  # measured yaw rate
        self.odom_stamp = 0.0
        # the only thing that commands this robot's motors (Plan A, feature F)
        self.arbiter = MotionArbiter(
            plan.robot_id, sc.MODEL, dt,
            tracker=UnicycleTracker(v_max=sc.SPEC.limits.v_max * 1.25, v_reverse=EMERGENCY.reverse_speed + 0.1),
            orca_enabled=orca)
        self.objects = ObjectTracker(prefix=f"{plan.robot_id}.T")
        self.scan: tuple[LaserScan, tuple] | None = None
        self.last_scan_stamp = None  # last scan used for tracking
        self.candidate: tuple[Obstacle, int] | None = None  # obstacle detection waiting for confirmation
        self.handled: set[str] = set()  # obstacle ids already given to the controller
        self.cmd = node.create_publisher(Twist, f"/{self.ns}/cmd_vel", 10)
        node.create_subscription(Odometry, f"/{self.ns}/odom", self._on_odom, 10)
        node.create_subscription(LaserScan, f"/{self.ns}/scan", self._on_scan, qos_profile_sensor_data)

    def _on_odom(self, msg: Odometry) -> None:
        x0, y0, yaw0 = self.spawn
        p = msg.pose.pose.position
        c, s = math.cos(yaw0), math.sin(yaw0)
        x, y, th = x0 + c * p.x - s * p.y, y0 + s * p.x + c * p.y, wrap(yaw0 + yaw_of(msg.pose.pose.orientation))
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # signed speed along the heading from ~0.1 s of pose history: it does not
        # depend on how the plugin fills the twist, and it sees a stop as it happens
        v = msg.twist.twist.linear.x
        old = next((h for h in reversed(self.history) if stamp - h[0] >= 0.1), None)
        if old is not None and stamp - old[0] < 0.5:
            ox, oy = old[1][0], old[1][1]
            v = ((x - ox) * math.cos(th) + (y - oy) * math.sin(th)) / (stamp - old[0])
        self.pose = (x, y, th, v)
        self.w = msg.twist.twist.angular.z
        self.odom_stamp = stamp
        self.history.append((stamp, self.pose))
        # world -> <ns>/base_footprint, so RViz can place the robot's sensor data
        x, y, yaw, _ = self.pose
        self.tf.sendTransform(transform(msg.header.stamp, "world", f"{self.ns}/base_footprint", x, y, 0.0, yaw))

    def _on_scan(self, msg: LaserScan) -> None:
        if self.pose is not None:
            # the pose when the scan was taken, not when it arrived: while turning,
            # 1 degree of heading error moves a shelf 12 cm at 7 m range
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.scan = (msg, self.pose_at(stamp))

    def pose_at(self, stamp: float):
        """Odometry pose closest to ``stamp`` (e.g. when a lidar scan was taken)."""
        if not self.history:
            return self.pose
        stamps = [h[0] for h in self.history]
        i = bisect.bisect_left(stamps, stamp)
        if i == 0:
            return self.history[0][1]
        if i == len(stamps):
            return self.history[-1][1]
        before, after = self.history[i - 1], self.history[i]
        return before[1] if stamp - before[0] <= after[0] - stamp else after[1]

    def heading_error(self, th: float) -> float:
        return abs(wrap(th - (0.0 if self.direction > 0 else math.pi)))

    # --- RobotInterface -----------------------------------------------------

    def state(self) -> RobotState:
        x, y, _, v = self.pose
        return RobotState(self.robot_id, x, abs(v), self.lane, self.direction, y, v < -0.02)

    def follow(self, trajectory: Trajectory, source: Source = Source.PLANNER, kind: Kind = Kind.FOLLOW) -> None:
        t = self.node.scenario_time()
        self.arbiter.request(MotionIntent(source, kind, trajectory, stamp=t), t)

    def _stop_trajectory(self, decel: float) -> Trajectory:
        t = self.node.scenario_time()
        x, y, _, v = self.pose
        lim = sc.SPEC.limits
        travel = self.direction if v >= 0 else -self.direction
        prof = Profile(t, travel * x, abs(v)).change_speed(0.0, lim.accel, decel)
        return Trajectory(prof, travel, y, [], sc.SPEC.lane_change, reverse=travel != self.direction)

    def stop(self, source: Source = Source.PLANNER) -> None:
        t = self.node.scenario_time()
        self.arbiter.request(MotionIntent(source, Kind.HOLD, self._stop_trajectory(sc.SPEC.limits.decel), stamp=t), t)

    def emergency_brake(self, source: Source = Source.EMERGENCY) -> None:
        """Maximum braking (the plugin's 1 m/s^2) through the arbiter; nothing else can override it."""
        t = self.node.scenario_time()
        self.arbiter.request(MotionIntent(source, Kind.BRAKE, None, stamp=t), t)

    def release(self, source: Source, handover: bool = False) -> None:
        self.arbiter.release(source, self.node.scenario_time(), handover)

    @property
    def trajectory(self) -> Trajectory | None:
        return self.arbiter.trajectory(self.node.scenario_time())

    @property
    def estop(self) -> bool:
        a = self.arbiter.active(self.node.scenario_time())
        return a is not None and a.kind is Kind.BRAKE

    @property
    def lateness(self) -> float:
        return self.arbiter.lateness

    @property
    def last_orca(self):
        return self.arbiter.last.orca if self.arbiter.last else None

    @property
    def intent_seq(self) -> int:
        a = self.arbiter.active(self.node.scenario_time())
        return a.seq if a else 0

    def reference_time(self, t: float) -> float:
        return self.arbiter.reference_time(t)

    # --- low level ----------------------------------------------------------

    def drive(self, t: float) -> None:
        """One control tick: the arbiter's command is the only one this robot receives."""
        cmd = Twist()
        if not self.parked:
            x, y, th, v = self.pose
            age = max(self.node.get_clock().now().nanoseconds * 1e-9 - self.odom_stamp, 0.0)
            out = self.arbiter.step(t, (x, y, th), Command(v, self.w), odom_age=age)
            cmd.linear.x, cmd.angular.z = float(out.v), float(out.w)
        self.cmd.publish(cmd)

    def park(self) -> None:
        self.parked = True
        self.cmd.publish(Twist())

    def body(self) -> OrientedBox:
        x, y, th, _ = self.pose
        return OrientedBox(x, y, BODY_LENGTH, BODY_WIDTH, th)

    def done(self) -> bool:
        x = self.pose[0]
        return self.direction * (x - self.plan.end_s) >= 0


class RosFleetReporter:
    def __init__(self, node: Node):
        self.node = node
        self.reports = []
        self.emergencies = []
        self.pub = node.create_publisher(String, "/visison/obstacle_reports", 10)
        self.emergency_pub = node.create_publisher(String, "/visison/emergency", 10)

    def report_obstacle(self, report) -> None:
        self.reports.append(report)
        self.node.get_logger().warning(
            f"OBSTACLE REPORT -> fleet: {report.obstacle_id} blocks lane(s) "
            f"{','.join(report.lanes_blocked)} at s=[{report.s_min:.2f}, {report.s_max:.2f}] m, "
            f"{report.reporter_id} action: {report.action}. Please remove it.")
        self.pub.publish(String(data=json.dumps(report.to_dict())))

    def report_schedule_conflicts(self, robot_id, conflicts) -> None:
        self.node.get_logger().warning(
            f"{robot_id}'s delay now overlaps {len(conflicts)} reservation(s) of "
            f"{sorted({c.robot_id for c in conflicts})}")

    def report_emergency(self, report) -> None:
        self.emergencies.append(report)
        log = self.node.get_logger()
        text = f"[t={report.t:6.2f}s] EMERGENCY #{report.standoff_id} {report.event.upper()}: {report.message}"
        if report.event in ("brake", "escalate"):  # rclpy: one severity per call site
            log.error(text)
        else:
            log.warning(text)
        self.emergency_pub.publish(String(data=json.dumps(report.to_dict())))
        self.node.status_pub.publish(String(data=text))


class HostRerouter:
    """Stand-in for the fleet's global planner (e.g. PIBT). Only its cost is used here."""

    def __init__(self, node: Node, robot: GazeboRobot, extra_time: float):
        self.node, self.robot, self.extra_time = node, robot, extra_time

    def estimate(self, state, obstacle):
        return RerouteEstimate(self.extra_time, 0.8 * self.extra_time)

    def execute(self, state, obstacle):
        self.node.get_logger().warning(
            f"REROUTE: handing {self.robot.robot_id} to the global planner (not simulated); stopping")
        self.robot.stop()


class BypassScenario(Node):
    def __init__(self, name: str = "visison_bypass_scenario", overrides: dict | None = None):
        """``overrides`` sets parameters directly (the demo director runs one node per case)."""
        from rclpy.parameter import Parameter
        super().__init__(name, parameter_overrides=[Parameter(k, value=v) for k, v in (overrides or {}).items()])
        self.result_lines: list[str] = []
        self.result_pass: bool | None = None
        self.declare_parameter("r2_arrival", 50.0)
        self.declare_parameter("r2_speed", 0.8)
        self.declare_parameter("r2_early", 0.0)
        self.declare_parameter("lane_b_robots", 1)
        self.declare_parameter("lane_a_followers", 0)
        self.declare_parameter("perception", "lidar")  # or "ground_truth"
        self.declare_parameter("sensing_range", 8.0)
        self.declare_parameter("assumed_depth", 1.0)
        self.declare_parameter("reroute_extra_time", 45.0)
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("orca", False)  # Plan A: ORCA-DD in every robot's arbiter
        self.declare_parameter("coop", False)  # Plan A: reciprocal bypass windows
        self.declare_parameter("rmf", False)  # Plan B: tasks and traffic through Open-RMF
        self.declare_parameter("rmf_share", "")  # visison_rmf share directory (configs, nav graph)
        self.declare_parameter("other_speed", 0.0)  # >0: the other fleet really drives this fast (RMF plans 0.8)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.rmf = bool(p("rmf"))
        self.rmf_share = str(p("rmf_share"))
        self.other_speed = float(p("other_speed"))

        self.layout = sc.fleet_layout(int(p("lane_b_robots")), int(p("lane_a_followers")),
                                      float(p("r2_arrival")), float(p("r2_speed")), float(p("r2_early")))
        self.r2_early = float(p("r2_early"))
        self.perception = str(p("perception"))
        self.orca, self.coop = bool(p("orca")), bool(p("coop"))
        dt = 1.0 / float(p("control_rate"))
        self.sensing_range = float(p("sensing_range"))
        self.assumed_depth = float(p("assumed_depth"))

        self.tf = TransformBroadcaster(self)
        self.status_pub = self.create_publisher(String, "/visison/status", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/visison/markers", 10)
        self.lidar_pub = self.create_publisher(MarkerArray, "/visison/lidar", 10)
        self.tracks_pub = self.create_publisher(MarkerArray, "/visison/tracks", 10)
        self.watcher = None
        if self.rmf:
            from visison_rmf.bridge import RmfReservations, RmfTrafficWatcher
            self.watcher = RmfTrafficWatcher(self, sc.AISLE, "visison_fleet", sc.SPEC.footprint.length,
                                             sc.CONFIG.margins.space, lambda ns: ns * 1e-9 - (self.t0 or 0.0))
            self.table = RmfReservations(self.watcher)
        else:
            self.table = InMemoryReservationTable()
        self.reporter = RosFleetReporter(self)
        self.coord = EmergencyCoordinator(BypassPlanner(sc.AISLE, sc.SPEC, sc.CONFIG), self.table,
                                          self.reporter, EMERGENCY)
        self.robots: dict[str, GazeboRobot] = {}
        self.ctls: dict[str, BypassController] = {}
        for plan in self.layout:
            rob = GazeboRobot(self, plan, self.tf, dt, self.orca)
            rob.arbiter.neighbours = self.coord.orca_inputs
            rob.arbiter.orca_cfg = self.coord.orca_cfg
            rerouter = HostRerouter(self, rob, float(p("reroute_extra_time"))) if plan.robot_id == "R1" else None
            ctl = BypassController(rob, BypassPlanner(sc.AISLE, sc.SPEC, sc.COOP_CONFIG if self.coop else sc.CONFIG),
                                   self.table, self.reporter, rerouter)
            if self.coop:
                ctl.shiftable = lambda other: other in self.robots  # our own fleet only
                ctl.coop_executor = self.coord.apply_delay
            self.robots[plan.robot_id], self.ctls[plan.robot_id] = rob, ctl
            other = self.rmf and plan.lane == "B"  # in RMF mode lane B belongs to another vendor's fleet
            self.coord.register(plan.robot_id, FleetMember(rob, sc.SPEC, None if other else ctl,
                                                           cruise_speed=plan.cruise_speed, model=sc.MODEL,
                                                           orca=self.orca and not other, cooperative=not other))
        self.other_fleet = {rid for rid, r in self.robots.items() if self.rmf and r.lane == "B"}
        self.bridges = []
        self.dispatch: list[tuple[float, str, str, str]] = []  # (t, fleet, robot, place)
        self.expected_tasks: set[str] = set()
        self.r1, self.ctl = self.robots["R1"], self.ctls["R1"]
        self.r2 = self.robots.get("R2")

        self.t0: float | None = None
        self.detected: list[Obstacle] = []  # fleet-wide list of confirmed obstacles
        self.first_decision = None  # what R1 decided for the fallen box
        self.n_events: dict[str, int] = {rid: 0 for rid in self.robots}
        self.collisions: list[str] = []
        self.r1_lane_b: list[float] = []  # [t_enter, t_leave]
        self.min_gap_in_b = math.inf
        self.min_gap = math.inf  # between any two robots sharing a lateral band
        self.min_gap_at: tuple[str, str, float] | None = None
        self.r2_passed_box: float | None = None
        self.done = False

        self.step_wall_max = self.step_wall_sum = self.odom_lag_max = 0.0  # loop timing (wall s / sim s)
        self.step_count = 0
        self.create_timer(1.0 / float(p("control_rate")), self._step)
        self.create_timer(0.25, self._publish_markers)
        lane_b = [pl for pl in self.layout if pl.lane == "B"]
        self.get_logger().info(
            f"Fallen box in lane A at s=[{sc.FALLEN_BOX.s_min}, {sc.FALLEN_BOX.s_max}] m. "
            f"{len(self.layout)} robots: {', '.join(pl.robot_id + '@' + pl.lane for pl in self.layout)}. "
            + (f"R2 leaves s={lane_b[0].spawn_s:.1f} m at t={lane_b[0].schedule.t_go:.1f} s and passes the "
               f"box at t={lane_b[0].schedule.arrival:.1f} s" if lane_b else "No robot in lane B")
            + (f" ({self.r2_early:.1f} s EARLIER than reserved)" if self.r2_early else "")
            + f". Perception: {self.perception}.")

    # --- time -----------------------------------------------------------------

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def scenario_time(self) -> float:
        return self._now() - (self.t0 or 0.0)

    # --- main loop ------------------------------------------------------------

    def _step(self) -> None:
        self._phase_times = []
        w0 = time.perf_counter()
        self._step_inner()
        w = time.perf_counter() - w0
        if w > 0.1 and self.t0 is not None:
            ph = getattr(self, "_phase_times", [])
            parts = ", ".join(f"{ph[k][0]} {1e3 * (ph[k][1] - ph[k - 1][1]):.0f}" for k in range(1, len(ph)))
            self.get_logger().warning(f"[t={self.scenario_time():6.2f}s] slow control step: {1e3 * w:.0f} ms (wall)"
                                      f" [{parts}]")
        self.step_wall_max = max(self.step_wall_max, w)
        self.step_wall_sum += w
        self.step_count += 1
        if self.t0 is not None:
            now = self.get_clock().now().nanoseconds * 1e-9
            lag = max((now - r.odom_stamp for r in self.robots.values() if r.odom_stamp), default=0.0)
            self.odom_lag_max = max(self.odom_lag_max, lag)

    def _step_inner(self) -> None:
        missing = [r.ns for r in self.robots.values() if r.pose is None]
        if missing:
            self.get_logger().info(f"waiting for odometry of {', '.join(missing)} ...", throttle_duration_sec=5.0)
            return
        if self.t0 is None:
            self._start()
        t = self.scenario_time()

        ph = self._phase_times = [("start", time.perf_counter())]
        self._perceive(t)
        ph.append(("perceive", time.perf_counter()))
        tracks = self._track()
        ph.append(("track", time.perf_counter()))
        self.coord.tick(t, tracks or None)
        ph.append(("coordinator", time.perf_counter()))
        for rid, ctl in self.ctls.items():
            if rid not in self.other_fleet:
                ctl.tick(t)
        ph.append(("controllers", time.perf_counter()))
        if self.rmf:
            self._dispatch_tasks(t)
            for b in self.bridges:
                b.step(t)
        self._log_events()
        ph.append(("rmf+log", time.perf_counter()))

        for rob in self.robots.values():
            if not self.rmf and not rob.parked and rob.done():
                rob.park()
            rob.drive(t)
        ph.append(("drive", time.perf_counter()))
        self._monitor(t)
        ph.append(("monitor", time.perf_counter()))

        if self.rmf:
            done = set().union(*(b.completed for b in self.bridges)) if self.bridges else set()
            finished = not self.dispatch and self.expected_tasks <= done
        else:
            finished = all(rob.parked or self.ctls[rid].state in (ControllerState.REROUTING, ControllerState.HOLDING)
                           for rid, rob in self.robots.items())
        if not self.done and finished and not self.coord.standoffs:
            self.done = True
            self._summary()

    def _start(self) -> None:
        self.t0 = self._now()
        if self.rmf:
            self._start_rmf()
            return
        for plan in self.layout:
            self.table.add(*plan.reservations())
            self.robots[plan.robot_id].follow(plan.trajectory())
        self.get_logger().info(f"t=0: {len(self.robots)} robots up, reservations loaded, driving")

    def _start_rmf(self) -> None:
        """Plan B: two RMF fleets; every motion goal comes from an RMF task."""
        import json
        import os
        import uuid

        from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
        from rmf_task_msgs.msg import ApiRequest
        from visison_rmf.bridge import RmfFleetBridge

        share = self.rmf_share
        nav = os.path.join(share, "maps", "aisle_nav.yaml")
        ours = {rid.lower(): r for rid, r in self.robots.items() if rid not in self.other_fleet}
        theirs = {rid.lower(): r for rid, r in self.robots.items() if rid in self.other_fleet}
        self.bridges = [RmfFleetBridge(self, "visison_fleet", os.path.join(share, "config", "visison_fleet.yaml"), nav,
                                       ours, sc.AISLE, sc.SPEC, self.robots,
                                       {rid.lower(): self.ctls[rid] for rid in self.robots if rid not in self.other_fleet},
                                       self.coord, visison=True)]
        if theirs:
            other = RmfFleetBridge(self, "other_fleet", os.path.join(share, "config", "other_fleet.yaml"), nav, theirs,
                                   sc.AISLE, sc.SPEC, self.robots, visison=False)
            other.v_max = self.other_speed or None
            self.bridges.append(other)
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10, reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._task_pub = self.create_publisher(ApiRequest, "/task_api_requests", qos)
        self._task_json, self._task_id = json, (lambda: "visison_" + uuid.uuid4().hex[:8])
        self.dispatch.append((2.0, "visison_fleet", "r1", "a_end"))
        for pl in self.layout:
            if pl.robot_id in self.other_fleet:
                # dispatched early enough that its RMF itinerary exists when R1 has to decide
                self.dispatch.append((max(pl.schedule.t_go, 2.5), "other_fleet", pl.ns, "b_end"))
        self.dispatch.sort()
        self.expected_tasks = {name for _, _, name, _ in self.dispatch}
        places = {"a_end": 39.0, "b_end": 1.0}  # maps/aisle_nav.yaml
        for b in self.bridges:
            b.targets = {name: places[place] for _, _, name, place in self.dispatch if name in b.robots}
        self.get_logger().info("t=0: RMF mode. Tasks: " + ", ".join(
            f"{n}->{place} at t={t:.1f} s ({f})" for t, f, n, place in self.dispatch))

    def _dispatch_tasks(self, t: float) -> None:
        from rmf_task_msgs.msg import ApiRequest
        while self.dispatch and self.dispatch[0][0] <= t:
            _, fleet, name, place = self.dispatch.pop(0)
            req = {"type": "robot_task_request", "robot": name, "fleet": fleet,
                   "request": {"unix_millis_earliest_start_time": 0, "category": "compose",
                               "description": {"category": "go_to_place", "phases": [{"activity": {
                                   "category": "sequence", "description": {"activities": [
                                       {"category": "go_to_place", "description": place}]}}}]}}}
            self._task_pub.publish(ApiRequest(json_msg=self._task_json.dumps(req), request_id=self._task_id()))
            self.get_logger().warning(f"[t={t:6.2f}s] RMF TASK -> {fleet}/{name}: go_to_place {place}")

    def _others(self, rid: str) -> list[tuple[float, float, float, float]]:
        return [(r.pose[0], r.pose[1], BODY_LENGTH / 2, BODY_WIDTH / 2)
                for o, r in self.robots.items() if o != rid]

    def _scan_points(self, rob: GazeboRobot):
        msg, (sx, sy, sth, _) = rob.scan
        return msg, (sx, sy, sth), scan_to_world(msg.ranges, msg.angle_min, msg.angle_increment,
                                                 msg.range_min, msg.range_max, sx, sy, sth)

    def _perceive(self, t: float) -> None:
        """Obstacle detection in each robot's own lane, for its BypassController."""
        for rid, rob in self.robots.items():
            if rob.parked or rid in self.other_fleet:
                continue
            x = rob.pose[0]
            lane = sc.AISLE.lane_bounds(rob.lane)
            if self.perception == "ground_truth":
                box = sc.FALLEN_BOX
                ahead = rob.direction * ((box.s_min if rob.direction > 0 else box.s_max) - x)
                ob = box if box.overlaps_lateral(*lane) and 0.0 < ahead <= self.sensing_range else None
            else:
                if rob.scan is None:
                    continue
                msg, (sx, sy, sth), pts = self._scan_points(rob)
                if rid == "R1" and msg.header.stamp != getattr(self, "_r1_lidar_stamp", None):
                    self._r1_lidar_stamp = msg.header.stamp
                    self._publish_lidar(t, msg, (sx, sy, sth), pts)
                if rob.heading_error(sth) > MAX_DETECT_HEADING:
                    # while turning, the body tilts and the 2-D lidar catches the floor
                    # and shelf edges; the controller would only queue a detection now anyway
                    rob.candidate = None
                    continue
                pts = drop_walls(drop_near(pts, self._others(rid)))  # robots and shelf faces are not obstacles
                ob = detect_in_lane(pts, sx, rob.direction, lane, self.sensing_range, self.assumed_depth,
                                    edge_margin=LIDAR_EDGE_MARGIN, obstacle_id=f"obstacle-{len(self.detected) + 1}")
            if ob is None:
                rob.candidate = None
                continue
            known = next((o for o in self.detected if ob.overlaps_longitudinal(o.s_min - 1.0, o.s_max + 1.0)
                          and ob.overlaps_lateral(o.y_min - 0.3, o.y_max + 0.3)), None)
            if known is not None and known.obstacle_id in rob.handled:
                rob.candidate = None
                continue
            if self.perception != "ground_truth" and known is None:
                # confirm in 3 consecutive scans at a consistent position before acting
                prev = rob.candidate
                n = prev[1] + 1 if prev and abs(prev[0].s_min - ob.s_min) < 0.3 else 1
                rob.candidate = (ob, n)
                if n < 3:
                    continue
                rob.candidate = None
            if known is None:
                self.detected.append(ob)
                self.coord.known_obstacles.append(ob)
                known = ob
                self.get_logger().warning(
                    f"[t={t:6.2f}s] {rid} at s={x:.2f} m sees an obstacle in lane {rob.lane}: "
                    f"s=[{ob.s_min:.2f}, {ob.s_max:.2f}] y=[{ob.y_min:.2f}, {ob.y_max:.2f}]")
            rob.handled.add(known.obstacle_id)
            ctl = self.ctls[rid]
            running = ctl.decision if ctl.active else None
            d = ctl.on_obstacle(known, t)
            if d is None or (running is not None and d is running):
                continue  # queued behind the running manoeuvre; the controller logs it
            if rid == "R1" and self.first_decision is None:
                self.first_decision = d
            self._log_decision(t, rid, d)
            if self.watcher is not None:
                foreign = self.watcher.reservations()
                who = sorted({r.robot_id for r in foreign})
                self.get_logger().warning(
                    f"    decided with {len(foreign)} bookings of other fleets from the RMF schedule"
                    + (f" ({', '.join(who)})" if who else ""))

    def _track(self) -> dict[str, list]:
        """Lidar clusters -> tracked objects with velocities, once per new scan and robot."""
        out = {}
        if self.perception == "ground_truth":
            return out
        for rid, rob in self.robots.items():
            if rob.scan is None or rob.scan[0].header.stamp == rob.last_scan_stamp:
                continue
            rob.last_scan_stamp = rob.scan[0].header.stamp
            msg, (sx, sy, sth), pts = self._scan_points(rob)
            pts = drop_walls(drop_near(pts, [(sx, sy, BODY_LENGTH / 2, BODY_WIDTH / 2)], pad=0.05))
            meas = detect_objects(pts, (0.0, 2 * sc.AISLE.lane_width), LIDAR_EDGE_MARGIN)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - self.t0
            out[rid] = rob.objects.update(stamp, meas)
        if out:
            self._publish_tracks(out)
        return out

    def _publish_tracks(self, tracks: dict) -> None:
        """What each robot's lidar tracker sees: a dot per tracked object and its velocity arrow."""
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for rid, trs in tracks.items():
            col = [float(v) for v in self.robots[rid].plan.color.split()]
            m = Marker()
            m.header.frame_id, m.header.stamp = "world", stamp
            m.ns, m.id, m.type, m.action = f"tracks_{rid}", 0, Marker.SPHERE_LIST, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.14
            m.color.r, m.color.g, m.color.b, m.color.a = col[0], col[1], col[2], 0.9
            m.points = [Point(x=float(tr.s), y=float(tr.y), z=0.6) for tr in trs]
            if not m.points:
                m.action = Marker.DELETE
            ma.markers.append(m)
            arrows = Marker()
            arrows.header = m.header
            arrows.ns, arrows.id, arrows.type = f"track_velocity_{rid}", 0, Marker.LINE_LIST
            arrows.action = Marker.ADD if trs else Marker.DELETE
            arrows.pose.orientation.w = 1.0
            arrows.scale.x = 0.04
            arrows.color = m.color
            for tr in trs:
                arrows.points += [Point(x=float(tr.s), y=float(tr.y), z=0.6),
                                  Point(x=float(tr.s + tr.vs), y=float(tr.y + tr.vy), z=0.6)]
            ma.markers.append(arrows)
        self.tracks_pub.publish(ma)

    def _log_decision(self, t: float, rid: str, d) -> None:
        log = self.get_logger()
        log.warning(f"[t={t:6.2f}s] {rid} DECISION: {d.action.value.upper()} - {d.reason}")
        plan = d.plan
        if plan is not None and plan.kind == "bypass":
            log.info(
                f"    plan: wait {plan.wait_time:.1f} s, lane change out at t={plan.t_out:.1f} s "
                f"(s={plan.s_out:.2f}), back in lane {plan.lane_from} at t={plan.t_back:.1f} s "
                f"(s={plan.s_back:.2f}), {len(plan.claims)} lane-{plan.lane_to} cells claimed, "
                f"delay {plan.delay:.1f} s")
            log.info(f"    cost: bypass {d.cost_bypass:.1f} s vs reroute {d.cost_reroute:.1f} s")
        self.status_pub.publish(String(data=f"t={t:.1f} {rid} {d.action.value}: {d.reason}"))

    def _log_events(self) -> None:
        for rid, ctl in self.ctls.items():
            for e in ctl.events[self.n_events[rid]:]:
                if e.state not in (ControllerState.DETECTED, ControllerState.EVALUATING) or e.message:
                    self.get_logger().info(f"[t={e.t:6.2f}s] {rid} {e.state.value}" + (f": {e.message}" if e.message else ""))
                if e.message.startswith("abort") or e.state is ControllerState.HOLDING:
                    self.status_pub.publish(String(data=f"t={e.t:.1f} {rid} {e.state.value}: {e.message}"))
            self.n_events[rid] = len(ctl.events)

    def _monitor(self, t: float) -> None:
        box = OrientedBox.of_obstacle(sc.FALLEN_BOX)
        bodies = {rid: rob.body() for rid, rob in self.robots.items()}
        ids = sorted(bodies)
        hits = [(boxes_overlap(bodies[a], box), f"{a} hit the fallen box") for a in ids]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                hits.append((boxes_overlap(bodies[a], bodies[b]), f"{a} hit {b}"))
                ya, yb = [y for _, y in bodies[a].corners()], [y for _, y in bodies[b].corners()]
                if min(ya) < max(yb) and min(yb) < max(ya):
                    xa, xb = [x for x, _ in bodies[a].corners()], [x for x, _ in bodies[b].corners()]
                    g = max(min(xb) - max(xa), min(xa) - max(xb))
                    if g < self.min_gap:
                        self.min_gap, self.min_gap_at = g, (a, b, t)
        for hit, what in hits:
            if hit and what not in self.collisions:
                self.collisions.append(what)
                self.get_logger().error(f"[t={t:6.2f}s] COLLISION: {what}")

        b1 = bodies["R1"]
        in_b = max(y for _, y in b1.corners()) > sc.AISLE.lane_width
        near = min((abs(r.pose[0] - self.r1.pose[0]) - BODY_LENGTH for rid, r in self.robots.items()
                    if r.lane == "B"), default=math.inf)
        if in_b and not self.r1_lane_b:
            self.r1_lane_b = [t, t]
            self.get_logger().info(f"[t={t:6.2f}s] R1 body enters lane B (nearest lane-B robot {near:.2f} m away)")
        elif in_b:
            self.r1_lane_b[1] = t
        if in_b:
            self.min_gap_in_b = min(self.min_gap_in_b, near)
        if self.r2 is not None and self.r2_passed_box is None and self.r2.pose[0] <= sc.BOX_CENTER_S:
            self.r2_passed_box = t
            self.get_logger().info(f"[t={t:6.2f}s] R2 passes the fallen box "
                                   f"(reserved for {self._r2_reserved():.1f} s)")

    def _r2_reserved(self) -> float:
        plan = next((pl for pl in self.layout if pl.robot_id == "R2"), None)
        return plan.reserved.arrival if plan is not None else math.nan

    def shutdown(self) -> None:
        """Stop the RMF adapters of this case (the demo director reuses the process)."""
        for b in self.bridges:
            b.shutdown()

    def _summary(self) -> None:
        d = self.first_decision
        plan = d.plan if d else None
        standoffs = self.coord.history + self.coord.standoffs
        lines = [
            "========== Visison scenario result ==========",
            f"robots             : {len(self.robots)} ({', '.join(sorted(self.robots, key=lambda r: int(r[1:])))})",
        ]
        if self.r2 is not None:
            lines.append(
                f"R2 arrival         : reserved {self._r2_reserved():.1f} s"
                + (f", driven {self.r2_early:.1f} s early" if self.r2_early else "")
                + " (R2 actually passed the box at "
                + (f"{self.r2_passed_box:.1f} s)" if self.r2_passed_box is not None else "- )"))
        lines.append(f"R1 decision        : {d.action.value.upper() if d else '-'} - {d.reason if d else ''}")
        if plan is not None and plan.kind == "bypass":
            lines.append(f"planned in lane B  : t={plan.t_out:.1f} .. {plan.t_back:.1f} s (wait {plan.wait_time:.1f} s)")
        if self.r1_lane_b:
            lines.append(f"actual in lane B   : t={self.r1_lane_b[0]:.1f} .. {self.r1_lane_b[1]:.1f} s")
            lines.append(f"closest gap to a lane-B robot while R1 was in lane B: {self.min_gap_in_b:.2f} m")
        lines.append(f"emergency stops    : {len(standoffs)}")
        for so in standoffs:
            moved = ", ".join(f"{k} {v:.2f} m" for k, v in so.moved.items()) or "nobody"
            lines.append(f"  #{so.sid}: {so.resolution.value if so.resolution else '-'}, robots "
                         f"{','.join(sorted(so.all_members))}, backed off: {moved}, {so.phase.value}")
        lines += [
            f"closest robot gap  : {self.min_gap:.2f} m (robots sharing a lane band"
            + (f": {self.min_gap_at[0]}-{self.min_gap_at[1]} at t={self.min_gap_at[2]:.1f} s)" if self.min_gap_at else ")"),
            f"obstacle reports   : {len(self.reporter.reports)}",
            f"control loop       : {1e3 * self.step_wall_sum / max(1, self.step_count):.1f} ms mean, "
            f"{1e3 * self.step_wall_max:.1f} ms max (wall); odometry lag max {self.odom_lag_max:.2f} s (sim)",
            f"collisions         : {', '.join(self.collisions) if self.collisions else 'none'}",
            "RESULT             : " + ("PASS" if not self.collisions and d and d.action is not Action.HOLD
                                       and all(so.phase.value == "done" for so in standoffs) else "CHECK"),
            "=============================================",
        ]
        self.result_lines = lines
        for line in lines:
            self.get_logger().info(line)
        self.status_pub.publish(String(data="\n".join(lines)))
        # set last: the demo director moves on as soon as result_pass is known
        self.result_pass = not self.collisions and bool(d) and d.action is not Action.HOLD and all(
            so.phase.value == "done" for so in standoffs)

    # --- visualisation ----------------------------------------------------------

    def _publish_lidar(self, t: float, msg: LaserScan, sensor_pose, pts) -> None:
        """Show what R1's detector sees: the watched stretch of lane A and rays to hits in it."""
        x, y, th = sensor_pose
        stamp = msg.header.stamp
        lx, ly = x + LIDAR_OFFSET[0] * math.cos(th), y + LIDAR_OFFSET[0] * math.sin(th)
        z = LIDAR_OFFSET[2]
        lo, hi = sc.AISLE.lane_bounds("A")
        hits = lane_points(drop_near(pts, self._others("R1")), x, +1, (lo, hi), self.sensing_range, LIDAR_EDGE_MARGIN)
        confirmed = any(o.s_min - 1.0 <= x + self.sensing_range and x <= o.s_max for o in self.detected)
        turning = abs(wrap(th)) > MAX_DETECT_HEADING

        ma = MarkerArray()

        def mk(ns, mtype, rgba, sx=0.0, sy=0.0, sz=0.0):
            m = Marker()
            m.header.frame_id, m.header.stamp = "world", stamp
            m.ns, m.id, m.type, m.action = ns, 0, mtype, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = float(sx), float(sy), float(sz)
            m.color.r, m.color.g, m.color.b, m.color.a = (float(c) for c in rgba)
            ma.markers.append(m)
            return m

        # the stretch of R1's lane the lidar is watching (green = clear, red = obstacle in it)
        zone_rgba = ((0.5, 0.5, 0.5, 0.12) if turning else
                     (0.9, 0.15, 0.1, 0.18) if hits else (0.1, 0.8, 0.2, 0.15))
        watch = mk("lidar_watch_zone", Marker.CUBE, zone_rgba,
                   self.sensing_range, hi - lo - 2 * LIDAR_EDGE_MARGIN, 0.01)
        watch.pose.position.x, watch.pose.position.y, watch.pose.position.z = x + self.sensing_range / 2, (lo + hi) / 2, 0.005

        ray_rgb = (0.55, 0.55, 0.55) if turning else (1.0, 0.85, 0.0)
        rays = mk("lidar_rays", Marker.LINE_LIST, (*ray_rgb, 0.9), 0.012)
        dots = mk("lidar_hits_in_lane", Marker.SPHERE_LIST, (*ray_rgb, 1.0), 0.08, 0.08, 0.08)
        for _, px, py in hits[:: max(1, len(hits) // 60)]:
            rays.points += [Point(x=lx, y=ly, z=z), Point(x=px, y=py, z=z)]
            dots.points.append(Point(x=px, y=py, z=z))
        if not hits:  # a marker needs at least one point to be valid
            rays.action = dots.action = Marker.DELETE

        label = mk("lidar_text", Marker.TEXT_VIEW_FACING, (0.6, 0.05, 0.0, 1.0) if hits else (0.05, 0.4, 0.05, 1.0),
                   sz=0.28)
        label.pose.position.x, label.pose.position.y, label.pose.position.z = x, y, 1.2
        if turning:
            label.text = "LIDAR: detection paused while turning"
            label.color.r, label.color.g, label.color.b = 0.35, 0.35, 0.35
        elif hits:
            label.text = (f"LIDAR: {len(hits)} hits in lane A, nearest {hits[0][0]:.1f} m"
                          + ("  -> obstacle confirmed" if confirmed else "  -> confirming..."))
        else:
            label.text = f"LIDAR: lane A clear for {self.sensing_range:.0f} m"
        self.lidar_pub.publish(ma)

    def _publish_markers(self) -> None:
        if self.t0 is None or any(r.pose is None for r in self.robots.values()):
            return
        t = self.scenario_time()
        stamp = self.get_clock().now().to_msg()
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        def mk(ns, mid, mtype, x, y, z, sx, sy, sz, rgba, yaw=0.0):
            m = Marker()
            m.header.frame_id, m.header.stamp = "world", stamp
            m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
            # ROS message fields reject ints, so everything goes through float()
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), float(z)
            m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            m.scale.x, m.scale.y, m.scale.z = float(sx), float(sy), float(sz)
            m.color.r, m.color.g, m.color.b, m.color.a = (float(c) for c in rgba)
            ma.markers.append(m)
            return m

        w, L = sc.AISLE.lane_width, sc.AISLE.length
        mk("lanes", 0, Marker.CUBE, L / 2, sc.LANE_A_Y, -0.01, L, w, 0.005, (0.75, 0.82, 0.95, 0.5))
        mk("lanes", 1, Marker.CUBE, L / 2, sc.LANE_B_Y, -0.01, L, w, 0.005, (0.92, 0.92, 0.92, 0.5))
        box = sc.FALLEN_BOX
        mk("fallen_box", 0, Marker.CUBE, (box.s_min + box.s_max) / 2, (box.y_min + box.y_max) / 2, 0.25,
           box.s_max - box.s_min, box.y_max - box.y_min, 0.5, (0.65, 0.45, 0.25, 1.0))
        for i, ob in enumerate(self.detected):
            mk("detected", i, Marker.CUBE, (ob.s_min + ob.s_max) / 2, (ob.y_min + ob.y_max) / 2, 0.3,
               ob.s_max - ob.s_min, ob.y_max - ob.y_min, 0.6, (0.9, 0.1, 0.1, 0.35))

        # lane-B reservations that are active now or start within the next 20 s
        horizon = 20.0
        for i, r in enumerate(self.table.query("B", range(sc.AISLE.n_cells), t)):
            if r.start > t + horizon:
                continue
            lo, hi = sc.AISLE.cell_bounds(r.cell)
            soon = 1.0 - max(r.start - t, 0.0) / horizon
            rgb = (0.1, 0.35, 0.9) if self.robots.get(r.robot_id) and self.robots[r.robot_id].lane == "A" \
                else (0.95, 0.5, 0.1)
            mk("reservations_B", i, Marker.CUBE, (lo + hi) / 2, sc.LANE_B_Y, 0.01,
               hi - lo - 0.04, w * 0.8, 0.02, (*rgb, 0.15 + 0.6 * soon))

        for k, (rid, ctl) in enumerate(self.ctls.items()):
            plan = ctl.plan
            if plan is not None and ctl.active and ctl.state is not ControllerState.EMERGENCY_STOP:
                m = mk("plans", k, Marker.LINE_STRIP, 0, 0, 0, 0.05, 0, 0, (0.1, 0.35, 0.9, 1.0))
                tt = max(t, plan.t_arrive)
                while tt <= plan.t_back + 2.0:
                    pose = plan.trajectory.sample(tt)
                    m.points.append(Point(x=pose.s, y=pose.y, z=0.05))
                    tt += 0.2

        # emergency standoffs: a red zone over the stopped group, and where robots back off to
        for so in self.coord.standoffs:
            xs = [self.robots[r].pose[0] for r in so.all_members | so.members if r in self.robots]
            if xs:
                lo, hi = min(xs) - 0.5, max(xs) + 0.5
                mk("standoff", so.sid, Marker.CUBE, (lo + hi) / 2, w, 0.02, hi - lo, 2 * w, 0.03, (0.95, 0.1, 0.1, 0.25))
                txt = mk("standoff_text", so.sid, Marker.TEXT_VIEW_FACING, (lo + hi) / 2, w, 1.8, 0, 0, 0.35,
                         (0.8, 0.0, 0.0, 1))
                txt.text = f"EMERGENCY #{so.sid}: {so.phase.value}" + (f" ({so.resolution.value})" if so.resolution else "")
            if so.reverse is not None and so.phase.value == "clearing":
                for j, (rid, s_t) in enumerate(so.reverse.targets.items()):
                    mk("backoff_targets", so.sid * 100 + j, Marker.CYLINDER, s_t, sc.LANE_B_Y, 0.02,
                       0.25, 0.25, 0.04, (0.95, 0.5, 0.0, 0.9))

        # the first two robots are drawn by RViz's RobotModel displays (URDF); the rest as boxes
        for i, (rid, rob) in enumerate(self.robots.items()):
            x, y, th, _ = rob.pose
            if i >= 2:
                c = [float(v) for v in rob.plan.color.split()]
                mk("robot_bodies", i, Marker.CUBE, x, y, 0.15, BODY_LENGTH, BODY_WIDTH, 0.3, c, th)
            state = self.ctls[rid].state
            txt = mk("robots", i, Marker.TEXT_VIEW_FACING, x, y, 0.75, 0, 0, 0.3,
                     (0.8, 0, 0, 1) if state is ControllerState.EMERGENCY_STOP else (0, 0, 0, 1))
            txt.text = rid + ("" if state in (ControllerState.CRUISING, ControllerState.REJOINED) else f" ({state.value})")
        for i, y_shelf in enumerate((-0.4, 2 * w + 0.4)):  # see-through shelves, as in the world
            mk("shelves", i, Marker.CUBE, L / 2, y_shelf, 1.0, L, 0.6, 2.0, (0.2, 0.35, 0.6, 0.18))

        d = self.ctl.decision
        info = mk("status", 0, Marker.TEXT_VIEW_FACING, L / 2, w, 3.0, 0, 0, 0.5, (0.05, 0.05, 0.05, 1))
        info.text = (f"t = {t:5.1f} s   R1: {self.ctl.state.value}"
                     + (f"   decision: {d.action.value}" if d else "")
                     + (f"   R2 reserved at box t = {self._r2_reserved():.0f} s" if self.r2 is not None else "")
                     + (f" (runs {self.r2_early:.0f} s early)" if self.r2_early else ""))
        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = BypassScenario()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl-C must not interrupt cleanup
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
