"""Open-RMF integration for the Visison Gazebo node (Plan B).

Roles, as in docs/plans/PLAN_B_OPEN_RMF.md::

    RMF task dispatcher -> RMF route planning + traffic schedule + negotiation (all fleets)
          | navigate(goal)                         ^ override_schedule(bypass / hold), states
          v                                        |
    RmfFleetBridge (EasyFullControl) -> Visison controller (mission) -> MotionArbiter -> robot

* RMF never commands a robot directly. A ``navigate`` goal becomes the robot's
  *mission*: its controller follows it as a PLANNER intent, and it is
  re-issued after any bypass or emergency standoff. ``stop`` is a FLEET hold.
* Whenever Visison changes what the robot will really do (a bypass through
  the other lane, a rejoin, an emergency hold), the bridge tells RMF with
  ``override_schedule``, so every other fleet's planner sees it.
* :class:`RmfTrafficWatcher` listens to the RMF traffic schedule topics and
  turns other fleets' itineraries into read-only Visison reservations
  (:class:`RmfReservations`), so bypass windows respect robots we cannot command.

A second, plain :class:`RmfFleetBridge` (``visison=False``) stands in for another
vendor's fleet: it only follows RMF, with a simple onboard safety stop.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import rmf_adapter as adpt
import rmf_adapter.easy_full_control as efc
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rmf_traffic_msgs.msg import ItineraryClear, ItineraryDelay, ItineraryExtend, ItinerarySet, Participants

from visison import ControllerState, InMemoryReservationTable, Reservation, Source

from .core import arrived, goal_trajectory, itinerary_reservations, override_path

MAP = "L1"
_BUSY = (ControllerState.DETECTED, ControllerState.EVALUATING, ControllerState.APPROACH, ControllerState.WAITING,
         ControllerState.CHANGE_OUT, ControllerState.PASSING, ControllerState.CHANGE_IN,
         ControllerState.EMERGENCY_STOP, ControllerState.REJOINING, ControllerState.HOLDING)

_rclcpp_ready = False


def init_rmf() -> None:
    """rmf_adapter runs its own rclcpp context next to rclpy; start it once per process."""
    global _rclcpp_ready
    if not _rclcpp_ready:
        adpt.init_rclcpp()
        _rclcpp_ready = True


@dataclass
class _Mission:
    goal_x: float
    execution: object
    traj: object = None
    plan_id: int | None = None  # id() of the Visison plan last sent to RMF
    held: bool = False


@dataclass
class RmfFleetBridge:
    node: Node
    fleet_name: str
    config_file: str
    nav_graph: str
    robots: dict  # rmf robot name -> GazeboRobot
    aisle: object = None
    spec: object = None
    all_robots: dict = field(default_factory=dict)  # every robot in the scene (for the onboard safety check)
    controllers: dict = field(default_factory=dict)  # rmf robot name -> BypassController (Visison fleet only)
    coordinator: object = None
    visison: bool = True
    safety_gap: float = 0.9  # m, onboard stop distance of a plain (non-Visison) robot

    def __post_init__(self):
        init_rmf()
        self.cfg = efc.FleetConfiguration.from_config_files(self.config_file, self.nav_graph, None)
        if self.cfg is None:
            raise RuntimeError(f"cannot load RMF fleet config {self.config_file}")
        self.adapter = adpt.Adapter.make(f"{self.fleet_name}_fleet_adapter")
        self.fleet = self.adapter.add_easy_fleet(self.cfg)
        self.adapter.start()
        self._lock = threading.Lock()
        self._inbox: list[tuple] = []
        self.missions: dict[str, _Mission] = {}
        self.handles = {}
        self.log: list[str] = []
        self.completed: set[str] = set()  # robots that reached their task's final place
        self.targets: dict[str, float] = {}  # robot -> x of its task's final place (set by the host)
        self.overrides = 0  # schedule overrides sent to RMF
        self.v_max: float | None = None  # plain fleets: really drive this fast (RMF plans with the config)
        for name, rob in self.robots.items():
            x, y, th, _ = rob.pose
            cb = efc.RobotCallbacks(
                lambda dest, exe, n=name: self._post(("navigate", n, dest, exe)),
                lambda activity, n=name: self._post(("stop", n, activity)),
                lambda cat, desc, exe, n=name: exe.finished(),
            )
            self.handles[name] = self.fleet.add_robot(name, efc.RobotState(MAP, [x, y, th], 1.0),
                                                      self.cfg.get_known_robot_configuration(name), cb)

    def _post(self, item) -> None:  # called from the rclcpp thread
        with self._lock:
            self._inbox.append(item)

    def _say(self, msg: str) -> None:
        self.log.append(msg)
        self.node.get_logger().info(f"[RMF {self.fleet_name}] {msg}")

    # ------------------------------------------------------------------ tick

    def step(self, t: float) -> None:
        with self._lock:
            inbox, self._inbox = self._inbox, []
        for kind, name, *rest in inbox:
            if kind == "navigate":
                dest, exe = rest
                self.robots[name].release(Source.FLEET)  # a new goal ends any earlier RMF stop
                self.missions[name] = _Mission(float(dest.xy[0]), exe)
                self._say(f"{name}: navigate to {dest.name or tuple(round(v, 2) for v in dest.xy)}")
            else:
                self.missions.pop(name, None)
                self.robots[name].stop(Source.FLEET)
                self._say(f"{name}: stop requested by RMF")
        for name, rob in self.robots.items():
            if rob.pose is None:
                continue
            st = rob.state()
            m = self.missions.get(name)
            if m is not None:
                self._run_mission(name, rob, st, m, t)
            x, y, th = self._reported_pose(name, rob, st)
            self.handles[name].update(efc.RobotState(MAP, [x, y, th], 1.0),
                                      m.execution.identifier if m is not None else None)

    def _in_manoeuvre(self, name, rob) -> bool:
        ctl = self.controllers.get(name)
        if not self.visison or ctl is None:
            return False
        in_standoff = self.coordinator is not None and self.coordinator.standoff_of(rob.robot_id) is not None
        return in_standoff or ctl.state in _BUSY

    def _reported_pose(self, name, rob, st) -> tuple[float, float, float]:
        """While Visison runs a bypass / emergency manoeuvre, RMF sees the robot on its own lane.

        RMF only knows the lanes of its graph: a robot reported half-way into
        the other lane would be snapped onto that lane and re-routed. The real
        sideways path reaches RMF through ``override_schedule`` instead.
        """
        x, y, th, _ = rob.pose
        if self._in_manoeuvre(name, rob):
            return x, self.aisle.lane_center(st.lane), 0.0 if st.direction > 0 else 3.141592653589793
        return x, y, th

    def _run_mission(self, name, rob, st, m: _Mission, t: float) -> None:
        if arrived(st, m.goal_x):
            m.execution.finished()
            self.missions.pop(name, None)
            final = name in self.targets and abs(self.targets[name] - m.goal_x) < 0.3
            if final:
                self.completed.add(name)
            self._say(f"{name}: arrived at x={st.s:.2f}" + (" - task complete" if final else " (leg done)"))
            return
        ctl = self.controllers.get(name)
        if self.visison and ctl is not None:
            in_standoff = self.coordinator is not None and self.coordinator.standoff_of(rob.robot_id) is not None
            if in_standoff:
                if not m.held:  # tell RMF this robot is held where it is
                    x, y, th, _ = rob.pose
                    m.execution.override_schedule(MAP, [[x, y, th]], 10.0)
                    m.held = True
                    self.overrides += 1
                    self._say(f"{name}: emergency standoff, holding (schedule overridden)")
                return
            m.held = False
            if ctl.state in _BUSY:
                plan = ctl.plan
                if plan is not None and id(plan) != m.plan_id and plan.kind in ("bypass", "rejoin", "return"):
                    m.plan_id = id(plan)
                    t_end = max(plan.t_back, t) + 2.0
                    m.execution.override_schedule(MAP, override_path(plan.trajectory, t, t_end), 0.0)
                    self.overrides += 1
                    self._say(f"{name}: {plan.kind} through lane {plan.lane_to} published to the RMF schedule "
                              f"(until t={t_end:.1f} s)")
                return
        elif not self.visison and self._safety_stop(name, rob, st, t):
            return
        # idle, cruising or just rejoined: (re)issue the mission from where the robot really is
        if m.traj is None or rob.trajectory is not m.traj:
            traj = goal_trajectory(self.aisle, st, self.spec, m.goal_x, t, self.v_max)
            if traj is None:
                m.execution.finished()  # goal behind: RMF's next leg takes over (turnaround lane)
                self.missions.pop(name, None)
                return
            m.traj = traj
            rob.follow(traj)

    def _safety_stop(self, name, rob, st, t) -> bool:
        """A plain vendor robot's own safety: stop if anything is close ahead in its lane band."""
        others = [o for o in self.all_robots.values() if o is not rob and o.pose is not None]
        y = st.y
        ahead = [st.direction * (o.pose[0] - st.s) - 0.5 for o in others
                 if abs(o.pose[1] - y) < 0.55 and st.direction * (o.pose[0] - st.s) > 0]
        gap = min(ahead, default=float("inf"))
        m = self.missions[name]
        if gap < self.safety_gap:
            if not m.held:
                rob.stop(Source.FLEET)
                m.held, m.traj = True, None
                self._say(f"{name}: onboard safety stop ({gap:.2f} m ahead)")
            return True
        if m.held and gap < self.safety_gap + 0.6:
            return True  # wait until clearly free
        m.held = False
        return False

    def shutdown(self) -> None:
        self.adapter.stop()


class RmfTrafficWatcher:
    """Other fleets' itineraries from the RMF traffic schedule, as Visison reservations."""

    def __init__(self, node: Node, aisle, own_fleet: str, length: float, space: float, time_offset):
        self.node, self.aisle, self.own = node, aisle, own_fleet
        self.length, self.space = length, space
        self.time_offset = time_offset  # callable: RMF time (ns) -> scenario time (s)
        self.names: dict[int, tuple[str, str]] = {}
        self.routes: dict[int, list] = {}
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=100, reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(Participants, "/rmf_traffic/participants", self._on_participants, qos)
        node.create_subscription(ItinerarySet, "/rmf_traffic/itinerary_set", self._on_set, 100)
        node.create_subscription(ItineraryExtend, "/rmf_traffic/itinerary_extend", self._on_extend, 100)
        node.create_subscription(ItineraryDelay, "/rmf_traffic/itinerary_delay", self._on_delay, 100)
        node.create_subscription(ItineraryClear, "/rmf_traffic/itinerary_clear", self._on_clear, 100)

    def _on_participants(self, msg):
        self.names = {p.id: (p.description.owner, p.description.name) for p in msg.participants}

    def _pts(self, routes):
        out = []
        for r in routes:
            out.append([(self.time_offset(w.time), w.position[0], w.position[1]) for w in r.trajectory.waypoints])
        return out

    def _on_set(self, msg):
        self.routes[msg.participant] = self._pts(msg.itinerary)

    def _on_extend(self, msg):
        self.routes.setdefault(msg.participant, []).extend(self._pts(msg.routes))

    def _on_delay(self, msg):
        dt = msg.delay * 1e-9
        self.routes[msg.participant] = [[(t + dt, x, y) for t, x, y in r] for r in self.routes.get(msg.participant, [])]

    def _on_clear(self, msg):
        self.routes.pop(msg.participant, None)

    def reservations(self) -> list[Reservation]:
        out = []
        for pid, routes in self.routes.items():
            owner, name = self.names.get(pid, ("?", f"p{pid}"))
            if owner == self.own:
                continue  # our own robots are booked precisely by Visison itself
            for r in routes:
                out += itinerary_reservations(self.aisle, name.upper(), r, self.length, self.space)
        return out


class RmfReservations(InMemoryReservationTable):
    """Visison's table for our fleet, plus read-only bookings of other fleets from RMF."""

    def __init__(self, watcher: RmfTrafficWatcher | None = None):
        super().__init__()
        self.watcher = watcher

    def _foreign(self, lane, cells, t_from):
        if self.watcher is None:
            return []
        cells = set(cells)
        return [r for r in self.watcher.reservations() if r.lane == lane and r.cell in cells and r.end > t_from]

    def query(self, lane, cells, t_from, exclude_robot=None):
        cells = list(cells)
        own = super().query(lane, cells, t_from, exclude_robot)
        return own + [r for r in self._foreign(lane, cells, t_from) if r.robot_id != exclude_robot]

    def try_reserve(self, claims):
        for c in claims:
            if any(c.conflicts_with(r) for r in self._foreign(c.lane, [c.cell], c.start - 1e-6)):
                return False
        return super().try_reserve(claims)
