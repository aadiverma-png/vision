"""Two-lane aisle simulator for the Visison controller, with any number of robots.

Run ``python -m sim.aisle_sim --scenario far`` (or near, both_blocked,
squeeze, miscalc_rejoin, miscalc_yield, convoy_reverse, fleet_10,
orca_small_error, coop_window, all); ``--orca`` / ``--coop`` switch the Plan A
features on for every scenario, and ``--compare`` prints baseline vs ORCA vs
ORCA + reciprocal windows side by side. Every
``dt`` the simulator checks the oriented footprints of all robots and
obstacles for overlap, so a scenario only passes if it is collision-free.
``--plot out.png`` saves a top view and a space-time diagram (needs matplotlib).

Every robot runs its own ``BypassController`` and a simulated 2-D lidar
(10 Hz, 12 m, 1 cm noise) whose clusters feed an ``ObjectTracker``. One
``EmergencyCoordinator`` covers the aisle. A robot's ``early`` setting makes it
drive ahead of (or, if negative, behind) its reservations. That is the
"miscalculated timing" the emergency layer has to catch.

Robots track their reference trajectories perfectly, which isolates the
planning logic from tracking-controller error.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, field

from visison import (
    Action,
    Command,
    DiffDrive,
    Kind,
    MotionArbiter,
    MotionIntent,
    Pose,
    Source,
    UnicycleTracker,
    Aisle,
    BypassController,
    BypassPlanner,
    ControllerState,
    EmergencyConfig,
    EmergencyCoordinator,
    FleetMember,
    Footprint,
    InMemoryReservationTable,
    LoggingReporter,
    MotionLimits,
    ObjectTracker,
    Obstacle,
    OrientedBox,
    PlannerConfig,
    Profile,
    QuinticLaneChange,
    RerouteEstimate,
    Resolution,
    RobotSpec,
    RobotState,
    Trajectory,
    boxes_overlap,
    reservations_for_path,
)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0


class SimRobot:
    """Implements :class:`visison.adapters.RobotInterface`; a :class:`MotionArbiter` owns its motion.

    Every ``follow``/``stop``/``emergency_brake`` is an intent for the arbiter.
    Without ORCA the robot replays the active intent's trajectory exactly
    (perfect tracking, which isolates the planning logic). With ``orca=True``
    it is a real unicycle: each ``sim_step`` the arbiter tracks, filters with
    ORCA-DD and gates, and the robot integrates the resulting (v, w).
    """

    def __init__(self, robot_id: str, spec: RobotSpec, aisle: Aisle, lane: str,
                 s0: float, v0: float, direction: int, clock: Clock, orca: bool = False,
                 model=None, dt: float = 0.05):
        self.robot_id, self.spec, self.aisle, self.lane = robot_id, spec, aisle, lane
        self.direction, self.clock, self.dt = direction, clock, dt
        self.model = model or sim_model(spec)
        self.arbiter = MotionArbiter(robot_id, self.model, dt,
                                     tracker=UnicycleTracker(v_max=spec.limits.v_max * 1.1, v_reverse=0.3),
                                     orca_enabled=orca)
        traj0 = Trajectory(Profile(0.0, direction * s0, v0), direction, aisle.lane_center(lane), [],
                           spec.lane_change)
        self.arbiter.request(MotionIntent(Source.PLANNER, Kind.FOLLOW, traj0, stamp=clock.t), clock.t)
        self._seq, self._prev = None, traj0
        self.orca = orca
        p = traj0.sample(clock.t)
        self._x, self._y, self._th, self._v, self._w = p.s, p.y, p.heading, p.v, 0.0

    # --- playback helpers ------------------------------------------------------

    @property
    def traj(self) -> Trajectory:
        """The trajectory being executed now (the arbiter's active intent)."""
        t = self.clock.t
        a = self.arbiter.active(t)
        if a is None:
            return self._prev
        if a.seq != self._seq:
            if a.kind in (Kind.HOLD, Kind.BRAKE) and self._seq is not None:
                # a stop that waited behind a higher intent starts from where the robot is now
                a.trajectory = self._stop_traj(self._prev,
                                               self.spec.limits.e_decel if a.kind is Kind.BRAKE else self.spec.limits.decel)
            self._seq = a.seq
        self._prev = a.trajectory
        return a.trajectory

    def pose(self, t: float | None = None):
        t = self.clock.t if t is None else t
        if self.orca:
            return Pose(t, self._x, self._y, self._v, self._th)
        return self.traj.sample(t)

    def _stop_traj(self, traj: Trajectory, decel: float) -> Trajectory:
        t = self.clock.t
        p = traj.sample(t) if not self.orca else self.pose(t)
        travel = self.direction if p.v >= 0 else -self.direction
        prof = Profile(t, travel * p.s, abs(p.v)).change_speed(0.0, self.spec.limits.accel, decel)
        return Trajectory(prof, travel, p.y, [], self.spec.lane_change, reverse=travel != self.direction)

    # --- RobotInterface --------------------------------------------------------

    def state(self) -> RobotState:
        p = self.pose()
        return RobotState(self.robot_id, p.s, abs(p.v), self.lane, self.direction, p.y, p.v < 0, p.heading)

    def follow(self, trajectory: Trajectory, source: Source = Source.PLANNER, kind: Kind = Kind.FOLLOW) -> None:
        self.arbiter.request(MotionIntent(source, kind, trajectory, stamp=self.clock.t), self.clock.t)

    def stop(self, source: Source = Source.PLANNER) -> None:
        self.arbiter.request(MotionIntent(source, Kind.HOLD, self._stop_traj(self.traj, self.spec.limits.decel),
                                          stamp=self.clock.t), self.clock.t)

    def emergency_brake(self, source: Source = Source.EMERGENCY) -> None:
        self.arbiter.request(MotionIntent(source, Kind.BRAKE, self._stop_traj(self.traj, self.spec.limits.e_decel),
                                          stamp=self.clock.t), self.clock.t)

    def release(self, source: Source, handover: bool = False) -> None:
        self.traj  # noqa: B018 - settle the active trajectory before the switch
        self.arbiter.release(source, self.clock.t, handover)

    @property
    def lateness(self) -> float:
        return self.arbiter.lateness

    @property
    def last_orca(self):
        return self.arbiter.last.orca if self.arbiter.last else None

    @property
    def intent_seq(self) -> int:
        a = self.arbiter.active(self.clock.t)
        return a.seq if a else 0

    def reference_time(self, t: float) -> float:
        return self.arbiter.reference_time(t)

    # --- simulation ------------------------------------------------------------

    def sim_step(self, t: float) -> None:
        """ORCA mode: one control period of the arbiter, then integrate the unicycle."""
        if not self.orca:
            return
        self.traj  # noqa: B018 - rebuild a pending stop from the current pose
        cmd = self.arbiter.step(t, (self._x, self._y, self._th), Command(self._v, self._w))
        dt = self.dt
        th_mid = self._th + 0.5 * cmd.w * dt
        self._x += cmd.v * math.cos(th_mid) * dt
        self._y += cmd.v * math.sin(th_mid) * dt
        self._th = math.atan2(math.sin(self._th + cmd.w * dt), math.cos(self._th + cmd.w * dt))
        self._v, self._w = cmd.v, cmd.w

    def box(self, t: float) -> OrientedBox:
        p = self.pose(t)
        fp = self.spec.footprint
        return OrientedBox(p.s, p.y, fp.length, fp.width, p.heading)


def sim_model(spec: RobotSpec) -> DiffDrive:
    """Differential drive for the generic sim robot, with headroom above the planner's limits."""
    lim = spec.limits
    return DiffDrive(track_width=0.8 * spec.footprint.width, v_wheel_max=1.4 * lim.v_max, a_acc=1.5 * lim.accel,
                     a_dec=lim.e_decel, alpha_max=3.0, w_max=1.5, v_rev_max=0.3, D=0.2)


class SimLidar:
    """Axis-aligned extents of everything within range, with Gaussian noise, like lidar clusters."""

    def __init__(self, max_range: float = 12.0, noise: float = 0.01, seed: int = 1):
        self.max_range, self.noise = max_range, noise
        self.rng = random.Random(seed)

    def scan(self, me: OrientedBox, others: list[OrientedBox]) -> list[tuple[float, float, float, float]]:
        out = []
        for b in others:
            if math.hypot(b.cx - me.cx, b.cy - me.cy) > self.max_range:
                continue
            xs = [x for x, _ in b.corners()]
            ys = [y for _, y in b.corners()]
            n = lambda: self.rng.gauss(0.0, self.noise)  # noqa: E731
            out.append(((min(xs) + max(xs)) / 2 + n(), (min(ys) + max(ys)) / 2 + n(),
                        max(xs) - min(xs), max(ys) - min(ys)))
        return out


@dataclass
class SimRerouter:
    """Stands in for PIBT/A*: a fixed detour estimate; executing it just stops."""

    robot: SimRobot
    extra_time: float = 45.0
    extra_distance: float = 60.0
    executed: bool = False

    def estimate(self, state, obstacle):
        return RerouteEstimate(self.extra_time, self.extra_distance)

    def execute(self, state, obstacle):
        self.executed = True
        self.robot.stop()  # the host planner now takes the robot out of the aisle (see run())


@dataclass(frozen=True)
class RobotSetup:
    robot_id: str
    lane: str
    s0: float  # where its reservations say it starts
    v0: float
    direction: int
    early: float = 0.0  # s it actually runs ahead of its reservations (negative: late)


@dataclass
class Scenario:
    name: str
    description: str
    obstacles: tuple[Obstacle, ...]
    robots: tuple[RobotSetup, ...]
    expected: Action  # first decision of the first robot
    resolutions: tuple[Resolution, ...] = ()  # standoff resolutions that must happen
    reroute: bool = True
    t_max: float = 90.0
    orca: bool = False  # robots run ORCA-DD in their arbiters (unicycle integration)
    coop: bool = False  # the planner may ask oncoming robots of our fleet to slow a little

    @property
    def obstacle(self) -> Obstacle:
        return self.obstacles[0]


AISLE = Aisle(lane_width=1.2, length=40.0, cell_size=0.5)
SPEC = RobotSpec(
    footprint=Footprint(length=0.9, width=0.6),
    limits=MotionLimits(v_max=1.5, accel=0.8, decel=1.2, emergency_decel=2.0),
    lane_change=QuinticLaneChange(max_lateral_accel=0.5),
    bypass_speed=1.0,
)
R1_START, SENSING_RANGE, R2_SPEED = 2.0, 12.0, 1.0
BOX = Obstacle(20.0, 20.6, 0.1, 1.1, obstacle_id="box-1")


def _r1() -> RobotSetup:
    return RobotSetup("R1", "A", R1_START, SPEC.limits.v_max, +1)


def _b(rid: str, s0: float, early: float = 0.0) -> RobotSetup:
    return RobotSetup(rid, "B", s0, R2_SPEED, -1, early)


SCENARIOS = {
    s.name: s
    for s in (
        Scenario("far", "R2 about 20 s from the bypass zone: bypass now",
                 (BOX,), (_r1(), _b("R2", 50.0)), Action.BYPASS_NOW),
        Scenario("near", "R2 about 5 s from the bypass zone: wait for it, then bypass",
                 (BOX,), (_r1(), _b("R2", 35.0)), Action.WAIT_THEN_BYPASS),
        Scenario("both_blocked", "the fallen load blocks both lanes: reroute",
                 (Obstacle(20.0, 20.8, 0.1, 2.3, obstacle_id="pallet-1"),), (_r1(),), Action.REROUTE),
        Scenario("squeeze", "small box at the lane edge: pass inside lane A",
                 (Obstacle(20.0, 20.4, 0.0, 0.3, obstacle_id="carton-1"),), (_r1(), _b("R2", 48.0)),
                 Action.SQUEEZE),
        Scenario("miscalc_rejoin",
                 "R2 runs 14 s ahead of its reservations and meets R1 mid-bypass: both brake, "
                 "R2 backs off, R1 rejoins lane A",
                 (BOX,), (_r1(), _b("R2", 50.0, early=14.0)), Action.BYPASS_NOW,
                 (Resolution.CHAIN_REVERSE,)),
        Scenario("miscalc_yield",
                 "R2 is 5 s late: R1 starts its bypass as R2 arrives, both brake, R1 gives way",
                 (BOX,), (_r1(), _b("R2", 35.0, early=-5.0)), Action.WAIT_THEN_BYPASS,
                 (Resolution.BYPASSER_YIELDS,)),
        Scenario("convoy_reverse",
                 "a convoy of 3 robots in lane B runs 14 s early: the whole chain backs off for R1",
                 (BOX,), (_r1(), _b("R2", 50.0, 14.0), _b("R3", 52.0, 14.0), _b("R4", 54.0, 14.0)),
                 Action.BYPASS_NOW, (Resolution.CHAIN_REVERSE,), t_max=120.0),
        Scenario("fleet_10",
                 "10 robots: R1 plus 2 followers in lane A, 7 in lane B, several of them early",
                 (BOX,),
                 (_r1(), RobotSetup("R2", "A", -6.0, 1.2, +1), RobotSetup("R3", "A", -14.0, 1.2, +1),
                  _b("R4", 50.0, 14.0), _b("R5", 52.0, 14.0), _b("R6", 58.0, 14.0), _b("R7", 64.0, 10.0),
                  _b("R8", 72.0, 6.0), _b("R9", 80.0), _b("R10", 88.0)),
                 Action.BYPASS_NOW, (Resolution.CHAIN_REVERSE,), t_max=200.0),
        Scenario("orca_small_error",
                 "R2 runs 8 s early: without ORCA that is an emergency brake; with ORCA-DD both robots "
                 "just slow down",
                 (BOX,), (_r1(), _b("R2", 50.0, early=8.0)), Action.BYPASS_NOW, orca=True),
        Scenario("coop_window",
                 "R2 would make R1 wait ~21 s; instead R2 slows by 2 s and R1 bypasses at once "
                 "(reciprocal window)",
                 (BOX,), (_r1(), _b("R2", 45.0)), Action.COOP_BYPASS, orca=True, coop=True),
    )
}


@dataclass
class SimResult:
    scenario: Scenario
    action: Action | None = None
    reason: str = ""
    orca: bool = False
    coop: bool = False
    fleet_delay: float = 0.0  # s, summed over robots that left the aisle
    avoid_events: int = 0  # robot pairs ORCA-DD handled without an emergency brake
    hard_decel_events: int = 0  # braking episodes harder than normal braking
    tick_cpu: float = 0.0  # s, slowest coordinator tick
    coop_events: list = field(default_factory=list)  # (t, robot, delay) of reciprocal windows
    collisions: list[tuple[float, str]] = field(default_factory=list)
    events: list = field(default_factory=list)
    reports: list = field(default_factory=list)
    emergencies: list = field(default_factory=list)  # EmergencyReport, in order
    standoffs: list = field(default_factory=list)
    max_y_r1: float = -math.inf
    r2_s_at_change_out: float | None = None
    min_gap: float = math.inf  # smallest bumper gap between two robots sharing a lateral band
    plan: object = None
    decision: object = None
    finished: dict[str, bool] = field(default_factory=dict)
    trace: list[tuple[float, float, float, float | None]] = field(default_factory=list)
    traces: dict[str, list[tuple[float, float, float]]] = field(default_factory=dict)

    @property
    def resolutions(self) -> list[Resolution]:
        return [so.resolution for so in self.standoffs if so.resolution is not None]

    @property
    def ok(self) -> bool:
        sc = self.scenario
        return (not self.collisions and self.action is sc.expected and bool(self.reports)
                and all(r in self.resolutions for r in sc.resolutions)
                and (bool(sc.resolutions) or not any(e.event == "brake" for e in self.emergencies))
                and Resolution.ESCALATE not in self.resolutions
                and all(self.finished.values()))


def _gap(a: OrientedBox, b: OrientedBox) -> float | None:
    """Bumper gap along the aisle if the two bodies share a lateral band."""
    ay, by = [y for _, y in a.corners()], [y for _, y in b.corners()]
    if not (min(ay) < max(by) and min(by) < max(ay)):
        return None
    ax, bx = [x for x, _ in a.corners()], [x for x, _ in b.corners()]
    return max(min(bx) - max(ax), min(ax) - max(bx))


def run(scenario: Scenario, dt: float = 0.05, t_max: float | None = None, quiet: bool = False,
        orca: bool | None = None, coop: bool | None = None) -> SimResult:
    """Run ``scenario``. ``orca``/``coop`` override the scenario's own settings (None: keep them)."""
    orca = scenario.orca if orca is None else orca
    coop = scenario.coop if coop is None else coop
    t_max = t_max or scenario.t_max
    clock = Clock()
    table = InMemoryReservationTable()
    cfg = PlannerConfig(coop_delays=(1.0, 2.0, 3.0, 4.0, 5.0) if coop else ())
    space = cfg.margins.space
    h = SPEC.footprint.length / 2
    reporter = LoggingReporter(logging.getLogger("visison.sim"))
    coord = EmergencyCoordinator(BypassPlanner(AISLE, SPEC, cfg), table, reporter, EmergencyConfig())

    robots: dict[str, SimRobot] = {}
    nominal_exit: dict[str, float] = {}  # when each robot would leave the aisle cruising undisturbed
    ctls: dict[str, BypassController] = {}
    rerouters: dict[str, SimRerouter] = {}
    trackers: dict[str, ObjectTracker] = {}
    for i, rs in enumerate(scenario.robots):
        d = rs.direction
        s_actual = rs.s0 + d * rs.v0 * rs.early
        rob = SimRobot(rs.robot_id, SPEC, AISLE, rs.lane, s_actual, rs.v0, d, clock, orca=orca, dt=dt)
        rob.arbiter.neighbours = coord.orca_inputs
        rob.arbiter.orca_cfg = coord.orca_cfg  # one ORCA configuration for the whole fleet
        nominal = Profile(0.0, d * rs.s0, rs.v0)
        table.add(*reservations_for_path(rs.robot_id, AISLE, rs.lane, nominal, d, d * rs.s0,
                                         h + (AISLE.length if d > 0 else 0.0), h + space))
        rr = SimRerouter(rob) if scenario.reroute else None
        ctl = BypassController(rob, BypassPlanner(AISLE, SPEC, cfg), table, reporter, rr)
        if coop:
            ctl.shiftable = lambda other: other in robots  # every robot in the sim is our own fleet
            ctl.coop_executor = coord.apply_delay
        robots[rs.robot_id], ctls[rs.robot_id] = rob, ctl
        if rr is not None:
            rerouters[rs.robot_id] = rr
        trackers[rs.robot_id] = ObjectTracker(prefix=f"{rs.robot_id}.T")
        coord.register(rs.robot_id, FleetMember(rob, SPEC, ctl, cruise_speed=rs.v0, model=rob.model, orca=orca))
        nominal_exit[rs.robot_id] = (((AISLE.length + 1.0) - s_actual) if d > 0 else (s_actual + 1.0)) / rs.v0

    lidar = SimLidar()
    first = scenario.robots[0].robot_id
    r1, ctl1 = robots[first], ctls[first]
    ob_boxes = [(ob, OrientedBox.of_obstacle(ob)) for ob in scenario.obstacles]
    handled: dict[str, set[str]] = {rid: set() for rid in robots}
    res = SimResult(scenario, traces={rid: [] for rid in robots})
    first_decision = None
    lidar_every = max(1, round(0.1 / dt))
    v_prev: dict[str, float] = {}
    hard_now: set[str] = set()
    exit_t: dict[str, float] = {}

    def out_of_aisle(rid: str, t: float) -> bool:
        s = robots[rid].pose(t).s
        return s > AISLE.length + 1.0 if robots[rid].direction > 0 else s < -1.0

    gone: set[str] = set()  # handed to the host planner's detour and out of this aisle

    def finished(rid: str, t: float) -> bool:
        return rid in gone or out_of_aisle(rid, t)

    for i in range(int(t_max / dt) + 1):
        t = clock.t = i * dt
        for rid, ctl in ctls.items():
            if rid not in gone and ctl.state is ControllerState.REROUTING and robots[rid].pose(t).v == 0:
                # the detour is not simulated: once stopped, the robot leaves the aisle
                gone.add(rid)
                coord.unregister(rid)
                table.release([r for r in table.all() if r.robot_id == rid])
        active = {rid: rob for rid, rob in robots.items() if rid not in gone}
        # 1. obstacle perception in each robot's own lane
        for rid, rob in active.items():
            p = rob.pose(t)
            lo, hi = AISLE.lane_bounds(rob.lane)
            for ob in scenario.obstacles:
                ahead = (ob.s_min - p.s) if rob.direction > 0 else (p.s - ob.s_max)
                if ob.obstacle_id in handled[rid] or not ob.overlaps_lateral(lo, hi):
                    continue
                if 0.0 < ahead <= SENSING_RANGE:
                    handled[rid].add(ob.obstacle_id)
                    dec = ctls[rid].on_obstacle(ob, t)
                    if rid == first and first_decision is None:
                        first_decision = dec
                        res.action, res.reason = dec.action, dec.reason
        # 2. lidar -> tracks -> collision risk -> emergency coordination
        boxes = {rid: rob.box(t) for rid, rob in active.items()}
        tracks = None
        if i % lidar_every == 0:
            tracks = {}
            for rid in active:
                others = [b for o, b in boxes.items() if o != rid] + [b for _, b in ob_boxes]
                tracks[rid] = trackers[rid].update(t, lidar.scan(boxes[rid], others))
        t_cpu = time.perf_counter()
        coord.tick(t, tracks)
        res.tick_cpu = max(res.tick_cpu, time.perf_counter() - t_cpu)
        # 3. controllers
        for rid in active:
            ctls[rid].tick(t)
        for rid in active:
            robots[rid].sim_step(t)  # ORCA mode: arbiter command, then unicycle integration
        for rid, rob in active.items():
            v = abs(rob.pose(t).v)
            if (v_prev.get(rid, v) - v) / dt > SPEC.limits.decel + 0.1:  # harder than normal braking
                if rid not in hard_now:
                    res.hard_decel_events += 1
                hard_now.add(rid)
            else:
                hard_now.discard(rid)
            v_prev[rid] = v
            if rid not in exit_t and out_of_aisle(rid, t):
                exit_t[rid] = t

        # 4. ground-truth safety check on every pair
        boxes = {rid: rob.box(t) for rid, rob in active.items()}
        ids = list(boxes)
        for a_i, a in enumerate(ids):
            for ob, obox in ob_boxes:
                if boxes_overlap(boxes[a], obox):
                    res.collisions.append((t, f"{a} hit {ob.obstacle_id}"))
            for b in ids[a_i + 1:]:
                if boxes_overlap(boxes[a], boxes[b]):
                    res.collisions.append((t, f"{a} hit {b}"))
                g = _gap(boxes[a], boxes[b])
                if g is not None:
                    res.min_gap = min(res.min_gap, g)
        b1 = r1.box(t)
        res.max_y_r1 = max(res.max_y_r1, max(y for _, y in b1.corners()))
        s2 = boxes["R2"].cx if "R2" in boxes and robots["R2"].lane != r1.lane else None
        plan = first_decision.plan if first_decision is not None else None
        if plan is not None and res.r2_s_at_change_out is None and t >= plan.t_out and s2 is not None:
            res.r2_s_at_change_out = s2
        res.trace.append((t, b1.cx, b1.cy, s2))
        for rid, b in boxes.items():
            res.traces[rid].append((t, b.cx, b.cy))

        if all(finished(rid, t) for rid in robots) and not coord.standoffs:
            break

    res.finished = {rid: finished(rid, clock.t) for rid in robots}
    res.fleet_delay = sum(exit_t[r] - nominal_exit[r] for r in exit_t)
    res.avoid_events = len({tuple(sorted(a.pair)) for a in coord.avoided})
    res.orca, res.coop = orca, coop
    res.coop_events = list(coord.coop_log)
    res.events, res.reports = ctl1.events, reporter.reports
    res.emergencies = coord.reports
    res.standoffs = coord.history + coord.standoffs
    res.decision = first_decision
    res.plan = first_decision.plan if first_decision is not None else None
    if not quiet:
        _print(res, rerouters.get(first), coord)
    return res


def _print(res: SimResult, rerouter: SimRerouter | None, coord: EmergencyCoordinator) -> None:
    sc, d = res.scenario, res.decision
    print(f"\n=== {sc.name}: {sc.description}")
    print(f"robots   : {len(sc.robots)} ({', '.join(r.robot_id + '@' + r.lane for r in sc.robots)})")
    print(f"decision : {res.action.value if res.action else '-'}  ({res.reason})")
    for e in res.events:
        print(f"  t={e.t:6.2f} s  {e.state.value:<14} {e.message}")
    p = res.plan
    if p is not None:
        print(f"timeline : wait {p.wait_time:.1f} s, lane change out at t={p.t_out:.1f} s, "
              f"back in lane {p.lane_from} at t={p.t_back:.1f} s, delay {p.delay:.1f} s")
        if rerouter is not None and d is not None and math.isfinite(d.cost_reroute):
            print(f"vs reroute: bypass cost {d.cost_bypass:.1f} s, reroute cost {d.cost_reroute:.1f} s "
                  f"-> saves {d.cost_reroute - d.cost_bypass:.1f} s")
    if res.r2_s_at_change_out is not None:
        print(f"R2 at s={res.r2_s_at_change_out:.1f} m when R1 started the lane change")
    if res.emergencies:
        print("emergency:")
        for e in res.emergencies:
            print(f"  t={e.t:6.2f} s  #{e.standoff_id} {e.event:<9} {e.message}")
        for so in res.standoffs:
            moved = ", ".join(f"{k} {v:.2f} m" for k, v in so.moved.items()) or "nobody"
            print(f"  standoff #{so.sid}: {so.resolution.value if so.resolution else '-'}, "
                  f"robots {sorted(so.all_members)}, backed off: {moved}, phase {so.phase.value}")
    mode = "ORCA-DD" + (" + reciprocal windows" if res.coop else "") if res.orca else "baseline"
    print(f"mode     : {mode}; fleet delay {res.fleet_delay:.1f} s, emergency brakes {brakes(res)}, "
          f"ORCA-avoided pairs {res.avoid_events}")
    for t, rid, dl in res.coop_events:
        print(f"coop     : t={t:.1f} s {rid} slowed by {dl:.0f} s to open R1's window")
    print(f"reports  : {len(res.reports)} obstacle report(s) sent")
    print(f"R1 max y : {res.max_y_r1:.2f} m (lane A ends at y={AISLE.lane_width:.2f} m)")
    if math.isfinite(res.min_gap):
        print(f"min gap  : {res.min_gap:.2f} m between robots sharing a lane band")
    unfinished = [r for r, ok in res.finished.items() if not ok]
    if unfinished:
        print(f"unfinished: {', '.join(unfinished)}")
    print("collisions: " + (", ".join(f"{m} at t={t:.2f}" for t, m in res.collisions[:3]) or "none"))
    print(f"result   : {'PASS' if res.ok else 'FAIL'} (expected {sc.expected.value}"
          + (f", {', '.join(r.value for r in sc.resolutions)}" if sc.resolutions else ", no emergency") + ")")


def plot(results: list[SimResult], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(len(results), 2, figsize=(13, 3.4 * len(results)), squeeze=False)
    colors = plt.cm.tab10.colors
    for (top, st), res in zip(axes, results):
        top.set_title(f"{res.scenario.name}: top view")
        top.axhspan(0, AISLE.lane_width, color="#e8eef7")
        top.axhspan(AISLE.lane_width, 2 * AISLE.lane_width, color="#f5f5f5")
        for ob in res.scenario.obstacles:
            top.add_patch(Rectangle((ob.s_min, ob.y_min), ob.s_max - ob.s_min, ob.y_max - ob.y_min,
                                    color="#b5472f"))
            st.axhspan(ob.s_min, ob.s_max, color="#b5472f", alpha=0.3)
        for k, (rid, tr) in enumerate(res.traces.items()):
            c = colors[k % len(colors)]
            top.plot([r[1] for r in tr], [r[2] for r in tr], color=c, lw=1.2, label=rid)
            st.plot([r[0] for r in tr], [r[1] for r in tr], color=c, lw=1.2, label=rid)
        for e in res.emergencies:
            if e.event == "brake":
                st.axvline(e.t, color="#d62728", lw=0.8, ls="--")
            elif e.event == "reverse":
                st.axvline(e.t, color="#ff7f0e", lw=0.8, ls=":")
        top.set_xlim(0, AISLE.length)
        top.set_ylim(0, 2 * AISLE.lane_width)
        top.set_xlabel("s [m]")
        top.set_ylabel("y [m]")
        top.legend(loc="upper left", fontsize=7, ncol=5)
        st.set_title(f"{res.scenario.name}: space-time (red: emergency brake, orange: back-off)")
        st.set_xlabel("t [s]")
        st.set_ylabel("s [m]")
        st.set_ylim(0, AISLE.length)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nplot saved to {path}")


def fleet_n(n: int) -> Scenario:
    """A stress scene: R1 plus n-1 robots spread over both lanes (CPU of the coordinator)."""
    robots = [_r1()]
    for i in range(2, n + 1):
        if i % 3 == 0:
            robots.append(RobotSetup(f"R{i}", "A", -3.0 * (i // 3), 1.2, +1))
        else:
            robots.append(_b(f"R{i}", 45.0 + 2.5 * i))
    return Scenario(f"fleet_{n}", f"{n} robots, both lanes (coordinator CPU stress test)", (BOX,),
                    tuple(robots), Action.BYPASS_NOW, t_max=60.0)


def brakes(res: SimResult) -> int:
    """Emergency brakes caused by collision risk (deadlock hand-overs are not brakes)."""
    return sum(e.event == "brake" and "deadlock" not in e.message for e in res.emergencies)


def compare(names: list[str], dt: float, stress: bool) -> None:
    """Each scenario three ways: baseline, + ORCA-DD, + ORCA-DD + reciprocal windows."""
    modes = (("baseline", False, False), ("+orca", True, False), ("+orca+coop", True, True))
    head = (f"{'scenario':<17}{'mode':<12}{'decision':<17}{'fleet delay':>12}{'e-brakes':>9}"
            f"{'standoffs':>10}{'avoided':>8}{'hard dec':>9}{'min gap':>8}{'coll':>5}{'cpu/tick':>10}")
    print(head)
    print("-" * len(head))
    scenes = [SCENARIOS[n] for n in names] + ([fleet_n(50)] if stress else [])
    for sc in scenes:
        for label, o, c in modes:
            r = run(sc, dt=dt, quiet=True, orca=o, coop=c)
            print(f"{sc.name:<17}{label:<12}{(r.action.value if r.action else '-'):<17}"
                  f"{r.fleet_delay:>11.1f}s{brakes(r):>9}{len(r.standoffs):>10}{r.avoid_events:>8}"
                  f"{r.hard_decel_events:>9}{(r.min_gap if math.isfinite(r.min_gap) else 99):>8.2f}"
                  f"{len(r.collisions):>5}{r.tick_cpu * 1e3:>8.1f}ms")
        print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="all", choices=[*SCENARIOS, "all"])
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--plot", metavar="PNG", help="save a top view and space-time plot")
    ap.add_argument("--orca", action="store_true", help="run every robot with ORCA-DD (unicycle integration)")
    ap.add_argument("--coop", action="store_true", help="allow reciprocal bypass windows")
    ap.add_argument("--compare", action="store_true", help="baseline vs +orca vs +orca+coop metrics table")
    ap.add_argument("--stress", action="store_true", help="with --compare: add a 50-robot CPU test")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.CRITICAL)
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    if args.compare:
        compare(names, args.dt, args.stress)
        return 0
    results = [run(SCENARIOS[n], dt=args.dt, orca=args.orca or None, coop=args.coop or None) for n in names]
    if args.plot:
        plot(results, args.plot)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
