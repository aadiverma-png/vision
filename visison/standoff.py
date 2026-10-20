"""Fleet-level emergency braking and standoff resolution for any number of robots.

The :class:`EmergencyCoordinator` sits with the fleet manager. Each tick:

1. Every registered robot's :class:`~visison.emergency.EmergencyMonitor`
   rates the objects in its lidar (MATH.md section 8).
2. BRAKE alerts form *standoffs*, which are groups of robots that must stop
   together. In a head-on pair both robots brake. In a same-direction pair
   only the robot that is closing brakes (braking the one in front would make
   it worse). Robots driving into a braking group within ``cascade_range``
   brake too, and alerts that link two standoffs merge them. There is no
   limit on the number of robots or standoffs.
3. Each standoff runs its own resolution (MATH.md section 9)::

       BRAKING --all stopped--> resolve
         robot between the lanes -> CLEARING (chain reverse) -> REJOINING -> resolve again
         robot about to bypass   -> it yields                -> RESUMING
         otherwise               -> RESUMING (queue release, front first)
         unknown object in the way -> WAIT_UNKNOWN -> resolve
         anything infeasible / timeout -> ESCALATED (retries, fleet notified)

**Chain reverse**: the robot between the lanes (the *bypasser*) needs the
other lane up to ``reach`` to get back into its own lane. Robots in
that lane ahead of it are pushed back one after another. Each one only as far
as needed, and the push stops at the first robot that does not have to move.
Afterwards everyone resumes, and each robot's reservations are rebuilt from how
it actually moves, so the table is right again.

A deadlock (facing robots stopped in each other's way for
``deadlock_time`` s) opens a standoff without any alert.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Sequence

from .adapters import FleetReporter, RobotInterface
from .arbiter import Kind, MotionIntent, Source
from .controller import BypassController, ControllerState
from .emergency import (
    EmergencyConfig,
    EmergencyMonitor,
    RiskAssessment,
    RiskLevel,
    TrackedObject,
    matches_obstacle,
)
from .geometry import Aisle, Obstacle
from .kinematics import EPS, MotionLimits, Profile
from .motion_models import DiffDrive, Line, MotionModel
from .orca import OrcaAgent, OrcaConfig, OrcaMode, wall_half_planes
from .planner import BypassPlan, BypassPlanner, LateralShift, RobotSpec, RobotState, Trajectory
from .reporting import EmergencyReport
from .reservation import ReservationProvider, reservations_for_path


class Resolution(str, Enum):
    CHAIN_REVERSE = "chain_reverse"
    BYPASSER_YIELDS = "bypasser_yields"
    QUEUE_RELEASE = "queue_release"
    WAIT_UNKNOWN = "wait_unknown"
    ESCALATE = "escalate"


class StandoffPhase(str, Enum):
    BRAKING = "braking"
    WAIT_UNKNOWN = "wait_unknown"
    CLEARING = "clearing"  # robots in the borrowed lane back off
    REJOINING = "rejoining"  # the bypasser returns to its lane
    RESUMING = "resuming"  # everyone else drives on, front of each queue first
    ESCALATED = "escalated"
    DONE = "done"


_MANOEUVRE = (ControllerState.APPROACH, ControllerState.WAITING, ControllerState.CHANGE_OUT,
              ControllerState.PASSING, ControllerState.CHANGE_IN, ControllerState.REJOINING)
_UNCOMMITTED = (ControllerState.DETECTED, ControllerState.EVALUATING, ControllerState.APPROACH,
                ControllerState.WAITING, ControllerState.CHANGE_OUT)
_PAST_RETURN = (ControllerState.PASSING, ControllerState.CHANGE_IN, ControllerState.REJOINING)


@dataclass
class FleetMember:
    robot: RobotInterface
    spec: RobotSpec
    controller: BypassController | None = None
    cruise_speed: float | None = None  # speed to resume to; defaults to ``spec.limits.v_max``
    model: MotionModel | None = None  # kinematics for ORCA-DD; defaults to ``default_model(spec)``
    orca: bool = False  # the robot's arbiter runs ORCA-DD
    cooperative: bool = True  # False for robots of other fleets: we cannot command them
    open_area: bool = False  # True where the robot drives freely (full 2-D ORCA-DD), not on a lane

    def motion_model(self) -> MotionModel:
        return self.model or default_model(self.spec)


def default_model(spec: RobotSpec) -> MotionModel:
    """A differential-drive model derived from the planner's robot spec."""
    lim = spec.limits
    return DiffDrive(track_width=0.8 * spec.footprint.width, v_wheel_max=1.3 * lim.v_max,
                     a_acc=1.5 * lim.accel, a_dec=lim.e_decel, alpha_max=3.0, w_max=1.5)


@dataclass
class ReversePlan:
    bypasser: str
    rejoin: BypassPlan | None
    moves: dict[str, float]  # robot id -> metres it backs off
    targets: dict[str, float]  # robot id -> ``s`` where it stops
    feasible: bool
    reason: str
    reach: float = math.nan  # ``s`` up to which the bypasser may still use the other lane


@dataclass
class Standoff:
    sid: int
    members: set[str]
    t_start: float
    phase: StandoffPhase = StandoffPhase.BRAKING
    t_phase: float = 0.0
    trigger: RiskAssessment | None = None
    attempts: int = 1
    resolution: Resolution | None = None
    bypasser: str | None = None
    reverse: ReversePlan | None = None
    all_members: set[str] = field(default_factory=set)
    v_before: dict[str, float] = field(default_factory=dict)
    move_end: dict[str, float] = field(default_factory=dict)
    unknown: set[str] = field(default_factory=set)
    unknown_clear_since: float | None = None
    retry_at: float = 0.0
    escalation: str = ""
    moved: dict[str, float] = field(default_factory=dict)  # every back-off, over all attempts
    rejoined: list[str] = field(default_factory=list)
    yielded: list[str] = field(default_factory=list)
    braking: set[str] = field(default_factory=set)  # told to brake and not given a new motion since


# ---------------------------------------------------------------- pure helpers


def conflict_groups(pairs: Iterable[tuple[str, str]]) -> list[set[str]]:
    """Connected components of the "must stop together" relation (union-find)."""
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        parent[find(a)] = find(b)
    groups: dict[str, set[str]] = {}
    for a in list(parent):
        groups.setdefault(find(a), set()).add(a)
    return sorted(groups.values(), key=lambda g: sorted(g))


def body_band(aisle: Aisle, state: RobotState, spec: RobotSpec, pad: float = 0.0) -> tuple[float, float]:
    y = state.y if state.y is not None else aisle.lane_center(state.lane)
    return y - spec.footprint.width / 2 - pad, y + spec.footprint.width / 2 + pad


def is_between_lanes(aisle: Aisle, state: RobotState, spec: RobotSpec, clearance: float,
                     other_width: float | None = None) -> bool:
    """True if the robot's body reaches into the corridor that robots of the other lane drive in."""
    lane_to = aisle.other_lane(state.lane)
    half = (other_width if other_width is not None else spec.footprint.width) / 2 + clearance
    yc = aisle.lane_center(lane_to)
    lo, hi = body_band(aisle, state, spec)
    return lo < yc + half and yc - half < hi


def plan_chain_reverse(
    planner: BypassPlanner,
    bypasser: RobotState,
    obstacle: Obstacle,
    others: dict[str, tuple[RobotState, RobotSpec]],
    t: float,
    cfg: EmergencyConfig,
    barriers: Sequence[tuple[str, float, float]] = (),
) -> ReversePlan:
    """How far each robot in the borrowed lane must back off so the bypasser can rejoin its lane.

    In the bypasser's travel frame ``u = d*s``, it may use the other lane up
    to its front ``F = u_back + h_k`` (``u_back`` from :meth:`BypassPlanner.rejoin_plan`).
    Robots ``j1, j2, ...`` in that lane ahead of it, nearest first, need::

        req_1     = F + g                          g = min_gap + space
        rev_i     = max(0, req_i + h_i - u_i)
        req_{i+1} = u_i + rev_i + h_i + g

    The chain stops at the first ``rev_i = 0``. It is infeasible if a robot would
    have to back off more than ``max_reverse``, leave the aisle, or run into a
    barrier ``(lane, s_lo, s_hi)`` such as a static obstacle.
    """
    aisle, d = planner.aisle, bypasser.direction
    rejoin = planner.rejoin_plan(bypasser, obstacle, t, cfg.creep_speed)
    if rejoin is None:
        return ReversePlan(bypasser.robot_id, None, {}, {}, False,
                           "no creep path back into its lane from here")
    h_k = planner.spec.footprint.length / 2
    reach_u = d * rejoin.s_back + h_k
    gap = cfg.min_gap + planner.cfg.margins.space
    lane_to = rejoin.lane_to
    u_k = d * bypasser.s
    ahead = sorted(
        (d * st.s, rid, st, sp) for rid, (st, sp) in others.items()
        if st.lane == lane_to and d * st.s > u_k
    )
    req = reach_u + gap
    moves: dict[str, float] = {}
    targets: dict[str, float] = {}
    for u, rid, st, sp in ahead:
        h = sp.footprint.length / 2
        rev = req + h - u
        if rev <= 1e-6:
            break
        if rev > cfg.max_reverse:
            return ReversePlan(bypasser.robot_id, rejoin, moves, targets, False,
                               f"{rid} would have to back off {rev:.2f} m > max_reverse "
                               f"{cfg.max_reverse:.1f} m", d * reach_u)
        new_u = u + rev
        s_new = d * new_u
        if s_new - h < -1e-6 or s_new + h > aisle.length + 1e-6:
            return ReversePlan(bypasser.robot_id, rejoin, moves, targets, False,
                               f"{rid} would have to back out of the aisle", d * reach_u)
        for lane, s_lo, s_hi in barriers:
            b_lo, b_hi = sorted((d * s_lo, d * s_hi))
            if lane == lane_to and b_lo < new_u + h + gap and b_hi > u + h:
                return ReversePlan(bypasser.robot_id, rejoin, moves, targets, False,
                                   f"{rid} cannot back off: something is behind it at "
                                   f"s=[{s_lo:.2f}, {s_hi:.2f}]", d * reach_u)
        moves[rid], targets[rid] = rev, s_new
        req = new_u + h + gap
    why = (", ".join(f"{rid} backs off {m:.2f} m" for rid, m in moves.items())
           or "the other lane is already clear enough")
    return ReversePlan(bypasser.robot_id, rejoin, moves, targets, True, why, d * reach_u)


def displacement_trajectory(aisle: Aisle, state: RobotState, spec: RobotSpec, distance: float,
                            sign: int, t: float, speed: float) -> Trajectory:
    """Move ``distance`` m along the lane towards ``sign`` (+1: increasing s), from rest to rest.

    If that is against the robot's heading, it drives backwards.
    """
    lim = spec.limits
    creep = MotionLimits(min(speed, lim.v_max), lim.accel, lim.decel)
    prof = Profile(t, sign * state.s, 0.0).move(distance, creep, v_end=0.0)
    y = state.y if state.y is not None else aisle.lane_center(state.lane)
    return Trajectory(prof, sign, y, [], spec.lane_change, reverse=sign != state.direction)


def resume_trajectory(aisle: Aisle, state: RobotState, spec: RobotSpec, t: float,
                      v_target: float) -> Trajectory:
    """From rest up to ``v_target`` in the robot's own lane, drifting back to the lane centre."""
    d, lim = state.direction, spec.limits
    y0 = state.y if state.y is not None else aisle.lane_center(state.lane)
    yc = aisle.lane_center(state.lane)
    prof = Profile(t, d * state.s, 0.0)
    shifts = []
    if v_target > EPS:
        prof.change_speed(min(v_target, lim.v_max), lim.accel, lim.decel)
        if abs(y0 - yc) > 0.02:
            shifts.append(LateralShift(prof.t_end, spec.lane_change.duration(yc - y0), y0, yc))
    return Trajectory(prof, d, y0, shifts, spec.lane_change)


def delayed_trajectory(traj: Trajectory, t: float, delay: float, spec: RobotSpec) -> Trajectory | None:
    """The same lane path, arriving ``delay`` s later: slow to half speed for a while, then speed up.

    Dropping from v to v/2 and back costs ``v/(8 a_dec) + v/(8 a_acc)`` s; the
    rest of the delay comes from cruising a distance ``x`` at v/2, which costs ``x/v``.
    A robot that stands still (e.g. parked until its departure) just leaves later.
    Returns None if ``delay`` is too small to be done smoothly.
    """
    lim = spec.limits
    u, v = traj.profile.state(t)
    y = traj.lateral(t)
    if v < 0.05:
        p0 = traj.profile
        prof = Profile(p0.t0 + delay, p0.u0, p0.v0)
        for seg in p0.segments:
            prof._push(seg.a, seg.dt)
        return Trajectory(prof, traj.direction, y, [], spec.lane_change, traj.reverse)
    ramps = v / (8 * lim.decel) + v / (8 * lim.accel)
    if delay < ramps:
        return None
    prof = Profile(t, u, v).change_speed(v / 2, lim.accel, lim.decel)
    prof.cruise((delay - ramps) * v).change_speed(v, lim.accel, lim.decel)
    return Trajectory(prof, traj.direction, y, [], spec.lane_change, traj.reverse)


def stop_distance(v: float, decel: float, cfg: EmergencyConfig) -> float:
    return v * cfg.t_react_ego + v * v / (2 * decel)


# ----------------------------------------------------------------- coordinator


class EmergencyCoordinator:
    """Emergency braking and standoff resolution for all robots in one aisle.

    ``register`` robots (any number, also while running), then call
    ``tick(t, tracks_by_robot)`` at the control rate with each robot's lidar
    tracks. A fleet with several aisles runs one coordinator per aisle.
    """

    def __init__(
        self,
        planner: BypassPlanner,
        reservations: ReservationProvider | None = None,
        reporter: FleetReporter | None = None,
        cfg: EmergencyConfig | None = None,
        orca_cfg: OrcaConfig | None = None,
    ):
        self.planner, self.aisle = planner, planner.aisle
        self.orca_cfg = orca_cfg or OrcaConfig()
        self.reservations, self.reporter = reservations, reporter
        self.cfg = cfg or EmergencyConfig()
        self.members: dict[str, FleetMember] = {}
        self.monitors: dict[str, EmergencyMonitor] = {}
        self.standoffs: list[Standoff] = []
        self.history: list[Standoff] = []
        self.reports: list[EmergencyReport] = []
        self.alerts: list[RiskAssessment] = []
        self.known_obstacles: list[Obstacle] = []
        self.states: dict[str, RobotState] = {}
        self.tracks: dict[str, list[TrackedObject]] = {}  # last lidar tracks per robot
        self._still: dict[frozenset, float] = {}
        self._sid = 0
        self._order: list[tuple[float, str]] = []  # robots sorted by s (per-aisle index)
        self._late_shift: dict[str, tuple[int, float]] = {}  # rid -> (intent seq, s already shifted)
        self.avoided: list[RiskAssessment] = []  # alerts ORCA-DD handled without a brake
        self.coop_log: list[tuple[float, str, float]] = []  # (t, robot, delay) of reciprocal windows

    # registry ---------------------------------------------------------------

    def register(self, robot_id: str, member: FleetMember) -> None:
        self.members[robot_id] = member
        self.monitors[robot_id] = EmergencyMonitor(robot_id, member.spec.limits.e_decel, self.cfg)

    def unregister(self, robot_id: str) -> None:
        self.members.pop(robot_id, None)
        self.monitors.pop(robot_id, None)
        self.tracks.pop(robot_id, None)
        for so in self.standoffs:
            so.members.discard(robot_id)

    def standoff_of(self, robot_id: str) -> Standoff | None:
        return next((so for so in self.standoffs if robot_id in so.members), None)

    # perception -------------------------------------------------------------

    def ego_object(self, robot_id: str) -> TrackedObject:
        st, fp = self.states[robot_id], self.members[robot_id].spec.footprint
        y = st.y if st.y is not None else self.aisle.lane_center(st.lane)
        return TrackedObject(st.s, y, st.vs, 0.0, fp.length, fp.width, robot_id)

    def ego_band(self, robot_id: str, t: float) -> tuple[float, float]:
        """Lateral band the robot occupies now and along its plan for ``lookahead`` s."""
        st, m = self.states[robot_id], self.members[robot_id]
        ys = [st.y if st.y is not None else self.aisle.lane_center(st.lane)]
        ctl = m.controller
        if ctl is not None and ctl.plan is not None and ctl.state in _MANOEUVRE:
            traj = ctl.plan.trajectory
            ref = getattr(m.robot, "reference_time", None)
            t0 = ref(t) if ref is not None else t  # where the robot really is on its plan
            ys += [traj.lateral(t0 + self.cfg.lookahead * i / 10) for i in range(11)]
        half = m.spec.footprint.width / 2 + self.cfg.lateral_clearance
        return min(ys) - half, max(ys) + half

    def identify(self, track: TrackedObject, observer: str) -> str | None:
        """The fleet robot a lidar track belongs to, or ``None`` for an unknown object."""
        best, best_d = None, math.inf
        g = self.cfg.associate_gate
        for rid in self.near(track.s, self._max_length() / 2 + g):
            st = self.states[rid]
            if rid == observer:
                continue
            fp = self.members[rid].spec.footprint
            y = st.y if st.y is not None else self.aisle.lane_center(st.lane)
            ds, dy = abs(track.s - st.s), abs(track.y - y)
            if ds <= fp.length / 2 + g and dy <= fp.width / 2 + g and ds + dy < best_d:
                best, best_d = rid, ds + dy
        return best

    def obstacles(self) -> list[Obstacle]:
        """Known static obstacles: reported ones plus everything a controller is handling."""
        obs = list(self.known_obstacles)
        for m in self.members.values():
            ctl = m.controller
            if ctl is not None:
                if ctl.obstacle is not None:
                    obs.append(ctl.obstacle)
                obs += [o for o, _ in ctl._pending]
        return obs

    def _reindex(self) -> None:
        self._order = sorted((st.s, rid) for rid, st in self.states.items())

    def near(self, s: float, rng: float) -> list[str]:
        """Robots whose centre lies within ``rng`` of ``s`` along the aisle: O(log n + k)."""
        lo = bisect.bisect_left(self._order, (s - rng, ""))
        hi = bisect.bisect_right(self._order, (s + rng, "\uffff"))
        return [rid for _, rid in self._order[lo:hi]]

    def _max_length(self) -> float:
        return max((m.spec.footprint.length for m in self.members.values()), default=0.0)

    def orca_inputs(self, rid: str, t: float, intent: MotionIntent):
        """Everything a robot's arbiter needs for ORCA-DD: ego, neighbours, walls and mode.

        On lanes (inside the aisle and on its approaches) the lanes already keep
        robots apart sideways, so ORCA is lane-locked (``PATH_SPEED_ONLY``,
        feature C): only robots and unknown objects whose lateral band overlaps
        this robot's band count, and they are projected onto its lane line with a
        radius of half their length. That is 1-D ORCA along the lane. Only a robot
        the host marks as being in an open area (``FleetMember.open_area``) gets
        full 2-D ORCA-DD on effective-centre discs. Mixing the two between the same
        robots would break ORCA's reciprocity assumption.
        """
        st, m = self.states.get(rid), self.members[rid]
        if st is None:
            st = m.robot.state()
        cfg, fp = self.orca_cfg, m.spec.footprint
        coop = lambda oid: (self.members[oid].orca and self.members[oid].cooperative  # noqa: E731
                            and self.standoff_of(oid) is None)
        obstacles = self.obstacles()
        unknown = [tr for tr in self.tracks.get(rid, ())
                   if self.identify(tr, rid) is None and not any(matches_obstacle(tr, ob, self.cfg) for ob in obstacles)]
        if m.open_area:
            ego = self._agent(rid, st, m, True)
            nbs = [self._agent(o, self.states[o], self.members[o], coop(o))
                   for o in self.near(st.s, cfg.neighbour_range) if o != rid and o in self.members]
            nbs += [OrcaAgent(f"track:{tr.track_id}", (tr.s, tr.y), (tr.vs, tr.vy),
                              max(tr.length, tr.width) / 2 + cfg.radius_margin, False) for tr in unknown]
            nbs.sort(key=lambda a: math.dist(a.p, ego.p))
            return ego, nbs[:cfg.max_neighbours], [], OrcaMode.NORMAL

        y = st.y if st.y is not None else self.aisle.lane_center(st.lane)
        band = self.ego_band(rid, t)
        # ORCA keeps at least the emergency layer's min_gap (half per robot, plus a margin), which
        # stays inside the larger gap (min_gap + space) the chain reverse and rejoin are planned with
        pad = cfg.radius_margin + self.cfg.min_gap / 2
        ego = OrcaAgent(rid, (st.s, y), (st.vs, 0.0), fp.length / 2 + pad)
        body = body_band(self.aisle, st, m.spec)
        overlap = lambda a, b: a[0] < b[1] and b[0] < a[1]  # noqa: E731
        own = self.standoff_of(rid)
        scripted = intent.kind is Kind.SCRIPTED and own is not None
        nbs = []
        for oid in self.near(st.s, cfg.neighbour_range):
            if oid == rid or oid not in self.members:
                continue
            if scripted and oid in own.members:
                continue  # the standoff's back-off / rejoin geometry already keeps its members apart
            o, om = self.states[oid], self.members[oid]
            o_body = body_band(self.aisle, o, om.spec)
            # Traffic rule: a robot reacts to where the other robot physically is, and to its own
            # planned path. So a robot about to change lanes yields to the lane's owner, and the
            # owner only reacts once the other robot is really in its lane.
            if not overlap(band, o_body):
                continue
            shares = coop(oid) and (overlap(body, o_body) or overlap(self.ego_band(oid, t), body))
            s_o = o.s
            if o.direction == st.direction and st.direction * (o.s - st.s) > 0:
                # a leader in the same direction: keep a time headway (as adaptive cruise control does),
                # so the follower could still stop if the leader braked hard
                s_o -= st.direction * cfg.headway * abs(st.v)
            nbs.append(OrcaAgent(oid, (s_o, y), (o.vs, 0.0), om.spec.footprint.length / 2 + pad, shares))
        for tr in unknown:
            lo, hi = tr.band()
            if lo < band[1] and band[0] < hi and abs(tr.s - st.s) <= cfg.neighbour_range:
                nbs.append(OrcaAgent(f"track:{tr.track_id}", (tr.s, y), (tr.vs, 0.0),
                                     tr.length / 2 + pad, False))
        nbs.sort(key=lambda a: abs(a.p[0] - st.s))
        lane_lo, lane_hi = self.aisle.lane_bounds(st.lane)
        walls = wall_half_planes((st.s, y), fp.width / 2 + cfg.wall_margin,
                                 (min(lane_lo, band[0]), max(lane_hi, band[1])), cfg.tau_obs)
        return ego, nbs[:cfg.max_neighbours], walls, OrcaMode.PATH_SPEED_ONLY

    def _agent(self, rid: str, st: RobotState, m: FleetMember, cooperative: bool) -> OrcaAgent:
        model = m.motion_model()
        th = st.heading if st.heading is not None else (0.0 if st.direction > 0 else math.pi)
        y = st.y if st.y is not None else self.aisle.lane_center(st.lane)
        v = st.v * (-1.0 if st.reversing else 1.0)
        p = (st.s + model.D * math.cos(th), y + model.D * math.sin(th))
        r = model.effective_radius(m.spec.footprint.length, m.spec.footprint.width, self.orca_cfg.radius_margin)
        return OrcaAgent(rid, p, (v * math.cos(th), v * math.sin(th)), r, cooperative)

    def assess(self, t: float, tracks_by_robot: dict[str, Sequence[TrackedObject]]) -> list[RiskAssessment]:
        obstacles = self.obstacles()
        out = []
        for rid, tracks in tracks_by_robot.items():
            if rid not in self.members:
                continue
            out += self.monitors[rid].assess(
                self.ego_object(rid), self.ego_band(rid, t), tracks,
                identify=lambda tr, rid=rid: self.identify(tr, rid),
                decel_of=lambda o: self.members[o].spec.limits.e_decel if o in self.members else None,
                ignore=obstacles,
            )
        return out

    # main loop --------------------------------------------------------------

    def tick(
        self,
        t: float,
        tracks_by_robot: dict[str, Sequence[TrackedObject]] | None = None,
        alerts: Iterable[RiskAssessment] = (),
    ) -> list[RiskAssessment]:
        """Rate risks, brake, and advance every standoff. Returns this tick's assessments."""
        self.states = {rid: m.robot.state() for rid, m in self.members.items()}
        self._reindex()
        found = list(alerts)
        if tracks_by_robot is not None:
            # robots may scan at different moments: keep the last tracks of the others
            self.tracks.update({rid: list(trs) for rid, trs in tracks_by_robot.items()})
            found += self.assess(t, tracks_by_robot)
        self.alerts = found
        for a in found:
            if a.level is RiskLevel.BRAKE:
                if a.probability < self.cfg.p_hard and self._orca_handles(a):
                    self.avoided.append(replace(a, level=RiskLevel.AVOID))
                    continue
                self._on_brake(t, a)
        self._detect_deadlocks(t)
        self._follow_lateness(t)
        for so in list(self.standoffs):
            self._step(t, so)
        return found

    # braking ----------------------------------------------------------------

    def _orca_handles(self, a: RiskAssessment) -> bool:
        """Graded response: ORCA-DD is already resolving this pair within every closer's limits."""
        if a.gap <= self.cfg.min_gap:
            return False
        for rid in self._closing(a):
            m = self.members.get(rid)
            if m is None or not m.orca or self.standoff_of(rid) is not None:
                return False
            if m.controller is not None and m.controller.claim_conflict:
                return False
            last = getattr(m.robot, "last_orca", None)
            if last is None:
                return False
            if not last.feasible and not self._yielding(rid, a, last):
                return False
        return True

    def _yielding(self, rid: str, a: RiskAssessment, last) -> bool:
        """Still clear of the other's lane band and already slowing down: this robot is giving way."""
        other = a.other_id if a.ego_id == rid else a.ego_id
        st = self.states[rid]
        mine = body_band(self.aisle, st, self.members[rid].spec)
        if other in self.states:
            theirs = body_band(self.aisle, self.states[other], self.members[other].spec)
        else:
            tr = next((t for trs in self.tracks.values() for t in trs if f"track:{t.track_id}" == other), None)
            if tr is None:
                return False
            theirs = tr.band()
        apart = not (mine[0] < theirs[1] and theirs[0] < mine[1])
        return apart and abs(last.cmd.v) <= st.v + 1e-6

    def _follow_lateness(self, t: float) -> None:
        """Feedback: when a cruising robot runs late (e.g. slowed by ORCA), its bookings move with it."""
        if self.reservations is None:
            return
        for rid, m in self.members.items():
            late = getattr(m.robot, "lateness", 0.0)
            seq = getattr(m.robot, "intent_seq", 0)
            ctl = m.controller
            if self.standoff_of(rid) is not None or (ctl is not None and ctl.state not in (
                    ControllerState.CRUISING, ControllerState.REJOINED)):
                continue
            prev_seq, shifted = self._late_shift.get(rid, (seq, 0.0))
            if prev_seq != seq:
                shifted = 0.0  # a new reference: its lateness starts from zero
            if late - shifted > self.cfg.late_tolerance:
                conflicts = self.reservations.shift(rid, late - shifted, after=t)
                if self.reporter is not None and conflicts:
                    self.reporter.report_schedule_conflicts(rid, conflicts)
                shifted = late
            self._late_shift[rid] = (seq, shifted)

    def apply_delay(self, rid: str, delay: float, t: float) -> bool:
        """Reciprocal window (feature A): robot ``rid`` of our fleet slips by ``delay`` s on its path."""
        m = self.members.get(rid)
        if m is None or not m.cooperative or self.standoff_of(rid) is not None:
            return False
        if m.controller is not None and m.controller.state not in (ControllerState.CRUISING,
                                                                     ControllerState.REJOINED):
            return False
        traj = getattr(m.robot, "traj", None) or getattr(m.robot, "trajectory", None)
        new = delayed_trajectory(traj, t, delay, m.spec) if traj is not None else None
        if new is None:
            return False
        m.robot.follow(new)
        self.coop_log.append((t, rid, delay))
        return True

    def _release(self, rid: str, t: float) -> None:
        """The emergency layer lets go of a robot; its own motion continues (arbiter handover)."""
        release = getattr(self.members[rid].robot, "release", None)
        if release is not None:
            release(Source.EMERGENCY, handover=True)

    def _closing(self, a: RiskAssessment) -> set[str]:
        """Robots of an alert that have to brake."""
        ego = self.states[a.ego_id]
        if not a.other_is_robot or a.other_id not in self.states:
            return {a.ego_id}
        other = self.states[a.other_id]
        side = 1.0 if other.s >= ego.s else -1.0
        thr = self.cfg.standstill_speed
        out = set()
        if side * ego.vs > thr:
            out.add(a.ego_id)
        if -side * other.vs > thr:
            out.add(a.other_id)
        if ego.direction != other.direction:  # head-on: both stop, both take part in the resolution
            out |= {a.ego_id, a.other_id}
        return out or {a.ego_id}

    def _on_brake(self, t: float, a: RiskAssessment) -> None:
        involved = self._closing(a)
        touched = [so for so in self.standoffs if so.members & involved]
        if a.ego_id in self.members and not touched:
            # an alert from inside a running standoff about a robot that already left it
            touched = [so for so in self.standoffs if a.ego_id in so.members]
        if not touched:
            self._sid += 1
            so = Standoff(self._sid, set(involved), t, t_phase=t, trigger=a)
            self.standoffs.append(so)
            if not a.other_is_robot:
                so.unknown.add(a.other_id)
            self._cascade(so)
            self._brake(t, so, so.members, f"collision risk {a.ego_id}->{a.other_id}: "
                        f"P={a.probability:.2f}, gap {a.gap:.2f} m, TTC {a.ttc:.1f} s")
            return
        so = touched[0]
        for other in touched[1:]:  # the alert links two standoffs: merge them
            self._merge(so, other)
        before = set(so.members)
        so.members |= involved
        if not a.other_is_robot:
            so.unknown.add(a.other_id)
        self._cascade(so)
        added = so.members - before
        busy = so.phase in (StandoffPhase.CLEARING, StandoffPhase.REJOINING, StandoffPhase.RESUMING)
        if (not added and so.phase in (StandoffPhase.CLEARING, StandoffPhase.REJOINING)
                and a.other_id in before and a.ego_id in before and a.probability < self.cfg.p_hard):
            # both robots are executing the coordinator's own back-off / rejoin, which is
            # designed to keep min_gap + space between them; only a hard alarm overrides it
            return
        if not added and busy and a.other_is_robot and not self._closing_only(a, self.cfg.standstill_speed):
            # the gap is small but opening (e.g. the other robot is backing off as planned):
            # braking would only stop the manoeuvre that makes room
            return
        why = f"collision risk {a.ego_id}->{a.other_id} (P={a.probability:.2f}) during {so.phase.value}"
        if self._rear_join(a) and (added or busy):
            # a robot ran up behind the stopped group
            # (or one queued robot is still closing on the next): only the closer stops,
            # the manoeuvre at the front goes on
            closers = (added | (involved & self._closing_only(a))) - so.braking
            if closers:
                self._brake(t, so, closers, ("joins the back of the queue: " if added else "") + why,
                            keep_phase=True)
        elif added or busy or len(touched) > 1:
            if busy:
                so.attempts += 1
            self._brake(t, so, so.members, "new " + why)

    def _closing_only(self, a: RiskAssessment, thr: float = 0.0) -> set[str]:
        e, o = self.states[a.ego_id], self.states[a.other_id]
        side = 1.0 if o.s >= e.s else -1.0
        out = {a.ego_id} if side * e.vs > thr else set()
        return out | ({a.other_id} if -side * o.vs > thr else set())

    def _rear_join(self, a: RiskAssessment) -> bool:
        """The alert is a same-direction robot closing on another from behind."""
        if not a.other_is_robot or a.other_id not in self.states:
            return False
        e, o = self.states[a.ego_id], self.states[a.other_id]
        return e.direction == o.direction and e.lane == o.lane

    def _merge(self, into: Standoff, other: Standoff) -> None:
        into.members |= other.members
        into.all_members |= other.all_members
        into.unknown |= other.unknown
        for k, v in other.v_before.items():
            into.v_before.setdefault(k, v)
        into.attempts = max(into.attempts, other.attempts)
        self.standoffs.remove(other)

    def _cascade(self, so: Standoff) -> None:
        """Robots driving into a stopping group within ``cascade_range`` stop too."""
        changed = True
        while changed:
            changed = False
            for rid, st in self.states.items():
                if rid in so.members or self.standoff_of(rid) or abs(st.v) <= self.cfg.standstill_speed:
                    continue
                if self.members[rid].orca or not self.members[rid].cooperative:
                    continue  # ORCA-DD keeps its gap to the stopped group by itself
                band = body_band(self.aisle, st, self.members[rid].spec)
                near = set(self.near(st.s, self.cfg.cascade_range))
                for mid in [m for m in so.members if m in near]:
                    ms = self.states[mid]
                    ahead = st.direction * (ms.s - st.s)
                    mb = body_band(self.aisle, ms, self.members[mid].spec)
                    if (ms.lane == st.lane and ms.direction == st.direction
                            and 0 < ahead <= self.cfg.cascade_range and band[0] < mb[1] and mb[0] < band[1]):
                        so.members.add(rid)
                        changed = True
                        break

    def _brake(self, t: float, so: Standoff, robots: Iterable[str], why: str,
               keep_phase: bool = False) -> None:
        robots = sorted(robots)
        so.braking |= set(robots)
        for rid in robots:
            st = self.states.get(rid)
            if st is not None and rid not in so.v_before:
                so.v_before[rid] = 0.0 if st.reversing else st.v
            m = self.members.get(rid)
            if m is None or not m.cooperative:
                continue  # another fleet's robot: we can only watch it (its own safety must act)
            if m.controller is not None:
                m.controller.emergency_stop(t, why)
            else:
                m.robot.emergency_brake()
        so.all_members |= set(robots)
        if so.attempts > self.cfg.max_attempts:
            self._escalate(t, so, f"{so.attempts - 1} emergency brakes in one standoff")
            return
        if not keep_phase:
            so.phase, so.t_phase = StandoffPhase.BRAKING, t
        a = so.trigger
        self._report(t, so, "brake", f"EMERGENCY BRAKE for {', '.join(robots)}: {why}",
                     trigger=a.pair if a else None, probability=a.probability if a else None,
                     ttc=a.ttc if a else None)

    # deadlocks --------------------------------------------------------------

    def _detect_deadlocks(self, t: float) -> None:
        """Facing robots stopped in each other's way for ``deadlock_time`` form a standoff.

        All pairs that are ready at the same time are grouped (union-find), so a
        robot facing a whole queue ends up in one standoff, not one per pair.
        """
        free = {rid for rid in self.states if self.standoff_of(rid) is None}
        seen, ready = set(), []
        reach = self.cfg.deadlock_gap + self._max_length()
        for a in sorted(free):
            sa = self.states[a]
            for b in self.near(sa.s, reach):
                if b not in free or b <= a:
                    continue
                sb = self.states[b]
                key = frozenset((a, b))
                still = max(abs(sa.v), abs(sb.v)) <= self.cfg.standstill_speed
                facing = (sa.direction != sb.direction and sa.direction * (sb.s - sa.s) > 0
                          and sb.direction * (sa.s - sb.s) > 0)
                ba = body_band(self.aisle, sa, self.members[a].spec)
                bb = body_band(self.aisle, sb, self.members[b].spec)
                la, lb = self.members[a].spec.footprint.length, self.members[b].spec.footprint.length
                gap = abs(sa.s - sb.s) - (la + lb) / 2
                if still and facing and ba[0] < bb[1] and bb[0] < ba[1] and gap < self.cfg.deadlock_gap:
                    seen.add(key)
                    since = self._still.setdefault(key, t)
                    if t - since >= self.cfg.deadlock_time:
                        ready.append((a, b, gap, t - since))
        for group in conflict_groups((a, b) for a, b, _, _ in ready):
            pairs = [r for r in ready if r[0] in group]
            self._sid += 1
            so = Standoff(self._sid, set(group), t, t_phase=t)
            self.standoffs.append(so)
            a, b, gap, waited = min(pairs, key=lambda r: r[2])
            self._brake(t, so, so.members, f"deadlock: {a} and {b} face each other {gap:.2f} m apart "
                                           f"for {waited:.1f} s" + (f" ({len(group)} robots)" if len(group) > 2 else ""))
            for r in pairs:
                seen.discard(frozenset(r[:2]))
        self._still = {k: v for k, v in self._still.items() if k in seen}

    # resolution -------------------------------------------------------------

    def _step(self, t: float, so: Standoff) -> None:
        cfg, ph = self.cfg, so.phase
        if ph in (StandoffPhase.BRAKING, StandoffPhase.WAIT_UNKNOWN, StandoffPhase.CLEARING,
                  StandoffPhase.REJOINING) and t - so.t_phase > cfg.standoff_timeout:
            self._escalate(t, so, f"no progress in phase {ph.value} for {cfg.standoff_timeout:.0f} s")
        elif ph is StandoffPhase.BRAKING:
            stopped = all(abs(self.states[r].v) <= cfg.standstill_speed for r in so.members if r in self.states)
            # never before the robots can physically have stopped: odometry may lag
            t_stop = max((so.v_before.get(r, 0.0) / self.members[r].spec.limits.e_decel
                          for r in so.members if r in self.members), default=0.0) + cfg.t_react_other
            if (stopped and t - so.t_phase >= t_stop) or t - so.t_phase > cfg.brake_timeout:
                self._resolve(t, so)
        elif ph is StandoffPhase.WAIT_UNKNOWN:
            # "gone" means out of every member's path, not merely "no alert": a
            # stopped robot gets no alert for a person standing in front of it
            seen = any(self._unknown_ahead(r) for r in so.members) or any(
                a.other_id in so.unknown and a.ego_id in so.members
                and a.level is not RiskLevel.NONE for a in self.alerts)
            if seen:
                so.unknown_clear_since = None
            elif so.unknown_clear_since is None:
                so.unknown_clear_since = t
            elif t - so.unknown_clear_since >= cfg.unknown_clear_time:
                so.unknown.clear()
                self._report(t, so, "resolve", "unknown object has cleared")
                self._resolve(t, so)
        elif ph is StandoffPhase.CLEARING:
            done = all(t >= so.move_end[r] and abs(self.states[r].v) <= cfg.standstill_speed
                       for r in so.move_end if r in self.states)
            if done:
                self._start_rejoin(t, so)
        elif ph is StandoffPhase.REJOINING:
            ctl = self.members[so.bypasser].controller
            if ctl.state not in (ControllerState.REJOINING, ControllerState.EMERGENCY_STOP):
                so.members.discard(so.bypasser)
                so.rejoined.append(so.bypasser)
                self._release(so.bypasser, t)
                lane = ctl.plan.lane_from if ctl.plan is not None else self.states[so.bypasser].lane
                self._report(t, so, "rejoin", f"{so.bypasser} is back in lane {lane} ({ctl.state.value})")
                so.bypasser = None
                self._resolve(t, so)
        elif ph is StandoffPhase.RESUMING:
            self._release_queue(t, so)
        elif ph is StandoffPhase.ESCALATED and t >= so.retry_at:
            so.retry_at = t + cfg.escalate_retry
            self._resolve(t, so)
        if not so.members and so in self.standoffs:
            so.phase = StandoffPhase.DONE
            self.standoffs.remove(so)
            self.history.append(so)
            self._report(t, so, "done", f"standoff #{so.sid} resolved in {t - so.t_start:.1f} s")

    def _committed(self, rid: str) -> bool:
        st, m = self.states[rid], self.members[rid]
        ctl = m.controller
        if ctl is not None and ctl._pre_emergency in _PAST_RETURN:
            return True
        lane_to = self.aisle.other_lane(st.lane)
        widths = [self.members[o].spec.footprint.width for o, s in self.states.items() if s.lane == lane_to]
        return is_between_lanes(self.aisle, st, m.spec, self.planner.cfg.clearance,
                                max(widths) if widths else None)

    def _barriers(self) -> list[tuple[str, float, float]]:
        out = []
        for ob in self.obstacles():
            for lane in self.aisle.lanes:
                if ob.overlaps_lateral(*self.aisle.lane_bounds(lane)):
                    out.append((lane, ob.s_min, ob.s_max))
        return out

    def _resolve(self, t: float, so: Standoff) -> None:
        so.members &= set(self.members)
        if not so.members:
            return
        if so.unknown:
            if so.phase is not StandoffPhase.WAIT_UNKNOWN:
                so.phase, so.t_phase, so.resolution = StandoffPhase.WAIT_UNKNOWN, t, Resolution.WAIT_UNKNOWN
                so.unknown_clear_since = None
                self._report(t, so, "resolve", f"unknown object {', '.join(sorted(so.unknown))} "
                             "in the way: holding until it is gone", resolution=Resolution.WAIT_UNKNOWN)
            return

        committed = [r for r in so.members if self._committed(r)]
        if committed:
            self._resolve_committed(t, so, committed)
            return

        yielders = [r for r in so.members if self.members[r].controller is not None
                    and self.members[r].controller._pre_emergency in _UNCOMMITTED
                    and self.members[r].controller.obstacle is not None]
        for rid in yielders:
            ctl, st = self.members[rid].controller, self.states[rid]
            y = st.y if st.y is not None else self.aisle.lane_center(st.lane)
            plan = None
            if abs(y - self.aisle.lane_center(st.lane)) > 0.05:
                plan = ctl.planner.rejoin_plan(st, ctl.obstacle, t, self.cfg.creep_speed, mode="return")
            if plan is not None:
                ctl.rejoin(t, plan, "emergency: back to the lane centre and give way")
            else:
                ctl.yield_lane(t, "emergency: give way, plan the bypass again later")
            so.members.discard(rid)
            so.yielded.append(rid)
            self._release(rid, t)
        so.resolution = Resolution.BYPASSER_YIELDS if yielders else (so.resolution or Resolution.QUEUE_RELEASE)
        so.phase, so.t_phase = StandoffPhase.RESUMING, t
        msg = (f"{', '.join(yielders)} gives way and plans again; " if yielders else "") + \
              f"{', '.join(sorted(so.members)) or 'nobody'} resume(s), front of each queue first"
        self._report(t, so, "resolve", msg, resolution=so.resolution)
        self._release_queue(t, so)

    def _resolve_committed(self, t: float, so: Standoff, committed: list[str]) -> None:
        options = []
        for rid in committed:
            ctl = self.members[rid].controller
            if ctl is None or ctl.obstacle is None:
                self._escalate(t, so, f"{rid} is between the lanes but has no known obstacle to pass")
                return
            others = {o: (s, self.members[o].spec) for o, s in self.states.items() if o != rid}
            rp = plan_chain_reverse(ctl.planner, self.states[rid], ctl.obstacle, others, t, self.cfg,
                                    self._barriers())
            foreign = [o for o in rp.moves if not self.members[o].cooperative]
            if rp.feasible and foreign:
                rp = replace(rp, feasible=False,
                             reason=f"{', '.join(foreign)} belong(s) to another fleet and cannot be asked to back "
                                    "off; the RMF schedule shows the hold")
            remaining = abs(rp.reach - self.states[rid].s) if rp.feasible else math.inf
            options.append((remaining, rid, rp))
        remaining, rid, rp = min(options, key=lambda o: o[0])
        if not rp.feasible:
            self._escalate(t, so, f"{rid} cannot rejoin its lane: {rp.reason}")
            return
        newcomers = set(rp.moves) - so.members
        if newcomers:  # robots outside the standoff are in the way: stop them first
            so.members |= newcomers
            self._brake(t, so, newcomers, f"needed to make room for {rid}")
            return
        so.bypasser, so.reverse, so.resolution = rid, rp, Resolution.CHAIN_REVERSE
        d = self.states[rid].direction
        so.move_end = {}
        for mid, dist in rp.moves.items():
            st, m = self.states[mid], self.members[mid]
            traj = displacement_trajectory(self.aisle, st, m.spec, dist, d, t, self.cfg.reverse_speed)
            m.robot.follow(traj, Source.EMERGENCY, Kind.SCRIPTED)  # its controller stays in EMERGENCY_STOP
            so.braking.discard(mid)
            so.move_end[mid] = traj.profile.t_end
            so.moved[mid] = so.moved.get(mid, 0.0) + dist
        so.phase, so.t_phase = StandoffPhase.CLEARING, t
        self._report(t, so, "reverse",
                     f"{rid} is between the lanes and has priority (needs lane "
                     f"{rp.rejoin.lane_to} up to s={rp.reach:.2f} m): {rp.reason}",
                     resolution=Resolution.CHAIN_REVERSE, moves=tuple(rp.moves.items()))

    def _start_rejoin(self, t: float, so: Standoff) -> None:
        rid = so.bypasser
        ctl, st = self.members[rid].controller, self.states[rid]
        plan = ctl.planner.rejoin_plan(st, ctl.obstacle, t, self.cfg.creep_speed)
        if plan is None:
            self._escalate(t, so, f"{rid} has no creep path back into its lane")
            return
        so.braking.discard(rid)
        ctl.rejoin(t, plan, f"emergency {plan.kind}: creep at {plan.v_bypass:.1f} m/s back into "
                            f"lane {plan.lane_from}")
        so.phase, so.t_phase = StandoffPhase.REJOINING, t
        self._report(t, so, "rejoin", f"{rid} creeps back into lane {plan.lane_from}, "
                     f"clear by s={plan.s_back:.2f} m at t={plan.t_back:.1f} s")

    def _nearest_ahead(self, rid: str) -> tuple[float, str, RobotState] | None:
        st, spec = self.states[rid], self.members[rid].spec
        band = body_band(self.aisle, st, spec, self.cfg.lateral_clearance)
        best = None
        i = bisect.bisect_left(self._order, (st.s, rid))
        order = self._order[i + 1:] if st.direction > 0 else reversed(self._order[:i])
        for _, oid in order:
            if best is not None and abs(self.states[oid].s - st.s) - self._max_length() > best[0]:
                break  # sorted by s: nothing further can be nearer
            os_ = self.states[oid]
            if oid == rid:
                continue
            ahead = st.direction * (os_.s - st.s)
            ob = body_band(self.aisle, os_, self.members[oid].spec)
            if ahead <= 0 or not (ob[0] < band[1] and band[0] < ob[1]):
                continue
            gap = ahead - (spec.footprint.length + self.members[oid].spec.footprint.length) / 2
            if best is None or gap < best[0]:
                best = (gap, oid, os_)
        return best

    def _unknown_ahead(self, rid: str) -> bool:
        """An unidentified lidar object in the robot's path, close enough to matter."""
        st, m = self.states[rid], self.members[rid]
        v = m.cruise_speed if m.cruise_speed is not None else m.spec.limits.v_max
        reach = m.spec.footprint.length / 2 + self.cfg.restart_gap + stop_distance(v, m.spec.limits.e_decel, self.cfg)
        lo, hi = body_band(self.aisle, st, m.spec, self.cfg.lateral_clearance)
        obstacles = self.obstacles()
        for tr in self.tracks.get(rid, ()):
            tlo, thi = tr.band()
            ahead = st.direction * (tr.s - st.s) - tr.length / 2
            if (tlo < hi and lo < thi and -m.spec.footprint.length / 2 < ahead <= reach
                    and self.identify(tr, rid) is None
                    and not any(matches_obstacle(tr, ob, self.cfg) for ob in obstacles)):
                return True
        return False

    def _release_queue(self, t: float, so: Standoff) -> None:
        """Resume the robots that have free road ahead, front of each queue first."""
        order = sorted(so.members, key=lambda r: -self.states[r].direction * self.states[r].s)
        for rid in order:
            st, m = self.states[rid], self.members[rid]
            if not m.cooperative:  # another fleet's robot is not ours to resume
                so.members.discard(rid)
                continue
            v = m.cruise_speed if m.cruise_speed is not None else m.spec.limits.v_max
            need = self.cfg.restart_gap + stop_distance(v, m.spec.limits.e_decel, self.cfg)
            if self._unknown_ahead(rid):
                continue
            ahead = self._nearest_ahead(rid)
            if ahead is not None and m.orca and ahead[0] >= self.cfg.min_gap:
                ahead = None  # ORCA-DD keeps the gap while the queue moves off together
            if ahead is not None:
                gap, oid, os_ = ahead
                v_away = st.direction * os_.vs  # > 0: moving away from us
                if oid in so.members or gap < need or v_away < -self.cfg.standstill_speed:
                    continue
                if gap < need + 2 * self.cfg.restart_gap + v * 2.0:
                    if v_away <= self.cfg.standstill_speed:
                        continue  # stopped robot close ahead: wait until it moves
                    o = self.members[oid]
                    v = min(v, o.cruise_speed if o.cruise_speed is not None else o.spec.limits.v_max)
            traj = resume_trajectory(self.aisle, st, m.spec, t, v)
            if m.controller is not None:
                m.controller.resume(t, traj)
            else:
                m.robot.follow(traj)
            self._rebook(t, rid, traj)
            so.members.discard(rid)
            self._release(rid, t)
            self._report(t, so, "resume", f"{rid} resumes at {v:.2f} m/s")

    def _rebook(self, t: float, rid: str, traj: Trajectory) -> None:
        """Replace the robot's remaining reservations with ones from how it will really move."""
        if self.reservations is None:
            return
        st, spec = self.states[rid], self.members[rid].spec
        lane, d = st.lane, st.direction
        old = [r for r in self.reservations.query(lane, range(self.aisle.n_cells), t) if r.robot_id == rid]
        if not old:
            return
        far = max(d * b for r in old for b in self.aisle.cell_bounds(r.cell))
        u0 = d * st.s
        if far <= u0 or traj.profile.v_end <= EPS:
            return
        h = spec.footprint.length / 2
        new = reservations_for_path(rid, self.aisle, lane, traj.profile, d, u0, far,
                                    h + self.planner.cfg.margins.space)
        self.reservations.release(old)
        if self.reservations.try_reserve(new):
            return
        self.reservations.try_reserve(old)
        delay = max(r.end for r in new) - max(r.end for r in old)
        conflicts = self.reservations.shift(rid, delay, after=t) if delay > 0 else []
        if self.reporter is not None and conflicts:
            self.reporter.report_schedule_conflicts(rid, conflicts)

    def _escalate(self, t: float, so: Standoff, why: str) -> None:
        so.phase, so.t_phase, so.resolution = StandoffPhase.ESCALATED, t, Resolution.ESCALATE
        so.retry_at = t + self.cfg.escalate_retry
        for rid in so.members:  # make sure nobody moves
            m = self.members[rid]
            if not m.cooperative:
                continue
            if m.controller is not None:
                m.controller.emergency_stop(t, "standoff escalated")
            else:
                m.robot.emergency_brake()
        if why != so.escalation:
            so.escalation = why
            self._report(t, so, "escalate", f"needs the fleet / an operator: {why}",
                         resolution=Resolution.ESCALATE)

    def _report(self, t: float, so: Standoff, event: str, message: str, **kw) -> None:
        if "resolution" in kw and kw["resolution"] is not None:
            kw["resolution"] = kw["resolution"].value
        rep = EmergencyReport(t, self.aisle.aisle_id, so.sid, event, tuple(sorted(so.all_members | so.members)),
                              message, **kw)
        self.reports.append(rep)
        if self.reporter is not None and hasattr(self.reporter, "report_emergency"):
            self.reporter.report_emergency(rep)
