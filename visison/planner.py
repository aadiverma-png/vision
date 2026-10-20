"""Reservation-aware obstacle bypass planner (the Visison decision core).

Given a blocked lane, the planner answers one question:

    Can the robot borrow the neighbouring lane, pass the obstacle, and be
    back in its own lane before anyone who has reserved that lane space
    needs it?

The answer uses closed-form kinematics plus an exact interval search:

1. Build the bypass motion (approach, lane change out, pass, lane change
   in). This gives, for every cell ``k`` of the other lane, the time
   interval ``[a_k, b_k]`` during which the robot's footprint covers it,
   measured relative to the moment the manoeuvre starts.
2. A reservation ``(r_s, r_e)`` on cell ``k`` rules out every start time in
   ``(r_s - b_k - tau, r_e - a_k + tau)``, where ``tau`` is the safety margin.
3. The earliest safe start ``t*`` is the smallest time not covered by the
   union of all those intervals. Sorting and merging gives it in O(n log n).

If the robot can bypass at ``t_now`` without stopping, the action is
BYPASS_NOW. Otherwise the robot stops before the obstacle and waits until
``t*`` (WAIT_THEN_BYPASS). If rerouting is cheaper overall, the action is
REROUTE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .geometry import Aisle, Footprint, Obstacle, squeeze_offset
from .intervals import Interval, earliest_free_point
from .kinematics import EPS, InfeasibleMotion, LaneChangeModel, MotionLimits, Profile
from .reservation import Reservation, ReservationProvider, sweep_occupancy


# --------------------------------------------------------------------------- types


@dataclass(frozen=True)
class RobotSpec:
    footprint: Footprint
    limits: MotionLimits
    lane_change: LaneChangeModel
    bypass_speed: float  # preferred longitudinal speed while in the other lane
    energy_per_meter: float = 1.0  # arbitrary energy units
    energy_per_second: float = 0.2  # hotel load (computers, sensors) while delayed


@dataclass(frozen=True)
class RobotState:
    robot_id: str
    s: float  # centre position along the aisle
    v: float  # speed, >= 0
    lane: str
    direction: int = 1  # +1 drives towards increasing s, -1 towards decreasing s
    y: float | None = None  # lateral centre; None means the lane centre
    reversing: bool = False  # True while the robot drives backwards (``v`` is still >= 0)
    heading: float | None = None  # body heading in the aisle frame; None means along ``direction``

    @property
    def vs(self) -> float:
        """Signed velocity along the aisle (``ds/dt``)."""
        return self.direction * self.v * (-1 if self.reversing else 1)


@dataclass(frozen=True)
class SafetyMargins:
    """Safety buffers.

    Time buffers are summed into ``tau``. Space buffers make the swept
    footprint longer.
    """

    comm_latency: float = 0.2  # s, fleet <-> robot message delay
    reservation_jitter: float = 0.5  # s, how early another robot may arrive
    other_brake_time: float = 0.75  # s, about v2 / (2 * a2_brake) of other robots
    localization_error: float = 0.05  # m
    min_gap: float = 0.30  # m, bumper-to-bumper gap to keep

    @property
    def tau(self) -> float:
        return self.comm_latency + self.reservation_jitter + self.other_brake_time

    @property
    def space(self) -> float:
        return self.localization_error + self.min_gap


@dataclass(frozen=True)
class PlannerConfig:
    clearance: float = 0.15  # m, lateral and longitudinal clearance to the obstacle
    margins: SafetyMargins = field(default_factory=SafetyMargins)
    speed_fractions: tuple[float, ...] = (1.0, 0.75, 0.5, 0.3)  # bypass speeds to try
    max_wait: float = 90.0  # s, beyond this, reroute if a reroute is available
    horizon: float = 300.0  # s, how far ahead to search for a window
    energy_weight: float = 0.0  # seconds of cost per energy unit
    reroute_penalty: float = 10.0  # s, cost of disturbing other robots' schedules
    coop_delays: tuple[float, ...] = ()  # s; e.g. (1, 2, 3) lets oncoming robots of our fleet slow a little
    coop_penalty: float = 1.0  # s, fixed cost of asking another robot to change its timing
    coop_lead: float = 2.0  # s, a shifted robot must have at least this + 2*delay before its window


@dataclass(frozen=True)
class RerouteEstimate:
    """Extra time and distance of the host planner's alternative, e.g. PIBT."""

    extra_time: float
    extra_distance: float


class Action(str, Enum):
    CONTINUE = "continue"  # obstacle is not in the robot's path
    SQUEEZE = "squeeze"  # pass inside the robot's own lane
    BYPASS_NOW = "bypass_now"
    WAIT_THEN_BYPASS = "wait_then_bypass"
    REROUTE = "reroute"
    HOLD = "hold"  # stop and wait for the obstacle to be removed
    REJOIN = "rejoin"  # emergency: creep back into the robot's own lane (see standoff.py)
    COOP_BYPASS = "coop_bypass"  # bypass now; oncoming robots of our fleet slow down a little


class Phase(str, Enum):
    APPROACH = "approach"
    WAITING = "waiting"
    CHANGE_OUT = "change_out"
    PASSING = "passing"
    CHANGE_IN = "change_in"
    REJOINED = "rejoined"


# ---------------------------------------------------------------- trajectory


@dataclass(frozen=True)
class Pose:
    t: float
    s: float
    y: float
    v: float
    heading: float  # angle of the velocity vector in the (s, y) plane, radians


@dataclass(frozen=True)
class LateralShift:
    t_start: float
    duration: float
    y_from: float
    y_to: float


class Trajectory:
    """Reference trajectory for the robot's tracking controller.

    Combines a longitudinal profile with a sequence of lateral shifts.
    ``direction`` is the direction of travel. With ``reverse=True`` the robot
    drives backwards: the body faces against the travel direction and the
    sampled speed is negative.
    """

    def __init__(
        self,
        profile: Profile,
        direction: int,
        y0: float,
        shifts: list[LateralShift],
        lane_change: LaneChangeModel,
        reverse: bool = False,
    ):
        self.profile, self.direction, self.y0 = profile, direction, y0
        self.shifts, self.lane_change, self.reverse = shifts, lane_change, reverse

    def lateral(self, t: float) -> float:
        y = self.y0
        for sh in self.shifts:
            if t <= sh.t_start:
                break
            x = (t - sh.t_start) / sh.duration if sh.duration > EPS else 1.0
            y = sh.y_from + (sh.y_to - sh.y_from) * self.lane_change.progress(sh.y_to - sh.y_from, x)
        return y

    def sample(self, t: float) -> Pose:
        u, v = self.profile.state(t)
        h = 1e-3
        dy = (self.lateral(t + h) - self.lateral(t - h)) / (2 * h)
        vx = self.direction * v
        heading = math.atan2(dy, vx) if abs(dy) + abs(vx) > EPS else (0.0 if self.direction > 0 else math.pi)
        if self.reverse:
            heading = math.atan2(-math.sin(heading), -math.cos(heading))
            v = -v
        return Pose(t, self.direction * u, self.lateral(t), v, heading)


# ---------------------------------------------------------------------- plan


@dataclass
class BypassPlan:
    kind: str  # "bypass" or "squeeze"
    lane_from: str
    lane_to: str
    v_bypass: float
    s_wait: float | None  # where the robot stops to wait, if it has to
    s_out: float  # where the lateral move out starts
    s_back: float  # where the robot is fully back in its lane
    t_arrive: float  # when it reaches the wait point (equals t_depart if rolling)
    t_depart: float  # t*: when the manoeuvre starts
    t_out: float
    t_back: float
    lane_change_time: float
    trajectory: Trajectory
    claims: list[Reservation]
    delay: float  # compared with driving the same stretch unobstructed
    extra_distance: float
    energy: float
    coop_shifts: dict[str, float] = field(default_factory=dict)  # robot id -> s it is asked to slip

    @property
    def wait_time(self) -> float:
        return self.t_depart - self.t_arrive

    @property
    def rolling(self) -> bool:
        return self.s_wait is None

    def phase(self, t: float) -> Phase:
        if t < self.t_arrive:
            return Phase.APPROACH
        if t < self.t_depart:
            return Phase.WAITING
        if t < self.t_out:
            return Phase.APPROACH
        if t < self.t_out + self.lane_change_time:
            return Phase.CHANGE_OUT
        if t < self.t_back - self.lane_change_time:
            return Phase.PASSING
        if t < self.t_back:
            return Phase.CHANGE_IN
        return Phase.REJOINED


@dataclass
class Decision:
    action: Action
    reason: str
    plan: BypassPlan | None = None
    cost_bypass: float = math.inf
    cost_reroute: float = math.inf
    candidates: int = 0


# ------------------------------------------------------------------- planner


class BypassPlanner:
    def __init__(self, aisle: Aisle, spec: RobotSpec, config: PlannerConfig | None = None):
        self.aisle, self.spec = aisle, spec
        self.cfg = config or PlannerConfig()

    # public ---------------------------------------------------------------

    def evaluate(
        self,
        state: RobotState,
        obstacle: Obstacle,
        t_now: float,
        reservations: ReservationProvider | None = None,
        other_lane_obstacles: tuple[Obstacle, ...] = (),
        reroute: RerouteEstimate | None = None,
        shiftable: Callable[[str], bool] | None = None,
    ) -> Decision:
        """Decide how to get past ``obstacle``.

        ``shiftable(robot_id)`` says which other robots may be asked to slow down
        (reciprocal bypass windows, ``cfg.coop_delays``); robots of other fleets
        must return False.
        """
        cfg, fp, d = self.cfg, self.spec.footprint, state.direction
        y0 = state.y if state.y is not None else self.aisle.lane_center(state.lane)
        u1 = d * state.s
        ou1, ou2 = sorted((d * obstacle.s_min, d * obstacle.s_max))
        c, h = cfg.clearance, fp.length / 2

        if ou2 <= u1 - h - c:
            return Decision(Action.CONTINUE, "obstacle is behind the robot")
        if not obstacle.overlaps_lateral(y0 - fp.width / 2 - c, y0 + fp.width / 2 + c):
            return Decision(Action.CONTINUE, "obstacle does not intersect the robot's path")

        # 0. Squeeze past inside the robot's own lane: no reservation needed.
        y_sq = squeeze_offset(self.aisle, state.lane, obstacle, fp, c)
        if y_sq is not None:
            plans = self._candidates(state, ou1, ou2, y0, y_sq, state.lane, t_now, None)
            if plans:
                best = min(plans, key=self._cost)
                return Decision(Action.SQUEEZE, f"lateral gap fits; nudge to y={y_sq:.2f} m",
                                best, self._cost(best), candidates=len(plans))

        # 1. Borrow the other lane.
        lane_to = self.aisle.other_lane(state.lane)
        y_to = self.aisle.lane_center(lane_to)
        band = (y_to - fp.width / 2 - c, y_to + fp.width / 2 + c)
        blocked = obstacle.overlaps_lateral(*band)
        if not blocked and other_lane_obstacles:
            reach = self._max_zone(ou1, ou2, y0, y_to)
            s_lo, s_hi = sorted((d * reach[0], d * reach[1]))
            blocked = any(
                o.overlaps_lateral(*band) and o.overlaps_longitudinal(s_lo, s_hi)
                for o in other_lane_obstacles
            )
        cost_reroute = self._cost_reroute(reroute)
        if blocked:
            return self._fallback(reroute, cost_reroute, "both lanes are blocked at the obstacle")

        plans = self._candidates(state, ou1, ou2, y0, y_to, lane_to, t_now, reservations)
        if cfg.coop_delays and reservations is not None and shiftable is not None:
            plans += self._coop_candidates(state, ou1, ou2, y0, y_to, lane_to, t_now, reservations, shiftable)
        if not plans:
            return self._fallback(
                reroute, cost_reroute,
                "no safe window within the horizon, or too close to change lanes safely",
            )
        best = min(plans, key=self._cost)
        cost_bypass = self._cost(best)

        if reroute is not None and (cost_reroute < cost_bypass or best.wait_time > cfg.max_wait):
            return Decision(
                Action.REROUTE,
                f"reroute cost {cost_reroute:.1f} s < bypass cost {cost_bypass:.1f} s"
                if cost_reroute < cost_bypass
                else f"wait {best.wait_time:.1f} s exceeds max_wait {cfg.max_wait:.0f} s",
                None, cost_bypass, cost_reroute, len(plans),
            )
        if best.coop_shifts:
            asked = ", ".join(f"{r} slows by {dl:.0f} s" for r, dl in best.coop_shifts.items())
            reason = (f"bypass now while {asked} (total fleet delay {cost_bypass:.1f} s instead of "
                      f"{min((self._cost(p) for p in plans if not p.coop_shifts), default=math.inf):.1f} s)")
            action = Action.COOP_BYPASS
        elif best.rolling:
            reason = (f"other lane is free long enough: clear by t={best.t_back:.1f} s "
                      f"({best.t_back - t_now:.1f} s from now) with margin tau={cfg.margins.tau:.2f} s")
            action = Action.BYPASS_NOW
        else:
            reason = (f"other lane is occupied now; wait {best.wait_time:.1f} s at "
                      f"s={best.s_wait:.2f} m, then bypass at t*={best.t_depart:.1f} s")
            action = Action.WAIT_THEN_BYPASS
        return Decision(action, reason, best, cost_bypass, cost_reroute, len(plans))

    def nominal_time(self, state: RobotState, s_target: float, t_now: float) -> float:
        """When the robot would reach ``s_target`` at full speed with no obstacle."""
        p = Profile(t_now, state.direction * state.s, min(state.v, self.spec.limits.v_max))
        lim = self.spec.limits
        p.change_speed(lim.v_max, lim.accel, lim.decel)
        return p.time_at(state.direction * s_target)

    def rejoin_plan(
        self, state: RobotState, obstacle: Obstacle, t_now: float, v_creep: float = 0.2,
        mode: str = "auto",
    ) -> BypassPlan | None:
        """Emergency plan that brings a stopped robot from between the lanes back into its own lane.

        The robot starts from rest and creeps at ``v_creep``, so its lane
        changes are as short as possible (``D = v_creep * T_lc``).

        * ``mode="return"`` shifts back to the lane centre *before* the
          obstacle and stops there (``kind="return"``). The robot then
          plans the bypass again.
        * ``mode="pass"`` completes the lane change out if the robot is still
          in front of the obstacle, creeps past it, and shifts back
          (``kind="rejoin"``).
        * ``mode="auto"`` picks whichever option uses less of the other lane.

        ``s_back`` is the furthest point the robot reaches while any part of it
        may still be in the other lane. Returns ``None`` if no option fits.
        """
        spec, lim, cfg, d = self.spec, self.spec.limits, self.cfg, state.direction
        fp, c, h = spec.footprint, cfg.clearance, spec.footprint.length / 2
        y_home = self.aisle.lane_center(state.lane)
        y_cur = state.y if state.y is not None else y_home
        lane_to = self.aisle.other_lane(state.lane)
        y_to = self.aisle.lane_center(lane_to)
        u1 = d * state.s
        ou1, ou2 = sorted((d * obstacle.s_min, d * obstacle.s_max))
        v_c = min(v_creep, lim.v_max)
        u_go = u1 + v_c**2 / (2 * lim.accel)  # creep speed is reached here
        past = u1 - h - c >= ou2
        options: list[tuple[str, Profile, list[LateralShift], float]] = []

        if mode in ("auto", "return"):
            T = spec.lane_change.duration(y_home - y_cur)
            u_stop = u_go + v_c * T + v_c**2 / (2 * lim.decel)
            if past or u_stop + h + c <= ou1:
                prof = Profile(t_now, u1, 0.0).change_speed(v_c, lim.accel, lim.decel)
                shifts = [LateralShift(prof.t_end, T, y_cur, y_home)]
                prof.cruise(v_c * T).change_speed(0.0, lim.accel, lim.decel)
                options.append(("return", prof, shifts, prof.t_end))

        if mode in ("auto", "pass"):
            prof = Profile(t_now, u1, 0.0).change_speed(v_c, lim.accel, lim.decel)
            shifts, y, ok = [], y_cur, True
            beside = obstacle.overlaps_lateral(y_cur - fp.width / 2 - c, y_cur + fp.width / 2 + c)
            if beside and not past:
                T_out = spec.lane_change.duration(y_to - y_cur)
                ok = u_go + v_c * T_out + h + c <= ou1
                shifts.append(LateralShift(prof.t_end, T_out, y_cur, y_to))
                prof.cruise(v_c * T_out)
                y = y_to
            if ok:
                prof.cruise(max(ou2 + c + h - prof.u_end, 0.0))
                T_back = spec.lane_change.duration(y_home - y)
                shifts.append(LateralShift(prof.t_end, T_back, y, y_home))
                prof.cruise(v_c * T_back)
                t_back = prof.t_end
                prof.change_speed(lim.v_max, lim.accel, lim.decel)
                options.append(("rejoin", prof, shifts, t_back))

        if not options:
            return None
        # the option that reaches least far along the aisle needs the least of the other lane
        kind, prof, shifts, t_back = min(options, key=lambda o: o[1].state(o[3])[0])
        u_back = prof.state(t_back)[0]
        tau, lo, hi = cfg.margins.tau, *self.aisle.lane_bounds(lane_to)
        ys = [y_cur, *(sh.y_to for sh in shifts)]
        intrudes = any(y - fp.width / 2 < hi and lo < y + fp.width / 2 for y in ys)
        occ = sweep_occupancy(self.aisle, prof, d, u1, u_back, h + cfg.margins.space) if intrudes else []
        claims = [Reservation(state.robot_id, lane_to, k, a_k - tau, b_k + tau) for k, a_k, b_k in occ]
        delay = max(t_back - self.nominal_time(state, d * u_back, t_now), 0.0)
        return BypassPlan(
            kind=kind, lane_from=state.lane, lane_to=lane_to, v_bypass=v_c,
            s_wait=None, s_out=state.s, s_back=d * u_back,
            t_arrive=t_now, t_depart=t_now, t_out=shifts[0].t_start, t_back=t_back,
            lane_change_time=shifts[-1].duration,
            trajectory=Trajectory(prof, d, y_cur, shifts, spec.lane_change),
            claims=claims, delay=delay, extra_distance=0.0,
            energy=spec.energy_per_second * delay,
        )

    # internals ------------------------------------------------------------

    def _cost(self, plan: BypassPlan) -> float:
        """Total fleet delay: this robot's delay, plus what it asks of the others."""
        coop = sum(plan.coop_shifts.values()) + (self.cfg.coop_penalty if plan.coop_shifts else 0.0)
        return plan.delay + self.cfg.energy_weight * plan.energy + coop

    def _coop_candidates(self, state, ou1, ou2, y0, y_to, lane_to, t_now, reservations, shiftable):
        """Rolling bypasses that become possible if the robots blocking the window slip by delta."""
        blockers: set[str] = set()
        v_b = min(self.spec.bypass_speed * max(self.cfg.speed_fractions), self.spec.limits.v_max)
        if self._build(state, ou1, ou2, y0, y_to, lane_to, v_b, True, t_now, reservations,
                       blockers_out=blockers) is not None or not blockers:
            return []  # nothing to ask for (or the window is already free)
        if not all(shiftable(r) for r in blockers):
            return []
        reach = self._max_zone(ou1, ou2, y0, y_to)
        zone = self.aisle.cells(*sorted((state.direction * reach[0], state.direction * reach[1])))
        out = []
        for delta in self.cfg.coop_delays:
            shift = {r: delta for r in blockers}
            if not self._coop_ok(reservations, shift, t_now, state.robot_id, lane_to, set(zone)):
                continue
            for f in self.cfg.speed_fractions:
                vb = min(self.spec.bypass_speed * f, self.spec.limits.v_max)
                plan = self._build(state, ou1, ou2, y0, y_to, lane_to, vb, True, t_now, reservations, coop=shift)
                if plan is not None:
                    plan.coop_shifts = dict(shift)
                    out.append(plan)
            if out:
                break  # the smallest delay that works
        return out

    def _coop_ok(self, reservations, shift: dict[str, float], t_now: float, me: str,
                 lane_to: str, zone: set[int]) -> bool:
        """Shifting these robots must leave them time to slow down before the zone and hit no third robot."""
        for rid, delta in shift.items():
            mine = [r for lane in self.aisle.lanes
                    for r in reservations.query(lane, range(self.aisle.n_cells), t_now) if r.robot_id == rid]
            future = [r for r in mine if r.start >= t_now]
            in_zone = [r.start for r in future if r.lane == lane_to and r.cell in zone]
            if not in_zone or min(in_zone) < t_now + self.cfg.coop_lead + 2 * delta:
                return False  # already in (or too close to) the conflict zone to slow down in time
            for r in future:
                moved = Reservation(r.robot_id, r.lane, r.cell, r.start + delta, r.end + delta)
                for o in reservations.query(r.lane, [r.cell], t_now):
                    if o.robot_id not in shift and o.robot_id != me and moved.conflicts_with(o):
                        return False
        return True

    def _cost_reroute(self, r: RerouteEstimate | None) -> float:
        if r is None:
            return math.inf
        e = self.spec.energy_per_meter * r.extra_distance + self.spec.energy_per_second * r.extra_time
        return r.extra_time + self.cfg.reroute_penalty + self.cfg.energy_weight * e

    def _fallback(self, reroute, cost_reroute, why: str) -> Decision:
        if reroute is not None:
            return Decision(Action.REROUTE, why, cost_reroute=cost_reroute)
        return Decision(Action.HOLD, why + "; no reroute available, holding for obstacle removal")

    def _max_zone(self, ou1, ou2, y0, y_to) -> tuple[float, float]:
        """The widest stretch of the other lane any candidate could use (travel frame)."""
        v = self.spec.bypass_speed * max(self.cfg.speed_fractions)
        D = v * self.spec.lane_change.duration(y_to - y0)
        pad = self.cfg.clearance + self.spec.footprint.length
        return ou1 - pad - D, ou2 + pad + D

    def _candidates(self, state, ou1, ou2, y0, y_to, lane_to, t_now, reservations) -> list[BypassPlan]:
        plans = []
        for f in self.cfg.speed_fractions:
            v_b = min(self.spec.bypass_speed * f, self.spec.limits.v_max)
            for rolling in (True, False):
                plan = self._build(state, ou1, ou2, y0, y_to, lane_to, v_b, rolling, t_now, reservations)
                if plan is not None:
                    plans.append(plan)
        return plans

    def _build(
        self, state: RobotState, ou1: float, ou2: float, y0: float, y_to: float, lane_to: str,
        v_b: float, rolling: bool, t_now: float, reservations: ReservationProvider | None,
        coop: dict[str, float] | None = None, blockers_out: set[str] | None = None,
    ) -> BypassPlan | None:
        spec, lim, cfg, d = self.spec, self.spec.limits, self.cfg, state.direction
        h, c = spec.footprint.length / 2, cfg.clearance
        T_lc = spec.lane_change.duration(y_to - y0)
        D_lc = v_b * T_lc
        u1, v1 = d * state.s, min(state.v, lim.v_max)
        u_a = ou1 - c - h - D_lc  # the lateral move out starts here
        u_b = ou2 + c + h + D_lc  # the robot is fully back in its lane here

        prof = Profile(t_now, u1, v1)
        try:
            if rolling:
                if u_a < u1 - EPS:
                    return None
                prof.move(u_a - u1, lim, v_end=v_b)
                u_w, t_arrive = None, t_now
                rel = Profile(prof.t_end, u_a, v_b)
            else:
                u_w = u_a - v_b**2 / (2 * lim.accel)
                if u_w < u1 - EPS:
                    return None
                prof.move(u_w - u1, lim, v_end=0.0)
                t_arrive = prof.t_end
                rel = Profile(0.0, u_w, 0.0).change_speed(v_b, lim.accel, lim.decel)
            rel.cruise(u_b - u_a)
        except InfeasibleMotion:
            return None

        # When the footprint covers each cell of the other lane. For a rolling
        # start the times are absolute; after a wait they are relative to t*.
        tau = cfg.margins.tau
        occ = []
        if lane_to != state.lane:  # a squeeze never leaves the robot's lane
            occ = sweep_occupancy(self.aisle, rel, d, u_a, u_b, h + cfg.margins.space)

        forbidden: list[Interval] = []
        if reservations is not None and occ:
            cells = [k for k, _, _ in occ]
            by_cell: dict[int, list[Reservation]] = {}
            for r in reservations.query(lane_to, cells, t_now - 3600, exclude_robot=state.robot_id):
                by_cell.setdefault(r.cell, []).append(r)
            offset = t_now if rolling else 0.0  # express everything as "start time"
            for k, a_k, b_k in occ:
                a_rel, b_rel = a_k - offset, b_k - offset
                for r in by_cell.get(k, ()):
                    dl = coop.get(r.robot_id, 0.0) if coop and r.start >= t_now else 0.0
                    iv = Interval(r.start + dl - b_rel - tau, r.end + dl - a_rel + tau)
                    forbidden.append(iv)
                    if blockers_out is not None and rolling and iv.lo < t_now < iv.hi:
                        blockers_out.add(r.robot_id)

        if rolling:
            if earliest_free_point(forbidden, t_now, t_now) is None:
                return None
            t_depart, shift = t_now, 0.0
        else:
            t_star = earliest_free_point(forbidden, t_arrive, t_now + cfg.horizon)
            if t_star is None:
                return None
            t_depart, shift = t_star, t_star
            prof.hold_until(t_star)
            prof.change_speed(v_b, lim.accel, lim.decel)

        # prof now ends at u_a with speed v_b: pass, then speed back up.
        prof.cruise(u_b - u_a)
        t_out, t_back = prof.time_at(u_a), prof.time_at(u_b)
        prof.change_speed(lim.v_max, lim.accel, lim.decel)

        claims = [
            Reservation(state.robot_id, lane_to, k, a_k + shift - tau, b_k + shift + tau)
            for k, a_k, b_k in occ
        ]
        shifts = [
            LateralShift(t_out, T_lc, y0, y_to),
            LateralShift(t_back - T_lc, T_lc, y_to, y0),
        ]
        traj = Trajectory(prof, d, y0, shifts, spec.lane_change)

        extra_distance = 2 * _arc_excess(spec.lane_change, y_to - y0, D_lc)
        delay = max(t_back - self.nominal_time(state, d * u_b, t_now), 0.0)
        energy = spec.energy_per_meter * extra_distance + spec.energy_per_second * delay
        return BypassPlan(
            kind="bypass" if lane_to != state.lane else "squeeze",
            lane_from=state.lane, lane_to=lane_to, v_bypass=v_b,
            s_wait=None if u_w is None else d * u_w,
            s_out=d * u_a, s_back=d * u_b,
            t_arrive=t_arrive, t_depart=t_depart, t_out=t_out, t_back=t_back,
            lane_change_time=T_lc, trajectory=traj, claims=claims,
            delay=delay, extra_distance=extra_distance, energy=energy,
        )


def oncoming_is_clear(
    t_cross: float, gap: float, v_other: float, margins: SafetyMargins, other_length: float = 0.0
) -> bool:
    """Closed-form rule for a single oncoming robot (MATH.md section 4).

    ``gap`` is the distance from the other robot's centre to the far edge of
    the bypass zone and ``t_cross`` is the time the bypass needs to clear it.
    The robot arrives after ``(gap - other_length/2) / v_other - jitter``;
    the bypass is safe if it finishes, plus latency, braking time and the
    space margin converted to time, before that. :meth:`BypassPlanner.evaluate`
    solves the general many-robot case exactly; this is the quick check.
    """
    if v_other <= EPS:
        return True
    t_arrive = (gap - other_length / 2) / v_other - margins.reservation_jitter
    t_safe = margins.comm_latency + margins.other_brake_time + margins.space / v_other
    return t_cross + t_safe <= t_arrive


def _arc_excess(model: LaneChangeModel, offset: float, D: float, n: int = 64) -> float:
    """Extra path length of one lane change compared with driving straight for ``D``."""
    if D <= EPS:
        return abs(offset)
    pts = [(D * i / n, offset * model.progress(offset, i / n)) for i in range(n + 1)]
    arc = sum(math.dist(p, q) for p, q in zip(pts, pts[1:]))
    return arc - D
