"""Runtime state machine that turns planner decisions into robot actions.

::

    CRUISING -> DETECTED -> EVALUATING -+-> APPROACH/WAITING -> CHANGE_OUT -> PASSING -> CHANGE_IN -> REJOINED
                                        +-> REROUTING
                                        +-> HOLDING (re-evaluates every ``hold_retry`` s)

    any state --emergency_stop--> EMERGENCY_STOP --rejoin--> REJOINING -> REJOINED (or HOLDING)
                                                 --yield_lane--> HOLDING
                                                 --resume--> CRUISING (or back to HOLDING)

The controller claims the bypass window atomically before moving. If the
claim is refused (another robot won the window), it plans again. Until the
lane change out starts, every tick checks that the claimed window is
still valid and that perception sees no new obstacle in the borrowed stretch;
if either check fails, it releases the claim and plans again. Once the
robot is partly in the other lane it completes the manoeuvre.

Emergencies: if the lidar-based collision prediction fires anyway (for
example another robot is not where its reservations say), the fleet's
:class:`~visison.standoff.EmergencyCoordinator` takes over through
``emergency_stop``, ``rejoin``, ``yield_lane`` and ``resume``. While it
owns the robot (EMERGENCY_STOP, REJOINING), ``tick`` does not move it and new
obstacles are queued.

Delay propagation: when a bypass is committed, the robot's later
reservations are shifted by the plan's delay so other robots see the new
timestamps immediately. Any extra lateness observed at rejoin is shifted as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable

from .adapters import FleetReporter, ReroutePlanner, RobotInterface
from .arbiter import Kind, Source
from .geometry import Obstacle
from .planner import Action, BypassPlan, BypassPlanner, Decision, Phase
from .reporting import ObstacleReport
from .reservation import ReservationProvider


class ControllerState(str, Enum):
    CRUISING = "cruising"
    DETECTED = "detected"
    EVALUATING = "evaluating"
    APPROACH = "approach"
    WAITING = "waiting"
    CHANGE_OUT = "change_out"
    PASSING = "passing"
    CHANGE_IN = "change_in"
    REJOINED = "rejoined"
    REROUTING = "rerouting"
    HOLDING = "holding"
    EMERGENCY_STOP = "emergency_stop"
    REJOINING = "rejoining"


_COORDINATED = (ControllerState.EMERGENCY_STOP, ControllerState.REJOINING)

_PHASE_STATE = {
    Phase.APPROACH: ControllerState.APPROACH,
    Phase.WAITING: ControllerState.WAITING,
    Phase.CHANGE_OUT: ControllerState.CHANGE_OUT,
    Phase.PASSING: ControllerState.PASSING,
    Phase.CHANGE_IN: ControllerState.CHANGE_IN,
    Phase.REJOINED: ControllerState.REJOINED,
}


@dataclass(frozen=True)
class Event:
    t: float
    state: ControllerState
    message: str = ""


@dataclass
class BypassController:
    robot: RobotInterface
    planner: BypassPlanner
    reservations: ReservationProvider | None = None
    reporter: FleetReporter | None = None
    rerouter: ReroutePlanner | None = None
    max_claim_attempts: int = 3
    hold_retry: float = 2.0  # s between re-evaluations while holding
    rejoin_tolerance: float = 0.5  # s of extra lateness before shifting again

    state: ControllerState = ControllerState.CRUISING
    obstacle: Obstacle | None = None
    decision: Decision | None = None
    events: list[Event] = field(default_factory=list)
    _reported: set[str] = field(default_factory=set)
    _pending: list[tuple[Obstacle, tuple[Obstacle, ...]]] = field(default_factory=list)
    _shifted: float = 0.0
    _t_commit: float = 0.0
    _retry_at: float = 0.0
    _pre_emergency: ControllerState | None = None
    shiftable: Callable[[str], bool] | None = None  # which other robots may be asked to slow (coop)
    coop_executor: Callable[[str, float, float], bool] | None = None  # (robot, delay, t) -> applied?
    _claim_ext: float = 0.0  # s the current claims have been extended by
    claim_conflict: bool = False  # an extension was refused: the coordinator watches this robot

    @property
    def plan(self) -> BypassPlan | None:
        return self.decision.plan if self.decision else None

    @property
    def active(self) -> bool:
        return self.state not in (ControllerState.CRUISING, ControllerState.REJOINED)

    # public ---------------------------------------------------------------

    def on_obstacle(
        self, obstacle: Obstacle, t: float, other_lane_obstacles: tuple[Obstacle, ...] = ()
    ) -> Decision:
        """Perception reports an obstacle ahead. Returns the decision taken.

        While a manoeuvre for another obstacle is running, a new obstacle is
        reported at once but only evaluated after the robot has rejoined its
        lane. Until then, the emergency layer handles anything in the way. The
        same happens while the emergency coordinator owns the robot.
        """
        if self.state in _COORDINATED or (self.active and self.obstacle is not None):
            if (self.obstacle is None or obstacle.obstacle_id != self.obstacle.obstacle_id) and not any(
                o.obstacle_id == obstacle.obstacle_id for o, _ in self._pending
            ):
                self._pending.append((obstacle, other_lane_obstacles))
                self.events.append(Event(t, self.state, f"obstacle {obstacle.obstacle_id} queued "
                                                        "until the current manoeuvre ends"))
                self._report_one(t, obstacle, "queued")
            return self.decision
        self.obstacle = obstacle
        self._set(t, ControllerState.DETECTED, f"obstacle {obstacle.obstacle_id} ahead")
        decision = self._decide(t, other_lane_obstacles)
        self._report(t, decision)
        return decision

    def tick(self, t: float, other_lane_obstacles: tuple[Obstacle, ...] = ()) -> ControllerState:
        """Advance the state machine. Call it periodically, e.g. at 10–50 Hz."""
        if self.state is ControllerState.EMERGENCY_STOP:
            return self.state
        tr = self._ref_time(t)  # where the robot really is on its plan (lags t when ORCA slowed it)
        if self.state is ControllerState.REJOINING:
            plan = self.plan
            self._extend_claims(t)
            if tr >= plan.t_back:
                if plan.kind == "rejoin":
                    self._finish(t, plan)
                else:  # "return": back in its lane before the obstacle; plan the bypass again
                    self._abort(t, "back at the lane centre before the obstacle", "abort bypass")
                    self._hold(t, "returned to own lane after emergency")
            return self.state
        if self.state is ControllerState.HOLDING:
            if t >= self._retry_at:
                self._decide(t, other_lane_obstacles)
            return self.state
        plan = self.plan
        if plan is None or not self.active or self.state is ControllerState.REROUTING:
            return self.state

        if tr < plan.t_out and plan.kind == "bypass":
            problem = self._window_problem(plan, t, other_lane_obstacles)
            if problem:
                self._abort(t, problem)
                self._decide(t, other_lane_obstacles)
                return self.state
        if plan.kind == "bypass" and tr >= plan.t_out:
            self._extend_claims(t)

        new = _PHASE_STATE[plan.phase(tr)]
        if new is ControllerState.REJOINED:
            self._finish(t, plan)
        elif new is not self.state:
            self._set(t, new)
        return self.state

    # emergency hooks (driven by visison.standoff.EmergencyCoordinator) ------

    def emergency_stop(self, t: float, reason: str) -> None:
        """Brake as hard as possible and wait for the coordinator. Claims are kept."""
        if self.state not in _COORDINATED:
            self._pre_emergency = self.state
        self.robot.emergency_brake()
        if self.state is not ControllerState.EMERGENCY_STOP:
            self._set(t, ControllerState.EMERGENCY_STOP, reason)

    def rejoin(self, t: float, plan: BypassPlan, reason: str = "") -> None:
        """Follow an emergency rejoin (or return) plan from :meth:`BypassPlanner.rejoin_plan`."""
        old = self.plan
        if self.reservations is not None and old is not None and old.claims:
            self.reservations.release(old.claims)
        if old is not None and old.kind == "bypass" and plan.kind == "rejoin":
            self._shift(t, max(plan.t_back - old.t_back, 0.0))  # the emergency made the robot later
        if self.reservations is not None:
            if plan.claims and not self.reservations.try_reserve(plan.claims):
                # the other robots were physically moved out of the way; the
                # table only lags behind, so report it and go on
                self.events.append(Event(t, self.state, "rejoin claim overlaps other reservations"))
        self.decision = Decision(Action.REJOIN, reason or f"{plan.kind} after emergency stop", plan)
        self._claim_ext = 0.0
        self.robot.follow(plan.trajectory, Source.EMERGENCY, Kind.SCRIPTED)  # driven by the emergency layer
        self._set(t, ControllerState.REJOINING, self.decision.reason)

    def yield_lane(self, t: float, reason: str) -> None:
        """Give up the planned bypass (release claims, undo the shift) and plan again later."""
        if self.plan is not None and self.plan.kind == "bypass":
            self._abort(t, reason, "abort bypass")
        self._hold(t, reason)

    def resume(self, t: float, trajectory) -> None:
        """Continue after an emergency stop in which this robot was not bypassing."""
        pre, self._pre_emergency = self._pre_emergency, None
        if pre is ControllerState.HOLDING or (pre is ControllerState.REROUTING):
            self._hold(t, "resumed after emergency stop; re-evaluating")
            return
        self.robot.follow(trajectory)
        self._set(t, ControllerState.CRUISING, "resumed after emergency stop")
        if self._pending:
            obstacle, others = self._pending.pop(0)
            self.on_obstacle(obstacle, t, others)

    # internals ------------------------------------------------------------

    def _ref_time(self, t: float) -> float:
        ref = getattr(self.robot, "reference_time", None)
        return ref(t) if ref is not None else t

    def _extend_claims(self, t: float) -> None:
        """Keep the other-lane claims ahead of the robot when it runs late (e.g. slowed by ORCA)."""
        plan = self.plan
        lateness = getattr(self.robot, "lateness", 0.0)
        if self.reservations is None or not plan.claims or lateness <= self._claim_ext + 0.25:
            return
        grow = lateness + 0.5 - self._claim_ext
        old = [c for c in plan.claims if c.end > t]
        new = [replace(c, end=c.end + grow) for c in old]
        self.reservations.release(old)
        if self.reservations.try_reserve(new):
            keep = [c for c in plan.claims if c.end <= t]
            plan.claims = keep + new
            self._claim_ext += grow
            self.claim_conflict = False
        else:
            self.reservations.try_reserve(old)
            if not self.claim_conflict:
                self.events.append(Event(t, self.state, f"running {lateness:.1f} s late; cannot extend "
                                                        "the other-lane claim"))
            self.claim_conflict = True

    def _hold(self, t: float, reason: str) -> None:
        self.robot.stop()
        self._retry_at = t + self.hold_retry
        self._set(t, ControllerState.HOLDING, reason)

    def _set(self, t: float, state: ControllerState, message: str = "") -> None:
        self.state = state
        self.events.append(Event(t, state, message))

    def _decide(self, t: float, other_lane_obstacles) -> Decision:
        self._set(t, ControllerState.EVALUATING)
        st = self.robot.state()
        reroute = self.rerouter.estimate(st, self.obstacle) if self.rerouter else None
        decision = None
        allow_coop = self.coop_executor is not None
        for attempt in range(self.max_claim_attempts):
            decision = self.planner.evaluate(
                st, self.obstacle, t, self.reservations, other_lane_obstacles, reroute,
                self.shiftable if allow_coop else None,
            )
            plan = decision.plan
            if plan is None or plan.kind != "bypass" or self.reservations is None:
                break
            if plan.coop_shifts:
                if self._claim_coop(t, plan):
                    break
                allow_coop = False  # plan again without asking the others
                continue
            if self.reservations.try_reserve(plan.claims):
                break
            self.events.append(Event(t, self.state, f"claim refused (attempt {attempt + 1}); replanning"))
        else:
            decision = Decision(Action.HOLD, "could not claim a bypass window; holding")
        self.decision = decision
        self._apply(t, decision, st)
        return decision

    def _claim_coop(self, t: float, plan: BypassPlan) -> bool:
        """Move the other robots' bookings, claim, and have them slow down; all or nothing."""
        done = []
        for rid, delta in plan.coop_shifts.items():
            self.reservations.shift(rid, delta, after=t)
            done.append((rid, delta))
        ok = self.reservations.try_reserve(plan.claims)
        if ok:
            ok = all(self.coop_executor(rid, delta, t) for rid, delta in plan.coop_shifts.items())
            if not ok:
                self.reservations.release(plan.claims)
        if not ok:
            for rid, delta in done:
                self.reservations.shift(rid, -delta, after=t + delta)
            self.events.append(Event(t, self.state, "cooperative window not granted; replanning"))
        return ok

    def _apply(self, t: float, d: Decision, st) -> None:
        if d.action is Action.CONTINUE:
            self._set(t, ControllerState.CRUISING, d.reason)
        elif d.action in (Action.SQUEEZE, Action.BYPASS_NOW, Action.WAIT_THEN_BYPASS, Action.COOP_BYPASS):
            plan = d.plan
            self.robot.follow(plan.trajectory)
            self._t_commit = t
            self._claim_ext, self.claim_conflict = 0.0, False
            self._shift(t, plan.delay, keep=plan.claims)
            self._set(t, _PHASE_STATE[plan.phase(t)], f"{d.action.value}: {d.reason}")
        elif d.action is Action.REROUTE:
            self._set(t, ControllerState.REROUTING, d.reason)
            self.rerouter.execute(st, self.obstacle)
        else:  # HOLD
            self.robot.stop()
            self._retry_at = t + self.hold_retry
            self._set(t, ControllerState.HOLDING, d.reason)

    def _shift(self, t: float, delay: float, keep=()) -> None:
        if self.reservations is None or abs(delay) < 1e-6:
            return
        conflicts = self.reservations.shift(self.robot.robot_id, delay, after=t, keep=keep)
        self._shifted += delay
        if self.reporter is not None and conflicts:
            self.reporter.report_schedule_conflicts(self.robot.robot_id, conflicts)

    def _window_problem(self, plan: BypassPlan, t: float, other_lane_obstacles) -> str | None:
        if self.reservations is not None and plan.claims:
            cells = sorted({c.cell for c in plan.claims})
            others = self.reservations.query(plan.lane_to, cells, t, exclude_robot=self.robot.robot_id)
            if any(c.conflicts_with(r) for c in plan.claims for r in others):
                return "claimed window was invaded by another reservation"
        s_lo, s_hi = sorted((plan.s_out, plan.s_back))
        lo, hi = self.planner.aisle.lane_bounds(plan.lane_to)
        for o in other_lane_obstacles:
            if o.overlaps_lateral(lo, hi) and o.overlaps_longitudinal(s_lo, s_hi):
                return f"new obstacle {o.obstacle_id} in the borrowed lane"
        return None

    def _abort(self, t: float, why: str, what: str = "abort before lane change") -> None:
        plan = self.plan
        if self.reservations is not None and plan is not None:
            self.reservations.release(plan.claims)
            if self._shifted:
                self.reservations.shift(
                    self.robot.robot_id, -self._shifted, after=self._t_commit + self._shifted
                )
                self._shifted = 0.0
        self.events.append(Event(t, self.state, f"{what}: {why}"))

    def _finish(self, t: float, plan: BypassPlan) -> None:
        if self.reservations is not None and plan.claims:
            self.reservations.release(plan.claims)  # the robot is back; free the other lane early
        late = t - plan.t_back
        if late > self.rejoin_tolerance:
            self._shift(t, late)
        self._set(t, ControllerState.REJOINED, f"back in lane {plan.lane_from}, delay {plan.delay:.1f} s")
        if self._pending:
            obstacle, others = self._pending.pop(0)
            self.on_obstacle(obstacle, t, others)

    def _report(self, t: float, decision: Decision) -> None:
        self._report_one(t, self.obstacle, decision.action.value)

    def _report_one(self, t: float, obstacle: Obstacle, action: str) -> None:
        if self.reporter is None or obstacle.obstacle_id in self._reported:
            return
        self._reported.add(obstacle.obstacle_id)
        self.reporter.report_obstacle(
            ObstacleReport.of(self.planner.aisle, obstacle, self.robot.robot_id, t, action)
        )
