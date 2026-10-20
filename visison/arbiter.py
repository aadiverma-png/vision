"""MotionArbiter: the only component that turns intents into a robot command.

Every layer above the robot says what it *wants*, as a :class:`MotionIntent`:
the planner and controller (``PLANNER``), a fleet manager such as Open-RMF
(``FLEET``, e.g. a pause), the emergency layer (``EMERGENCY``: brake, hold,
or scripted back-off / rejoin) and an operator (``OPERATOR``). The arbiter
keeps the latest intent of each source and, once per control tick:

1. picks the active intent by priority (OPERATOR > EMERGENCY > FLEET > PLANNER),
   dropping intents whose ``ttl`` has run out;
2. tracks its trajectory at a *time-scaled* reference time ``tau`` (with
   ORCA on): when the robot is slowed, the reference is delayed instead of
   demanding a catch-up surge, and ``lateness = t - tau`` is reported;
3. filters the tracker's command with ORCA-DD (skipped while braking;
   speed-only for scripted motion and lane-locked driving);
4. applies the safety gate: the command must satisfy the motion model's
   hard limits (otherwise it is projected onto them), and if odometry is
   stale the robot brakes;
5. publishes exactly one command.

A lower-priority intent that arrives while a higher one is active is stored,
not executed, and takes over when the higher one is released. When the
emergency layer releases a robot whose own motion (a rejoin, for example)
should simply continue, ``release(..., handover=True)`` turns that motion
into the planner's intent, unless the planner has asked for something newer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Callable, Sequence

from .motion_models import Command, Line, MotionModel, validate_command
from .orca import OrcaAgent, OrcaConfig, OrcaMode, OrcaResult, linear_program_2d, solve
from .planner import Trajectory
from .tracking import UnicycleTracker, reference

log = logging.getLogger("visison.arbiter")


class Source(IntEnum):
    PLANNER = 1
    FLEET = 2
    EMERGENCY = 3
    OPERATOR = 4


class Kind(str, Enum):
    FOLLOW = "follow"
    HOLD = "hold"
    BRAKE = "brake"
    SCRIPTED = "scripted"  # emergency motion that may reverse (back-off, rejoin)


@dataclass
class MotionIntent:
    source: Source
    kind: Kind
    trajectory: Trajectory | None = None
    stamp: float = 0.0
    ttl: float = math.inf
    seq: int = 0

    def alive(self, t: float) -> bool:
        return t <= self.stamp + self.ttl


# (ego, neighbours, static lines, suggested mode) for one robot at time t
NeighbourProvider = Callable[[str, float, MotionIntent], tuple[OrcaAgent, Sequence[OrcaAgent], Sequence[Line], OrcaMode]]


@dataclass
class ArbiterOutput:
    t: float
    cmd: Command
    intent: MotionIntent | None
    orca: OrcaResult | None = None
    lateness: float = 0.0
    gated: list[str] = field(default_factory=list)  # limits the gate had to enforce


class MotionArbiter:
    def __init__(
        self,
        robot_id: str,
        model: MotionModel,
        dt: float,
        tracker: UnicycleTracker | None = None,
        neighbours: NeighbourProvider | None = None,
        orca_cfg: OrcaConfig | None = None,
        orca_enabled: bool = False,
        publish: Callable[[Command], None] | None = None,
        odom_timeout: float | None = None,
    ):
        self.robot_id, self.model, self.dt = robot_id, model, dt
        self.tracker = tracker
        self.neighbours = neighbours
        self.orca_cfg = orca_cfg or OrcaConfig()
        self.orca_enabled = orca_enabled
        self.publish = publish
        self.odom_timeout = odom_timeout if odom_timeout is not None else 2 * dt
        self.intents: dict[Source, MotionIntent] = {}
        self._tau: dict[int, float] = {}  # virtual reference time per intent (by seq)
        self._seq = 0
        self._active_seq: int | None = None
        self.last: ArbiterOutput | None = None
        self.log: list[tuple[float, str]] = []  # rejected / shadowed requests and switches

    # intents -------------------------------------------------------------------

    def request(self, intent: MotionIntent, t: float) -> bool:
        """Store ``intent``. Returns True if it is the active one now."""
        self._seq += 1
        intent.seq = self._seq
        intent.stamp = t if intent.stamp == 0.0 else intent.stamp
        self.intents[intent.source] = intent
        self._tau[intent.seq] = intent.stamp
        active = self.active(t)
        if active is not intent:
            self.log.append((t, f"{intent.source.name} {intent.kind.value} stored behind "
                                f"{active.source.name} {active.kind.value}"))
            return False
        return True

    def release(self, source: Source, t: float, handover: bool = False) -> None:
        """Drop ``source``'s intent. With ``handover``, its trajectory continues as the planner's."""
        old = self.intents.pop(source, None)
        if old is None or not handover or old.trajectory is None:
            return
        planner = self.intents.get(Source.PLANNER)
        if planner is None or planner.seq < old.seq:
            new = MotionIntent(Source.PLANNER, Kind.FOLLOW, old.trajectory, old.stamp)
            self._seq += 1
            new.seq = self._seq
            self._tau[new.seq] = self._tau.get(old.seq, t)
            self.intents[Source.PLANNER] = new

    def active(self, t: float) -> MotionIntent | None:
        live = [i for i in self.intents.values() if i.alive(t)]
        return max(live, key=lambda i: (i.source, i.seq)) if live else None

    def holder(self, t: float) -> Source | None:
        a = self.active(t)
        return a.source if a else None

    def trajectory(self, t: float) -> Trajectory | None:
        a = self.active(t)
        return a.trajectory if a else None

    def reference_time(self, t: float) -> float:
        a = self.active(t)
        return self._tau.get(a.seq, t) if a is not None and self.orca_enabled else t

    @property
    def lateness(self) -> float:
        return self.last.lateness if self.last else 0.0

    # control tick ----------------------------------------------------------------

    def step(self, t: float, pose: tuple[float, float, float], current: Command,
             odom_age: float = 0.0) -> Command:
        """One control tick: pose is (x, y, heading), current is the measured (v, w)."""
        intent = self.active(t)
        if intent is not None and intent.seq != self._active_seq:
            self.log.append((t, f"active: {intent.source.name} {intent.kind.value}"))
            self._active_seq = intent.seq
        orca_res, gated = None, []
        tau = self._tau.get(intent.seq, t) if intent is not None else t
        brake = (intent is None or intent.kind is Kind.BRAKE or intent.trajectory is None
                 or odom_age > self.odom_timeout)
        if brake:
            cmd = self.model.brake_command(current, self.dt)
            if odom_age > self.odom_timeout:
                gated.append("stale_odometry")
        else:
            ref_t = tau if self.orca_enabled else t
            x_r, y_r, th_r, v_r, w_r = reference(intent.trajectory, ref_t)
            if self.tracker is not None:
                v, w = self.tracker.command(pose[0], pose[1], pose[2], x_r, y_r, th_r, v_r, w_r)
            else:
                v, w = v_r, w_r
            cmd = Command(v, w)
            allow_reverse = intent.kind is Kind.SCRIPTED or v_r < 0
            if self.orca_enabled and self.neighbours is not None:
                ego, nbs, statics, mode = self.neighbours(self.robot_id, t, intent)
                if intent.kind is Kind.SCRIPTED and mode is OrcaMode.NORMAL:
                    mode = OrcaMode.PATH_SPEED_ONLY
                orca_res = solve(self.model, current, cmd, pose[2], ego, nbs, statics, self.orca_cfg,
                                 self.dt, mode, allow_reverse)
                cmd = orca_res.cmd
            cmd, gated = self._gate(current, cmd, allow_reverse)
            if self.orca_enabled:  # time-scaled reference: advance as fast as the robot really progresses
                speed_ref = abs(v_r)
                kappa = 1.0 if speed_ref < 0.05 else min(max(abs(current.v) / speed_ref, 0.0), 1.0)
                if math.hypot(x_r - pose[0], y_r - pose[1]) < 0.02:
                    kappa = 1.0  # on the reference: time runs normally
                self._tau[intent.seq] = tau + self.dt * kappa
        lateness = (t - self._tau.get(intent.seq, t)) if (intent is not None and self.orca_enabled) else 0.0
        self.last = ArbiterOutput(t, cmd, intent, orca_res, max(lateness, 0.0), gated)
        if self.publish is not None:
            self.publish(cmd)
        return cmd

    def _gate(self, current: Command, cmd: Command, allow_reverse: bool) -> tuple[Command, list[str]]:
        """Project onto the model's hard limits if anything is broken (never adds speed unless forced)."""
        broken = validate_command(self.model, current, cmd, self.dt, allow_reverse)
        if not broken:
            return cmd, []
        c = self.model.to_c(cmd)
        best = None
        for hard in self.model.constraint_sets(current, self.dt, allow_reverse):
            cand, n_ok = linear_program_2d(hard, 1e3, c)
            if n_ok == len(hard):
                d = math.dist(cand, c)
                if best is None or d < best[0]:
                    best = (d, cand)
        if best is None:
            return self.model.brake_command(current, self.dt), broken + ["brake"]
        out = self.model.from_c(best[1], cmd.w)
        clip_w = getattr(self.model, "clip_w", None)
        if clip_w is not None:
            out = Command(out.v, clip_w(current, out.w, self.dt), out.vy)
        return out, broken
