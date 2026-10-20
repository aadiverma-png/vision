"""Lidar-based collision prediction for emergency braking (MATH.md section 8).

Every robot tracks what its lidar sees (:class:`ObjectTracker`) and rates
each track with :func:`collision_risk`. The risk is the probability that
the two bodies still touch after both have braked as hard as they can:

    g      = |s_o - s_e| - (L_e + L_o) / 2                 bumper gap
    x      = v*t_r + v*|v| / (2 b)                         signed travel until stopped
    d_stop = sign(s_o - s_e) * (x_e - x_o)                 gap used up while both stop
    m      = g - d_stop - min_gap                          margin left
    P      = Phi(-m / sigma)                               collision probability

The ego reacts after ``t_re`` (lidar period + compute). A cooperative robot
(known to the fleet) coming towards the ego brakes after the extra fleet
message latency, ``t_ro = t_re + comm_latency``. An object moving away is assumed to
brake at once as hard as it can, which is the worst case for a follower. An unknown object
coming towards the ego is assumed to keep going: ``x_o = v_o * t_stop_ego``.
Only pairs that are closing (``v_c > 0``) are rated.
``sigma`` grows with the time it takes to stop, because a velocity error
of ``sigma_v`` adds up over that time.

This module only rates risk. :mod:`visison.standoff` decides who brakes,
who reverses and who goes first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Sequence

from .geometry import Obstacle


@dataclass(frozen=True)
class EmergencyConfig:
    # perception and reaction
    scan_period: float = 0.1  # s, lidar period
    compute_latency: float = 0.05  # s, tracking + risk evaluation
    comm_latency: float = 0.2  # s, fleet message to the other robot
    sigma_pos: float = 0.05  # m, position error of a lidar track
    sigma_vel: float = 0.10  # m/s, velocity error of a lidar track
    # risk
    min_gap: float = 0.15  # m, bumper gap that must remain after both have stopped
    p_warn: float = 0.05
    p_brake: float = 0.25
    p_hard: float = 0.8  # at or above this, always brake, even if ORCA-DD could still avoid
    lookahead: float = 2.0  # s, how far along its own plan the ego's lateral band reaches
    lateral_clearance: float = 0.05  # m, added to the ego's half width
    associate_gate: float = 0.6  # m, a track this close to a fleet robot's body is that robot
    # standoff resolution
    creep_speed: float = 0.2  # m/s, speed of the rejoin manoeuvre
    reverse_speed: float = 0.2  # m/s, speed of robots backing off
    max_reverse: float = 6.0  # m, no robot is asked to back off further
    standstill_speed: float = 0.03  # m/s, below this a robot counts as stopped
    brake_timeout: float = 10.0  # s, resolve anyway if someone does not report standstill
    restart_gap: float = 1.0  # m, free space a stopped robot needs ahead before it resumes
    deadlock_time: float = 3.0  # s, facing robots stopped this long form a deadlock
    deadlock_gap: float = 3.0  # m
    cascade_range: float = 12.0  # m, robots driving into a braking group within this range brake too
    unknown_clear_time: float = 1.0  # s, an unknown object must be gone this long
    standoff_timeout: float = 60.0  # s, then escalate to the fleet / an operator
    escalate_retry: float = 5.0  # s between re-evaluations of an escalated standoff
    max_attempts: int = 3  # emergency brakes within one standoff before it is escalated
    late_tolerance: float = 1.0  # s a cruising robot may run behind its reference before its bookings move

    @property
    def t_react_ego(self) -> float:
        return self.scan_period + self.compute_latency

    @property
    def t_react_other(self) -> float:
        return self.t_react_ego + self.comm_latency


class RiskLevel(str, Enum):
    NONE = "none"
    AVOID = "avoid"  # would be BRAKE, but ORCA-DD is already avoiding it within the robots' limits
    WARN = "warn"
    BRAKE = "brake"


@dataclass(frozen=True)
class TrackedObject:
    """Something seen by the lidar (or the ego itself), in the aisle frame."""

    s: float
    y: float
    vs: float = 0.0  # ds/dt
    vy: float = 0.0
    length: float = 0.1  # visible extent along s
    width: float = 0.1
    track_id: str = ""

    def band(self) -> tuple[float, float]:
        return self.y - self.width / 2, self.y + self.width / 2


@dataclass(frozen=True)
class RiskAssessment:
    ego_id: str
    other_id: str  # a fleet robot id, or ``track:<id>`` for an unknown object
    level: RiskLevel
    probability: float
    gap: float
    closing_speed: float
    ttc: float
    stop_distance: float
    margin: float
    other_is_robot: bool = True

    @property
    def pair(self) -> tuple[str, str]:
        return self.ego_id, self.other_id


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def collision_risk(
    ego: TrackedObject,
    ego_band: tuple[float, float],
    ego_decel: float,
    obj: TrackedObject,
    obj_decel: float | None,
    cfg: EmergencyConfig,
    ego_id: str = "",
    other_id: str = "",
) -> RiskAssessment:
    """Probability that ``ego`` and ``obj`` collide if both brake now.

    ``obj_decel=None`` means the object does not brake (unknown object).
    """
    none = RiskAssessment(ego_id, other_id, RiskLevel.NONE, 0.0, math.inf, 0.0, math.inf, 0.0,
                          math.inf, obj_decel is not None)
    lo, hi = obj.band()
    if not (lo < ego_band[1] and ego_band[0] < hi):
        return none
    ds = obj.s - ego.s
    side = 1.0 if ds >= 0 else -1.0
    gap = abs(ds) - (ego.length + obj.length) / 2
    v_ec = side * ego.vs  # > 0: ego moves towards the object
    v_oc = -side * obj.vs  # > 0: object moves towards the ego
    v_c = v_ec + v_oc
    if v_c <= 1e-3:  # standing still or moving apart: nothing to brake for
        return RiskAssessment(ego_id, other_id, RiskLevel.NONE, 0.0, gap, v_c, math.inf, 0.0,
                              gap - cfg.min_gap, obj_decel is not None)
    if gap <= cfg.min_gap:
        return RiskAssessment(ego_id, other_id, RiskLevel.BRAKE, 1.0, gap, v_c, max(gap, 0.0) / v_c,
                              0.0, gap - cfg.min_gap, obj_decel is not None)
    # signed displacement along s of each body until it has stopped
    ve, vo = ego.vs, obj.vs
    t_se = cfg.t_react_ego + abs(ve) / ego_decel
    x_e = ve * cfg.t_react_ego + ve * abs(ve) / (2 * ego_decel)
    if v_oc <= 0:  # moving away: the worst case is that it brakes at once, as hard as it can
        b = obj_decel or ego_decel
        t_so = abs(vo) / b
        x_o = vo * abs(vo) / (2 * b)
    elif obj_decel:  # cooperative robot coming towards us: brakes after the fleet message
        t_so = cfg.t_react_other + abs(vo) / obj_decel
        x_o = vo * cfg.t_react_other + vo * abs(vo) / (2 * obj_decel)
    else:  # unknown object coming towards us: assume it keeps going
        t_so = t_se
        x_o = vo * t_se
    d_stop = side * (x_e - x_o)  # how much of the gap is used up
    margin = gap - d_stop - cfg.min_gap
    sigma = math.hypot(cfg.sigma_pos, cfg.sigma_vel * max(t_se, t_so))
    p = _phi(-margin / sigma)
    level = RiskLevel.BRAKE if p >= cfg.p_brake else RiskLevel.WARN if p >= cfg.p_warn else RiskLevel.NONE
    return RiskAssessment(ego_id, other_id, level, p, gap, v_c, gap / v_c, d_stop, margin,
                          obj_decel is not None)


# --------------------------------------------------------------------- tracking


@dataclass
class _Track:
    track_id: str
    s: float
    y: float
    length: float
    width: float
    t: float
    vs: float = 0.0
    vy: float = 0.0
    hits: int = 1
    missed: int = 0


@dataclass
class ObjectTracker:
    """Nearest-neighbour association plus an alpha-beta filter per track.

    ``update`` takes lidar clusters as ``(s, y, length, width)`` and returns
    the confirmed tracks (seen at least twice) with estimated velocities.
    """

    alpha: float = 0.6
    beta: float = 0.3
    gate: float = 1.0  # m, predicted-to-measured distance for association
    max_missed: int = 3
    prefix: str = "T"
    tracks: dict[str, _Track] = field(default_factory=dict)
    _next: int = 0

    def update(self, t: float, measurements: Iterable[Sequence[float]]) -> list[TrackedObject]:
        meas = [tuple(m) for m in measurements]
        free = set(self.tracks)
        pairs = []
        for i, (s, y, *_rest) in enumerate(meas):
            for tid in free:
                tr = self.tracks[tid]
                dt = max(t - tr.t, 0.0)
                d = math.hypot(tr.s + tr.vs * dt - s, tr.y + tr.vy * dt - y)
                if d <= self.gate:
                    pairs.append((d, i, tid))
        used_m, used_t = set(), set()
        for d, i, tid in sorted(pairs):
            if i in used_m or tid in used_t:
                continue
            used_m.add(i)
            used_t.add(tid)
            self._correct(self.tracks[tid], t, meas[i])
        for tid in free - used_t:
            self.tracks[tid].missed += 1
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]
        for i, m in enumerate(meas):
            if i not in used_m:
                self._next += 1
                tid = f"{self.prefix}{self._next}"
                self.tracks[tid] = _Track(tid, m[0], m[1], m[2], m[3], t)
        return self.confirmed()

    def confirmed(self) -> list[TrackedObject]:
        return [
            TrackedObject(tr.s, tr.y, tr.vs, tr.vy, tr.length, tr.width, tr.track_id)
            for tr in self.tracks.values()
            if tr.hits >= 2 and tr.missed == 0
        ]

    def _correct(self, tr: _Track, t: float, m) -> None:
        dt = t - tr.t
        if dt <= 1e-9:
            return
        s, y, length, width = m
        if tr.hits == 1:  # two-point initialisation
            tr.vs, tr.vy = (s - tr.s) / dt, (y - tr.y) / dt
            tr.s, tr.y = s, y
        else:
            ps, py = tr.s + tr.vs * dt, tr.y + tr.vy * dt
            rs, ry = s - ps, y - py
            tr.s, tr.y = ps + self.alpha * rs, py + self.alpha * ry
            tr.vs += self.beta * rs / dt
            tr.vy += self.beta * ry / dt
        tr.length, tr.width, tr.t = length, width, t
        tr.hits += 1
        tr.missed = 0


# ---------------------------------------------------------------------- monitor


def matches_obstacle(track: TrackedObject, obstacle: Obstacle, cfg: EmergencyConfig, pad: float = 0.3) -> bool:
    """A track inside a known obstacle's box (plus ``pad``) is that obstacle.

    Its speed is not checked: the centre of a static box's lidar cluster
    "moves" while a robot drives past and sees another face of it. Fleet robots
    are identified before this check, so they are never hidden by it.
    """
    return (
        obstacle.s_min - pad <= track.s <= obstacle.s_max + pad
        and obstacle.y_min - pad <= track.y <= obstacle.y_max + pad
    )


@dataclass
class EmergencyMonitor:
    """One robot's collision watch. Stateless apart from its configuration."""

    robot_id: str
    decel: float  # the ego's emergency deceleration
    cfg: EmergencyConfig = field(default_factory=EmergencyConfig)

    def assess(
        self,
        ego: TrackedObject,
        ego_band: tuple[float, float],
        tracks: Iterable[TrackedObject],
        identify: Callable[[TrackedObject], str | None] = lambda tr: None,
        decel_of: Callable[[str], float | None] = lambda rid: None,
        ignore: Sequence[Obstacle] = (),
    ) -> list[RiskAssessment]:
        """Rate every track. ``identify`` maps a track to a fleet robot id (or ``None``)."""
        out = []
        for tr in tracks:
            rid = identify(tr)
            if rid == self.robot_id:
                continue
            if rid is None and any(matches_obstacle(tr, ob, self.cfg) for ob in ignore):
                continue
            other = rid if rid is not None else f"track:{tr.track_id}"
            out.append(collision_risk(ego, ego_band, self.decel, tr,
                                      decel_of(rid) if rid is not None else None,
                                      self.cfg, self.robot_id, other))
        return out
