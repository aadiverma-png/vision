"""ORCA-DD: reciprocal collision avoidance solved in the robot's command space.

Pipeline for one robot and one control period (MATH.md section 10):

1. Every neighbour gives an ORCA half-plane on the velocity of the effective
   centre ``P`` (RVO2 construction: cut-off circle, legs, or the overlap case).
   A cooperative neighbour takes half of the avoidance (responsibility 0.5),
   and a non-cooperative one (unknown object, other fleet, a robot that is
   braking or under emergency control) leaves all of it to us (1.0).
2. Lane walls and known obstacles give conservative static half-planes.
3. All of these are rotated into command space ``c = R(-th) dP/dt`` and
   joined with the motion model's hard limits (:mod:`visison.motion_models`).
4. The RVO2 2-D linear program finds the command closest to the preferred
   one. If the neighbour constraints cannot all be met, the 3-D program
   relaxes *only* them (least penetration); limits, walls and obstacles stay
   hard. If even the hard set is empty, the result is the model's maximum
   braking command.

Modes: ``NORMAL`` (full 2-D), ``PATH_SPEED_ONLY`` (the command may only be
scaled along the preferred one: used for scripted emergency motion and
lane changes next to obstacles) and ``OFF`` (braking: ORCA must not dodge).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from .motion_models import EPS, Command, Line, MotionModel, rotate


class OrcaMode(str, Enum):
    NORMAL = "normal"
    PATH_SPEED_ONLY = "path_speed_only"
    OFF = "off"


@dataclass(frozen=True)
class OrcaConfig:
    tau: float = 2.0  # s, horizon for other robots
    tau_obs: float = 1.0  # s, horizon for walls and static obstacles
    neighbour_range: float = 6.0  # m
    max_neighbours: int = 8
    radius_margin: float = 0.1  # m, added to every radius
    wall_margin: float = 0.02  # m
    headway: float = 0.4  # s, extra following distance per m/s of own speed (lane-locked, same direction)


@dataclass(frozen=True)
class OrcaAgent:
    agent_id: str
    p: tuple[float, float]  # effective centre, world frame
    p_dot: tuple[float, float]  # its velocity
    radius: float
    cooperative: bool = True


@dataclass(frozen=True)
class OrcaResult:
    cmd: Command
    feasible: bool  # every neighbour constraint holds
    relaxed: bool = False  # the 3-D program was needed
    braked: bool = False  # the hard limits left no room: maximum braking
    u: dict[str, float] = field(default_factory=dict)  # |u| of every neighbour constraint the preferred command broke

    @property
    def max_u(self) -> float:
        return max(self.u.values(), default=0.0)


# ------------------------------------------------------------------ vector helpers


def _det(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _mul(a, k):
    return a[0] * k, a[1] * k


def _norm(a):
    n = math.hypot(*a)
    return (a[0] / n, a[1] / n) if n > EPS else (0.0, 0.0)


# ------------------------------------------------------------ RVO2 linear programs


def _lp1(lines: Sequence[Line], no: int, radius: float, opt, direction_opt: bool):
    ln = lines[no]
    dot = _dot(ln.point, ln.direction)
    disc = dot * dot + radius * radius - _dot(ln.point, ln.point)
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    t_left, t_right = -dot - sq, -dot + sq
    for i in range(no):
        denom = _det(ln.direction, lines[i].direction)
        numer = _det(lines[i].direction, _sub(ln.point, lines[i].point))
        if abs(denom) <= EPS:
            if numer < 0.0:
                return None
            continue
        t = numer / denom
        if denom >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return None
    if direction_opt:
        t = t_right if _dot(opt, ln.direction) > 0.0 else t_left
    else:
        t = min(max(_dot(ln.direction, _sub(opt, ln.point)), t_left), t_right)
    return _add(ln.point, _mul(ln.direction, t))


def linear_program_2d(lines: Sequence[Line], radius: float, opt, direction_opt: bool = False):
    """Closest point to ``opt`` inside all half-planes and the disc of ``radius``.

    Returns ``(result, n_ok)``; ``n_ok < len(lines)`` means line ``n_ok`` made it infeasible.
    """
    if direction_opt:
        result = _mul(opt, radius)
    elif _dot(opt, opt) > radius * radius:
        result = _mul(_norm(opt), radius)
    else:
        result = opt
    for i, ln in enumerate(lines):
        if _det(ln.direction, _sub(ln.point, result)) > 0.0:
            new = _lp1(lines, i, radius, opt, direction_opt)
            if new is None:
                return result, i
            result = new
    return result, len(lines)


def linear_program_3d(lines: Sequence[Line], n_hard: int, begin: int, radius: float, result):
    """Least-penetration point: lines ``[0, n_hard)`` stay hard, the rest are relaxed evenly."""
    distance = 0.0
    for i in range(begin, len(lines)):
        li = lines[i]
        if _det(li.direction, _sub(li.point, result)) > distance:
            proj = list(lines[:n_hard])
            for j in range(n_hard, i):
                lj = lines[j]
                det = _det(li.direction, lj.direction)
                if abs(det) <= EPS:
                    if _dot(li.direction, lj.direction) > 0.0:
                        continue
                    point = _mul(_add(li.point, lj.point), 0.5)
                else:
                    point = _add(li.point, _mul(li.direction, _det(lj.direction, _sub(li.point, lj.point)) / det))
                proj.append(Line(point, _norm(_sub(lj.direction, li.direction))))
            new, n_ok = linear_program_2d(proj, radius, (-li.direction[1], li.direction[0]), True)
            if n_ok >= len(proj):
                result = new
            distance = _det(li.direction, _sub(li.point, result))
    return result


# ------------------------------------------------------------------ constraints


def orca_half_plane(a: OrcaAgent, b: OrcaAgent, tau: float, dt: float, responsibility: float = 0.5):
    """ORCA half-plane for ``a``'s velocity (world frame) and the full change vector ``u``."""
    rel_p = _sub(b.p, a.p)
    rel_v = _sub(a.p_dot, b.p_dot)
    dist_sq = _dot(rel_p, rel_p)
    r = a.radius + b.radius
    r_sq = r * r
    if dist_sq > r_sq:
        w = _sub(rel_v, _mul(rel_p, 1.0 / tau))
        w_len_sq = _dot(w, w)
        dot1 = _dot(w, rel_p)
        if dot1 < 0.0 and dot1 * dot1 > r_sq * w_len_sq:  # project on the cut-off circle
            w_len = math.sqrt(w_len_sq)
            unit = _mul(w, 1.0 / w_len)
            direction = (unit[1], -unit[0])
            u = _mul(unit, r / tau - w_len)
        else:  # project on a leg
            leg = math.sqrt(dist_sq - r_sq)
            if _det(rel_p, w) > 0.0:
                direction = ((rel_p[0] * leg - rel_p[1] * r) / dist_sq, (rel_p[0] * r + rel_p[1] * leg) / dist_sq)
            else:
                direction = (-(rel_p[0] * leg + rel_p[1] * r) / dist_sq, -(-rel_p[0] * r + rel_p[1] * leg) / dist_sq)
            u = _sub(_mul(direction, _dot(rel_v, direction)), rel_v)
    else:  # already overlapping: get apart within one time step
        w = _sub(rel_v, _mul(rel_p, 1.0 / dt))
        w_len = math.hypot(*w)
        unit = _mul(w, 1.0 / w_len) if w_len > EPS else _norm((-rel_p[0], -rel_p[1]))
        direction = (unit[1], -unit[0])
        u = _mul(unit, r / dt - w_len)
    return Line(_add(a.p_dot, _mul(u, responsibility)), direction, f"orca:{b.agent_id}"), u


def static_half_plane(p, radius: float, closest, tau_obs: float, name: str = "static") -> Line:
    """Do not approach ``closest`` (a point of an obstacle or wall) faster than the free distance allows."""
    n = _norm(_sub(p, closest))
    free = max(math.dist(p, closest) - radius, 0.0)
    # n . p_dot >= -free / tau_obs   <=>   (-n) . p_dot <= free / tau_obs
    return Line.from_inequality((-n[0], -n[1]), free / tau_obs, name)


def wall_half_planes(p, half_width: float, band: tuple[float, float], tau_obs: float) -> list[Line]:
    """Keep ``p`` laterally inside ``band`` (a y-interval), never pushing a robot that is already outside."""
    lo, hi = band[0] + half_width, band[1] - half_width
    return [
        Line.from_inequality((0.0, 1.0), max((hi - p[1]) / tau_obs, 0.0), "wall_high"),
        Line.from_inequality((0.0, -1.0), max((p[1] - lo) / tau_obs, 0.0), "wall_low"),
    ]


def closest_on_box(p, s_min, s_max, y_min, y_max):
    return min(max(p[0], s_min), s_max), min(max(p[1], y_min), y_max)


def _to_command_space(line: Line, th: float) -> Line:
    """``c = R(-th) p_dot``: rotating point and direction keeps the half-plane (rotations keep det)."""
    return Line(rotate(*line.point, -th), rotate(*line.direction, -th), line.name)


# ------------------------------------------------------------------- the solver


def _speed_only(c_pref, lines: Sequence[Line]) -> tuple[float, float] | None:
    """Interval of s in [0, 1] with ``s * c_pref`` inside every line, or None."""
    lo, hi = 0.0, 1.0
    for ln in lines:
        k = _det(ln.direction, c_pref)
        rhs = _det(ln.direction, ln.point)
        if abs(k) <= EPS:
            if rhs > EPS:
                return None
            continue
        if k > 0:
            lo = max(lo, rhs / k)
        else:
            hi = min(hi, rhs / k)
    return (lo, hi) if lo <= hi + 1e-12 else None


def solve(
    model: MotionModel,
    current: Command,
    preferred: Command,
    heading: float,
    ego: OrcaAgent,
    neighbours: Sequence[OrcaAgent],
    statics: Sequence[Line],
    cfg: OrcaConfig,
    dt: float,
    mode: OrcaMode = OrcaMode.NORMAL,
    allow_reverse: bool = False,
) -> OrcaResult:
    """The executable command closest to ``preferred`` that avoids ``neighbours`` for ``cfg.tau``."""
    if mode is OrcaMode.OFF:
        return OrcaResult(preferred, True)
    orca_lines, u = [], {}
    c_pref = model.to_c(preferred)
    for nb in neighbours:
        line, uu = orca_half_plane(ego, nb, cfg.tau, dt, 0.5 if nb.cooperative else 1.0)
        cl = _to_command_space(line, heading)
        orca_lines.append(cl)
        if cl.violation(c_pref) > 1e-9:
            u[nb.agent_id] = math.hypot(*uu)
    statics_c = [_to_command_space(s, heading) for s in statics]
    radius = 10.0 * (1.0 + math.hypot(*c_pref))
    best = None
    for hard_set in model.constraint_sets(current, dt, allow_reverse):
        hard = list(hard_set) + statics_c
        c_ok, n_ok = linear_program_2d(hard, radius, c_pref)  # the preferred command, made executable
        if n_ok < len(hard):
            continue
        if mode is OrcaMode.PATH_SPEED_ONLY:
            s_hard = _speed_only(c_ok, hard)
            if s_hard is None:
                continue
            s_all = _speed_only(c_ok, hard + orca_lines)
            if s_all is not None:
                cand, relaxed = _mul(c_ok, s_all[1]), False
            else:  # cannot slow down enough along the path this period: as slow as the limits allow
                cand, relaxed = _mul(c_ok, s_hard[0]), True
        else:
            lines = hard + orca_lines
            cand, n_ok = linear_program_2d(lines, radius, c_pref)
            relaxed = n_ok < len(lines)
            if relaxed:
                cand = linear_program_3d(lines, len(hard), n_ok, radius, cand)
        cost = math.dist(cand, c_pref) + (1e3 if relaxed else 0.0)
        if best is None or cost < best[0]:
            best = (cost, cand, relaxed)
    if best is None:
        return OrcaResult(model.brake_command(current, dt), False, True, True, u)
    _, cand, relaxed = best
    cmd = model.from_c(cand, preferred.w)
    clip_w = getattr(model, "clip_w", None)
    if clip_w is not None:
        cmd = Command(cmd.v, clip_w(current, cmd.w, dt), cmd.vy)
    return OrcaResult(cmd, not relaxed, relaxed, False, u)


def orca_1d(v_pref: float, v: float, gap: float, v_other: float, tau: float, responsibility: float = 0.5) -> float:
    """Lane-locked ORCA for one neighbour ahead: the largest allowed speed.

    The pair must not close the bumper ``gap`` within ``tau``: closing speed
    ``v - v_other <= gap / tau``. The excess is shared by ``responsibility``.
    """
    excess = (v - v_other) - gap / tau
    limit = v - responsibility * excess if excess > 0 else math.inf
    return min(v_pref, limit)
