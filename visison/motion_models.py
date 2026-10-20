"""Kinematic models for ORCA-DD (MATH.md section 10).

ORCA reasons about a holonomic point. A differential-drive or car-like
robot cannot move its axle centre sideways, but a point ``P`` at distance
``D > 0`` in front of the axle *can* move in any direction:

    P     = (x, y) + D * (cos th, sin th)
    dP/dt = R(th) @ (v, D*w)            R(th) = rotation by the heading

so the velocity of ``P`` and the command ``(v, w)`` are related by an exact,
invertible linear map (Snape et al., "Smooth and collision-free navigation
for multiple robots under differential-drive constraints", 2010). Collision
avoidance is solved directly for the *command* ``c = (v, D*w)``: every ORCA
half-plane on ``dP/dt`` becomes a half-plane on ``c``, and the robot's
limits (wheel speeds, acceleration, turn rate, turning radius, reverse) are
more half-planes on ``c``. The answer is therefore always executable; there
is no after-the-fact holonomic-to-unicycle conversion.

An omnidirectional robot uses ``D = 0`` and ``c = (vx, vy)`` in its body frame.

Half-planes are :class:`Line` objects (RVO2 convention): the feasible side
is to the *left* of ``direction``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

EPS = 1e-9


@dataclass(frozen=True)
class Line:
    """Half-plane ``det(direction, c - point) >= 0``."""

    point: tuple[float, float]
    direction: tuple[float, float]
    name: str = ""

    @classmethod
    def from_inequality(cls, a: tuple[float, float], b: float, name: str = "") -> Line:
        """The half-plane ``a . c <= b``."""
        n = math.hypot(*a)
        ax, ay = a[0] / n, a[1] / n
        return cls((ax * b / n, ay * b / n), (-ay, ax), name)

    def violation(self, c: tuple[float, float]) -> float:
        """How far ``c`` is outside (> 0) or inside (<= 0) the half-plane."""
        dx, dy = self.direction
        return dx * (self.point[1] - c[1]) - dy * (self.point[0] - c[0])


@dataclass(frozen=True)
class Command:
    """A motion command. ``vy`` is only used by omnidirectional robots."""

    v: float
    w: float
    vy: float = 0.0


def rotate(x: float, y: float, th: float) -> tuple[float, float]:
    c, s = math.cos(th), math.sin(th)
    return c * x - s * y, s * x + c * y


def _window(v0: float, up: float, down: float) -> tuple[float, float]:
    """Speeds reachable within one period from ``v0``: slower by ``down``, faster by ``up``."""
    if v0 > EPS:
        return v0 - down, v0 + up
    if v0 < -EPS:
        return v0 - up, v0 + down
    return -up, up


def _interval_lines(lo: float, hi: float, axis: int, name: str) -> list[Line]:
    a_hi = (1.0, 0.0) if axis == 0 else (0.0, 1.0)
    a_lo = (-1.0, 0.0) if axis == 0 else (0.0, -1.0)
    return [Line.from_inequality(a_hi, hi, name + "_max"), Line.from_inequality(a_lo, -lo, name + "_min")]


def _clip_interval(lim: tuple[float, float], win: tuple[float, float]) -> tuple[float, float]:
    """``lim`` intersected with the reachable window; if they miss, the reachable point nearest to ``lim``."""
    lo, hi = max(lim[0], win[0]), min(lim[1], win[1])
    if lo <= hi:
        return lo, hi
    x = win[1] if win[1] < lim[0] else win[0]
    return x, x


class MotionModel(Protocol):
    D: float  # distance of the effective centre ahead of the rotation centre (0 for omni)

    def effective_radius(self, length: float, width: float, margin: float = 0.0) -> float: ...

    def to_c(self, cmd: Command) -> tuple[float, float]: ...

    def from_c(self, c: tuple[float, float], w_pref: float = 0.0) -> Command: ...

    def constraint_sets(self, cur: Command, dt: float, allow_reverse: bool) -> list[list[Line]]:
        """Convex sets of hard half-planes on ``c``. A command is valid if it is in any one of them."""
        ...

    def brake_command(self, cur: Command, dt: float) -> Command: ...

    def validate(self, cur: Command, cmd: Command, dt: float, allow_reverse: bool) -> list[str]: ...


def _corner_radius(D: float, length: float, width: float, margin: float) -> float:
    """Largest distance from the effective centre to a corner of a footprint centred on the axle."""
    return max(math.hypot(sx * length / 2 - D, sy * width / 2) for sx in (-1, 1) for sy in (-1, 1)) + margin


@dataclass(frozen=True)
class DiffDrive:
    track_width: float
    v_wheel_max: float
    a_acc: float
    a_dec: float
    alpha_max: float  # rad/s^2
    w_max: float  # rad/s
    v_rev_max: float = 0.2  # only used when reversing is allowed (scripted back-off)
    D: float = 0.15

    def effective_radius(self, length, width, margin=0.0):
        return _corner_radius(self.D, length, width, margin)

    def to_c(self, cmd):
        return cmd.v, self.D * cmd.w

    def from_c(self, c, w_pref=0.0):
        return Command(c[0], c[1] / self.D)

    def _limit_lines(self, cur, dt, allow_reverse):
        D, k = self.D, self.track_width / (2 * self.D)
        vw = self.v_wheel_max
        lines = [  # |v +- (b/2) w| <= v_wheel_max, written in c = (v, D w)
            Line.from_inequality((1.0, k), vw, "wheel_right"),
            Line.from_inequality((-1.0, -k), vw, "wheel_right"),
            Line.from_inequality((1.0, -k), vw, "wheel_left"),
            Line.from_inequality((-1.0, k), vw, "wheel_left"),
        ]
        v_rev = self.v_rev_max if allow_reverse else 0.0
        v_lo, v_hi = _clip_interval((-v_rev, math.inf), _window(cur.v, self.a_acc * dt, self.a_dec * dt))
        w_lo, w_hi = _clip_interval((-self.w_max, self.w_max),
                                    (cur.w - self.alpha_max * dt, cur.w + self.alpha_max * dt))
        lines += _interval_lines(v_lo, v_hi, 0, "v")
        lines += _interval_lines(D * w_lo, D * w_hi, 1, "w")
        return lines

    def constraint_sets(self, cur, dt, allow_reverse):
        return [self._limit_lines(cur, dt, allow_reverse)]

    def brake_command(self, cur, dt):
        dv = self.a_dec * dt
        v = max(cur.v - dv, 0.0) if cur.v >= 0 else min(cur.v + dv, 0.0)
        dw = self.alpha_max * dt
        w = max(cur.w - dw, 0.0) if cur.w >= 0 else min(cur.w + dw, 0.0)
        return Command(v, w)

    def validate(self, cur, cmd, dt, allow_reverse, tol=1e-6):
        out = []
        for side, s in (("wheel_left", -1), ("wheel_right", 1)):
            if abs(cmd.v + s * cmd.w * self.track_width / 2) > self.v_wheel_max + tol:
                out.append(side)
        up, down = self.a_acc * dt, self.a_dec * dt
        lo, hi = _window(cur.v, up, down)
        if not lo - tol <= cmd.v <= hi + tol:
            out.append("accel")
        if abs(cmd.w - cur.w) > self.alpha_max * dt + tol:
            out.append("alpha")
        if abs(cmd.w) > self.w_max + tol and abs(cmd.w) > abs(cur.w) - self.alpha_max * dt + tol:
            out.append("w_max")
        v_rev = self.v_rev_max if allow_reverse else 0.0
        if cmd.v < -v_rev - tol and cmd.v < hi - tol:  # reversing faster than allowed and not braking out of it
            out.append("reverse")
        return out


@dataclass(frozen=True)
class Ackermann:
    wheelbase: float
    R_min: float  # minimum turning radius
    v_max: float
    a_acc: float
    a_dec: float
    yaw_accel_max: float  # rad/s^2, stands in for the steering-rate limit
    v_rev_max: float = 0.2
    D: float = 0.3

    def effective_radius(self, length, width, margin=0.0):
        return _corner_radius(self.D, length, width, margin)

    def to_c(self, cmd):
        return cmd.v, self.D * cmd.w

    def from_c(self, c, w_pref=0.0):
        return Command(c[0], c[1] / self.D)

    def constraint_sets(self, cur, dt, allow_reverse):
        k = self.D / self.R_min  # |D w| <= k |v|
        v_lo, v_hi = _clip_interval((-(self.v_rev_max if allow_reverse else 0.0), self.v_max),
                                    _window(cur.v, self.a_acc * dt, self.a_dec * dt))
        w_lo, w_hi = cur.w - self.yaw_accel_max * dt, cur.w + self.yaw_accel_max * dt
        common = _interval_lines(v_lo, v_hi, 0, "v") + _interval_lines(self.D * w_lo, self.D * w_hi, 1, "w")
        sets = []
        if v_hi >= 0:  # forward: v >= 0, -k v <= D w <= k v
            sets.append(common + [Line.from_inequality((-1.0, 0.0), 0.0, "forward"),
                                  Line.from_inequality((-k, 1.0), 0.0, "turn_radius"),
                                  Line.from_inequality((-k, -1.0), 0.0, "turn_radius")])
        if v_lo <= 0:  # reverse: v <= 0, k v <= D w <= -k v
            sets.append(common + [Line.from_inequality((1.0, 0.0), 0.0, "reverse"),
                                  Line.from_inequality((k, 1.0), 0.0, "turn_radius"),
                                  Line.from_inequality((k, -1.0), 0.0, "turn_radius")])
        return sets

    def brake_command(self, cur, dt):
        dv = self.a_dec * dt
        v = max(cur.v - dv, 0.0) if cur.v >= 0 else min(cur.v + dv, 0.0)
        return Command(v, cur.w * (v / cur.v) if abs(cur.v) > EPS else 0.0)  # keep the curvature

    def validate(self, cur, cmd, dt, allow_reverse, tol=1e-6):
        out = []
        lo, hi = _window(cur.v, self.a_acc * dt, self.a_dec * dt)
        if not lo - tol <= cmd.v <= hi + tol:
            out.append("accel")
        if cmd.v > self.v_max + tol:
            out.append("v_max")
        if abs(cmd.w) > abs(cmd.v) / self.R_min + tol:
            out.append("turn_radius")
        if abs(cmd.w - cur.w) > self.yaw_accel_max * dt + tol:
            out.append("steer_rate")
        if cmd.v < -(self.v_rev_max if allow_reverse else 0.0) - tol and cmd.v < hi - tol:
            out.append("reverse")
        return out


@dataclass(frozen=True)
class Omni:
    vx_max: float
    vy_max: float
    w_max: float
    a_max: float
    alpha_max: float
    v_rev_max: float = 0.2
    D: float = 0.0

    def effective_radius(self, length, width, margin=0.0):
        return math.hypot(length / 2, width / 2) + margin

    def to_c(self, cmd):
        return cmd.v, cmd.vy

    def from_c(self, c, w_pref=0.0):
        return Command(c[0], w_pref, c[1])

    def clip_w(self, cur, w, dt):
        """Rotation is independent of translation for an omni robot: only its own limits apply."""
        lo, hi = _clip_interval((-self.w_max, self.w_max), (cur.w - self.alpha_max * dt, cur.w + self.alpha_max * dt))
        return min(max(w, lo), hi)

    def constraint_sets(self, cur, dt, allow_reverse):
        a = self.a_max * dt
        vx = _clip_interval((-(self.v_rev_max if allow_reverse else 0.0), self.vx_max), (cur.v - a, cur.v + a))
        vy = _clip_interval((-self.vy_max, self.vy_max), (cur.vy - a, cur.vy + a))
        return [_interval_lines(*vx, 0, "vx") + _interval_lines(*vy, 1, "vy")]

    def brake_command(self, cur, dt):
        a = self.a_max * dt
        sh = lambda x: max(x - a, 0.0) if x >= 0 else min(x + a, 0.0)  # noqa: E731
        return Command(sh(cur.v), sh(cur.w), sh(cur.vy))

    def validate(self, cur, cmd, dt, allow_reverse, tol=1e-6):
        out = []
        a = self.a_max * dt
        if abs(cmd.v - cur.v) > a + tol or abs(cmd.vy - cur.vy) > a + tol:
            out.append("accel")
        if cmd.v > self.vx_max + tol or abs(cmd.vy) > self.vy_max + tol:
            out.append("v_max")
        if abs(cmd.w) > self.w_max + tol:
            out.append("w_max")
        if cmd.v < -(self.v_rev_max if allow_reverse else 0.0) - tol and cmd.v < cur.v + a - tol:  # window top
            out.append("reverse")
        return out


def validate_command(model: MotionModel, cur: Command, cmd: Command, dt: float,
                     allow_reverse: bool = False) -> list[str]:
    """Every limit of ``model`` that ``cmd`` breaks when issued from ``cur`` for one period ``dt``."""
    return model.validate(cur, cmd, dt, allow_reverse)
