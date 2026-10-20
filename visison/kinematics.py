"""1-D motion profiles and lateral lane-change models.

Longitudinal motion is described in the robot's *travel frame* ``u``. Here
``u`` grows in the direction the robot drives and speed is ``v >= 0``. A
:class:`Profile` is a chain of constant-acceleration segments. It can be
evaluated at any time and inverted at any position; the inverse is used to
find when the robot's footprint enters and leaves each reservation cell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

EPS = 1e-9


class InfeasibleMotion(ValueError):
    """The requested move cannot be done within the robot's limits."""


@dataclass(frozen=True)
class MotionLimits:
    v_max: float  # m/s
    accel: float  # m/s^2, positive
    decel: float  # m/s^2, positive (braking capability)
    emergency_decel: float | None = None  # m/s^2, hardest braking; None means ``decel``

    @property
    def e_decel(self) -> float:
        return self.emergency_decel or self.decel


@dataclass(frozen=True)
class Segment:
    t0: float
    u0: float
    v0: float
    a: float
    dt: float

    @property
    def t1(self) -> float:
        return self.t0 + self.dt

    @property
    def u1(self) -> float:
        return self.u0 + self.v0 * self.dt + 0.5 * self.a * self.dt**2

    @property
    def v1(self) -> float:
        return max(0.0, self.v0 + self.a * self.dt)

    def at(self, t: float) -> tuple[float, float]:
        tau = t - self.t0
        return self.u0 + self.v0 * tau + 0.5 * self.a * tau**2, max(0.0, self.v0 + self.a * tau)


def trapezoid(
    d: float, v0: float, v_end: float, v_max: float, accel: float, decel: float
) -> list[tuple[float, float]]:
    """Time-optimal ``(accel, duration)`` pieces that cover distance ``d``.

    The speed goes from ``v0`` to ``v_end`` without exceeding ``v_max``. The
    peak speed of a triangular profile follows from the energy balance
    ``vp^2 = (2*a*b*d + b*v0^2 + a*v_end^2) / (a + b)``, where ``a`` is
    ``accel`` and ``b`` is ``decel``.
    """
    if d < -EPS:
        raise InfeasibleMotion(f"negative distance {d:.3f}")
    d = max(d, 0.0)
    if v0 > v_max + 1e-6 or v_end > v_max + 1e-6:
        raise InfeasibleMotion("boundary speed above v_max")
    if v_end > v0 and v_end**2 - v0**2 > 2 * accel * d + 1e-6:
        raise InfeasibleMotion("not enough distance to reach the target speed")
    if v0 > v_end and v0**2 - v_end**2 > 2 * decel * d + 1e-6:
        raise InfeasibleMotion("not enough distance to brake")
    vp2 = (2 * accel * decel * d + decel * v0**2 + accel * v_end**2) / (accel + decel)
    vp = max(min(math.sqrt(max(vp2, 0.0)), v_max), v0, v_end)
    d_acc = (vp**2 - v0**2) / (2 * accel)
    d_dec = (vp**2 - v_end**2) / (2 * decel)
    d_cruise = max(d - d_acc - d_dec, 0.0)
    return [
        (accel, (vp - v0) / accel),
        (0.0, d_cruise / vp if vp > EPS else 0.0),
        (-decel, (vp - v_end) / decel),
    ]


def travel_time(d: float, v0: float, v_end: float, limits: MotionLimits) -> float:
    """Minimum time to cover ``d`` metres going from ``v0`` to ``v_end``."""
    return sum(dt for _, dt in trapezoid(d, v0, v_end, limits.v_max, limits.accel, limits.decel))


class Profile:
    """Piecewise constant-acceleration motion ``u(t)``.

    Before ``t0`` the robot is at ``u0``. After the last segment it keeps
    its final speed.
    """

    def __init__(self, t0: float = 0.0, u0: float = 0.0, v0: float = 0.0):
        self.t0, self.u0, self.v0 = t0, u0, v0
        self.segments: list[Segment] = []

    @property
    def t_end(self) -> float:
        return self.segments[-1].t1 if self.segments else self.t0

    @property
    def u_end(self) -> float:
        return self.segments[-1].u1 if self.segments else self.u0

    @property
    def v_end(self) -> float:
        return self.segments[-1].v1 if self.segments else self.v0

    def _push(self, a: float, dt: float) -> Profile:
        if dt > EPS:
            self.segments.append(Segment(self.t_end, self.u_end, self.v_end, a, dt))
        return self

    def hold(self, dt: float) -> Profile:
        """Keep the current speed for ``dt`` seconds. At speed 0 this is a wait."""
        return self._push(0.0, dt)

    def hold_until(self, t: float) -> Profile:
        return self.hold(t - self.t_end)

    def change_speed(self, v: float, accel: float, decel: float) -> Profile:
        dv = v - self.v_end
        a = accel if dv > 0 else -decel
        return self._push(a, abs(dv) / abs(a)) if abs(dv) > EPS else self

    def cruise(self, d: float) -> Profile:
        if d <= EPS:
            return self
        if self.v_end <= EPS:
            raise InfeasibleMotion("cannot cruise at zero speed")
        return self._push(0.0, d / self.v_end)

    def move(self, d: float, limits: MotionLimits, v_end: float = 0.0) -> Profile:
        for a, dt in trapezoid(d, self.v_end, v_end, limits.v_max, limits.accel, limits.decel):
            self._push(a, dt)
        return self

    def state(self, t: float) -> tuple[float, float]:
        """``(u, v)`` at time ``t``."""
        if t <= self.t0:
            return self.u0, self.v0
        for seg in self.segments:
            if t <= seg.t1:
                return seg.at(t)
        return self.u_end + self.v_end * (t - self.t_end), self.v_end

    def time_at(self, u: float) -> float:
        """First time at which ``u(t) >= u``. Returns ``inf`` if it never happens."""
        if u <= self.u0:
            return self.t0
        for seg in self.segments:
            if seg.u1 >= u:
                du = max(u - seg.u0, 0.0)
                disc = math.sqrt(max(seg.v0**2 + 2 * seg.a * du, 0.0))
                denom = seg.v0 + disc  # numerically stable root of 0.5a*t^2 + v0*t = du
                return seg.t0 + (2 * du / denom if denom > EPS else 0.0)
        if self.v_end > EPS:
            return self.t_end + (u - self.u_end) / self.v_end
        return math.inf


class LaneChangeModel(Protocol):
    """Lateral motion used to move from one lane centre to another."""

    def duration(self, offset: float) -> float:
        """Seconds needed to shift sideways by ``offset`` metres."""
        ...

    def progress(self, offset: float, x: float) -> float:
        """Lateral progress in [0, 1] at normalised time ``x`` in [0, 1]."""
        ...


@dataclass(frozen=True)
class QuinticLaneChange:
    """Smooth lane change for differential-drive and car-like robots.

    ``y(x) = w * (10x^3 - 15x^4 + 6x^5)`` starts and ends with zero lateral
    velocity and acceleration. Its peak lateral acceleration is
    ``(10/sqrt(3)) * w / T^2``, which gives
    ``T = sqrt(10 * w / (sqrt(3) * a_lat_max))``.
    """

    max_lateral_accel: float
    PEAK_FACTOR = 10.0 / math.sqrt(3.0)

    def duration(self, offset: float) -> float:
        return math.sqrt(self.PEAK_FACTOR * abs(offset) / self.max_lateral_accel)

    def progress(self, offset: float, x: float) -> float:
        x = min(max(x, 0.0), 1.0)
        return x**3 * (10 - 15 * x + 6 * x**2)


@dataclass(frozen=True)
class HolonomicLaneChange:
    """Sideways translation for mecanum or omni robots (trapezoidal in ``y``)."""

    max_lateral_speed: float
    max_lateral_accel: float

    def _peak(self, offset: float) -> float:
        return min(self.max_lateral_speed, math.sqrt(self.max_lateral_accel * abs(offset)))

    def duration(self, offset: float) -> float:
        if abs(offset) < EPS:
            return 0.0
        vp, a = self._peak(offset), self.max_lateral_accel
        return 2 * vp / a + (abs(offset) - vp**2 / a) / vp

    def progress(self, offset: float, x: float) -> float:
        w = abs(offset)
        if w < EPS:
            return 1.0
        vp, a, T = self._peak(offset), self.max_lateral_accel, self.duration(offset)
        t, ta = min(max(x, 0.0), 1.0) * T, vp / a
        if t < ta:
            y = 0.5 * a * t**2
        elif t < T - ta:
            y = 0.5 * a * ta**2 + vp * (t - ta)
        else:
            y = w - 0.5 * a * (T - t) ** 2
        return y / w
