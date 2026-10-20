"""Trajectory tracking for a differential-drive robot (Kanayama et al., 1990).

The error between the reference pose and the actual pose is taken in the
robot frame:

    e_x =  cos(th)*(x_r - x) + sin(th)*(y_r - y)
    e_y = -sin(th)*(x_r - x) + cos(th)*(y_r - y)
    e_th = th_r - th

Control law:

    v = v_r*cos(e_th) + k_x*e_x
    w = w_r + v_r*(k_y*e_y + k_th*sin(e_th)) + k_0*sin(e_th)

The ``k_0`` term keeps the heading corrected while the reference is standing
still (for example, while waiting for the bypass window). The same law also
tracks a reversing reference (``v_r < 0``, used when a robot backs off to make
room in an emergency standoff). ``v_reverse`` must then be at least the
reverse speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class UnicycleTracker:
    v_max: float
    w_max: float = 1.2
    k_x: float = 2.0
    k_y: float = 6.0
    k_th: float = 3.0
    k_0: float = 0.8
    v_reverse: float = 0.1  # m/s, fastest allowed backwards motion

    def command(
        self, x: float, y: float, th: float,
        x_r: float, y_r: float, th_r: float, v_r: float, w_r: float,
    ) -> tuple[float, float]:
        dx, dy = x_r - x, y_r - y
        e_x = math.cos(th) * dx + math.sin(th) * dy
        e_y = -math.sin(th) * dx + math.cos(th) * dy
        e_th = wrap(th_r - th)
        v = v_r * math.cos(e_th) + self.k_x * e_x
        w = w_r + v_r * (self.k_y * e_y + self.k_th * math.sin(e_th)) + self.k_0 * math.sin(e_th)
        return (
            min(max(v, -self.v_reverse), self.v_max),
            min(max(w, -self.w_max), self.w_max),
        )


def reference(trajectory, t: float, h: float = 0.02) -> tuple[float, float, float, float, float]:
    """``(x, y, heading, v, omega)`` of a visison Trajectory at time ``t``."""
    p = trajectory.sample(t)
    h0, h1 = trajectory.sample(t - h).heading, trajectory.sample(t + h).heading
    omega = wrap(h1 - h0) / (2 * h) if abs(p.v) > 1e-3 else 0.0
    return p.s, p.y, p.heading, p.v, omega
