"""ROS-free pieces of the Open-RMF integration (unit tested without RMF installed).

* :func:`goal_trajectory`: an RMF ``navigate`` goal becomes a lane trajectory
  for the robot's controller: drive along its own lane and stop at the goal.
* :func:`override_path`: a Visison plan (bypass, rejoin or hold) becomes the
  ``[x, y, yaw]`` point list for ``CommandExecution.override_schedule``, so RMF,
  and through it every other fleet, sees where the robot will really be,
  including the excursion into the other lane.
* :func:`itinerary_reservations`: another fleet's RMF itinerary (timed
  waypoints from the traffic schedule) becomes read-only Visison reservations
  on the aisle's cells, so the planner picks bypass windows that respect robots
  it cannot command.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable, Sequence

from visison import (
    Aisle,
    Profile,
    Reservation,
    RobotSpec,
    RobotState,
    Trajectory,
)
from visison.kinematics import EPS
from visison.planner import LateralShift


def goal_trajectory(aisle: Aisle, state: RobotState, spec: RobotSpec, goal_x: float, t: float,
                    v_max: float | None = None) -> Trajectory | None:
    """Drive along the robot's own lane from where it is to ``goal_x`` and stop there.

    Returns None if the goal is not ahead in the robot's direction (RMF then
    needs a turnaround lane, which is outside the aisle).
    """
    d, lim = state.direction, spec.limits
    dist = d * (goal_x - state.s)
    y0 = state.y if state.y is not None else aisle.lane_center(state.lane)
    yc = aisle.lane_center(state.lane)
    if dist < -0.05:
        return None
    limits = lim if v_max is None else replace(lim, v_max=v_max)  # v_max may exceed the plan (a deviation)
    v0 = min(max(state.v, 0.0), limits.v_max)
    prof = Profile(t, d * state.s, v0)
    stop = v0 * v0 / (2 * limits.decel)
    if dist > stop + EPS:
        prof.move(dist, limits, v_end=0.0)
    else:
        prof.change_speed(0.0, limits.accel, limits.decel)  # too close: just stop
    shifts = []
    if abs(y0 - yc) > 0.02 and prof.t_end > t:
        shifts.append(LateralShift(t, min(spec.lane_change.duration(yc - y0), prof.t_end - t), y0, yc))
    return Trajectory(prof, d, y0, shifts, spec.lane_change)


def override_path(traj: Trajectory, t_from: float, t_to: float, step: float = 0.5) -> list[list[float]]:
    """``[x, y, yaw]`` points of a trajectory, every ``step`` s, for ``override_schedule``."""
    pts, t = [], t_from
    while t <= t_to + 1e-9:
        p = traj.sample(t)
        pts.append([p.s, p.y, p.heading])
        t += step
    return pts


def itinerary_reservations(
    aisle: Aisle,
    participant: str,
    waypoints: Sequence[tuple[float, float, float]],
    length: float,
    space: float,
    lane_width: float | None = None,
) -> list[Reservation]:
    """Timed waypoints ``(t, x, y)`` of another fleet's robot -> reservations of the cells it sweeps.

    Between two waypoints the robot is assumed to move linearly (RMF's
    trajectories are piecewise cubic; the linear hull plus ``space`` covers it
    for lane driving). Waypoints outside the aisle's lanes are ignored.
    """
    out: dict[tuple[str, int], list[float]] = {}
    half = length / 2 + space
    for (t0, x0, y0), (t1, x1, y1) in zip(waypoints, waypoints[1:]):
        if t1 <= t0:
            continue
        n = max(1, math.ceil(abs(x1 - x0) / (aisle.cell_size / 2)))
        for i in range(n + 1):
            f = i / n
            t, x, y = t0 + f * (t1 - t0), x0 + f * (x1 - x0), y0 + f * (y1 - y0)
            lane = _lane_of(aisle, y)
            if lane is None:
                continue
            for k in aisle.cells(x - half, x + half):
                span = out.setdefault((lane, k), [t, t])
                span[0], span[1] = min(span[0], t), max(span[1], t)
    dt_pad = 0.5  # the robot sweeps a cell a little before/after its centre does
    return [Reservation(participant, lane, k, lo - dt_pad, hi + dt_pad) for (lane, k), (lo, hi) in sorted(out.items())]


def _lane_of(aisle: Aisle, y: float) -> str | None:
    for lane in aisle.lanes:
        lo, hi = aisle.lane_bounds(lane)
        if lo <= y <= hi:
            return lane
    return None


def arrived(state: RobotState, goal_x: float, tol: float = 0.15) -> bool:
    return abs(state.s - goal_x) <= tol and state.v < 0.05


def merge(reservations: Iterable[Reservation]) -> list[Reservation]:
    return list(reservations)
