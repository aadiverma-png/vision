"""Scenario 1 in Gazebo: lane A is blocked by a fallen box, R2 comes down lane B.

The world frame is the aisle frame: ``x = s`` along the aisle and ``y``
across it. Lane A is ``y`` in [0, 1.2] and lane B is ``y`` in [1.2, 2.4].
R1 drives lane A towards +x and R2 drives lane B towards -x.

``r2_arrival`` is the scenario time, in seconds after both robots are up, at
which R2's centre passes the fallen box. Changing it moves the whole R2
schedule, so the same scene can produce BYPASS_NOW or WAIT_THEN_BYPASS.

:func:`fleet_layout` scales the scene to any number of robots: R1 plus
followers in lane A, and a convoy R2, R3, ... in lane B. ``r2_early`` makes the
lane-B robots drive that many seconds ahead of their reservations. That is the
timing mistake the emergency layer (``visison.standoff``) has to catch.

Everything here is plain Python. The launch file (spawn poses) and the node
(schedules, reservations) both use it, so they cannot disagree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from visison import (
    Aisle,
    DiffDrive,
    Footprint,
    MotionLimits,
    Obstacle,
    PlannerConfig,
    Profile,
    QuinticLaneChange,
    RobotSpec,
    Trajectory,
    reservations_for_path,
)

AISLE = Aisle(lane_width=1.2, length=40.0, cell_size=0.5, aisle_id="gazebo-aisle")

# amr_sim robot: 0.50 x 0.49 m incl. wheels, diff drive limited to 1 m/s^2.
SPEC = RobotSpec(
    footprint=Footprint(length=0.6, width=0.6),
    limits=MotionLimits(v_max=0.8, accel=0.4, decel=0.6, emergency_decel=1.0),
    lane_change=QuinticLaneChange(max_lateral_accel=0.3),
    bypass_speed=0.6,
)
CONFIG = PlannerConfig()
COOP_CONFIG = PlannerConfig(coop_delays=(1.0, 2.0, 3.0, 4.0, 5.0))

# The amr_sim diff drive (urdf/amr.urdf.xacro): 0.45 m track, plugin limits 1 m/s^2 and 2 rad/s^2.
MODEL = DiffDrive(track_width=0.45, v_wheel_max=1.2, a_acc=1.0, a_dec=1.0, alpha_max=2.0, w_max=1.2,
                  v_rev_max=0.3, D=0.15)

LANE_A_Y = AISLE.lane_center("A")
LANE_B_Y = AISLE.lane_center("B")

R1_START_S = 2.0
R1_END_S = 38.0  # R1 parks here when the scenario is over
R2_PARK_S = 38.0  # R2 waits here until it has to leave to meet r2_arrival
R2_EXIT_S = -1.0

# Must match the fallen_box model in worlds/aisle_world.sdf.
FALLEN_BOX = Obstacle(20.0, 20.8, 0.1, 1.1, obstacle_id="fallen-box")
BOX_CENTER_S = 0.5 * (FALLEN_BOX.s_min + FALLEN_BOX.s_max)


def _time_to_cover(d: float, v: float, a: float) -> float:
    """From rest, accelerate at ``a`` up to ``v`` and cruise: time for distance ``d``."""
    d_ramp = v * v / (2 * a)
    return v / a + (d - d_ramp) / v if d >= d_ramp else math.sqrt(2 * d / a)


@dataclass(frozen=True)
class R2Schedule:
    """R2 stands at ``spawn_s`` until ``t_go``, then drives down lane B at ``speed``."""

    arrival: float
    spawn_s: float
    t_go: float
    speed: float
    accel: float

    def profile(self) -> Profile:
        p = Profile(0.0, -self.spawn_s, 0.0)  # travel frame u = -s
        p.hold_until(self.t_go)
        return p.change_speed(self.speed, self.accel, self.accel)

    def trajectory(self) -> Trajectory:
        return Trajectory(self.profile(), -1, LANE_B_Y, [], SPEC.lane_change)

    def reservations(self):
        h = SPEC.footprint.length / 2
        return reservations_for_path("R2", AISLE, "B", self.profile(), -1,
                                     -self.spawn_s, -R2_EXIT_S, h + CONFIG.margins.space)


def r2_schedule(arrival: float, speed: float = 0.8, accel: float = 0.4,
                park_s: float = R2_PARK_S) -> R2Schedule:
    """Schedule that makes R2's centre pass the fallen box at ``t = arrival``.

    If there is enough time, R2 parks at ``park_s`` (the far end of the aisle) and leaves
    at the right moment. Otherwise it spawns closer and leaves at t = 0.
    """
    if arrival <= speed / accel:
        raise ValueError(f"r2_arrival must be > {speed / accel:.1f} s (R2's acceleration time)")
    t_go = arrival - _time_to_cover(park_s - BOX_CENTER_S, speed, accel)
    if t_go >= 0:
        return R2Schedule(arrival, park_s, t_go, speed, accel)
    d = speed * speed / (2 * accel) + speed * (arrival - speed / accel)
    return R2Schedule(arrival, BOX_CENTER_S + d, 0.0, speed, accel)


def r1_nominal_profile(start_s: float = R1_START_S) -> Profile:
    """R1's plan before the box falls: accelerate to v_max along lane A."""
    lim = SPEC.limits
    return Profile(0.0, start_s, 0.0).change_speed(lim.v_max, lim.accel, lim.decel)


def r1_nominal_trajectory(start_s: float = R1_START_S) -> Trajectory:
    return Trajectory(r1_nominal_profile(start_s), +1, LANE_A_Y, [], SPEC.lane_change)


def r1_reservations(robot_id: str = "R1", start_s: float = R1_START_S, end_s: float = R1_END_S):
    h = SPEC.footprint.length / 2
    return reservations_for_path(robot_id, AISLE, "A", r1_nominal_profile(start_s), +1,
                                 start_s, end_s + h, h + CONFIG.margins.space)


# ------------------------------------------------------------------ N robots

COLORS = ("0.1 0.3 0.8 1", "0.95 0.5 0.1 1", "0.2 0.7 0.3 1", "0.7 0.2 0.7 1", "0.9 0.8 0.1 1",
          "0.1 0.7 0.8 1", "0.6 0.35 0.2 1", "0.5 0.5 0.5 1", "0.9 0.3 0.4 1", "0.3 0.3 0.9 1")


@dataclass(frozen=True)
class RobotPlan:
    """One robot of the scene: where it spawns, how it drives, and what it has reserved."""

    robot_id: str  # "R1", "R2", ...
    ns: str  # "r1", "r2", ...
    lane: str
    direction: int
    spawn_s: float
    end_s: float  # it parks here when its run is over
    color: str
    cruise_speed: float
    schedule: R2Schedule | None = None  # lane B: how it really drives
    reserved: R2Schedule | None = None  # lane B: what the fleet table believes

    @property
    def spawn(self) -> tuple[float, float, float]:
        y = LANE_A_Y if self.lane == "A" else LANE_B_Y
        return self.spawn_s, y, 0.0 if self.direction > 0 else math.pi

    def trajectory(self) -> Trajectory:
        if self.schedule is not None:
            return self.schedule.trajectory()
        return r1_nominal_trajectory(self.spawn_s)

    def reservations(self):
        if self.reserved is not None:
            h = SPEC.footprint.length / 2
            return reservations_for_path(self.robot_id, AISLE, "B", self.reserved.profile(), -1,
                                         -self.reserved.spawn_s, -self.end_s, h + CONFIG.margins.space)
        return r1_reservations(self.robot_id, self.spawn_s, self.end_s)


def _color(robot_id: str) -> str:
    return COLORS[(int(robot_id[1:]) - 1) % len(COLORS)]


def fleet_layout(lane_b_robots: int = 1, lane_a_followers: int = 0, r2_arrival: float = 50.0,
                 r2_speed: float = 0.8, r2_early: float = 0.0, convoy_spacing: float = 2.5,
                 follow_spacing: float = 3.0) -> list[RobotPlan]:
    """R1 plus ``lane_a_followers`` behind it in lane A, and a lane-B convoy R2, R3, ...

    The convoy robots park one behind the other beyond the aisle end and pass
    the box ``convoy_spacing / r2_speed`` s apart, R2 first at ``r2_arrival``.
    With ``r2_early > 0`` they really drive that much earlier than reserved.
    Lane-A robots park past the aisle end, the leader furthest.
    """
    if lane_b_robots < 0 or lane_a_followers < 0:
        raise ValueError("robot counts must be >= 0")
    plans = []
    n_a = 1 + lane_a_followers
    for k in range(n_a):
        rid = f"R{1 if k == 0 else 1 + lane_b_robots + k}"
        plans.append(RobotPlan(rid, rid.lower(), "A", +1, R1_START_S - k * follow_spacing,
                               R1_END_S + (n_a - 1 - k) * follow_spacing, _color(rid), SPEC.limits.v_max))
    for k in range(lane_b_robots):
        rid = f"R{2 + k}"
        arrival = r2_arrival + k * convoy_spacing / r2_speed
        park = R2_PARK_S + k * convoy_spacing
        reserved = r2_schedule(arrival, r2_speed, park_s=park)
        actual = r2_schedule(arrival - r2_early, r2_speed, park_s=park)
        plans.append(RobotPlan(rid, rid.lower(), "B", -1, actual.spawn_s,
                               R2_EXIT_S - (lane_b_robots - 1 - k) * convoy_spacing,
                               _color(rid), r2_speed, actual, reserved))
    return plans
