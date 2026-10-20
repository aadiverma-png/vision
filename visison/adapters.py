"""Interfaces a host system implements to run Visison on its robots.

Nothing here depends on a specific robot, middleware or fleet manager. A
ROS 2 robot, a simulator or a vendor SDK plugs in by implementing these
protocols (see ``examples/ros2_adapter.py``).
"""

from __future__ import annotations

from typing import Protocol

from .arbiter import Kind, Source
from .geometry import Obstacle
from .planner import RerouteEstimate, RobotState, Trajectory
from .reporting import EmergencyReport, ObstacleReport
from .reservation import Reservation


class RobotInterface(Protocol):
    robot_id: str

    def state(self) -> RobotState:
        """Current pose and speed in the aisle frame."""
        ...

    # Every call is an *intent* for the robot's MotionArbiter (visison.arbiter),
    # which is the only thing that commands the motors. ``source`` decides priority.

    def follow(self, trajectory: Trajectory, source: Source = Source.PLANNER, kind: Kind = Kind.FOLLOW) -> None:
        """Track ``trajectory`` (sample it with ``trajectory.sample(t)``)."""
        ...

    def stop(self, source: Source = Source.PLANNER) -> None:
        """Brake to a controlled stop in the current lane."""
        ...

    def emergency_brake(self, source: Source = Source.EMERGENCY) -> None:
        """Brake as hard as the robot can (``limits.e_decel``) and stay stopped."""
        ...

    def release(self, source: Source, handover: bool = False) -> None:
        """Withdraw ``source``'s intent; with ``handover`` its trajectory continues as the planner's."""
        ...


class ReroutePlanner(Protocol):
    """The host's global planner (PIBT, A*, ...), used as the fallback."""

    def estimate(self, state: RobotState, obstacle: Obstacle) -> RerouteEstimate | None:
        """Extra time and distance of the best detour, or ``None`` if there is none."""
        ...

    def execute(self, state: RobotState, obstacle: Obstacle) -> None:
        """Take over the robot and drive the detour."""
        ...


class FleetReporter(Protocol):
    def report_obstacle(self, report: ObstacleReport) -> None:
        ...

    def report_schedule_conflicts(self, robot_id: str, conflicts: list[Reservation]) -> None:
        """Another robot's reservation now overlaps this robot's delayed schedule."""
        ...

    def report_emergency(self, report: EmergencyReport) -> None:
        """An emergency stop happened, was resolved, or needs an operator."""
        ...
