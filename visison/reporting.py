"""Obstacle reports sent to the fleet so a person can clear the aisle."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from .geometry import Aisle, Obstacle
from .reservation import Reservation

log = logging.getLogger("visison")


@dataclass(frozen=True)
class ObstacleReport:
    obstacle_id: str
    aisle_id: str
    lanes_blocked: tuple[str, ...]
    s_min: float
    s_max: float
    y_min: float
    y_max: float
    confidence: float
    reporter_id: str
    t_detected: float
    action: str = ""  # what the reporting robot decided to do

    @classmethod
    def of(
        cls, aisle: Aisle, obstacle: Obstacle, reporter_id: str, t: float, action: str = ""
    ) -> ObstacleReport:
        lanes = tuple(
            lane for lane in aisle.lanes if obstacle.overlaps_lateral(*aisle.lane_bounds(lane))
        )
        return cls(
            obstacle.obstacle_id, aisle.aisle_id, lanes,
            obstacle.s_min, obstacle.s_max, obstacle.y_min, obstacle.y_max,
            obstacle.confidence, reporter_id, t, action,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EmergencyReport:
    """One step of an emergency standoff, for the fleet and for logs."""

    t: float
    aisle_id: str
    standoff_id: int
    event: str  # "brake", "resolve", "reverse", "rejoin", "resume", "escalate", "done"
    robots: tuple[str, ...]
    message: str
    trigger: tuple[str, str] | None = None  # (ego, other) that raised the alarm
    probability: float | None = None
    ttc: float | None = None
    resolution: str = ""
    moves: tuple[tuple[str, float], ...] = ()  # (robot, metres backed off)

    def to_dict(self) -> dict:
        return asdict(self)


class LoggingReporter:
    """:class:`~visison.adapters.FleetReporter` that logs and keeps a history."""

    def __init__(self, logger: logging.Logger = log) -> None:
        self.logger = logger
        self.reports: list[ObstacleReport] = []
        self.conflicts: list[Reservation] = []
        self.emergencies: list[EmergencyReport] = []

    def report_obstacle(self, report: ObstacleReport) -> None:
        self.reports.append(report)
        self.logger.warning(
            "obstacle %s in %s lanes %s at s=[%.2f, %.2f] reported by %s (action: %s)",
            report.obstacle_id, report.aisle_id, ",".join(report.lanes_blocked),
            report.s_min, report.s_max, report.reporter_id, report.action or "-",
        )

    def report_schedule_conflicts(self, robot_id: str, conflicts: list[Reservation]) -> None:
        self.conflicts.extend(conflicts)
        if conflicts:
            self.logger.warning(
                "delay of %s now overlaps %d reservation(s) of %s",
                robot_id, len(conflicts), sorted({c.robot_id for c in conflicts}),
            )

    def report_emergency(self, report: EmergencyReport) -> None:
        self.emergencies.append(report)
        log = self.logger.error if report.event in ("brake", "escalate") else self.logger.warning
        log("EMERGENCY #%d %s [%s]: %s", report.standoff_id, report.event.upper(),
            ",".join(report.robots), report.message)
