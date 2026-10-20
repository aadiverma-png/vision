"""Visison: reservation-aware obstacle bypass for autonomous mobile robots."""

from .arbiter import ArbiterOutput, Kind, MotionArbiter, MotionIntent, Source
from .controller import BypassController, ControllerState, Event
from .geometry import Aisle, Footprint, Obstacle, OrientedBox, boxes_overlap, squeeze_offset
from .intervals import Interval, earliest_free_point, merge
from .kinematics import (
    HolonomicLaneChange,
    InfeasibleMotion,
    MotionLimits,
    Profile,
    QuinticLaneChange,
    travel_time,
)
from .planner import (
    Action,
    BypassPlan,
    BypassPlanner,
    Decision,
    Phase,
    PlannerConfig,
    Pose,
    RerouteEstimate,
    RobotSpec,
    RobotState,
    SafetyMargins,
    Trajectory,
    oncoming_is_clear,
)
from .emergency import (
    EmergencyConfig,
    EmergencyMonitor,
    ObjectTracker,
    RiskAssessment,
    RiskLevel,
    TrackedObject,
    collision_risk,
)
from .motion_models import Ackermann, Command, DiffDrive, Line, MotionModel, Omni, validate_command
from .orca import (
    OrcaAgent,
    OrcaConfig,
    OrcaMode,
    OrcaResult,
    orca_1d,
    orca_half_plane,
    solve as orca_solve,
)
from .reporting import EmergencyReport, LoggingReporter, ObstacleReport
from .reservation import (
    InMemoryReservationTable,
    Reservation,
    ReservationProvider,
    reservations_for_path,
    sweep_occupancy,
)

from .standoff import (
    EmergencyCoordinator,
    FleetMember,
    Resolution,
    ReversePlan,
    Standoff,
    StandoffPhase,
    conflict_groups,
    default_model,
    displacement_trajectory,
    is_between_lanes,
    plan_chain_reverse,
    resume_trajectory,
)

from .tracking import UnicycleTracker, reference

__version__ = "0.1.0"

__all__ = [name for name in dir() if not name.startswith("_")]
