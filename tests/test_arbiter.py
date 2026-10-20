import math

import pytest

from sim.aisle_sim import AISLE, SPEC, Clock, SimRobot
from visison import (
    BypassController,
    BypassPlanner,
    Command,
    ControllerState,
    DiffDrive,
    InMemoryReservationTable,
    Kind,
    MotionArbiter,
    MotionIntent,
    Obstacle,
    OrcaAgent,
    OrcaMode,
    Profile,
    Source,
    Trajectory,
    UnicycleTracker,
    validate_command,
)

DD = DiffDrive(track_width=0.4, v_wheel_max=2.0, a_acc=1.2, a_dec=2.0, alpha_max=3.0, w_max=1.5)
DT = 0.05


def cruise(v=1.0, t0=0.0, s0=0.0):
    return Trajectory(Profile(t0, s0, v), +1, 0.6, [], SPEC.lane_change)


def arbiter(**kw):
    published = []
    arb = MotionArbiter("R1", DD, DT, tracker=UnicycleTracker(v_max=1.6, v_reverse=0.3),
                        publish=published.append, **kw)
    return arb, published


def test_exactly_one_command_per_step():
    arb, pub = arbiter()
    arb.request(MotionIntent(Source.PLANNER, Kind.FOLLOW, cruise()), 0.0)
    for k in range(10):
        arb.step(k * DT, (k * DT, 0.6, 0.0), Command(1.0, 0.0))
    assert len(pub) == 10


def test_emergency_locks_out_the_planner_until_released():
    arb, _ = arbiter()
    arb.request(MotionIntent(Source.PLANNER, Kind.FOLLOW, cruise()), 0.0)
    arb.request(MotionIntent(Source.EMERGENCY, Kind.BRAKE), 1.0)
    later = cruise(0.5, 2.0, 2.0)
    assert arb.request(MotionIntent(Source.PLANNER, Kind.FOLLOW, later), 2.0) is False  # stored, not active
    assert arb.active(2.0).source is Source.EMERGENCY
    assert any("stored behind EMERGENCY" in m for _, m in arb.log)
    arb.release(Source.EMERGENCY, 3.0)
    assert arb.active(3.0).trajectory is later  # the newest planner intent takes over


def test_handover_continues_the_emergency_motion_unless_the_planner_asked_later():
    arb, _ = arbiter()
    arb.request(MotionIntent(Source.PLANNER, Kind.FOLLOW, cruise()), 0.0)
    rejoin = cruise(0.2, 5.0, 10.0)
    arb.request(MotionIntent(Source.EMERGENCY, Kind.SCRIPTED, rejoin), 5.0)
    arb.release(Source.EMERGENCY, 8.0, handover=True)
    assert arb.active(8.0).source is Source.PLANNER and arb.active(8.0).trajectory is rejoin
    # a planner request made during the emergency wins over the handover
    arb.request(MotionIntent(Source.EMERGENCY, Kind.BRAKE), 9.0)
    newer = cruise(0.0, 9.5, 11.0)
    arb.request(MotionIntent(Source.PLANNER, Kind.HOLD, newer), 9.5)
    arb.release(Source.EMERGENCY, 10.0, handover=True)
    assert arb.active(10.0).trajectory is newer


def test_brake_skips_orca_and_decelerates_at_the_limit():
    calls = []
    arb, _ = arbiter(orca_enabled=True, neighbours=lambda *a: calls.append(a) or (None, [], [], OrcaMode.NORMAL))
    arb.request(MotionIntent(Source.EMERGENCY, Kind.BRAKE), 0.0)
    cmd = arb.step(0.0, (0.0, 0.6, 0.0), Command(1.0, 0.0))
    assert not calls
    assert cmd.v == pytest.approx(1.0 - DD.a_dec * DT)


def test_gate_projects_unexecutable_commands_and_never_adds_speed():
    arb, _ = arbiter()
    # a reference far ahead makes the tracker ask for a big jump in speed
    arb.request(MotionIntent(Source.PLANNER, Kind.FOLLOW, cruise(1.5, 0.0, 5.0)), 0.0)
    cur = Command(0.2, 0.0)
    cmd = arb.step(0.0, (0.0, 0.6, 0.0), cur)
    assert not validate_command(DD, cur, cmd, DT)
    assert cmd.v <= cur.v + DD.a_acc * DT + 1e-9
    assert "accel" in arb.last.gated


def test_watchdog_brakes_on_stale_odometry_and_expired_intents():
    arb, _ = arbiter()
    arb.request(MotionIntent(Source.PLANNER, Kind.FOLLOW, cruise(), ttl=1.0), 0.0)
    cmd = arb.step(0.5, (0.5, 0.6, 0.0), Command(1.0, 0.0), odom_age=1.0)
    assert cmd.v < 1.0 and "stale_odometry" in arb.last.gated
    cmd = arb.step(2.0, (2.0, 0.6, 0.0), Command(1.0, 0.0))  # the intent expired at t = 1
    assert arb.active(2.0) is None and cmd.v < 1.0


def test_time_scaled_reference_turns_a_slowdown_into_lateness_not_a_surge():
    blocker = OrcaAgent("B", (3.0, 0.6), (0.0, 0.0), 0.6, cooperative=False)
    arb, _ = arbiter(orca_enabled=True,
                     neighbours=lambda rid, t, i: (OrcaAgent("R1", (x[0], 0.6), (v[0], 0.0), 0.6), [blocker] if t < 4 else [],
                                                   [], OrcaMode.PATH_SPEED_ONLY))
    arb.request(MotionIntent(Source.PLANNER, Kind.FOLLOW, cruise(1.0)), 0.0)
    x, v, speeds = [0.0], [1.0], []
    for k in range(int(8.0 / DT)):
        t = k * DT
        cmd = arb.step(t, (x[0], 0.6, 0.0), Command(v[0], 0.0))
        x[0] += cmd.v * DT
        v[0] = cmd.v
        speeds.append(cmd.v)
    assert arb.lateness > 1.0  # it was held back behind the blocker
    assert max(speeds) <= 1.0 + 0.05 * 1.6 + 1e-6  # never a catch-up surge above the reference speed


def test_controller_hold_during_a_scripted_reverse_does_not_interfere():
    """Regression for command conflicts: two layers speaking to one robot in the same tick."""
    clock = Clock()
    robot = SimRobot("R2", SPEC, AISLE, "B", 25.0, 0.0, -1, clock)
    back = Trajectory(Profile(0.0, 25.0, 0.0).move(1.5, SPEC.limits, 0.0), +1, 1.8, [], SPEC.lane_change,
                      reverse=True)
    robot.follow(back, Source.EMERGENCY, Kind.SCRIPTED)
    robot.stop()  # the controller holds at the same moment (PLANNER)
    clock.t = 3.0
    assert robot.arbiter.active(3.0).source is Source.EMERGENCY
    assert robot.pose().s > 25.0  # still backing off as the coordinator planned


def test_bypass_controller_phases_follow_the_reference_time():
    clock = Clock()
    robot = SimRobot("R1", SPEC, AISLE, "A", 2.0, SPEC.limits.v_max, +1, clock)
    ctl = BypassController(robot, BypassPlanner(AISLE, SPEC), InMemoryReservationTable())
    d = ctl.on_obstacle(Obstacle(20.0, 20.6, 0.1, 1.1), 0.0)
    robot.reference_time = lambda t: min(t, d.plan.t_out - 1.0)  # stalled before the lane change
    clock.t = d.plan.t_back + 5.0
    ctl.tick(clock.t)
    assert ctl.state is ControllerState.APPROACH  # not "rejoined" just because wall time passed
