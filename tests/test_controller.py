import logging

import pytest

from sim.aisle_sim import AISLE, SPEC, Clock, SimRerouter, SimRobot
from visison import (
    Action,
    BypassController,
    BypassPlanner,
    ControllerState,
    InMemoryReservationTable,
    LoggingReporter,
    Obstacle,
    Reservation,
    reservations_for_path,
)

BOX = Obstacle(20.0, 20.6, 0.1, 1.1, obstacle_id="box")
H = SPEC.footprint.length / 2


def make_controller(table=None, rerouter=False):
    clock = Clock()
    clock.t = 4.0
    robot = SimRobot("R1", SPEC, AISLE, "A", 2.0, SPEC.limits.v_max, +1, clock)
    table = table if table is not None else InMemoryReservationTable()
    table.add(*reservations_for_path("R1", AISLE, "A", robot.traj.profile, +1, 2.0, 40.0, H))
    reporter = LoggingReporter(logging.getLogger("test"))
    ctl = BypassController(robot, BypassPlanner(AISLE, SPEC), table, reporter,
                           SimRerouter(robot) if rerouter else None)
    return clock, robot, table, reporter, ctl


def run_until(ctl, clock, t_end, dt=0.05, **kw):
    while clock.t < t_end:
        clock.t += dt
        ctl.tick(clock.t, **kw)


def test_bypass_claims_shifts_reports_and_releases():
    clock, robot, table, reporter, ctl = make_controller()
    lane_a_before = {r.cell: r for r in table.all() if r.lane == "A"}
    d = ctl.on_obstacle(BOX, clock.t)
    assert d.action is Action.BYPASS_NOW
    assert all(c in table.all() for c in d.plan.claims)
    # later lane-A reservations moved by the plan's delay
    k = AISLE.cells(30, 30.1)[0]
    moved = next(r for r in table.all() if r.lane == "A" and r.cell == k)
    assert moved.start == pytest.approx(lane_a_before[k].start + d.plan.delay)
    assert len(reporter.reports) == 1 and reporter.reports[0].lanes_blocked == ("A",)
    assert reporter.reports[0].action == "bypass_now"

    run_until(ctl, clock, d.plan.t_back + 0.2)
    assert ctl.state is ControllerState.REJOINED
    assert not any(r.lane == "B" for r in table.all())  # other lane released early
    states = [e.state for e in ctl.events]
    for s in (ControllerState.CHANGE_OUT, ControllerState.PASSING, ControllerState.CHANGE_IN):
        assert s in states


def test_same_obstacle_is_handled_and_reported_once():
    clock, robot, table, reporter, ctl = make_controller()
    d1 = ctl.on_obstacle(BOX, clock.t)
    d2 = ctl.on_obstacle(BOX, clock.t + 0.1)
    assert d1 is d2 and len(reporter.reports) == 1


class FlakyTable(InMemoryReservationTable):
    """Refuses the first claim, as if another robot won the window."""

    def __init__(self):
        super().__init__()
        self.refusals = 1

    def try_reserve(self, claims):
        if self.refusals and any(c.lane == "B" for c in claims):
            self.refusals -= 1
            return False
        return super().try_reserve(claims)


def test_refused_claim_is_replanned():
    clock, robot, table, reporter, ctl = make_controller(FlakyTable())
    d = ctl.on_obstacle(BOX, clock.t)
    assert d.action is Action.BYPASS_NOW
    assert any("claim refused" in e.message for e in ctl.events)


def test_invaded_window_aborts_before_lane_change_and_replans():
    clock, robot, table, reporter, ctl = make_controller()
    d = ctl.on_obstacle(BOX, clock.t)
    run_until(ctl, clock, 6.0)
    first = d.plan
    intruder = [Reservation("R7", "B", c.cell, c.start, c.end) for c in first.claims]
    table.add(*intruder)
    run_until(ctl, clock, 6.1)
    assert any("abort" in e.message for e in ctl.events)
    assert not any(c in table.all() for c in first.claims)
    assert ctl.decision.action is Action.WAIT_THEN_BYPASS
    assert not any(c.conflicts_with(r) for c in ctl.plan.claims for r in intruder)
    assert all(c in table.all() for c in ctl.plan.claims)


def test_new_obstacle_in_other_lane_aborts():
    clock, robot, table, reporter, ctl = make_controller()
    ctl.on_obstacle(BOX, clock.t)
    run_until(ctl, clock, 5.0)
    blocker = Obstacle(21, 22, 1.4, 2.2, obstacle_id="cart")
    run_until(ctl, clock, 5.1, other_lane_obstacles=(blocker,))
    assert any("new obstacle cart" in e.message for e in ctl.events)
    assert ctl.state is ControllerState.HOLDING
    assert robot.traj.sample(clock.t + 10).v == 0  # braked


def test_hold_retries_periodically():
    clock, robot, table, reporter, ctl = make_controller()
    wide = Obstacle(20, 20.8, 0.1, 2.3, obstacle_id="pallet")
    assert ctl.on_obstacle(wide, clock.t).action is Action.HOLD
    run_until(ctl, clock, clock.t + 5.0)
    assert sum(e.state is ControllerState.EVALUATING for e in ctl.events) >= 3
    assert reporter.reports[0].lanes_blocked == ("A", "B")


def test_reroute_hands_over_to_host_planner():
    clock, robot, table, reporter, ctl = make_controller(rerouter=True)
    wide = Obstacle(20, 20.8, 0.1, 2.3, obstacle_id="pallet")
    assert ctl.on_obstacle(wide, clock.t).action is Action.REROUTE
    assert ctl.rerouter.executed and ctl.state is ControllerState.REROUTING


def test_new_obstacle_during_manoeuvre_is_queued_until_rejoined():
    clock, robot, table, reporter, ctl = make_controller()
    d = ctl.on_obstacle(BOX, clock.t)
    run_until(ctl, clock, d.plan.t_out + 1.0)  # committed: changing lanes
    second = Obstacle(35.0, 35.5, 0.0, 0.2, obstacle_id="carton")
    assert ctl.on_obstacle(second, clock.t) is d  # the running bypass is kept
    assert ctl.state is ControllerState.CHANGE_OUT
    assert [r.obstacle_id for r in reporter.reports] == ["box", "carton"]  # reported at once
    run_until(ctl, clock, d.plan.t_back + 0.2)
    assert ctl.obstacle.obstacle_id == "carton"  # evaluated after rejoining
    assert ctl.decision.action is Action.SQUEEZE
