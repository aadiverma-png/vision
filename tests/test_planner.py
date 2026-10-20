import dataclasses
import random

import pytest

from visison import (
    Action,
    Aisle,
    BypassPlanner,
    Footprint,
    HolonomicLaneChange,
    InMemoryReservationTable,
    MotionLimits,
    Obstacle,
    PlannerConfig,
    Profile,
    QuinticLaneChange,
    RerouteEstimate,
    Reservation,
    RobotSpec,
    RobotState,
    oncoming_is_clear,
    reservations_for_path,
)

AISLE = Aisle(lane_width=1.2, length=40.0)
SPEC = RobotSpec(Footprint(0.9, 0.6), MotionLimits(1.5, 0.8, 1.2), QuinticLaneChange(0.5), bypass_speed=1.0)
BOX = Obstacle(20.0, 20.6, 0.1, 1.1)
R1 = RobotState("R1", s=8.0, v=1.5, lane="A")
CFG = PlannerConfig()


def oncoming(s_start, v=1.0):
    table = InMemoryReservationTable()
    h = SPEC.footprint.length / 2
    table.add(*reservations_for_path("R2", AISLE, "B", Profile(0, -s_start, v), -1, -s_start, h,
                                     h + CFG.margins.space))
    return table


def test_continue_when_obstacle_not_in_path():
    p = BypassPlanner(AISLE, SPEC)
    assert p.evaluate(R1, Obstacle(2, 3, 0.1, 1.1), 0).action is Action.CONTINUE  # behind
    assert p.evaluate(R1, Obstacle(20, 21, 1.5, 2.3), 0).action is Action.CONTINUE  # other lane


def test_squeeze_stays_in_lane():
    d = BypassPlanner(AISLE, SPEC).evaluate(R1, Obstacle(20, 20.4, 0.0, 0.3), 0)
    assert d.action is Action.SQUEEZE
    assert d.plan.lane_to == "A" and not d.plan.claims
    ys = [d.plan.trajectory.lateral(t / 10) for t in range(300)]
    assert max(ys) + SPEC.footprint.width / 2 <= AISLE.lane_width + 1e-9


@pytest.mark.parametrize("s_start,expected,clear", [(56.0, Action.BYPASS_NOW, True), (31.0, Action.WAIT_THEN_BYPASS, False)])
def test_far_and_near_oncoming(s_start, expected, clear):
    p = BypassPlanner(AISLE, SPEC)
    d = p.evaluate(R1, BOX, 0.0, oncoming(s_start))
    assert d.action is expected
    # the closed-form single-robot rule agrees with the exact interval search
    rolling = p.evaluate(R1, BOX, 0.0).plan
    zone_far_edge = rolling.s_back + SPEC.footprint.length / 2 + CFG.margins.space
    assert oncoming_is_clear(rolling.t_back, s_start - zone_far_edge, 1.0, CFG.margins,
                             SPEC.footprint.length) is clear


def test_claims_are_grantable_and_do_not_touch_existing_reservations():
    table = oncoming(31.0)
    d = BypassPlanner(AISLE, SPEC).evaluate(R1, BOX, 0.0, table)
    assert table.try_reserve(d.plan.claims)


def test_both_blocked_holds_without_reroute_and_reroutes_with_one():
    wide = Obstacle(20, 20.8, 0.1, 2.3)
    p = BypassPlanner(AISLE, SPEC)
    assert p.evaluate(R1, wide, 0).action is Action.HOLD
    assert p.evaluate(R1, wide, 0, reroute=RerouteEstimate(40, 50)).action is Action.REROUTE


def test_other_lane_obstacle_blocks_bypass():
    p = BypassPlanner(AISLE, SPEC)
    d = p.evaluate(R1, BOX, 0, other_lane_obstacles=(Obstacle(21, 22, 1.5, 2.0),))
    assert d.action is Action.HOLD


def test_cheap_reroute_and_long_wait_prefer_reroute():
    p = BypassPlanner(AISLE, SPEC)
    assert p.evaluate(R1, BOX, 0, reroute=RerouteEstimate(0.0, 0.0)).action is Action.BYPASS_NOW  # penalty 10 s
    cheap = dataclasses.replace(CFG, reroute_penalty=0.0)
    assert BypassPlanner(AISLE, SPEC, cheap).evaluate(R1, BOX, 0, reroute=RerouteEstimate(0.5, 0)).action is Action.REROUTE
    table = InMemoryReservationTable()
    table.add(*(Reservation("R9", "B", k, 0, 200) for k in range(AISLE.n_cells)))
    impatient = dataclasses.replace(CFG, max_wait=30.0, horizon=400)
    d = BypassPlanner(AISLE, SPEC, impatient).evaluate(R1, BOX, 0, table, reroute=RerouteEstimate(300, 50))
    assert d.action is Action.REROUTE and "max_wait" in d.reason


def test_too_close_to_change_lanes():
    stuck = RobotState("R1", s=19.0, v=0.0, lane="A")
    assert BypassPlanner(AISLE, SPEC).evaluate(stuck, BOX, 0).action is Action.HOLD


def test_mirrored_direction_gives_mirrored_plan():
    p = BypassPlanner(AISLE, SPEC)
    fwd = p.evaluate(R1, BOX, 0).plan
    mirror = Obstacle(40 - BOX.s_max, 40 - BOX.s_min, BOX.y_min, BOX.y_max)
    back = p.evaluate(RobotState("R1", 40 - R1.s, R1.v, "A", direction=-1), mirror, 0).plan
    assert back.s_out == pytest.approx(40 - fwd.s_out)
    assert back.s_back == pytest.approx(40 - fwd.s_back)
    assert back.t_back == pytest.approx(fwd.t_back)
    assert sorted(c.cell for c in back.claims) == sorted(AISLE.n_cells - 1 - c.cell for c in fwd.claims)


def test_holonomic_robot_plans():
    spec = dataclasses.replace(SPEC, lane_change=HolonomicLaneChange(0.5, 0.4))
    d = BypassPlanner(AISLE, spec).evaluate(R1, BOX, 0)
    assert d.action is Action.BYPASS_NOW
    assert d.plan.trajectory.lateral(d.plan.t_out + d.plan.lane_change_time + 0.01) == pytest.approx(1.8)


def conflicts(claims, reservations, shift=0.0, eps=0.0):
    """Open-interval overlap of the claims moved by ``shift`` and shrunk by ``eps``."""
    return any(
        c.cell == r.cell and c.start + shift + eps < r.end and r.start < c.end + shift - eps
        for c in claims for r in reservations
    )


def test_t_star_matches_brute_force_on_random_tables():
    """The exact earliest window equals a fine time scan over random reservation tables."""
    rng = random.Random(3)
    cfg = dataclasses.replace(CFG, speed_fractions=(1.0,), horizon=500)
    planner = BypassPlanner(AISLE, SPEC, cfg)
    for trial in range(60):
        table = InMemoryReservationTable()
        rs = [Reservation("blocker", "B", k, 0.0, 6.0) for k in range(20, 60)]  # forces a wait
        for i in range(rng.randint(1, 12)):
            k0 = rng.randint(20, 55)
            t0 = rng.uniform(0, 60)
            rs += [Reservation(f"X{i}", "B", k, t0, t0 + rng.uniform(0.5, 6)) for k in range(k0, k0 + rng.randint(1, 6))]
        table.add(*rs)
        d = planner.evaluate(R1, BOX, 0.0, table)
        assert d.action is Action.WAIT_THEN_BYPASS, trial
        plan = d.plan
        assert not conflicts(plan.claims, rs, eps=1e-7), trial  # t* itself is safe
        step, t = 0.01, plan.t_arrive
        while t < plan.t_depart - step:  # and nothing earlier is
            assert conflicts(plan.claims, rs, shift=t - plan.t_depart), (trial, t)
            t += step


# ------------------------------------------------------------- reciprocal windows (Plan A, feature A)

def _oncoming(table, rid, s0, v=1.0, t0=0.0):
    prof = Profile(t0, -s0, v)
    table.add(*reservations_for_path(rid, AISLE, "B", prof, -1, -s0, 0.45, 0.45 + CFG.margins.space))


def coop_planner():
    return BypassPlanner(AISLE, SPEC, dataclasses.replace(CFG, coop_delays=(1.0, 2.0, 3.0)))


def test_coop_bypass_beats_a_long_wait_when_a_small_slip_opens_the_window():
    table = InMemoryReservationTable()
    _oncoming(table, "R2", 41.0, t0=4.0)
    plain = BypassPlanner(AISLE, SPEC).evaluate(R1, BOX, 4.0, table)
    coop = coop_planner().evaluate(R1, BOX, 4.0, table, shiftable=lambda r: True)
    assert plain.action is Action.WAIT_THEN_BYPASS
    assert coop.action is Action.COOP_BYPASS
    assert coop.plan.rolling and set(coop.plan.coop_shifts) == {"R2"}
    assert coop.cost_bypass < plain.cost_bypass


def test_coop_never_asks_robots_of_another_fleet():
    table = InMemoryReservationTable()
    _oncoming(table, "R2", 41.0, t0=4.0)
    d = coop_planner().evaluate(R1, BOX, 4.0, table, shiftable=lambda r: False)
    assert d.action is Action.WAIT_THEN_BYPASS


def test_coop_is_refused_if_the_shift_would_hit_a_third_robot():
    table = InMemoryReservationTable()
    _oncoming(table, "R2", 41.0, t0=4.0)
    _oncoming(table, "R3", 42.2, t0=4.0)  # right behind R2 in lane B: R2 cannot slip
    d = coop_planner().evaluate(R1, BOX, 4.0, table, shiftable=lambda r: r == "R2")
    assert d.action is not Action.COOP_BYPASS


def test_coop_claim_is_all_or_nothing():
    from sim.aisle_sim import Clock, SimRobot
    from visison import BypassController

    clock = Clock()
    clock.t = 4.0
    table = InMemoryReservationTable()
    _oncoming(table, "R2", 41.0, t0=4.0)
    robot = SimRobot("R1", SPEC, AISLE, "A", 8.0, SPEC.limits.v_max, +1, clock)
    ctl = BypassController(robot, coop_planner(), table)
    before = sorted((r.cell, r.start) for r in table.all() if r.robot_id == "R2")
    ctl.shiftable, ctl.coop_executor = (lambda r: True), (lambda rid, dl, t: False)  # R2 cannot slow
    d = ctl.on_obstacle(BOX, 4.0)
    assert d.action is not Action.COOP_BYPASS  # replanned without asking
    assert sorted((r.cell, r.start) for r in table.all() if r.robot_id == "R2") == before  # rolled back
