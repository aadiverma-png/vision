import math
import random

import pytest

from sim.aisle_sim import AISLE, SPEC, Clock, SimRobot
from visison import (
    BypassController,
    BypassPlanner,
    ControllerState,
    EmergencyConfig,
    EmergencyCoordinator,
    FleetMember,
    InMemoryReservationTable,
    LoggingReporter,
    ObjectTracker,
    Obstacle,
    Profile,
    RiskLevel,
    RobotState,
    StandoffPhase,
    TrackedObject,
    Trajectory,
    collision_risk,
    conflict_groups,
    plan_chain_reverse,
)

CFG = EmergencyConfig()
BOX = Obstacle(20.0, 20.6, 0.1, 1.1, obstacle_id="box")
Y_A, Y_B = AISLE.lane_center("A"), AISLE.lane_center("B")
H = SPEC.footprint.length / 2
GAP = CFG.min_gap + BypassPlanner(AISLE, SPEC).cfg.margins.space


def robot(s, vs, y=Y_A, length=0.9):
    return TrackedObject(s, y, vs, 0.0, length, 0.6)


def band(y=Y_A):
    return y - 0.35, y + 0.35


def risk(gap, v_e=1.0, v_o=-1.0, decel=2.0, **kw):
    ego, other = robot(10.0, v_e), robot(10.0 + 0.9 + gap, v_o)
    return collision_risk(ego, band(), decel, other, kw.get("obj_decel", decel), CFG, "E", "O")


# --------------------------------------------------------------- collision risk


def test_head_on_too_close_brakes_and_far_away_is_safe():
    assert risk(0.5).level is RiskLevel.BRAKE and risk(0.5).probability > 0.99
    far = risk(10.0)
    assert far.level is RiskLevel.NONE and far.probability < 1e-6
    assert far.ttc == pytest.approx(10.0 / 2.0)


def test_probability_is_one_half_at_zero_margin_and_falls_with_the_gap():
    r = risk(3.0)
    at_zero = risk(3.0 - r.margin)
    assert at_zero.margin == pytest.approx(0.0, abs=1e-9)
    assert at_zero.probability == pytest.approx(0.5, abs=1e-9)
    ps = [risk(g).probability for g in (0.4, 0.8, 1.2, 1.6, 2.0, 3.0)]
    assert all(a > b for a, b in zip(ps, ps[1:]))


def test_head_on_stop_distance_matches_the_closed_form():
    r = risk(3.0)
    v, b = 1.0, 2.0
    expected = v * CFG.t_react_ego + v * v / (2 * b) + v * CFG.t_react_other + v * v / (2 * b)
    assert r.stop_distance == pytest.approx(expected)


def test_diverging_standing_or_laterally_separated_is_no_risk():
    assert risk(0.3, v_e=-1.0, v_o=1.0).level is RiskLevel.NONE  # moving apart
    assert risk(0.3, v_e=0.0, v_o=0.0).level is RiskLevel.NONE  # both standing
    ego, other = robot(10.0, 1.0), robot(11.5, -1.0, y=Y_B)
    assert collision_risk(ego, band(), 2.0, other, 2.0, CFG).level is RiskLevel.NONE


def test_same_direction_follower_is_judged_on_relative_speed():
    ego, leader = robot(10.0, 1.0), robot(12.0, 1.0)
    assert collision_risk(ego, band(), 2.0, leader, 2.0, CFG).level is RiskLevel.NONE
    slow = robot(11.3, 0.2)  # 0.4 m ahead, much slower
    assert collision_risk(ego, band(), 2.0, slow, 2.0, CFG).level is RiskLevel.BRAKE


def test_unknown_object_is_assumed_not_to_brake():
    coop = risk(2.2, obj_decel=2.0)
    unknown = collision_risk(robot(10.0, 1.0), band(), 2.0, robot(10.9 + 2.2, -1.0), None, CFG)
    assert unknown.stop_distance > coop.stop_distance
    assert not unknown.other_is_robot


# ------------------------------------------------------------------- tracking


def test_tracker_estimates_velocity_of_a_moving_cluster():
    rng = random.Random(3)
    tr = ObjectTracker()
    tracks = []
    for k in range(30):
        t = 0.1 * k
        tracks = tr.update(t, [(30.0 - 0.8 * t + rng.gauss(0, 0.01), Y_B + rng.gauss(0, 0.01), 0.9, 0.6),
                               (20.3, 0.6, 0.6, 1.0)])
    by_s = sorted(tracks, key=lambda x: x.s)
    assert len(tracks) == 2
    assert by_s[0].vs == pytest.approx(0.0, abs=0.05)  # the static box
    assert by_s[1].vs == pytest.approx(-0.8, abs=0.05)
    assert tr.update(3.1, []) == []  # missed scan: nothing confirmed this time


# ------------------------------------------------------------ trajectories


def test_reverse_trajectory_faces_forward_and_moves_backwards():
    prof = Profile(0.0, 25.0, 0.0).change_speed(0.2, 0.8, 1.2)  # travel towards +s
    traj = Trajectory(prof, +1, Y_B, [], SPEC.lane_change, reverse=True)
    p = traj.sample(2.0)
    assert p.v < 0 and p.s > 25.0
    assert abs(math.cos(p.heading) + 1.0) < 1e-9  # body faces -s, like R2


def test_rejoin_plan_passes_the_box_and_ends_at_the_lane_centre():
    pl = BypassPlanner(AISLE, SPEC)
    st = RobotState("R1", 20.3, 0.0, "A", 1, Y_B)  # beside the box, in lane B
    plan = pl.rejoin_plan(st, BOX, 10.0, 0.2)
    assert plan.kind == "rejoin"
    end = plan.trajectory.sample(plan.t_back)
    assert end.y == pytest.approx(Y_A, abs=1e-6)
    start_in = plan.trajectory.shifts[-1].t_start
    assert plan.trajectory.sample(start_in).s - H >= BOX.s_max + pl.cfg.clearance - 1e-6
    assert plan.claims and all(c.lane == "B" for c in plan.claims)


def test_rejoin_plan_returns_before_the_box_when_that_is_shorter():
    pl = BypassPlanner(AISLE, SPEC)
    st = RobotState("R1", 15.0, 0.0, "A", 1, Y_A + 0.5)  # just started to change out
    plan = pl.rejoin_plan(st, BOX, 0.0, 0.2)
    assert plan.kind == "return"
    assert plan.s_back + H < BOX.s_min
    assert plan.trajectory.sample(plan.t_back + 5).v == 0.0


# ---------------------------------------------------------------- chain reverse


def lane_b(*positions):
    return {f"R{i + 2}": (RobotState(f"R{i + 2}", s, 0.0, "B", -1, Y_B), SPEC) for i, s in enumerate(positions)}


R1_BESIDE = RobotState("R1", 20.3, 0.0, "A", 1, Y_B)


@pytest.mark.parametrize("n", [1, 3, 6])
def test_chain_reverse_matches_the_recurrence(n):
    pl = BypassPlanner(AISLE, SPEC)
    others = lane_b(*(22.4 + 1.4 * i for i in range(n)))  # a tight queue in front of R1
    rp = plan_chain_reverse(pl, R1_BESIDE, BOX, others, 0.0, CFG)
    assert rp.feasible, rp.reason
    req = rp.reach + GAP
    for rid, (st, _) in sorted(others.items(), key=lambda kv: kv[1][0].s):
        need = req + H - st.s
        if need <= 1e-6:
            assert rid not in rp.moves
            break
        assert rp.moves[rid] == pytest.approx(need)
        req = st.s + need + H + GAP
    # after the moves every neighbour pair keeps the gap
    final = sorted([rp.reach - H] + [rp.targets.get(r, st.s) for r, (st, _) in others.items()])
    assert all(b - a >= 2 * H + GAP - 1e-9 for a, b in zip(final, final[1:]))


def test_chain_stops_at_the_first_robot_that_does_not_have_to_move():
    rp = plan_chain_reverse(BypassPlanner(AISLE, SPEC), R1_BESIDE, BOX, lane_b(22.4, 30.0, 31.4), 0.0, CFG)
    assert set(rp.moves) == {"R2"}


def test_chain_reverse_is_infeasible_beyond_max_reverse_or_the_aisle_end():
    pl = BypassPlanner(AISLE, SPEC)
    queue = lane_b(*(22.4 + 1.0 * i for i in range(16)))  # pushed past s=40
    small = EmergencyConfig(max_reverse=0.5)
    assert not plan_chain_reverse(pl, R1_BESIDE, BOX, lane_b(22.4), 0.0, small).feasible
    rp = plan_chain_reverse(pl, R1_BESIDE, BOX, lane_b(22.4, 38.9), 0.0, CFG)
    assert rp.feasible  # R3 is far enough to stay put
    blocked = plan_chain_reverse(pl, R1_BESIDE, BOX, queue, 0.0, EmergencyConfig(max_reverse=20))
    assert not blocked.feasible and "aisle" in blocked.reason
    wall = plan_chain_reverse(pl, R1_BESIDE, BOX, lane_b(22.4), 0.0, CFG, barriers=[("B", 23.8, 24.5)])
    assert not wall.feasible


def test_conflict_groups_merge_transitively():
    groups = conflict_groups([("R1", "R2"), ("R3", "R4"), ("R2", "R5"), ("R6", "R6")])
    assert groups == [{"R1", "R2", "R5"}, {"R3", "R4"}, {"R6"}]


# ------------------------------------------------------ coordinator end to end


def fleet(*setups):
    clock = Clock()
    table = InMemoryReservationTable()
    reporter = LoggingReporter()
    coord = EmergencyCoordinator(BypassPlanner(AISLE, SPEC), table, reporter, CFG)
    robots, ctls = {}, {}
    for rid, lane, s, v, d in setups:
        rob = SimRobot(rid, SPEC, AISLE, lane, s, v, d, clock)
        ctl = BypassController(rob, BypassPlanner(AISLE, SPEC), table, reporter)
        coord.register(rid, FleetMember(rob, SPEC, ctl, cruise_speed=v))
        robots[rid], ctls[rid] = rob, ctl
    return clock, coord, robots, ctls, table


def step(clock, coord, ctls, t_end, dt=0.05, alerts=lambda t: ()):
    while clock.t < t_end:
        clock.t += dt
        coord.tick(clock.t, alerts=alerts(clock.t))
        for c in ctls.values():
            c.tick(clock.t)


def test_controller_emergency_rejoin_releases_its_claims():
    clock, coord, robots, ctls, table = fleet(("R1", "A", 2.0, 1.5, 1))
    ctl = ctls["R1"]
    d = ctl.on_obstacle(BOX, 0.0)
    step(clock, coord, ctls, d.plan.t_out + 5.0)  # in the other lane now
    ctl.emergency_stop(clock.t, "test")
    assert ctl.state is ControllerState.EMERGENCY_STOP
    assert all(c in table.all() for c in d.plan.claims)  # still owns the lane
    step(clock, coord, ctls, clock.t + 3.0)
    st = robots["R1"].state()
    plan = ctl.planner.rejoin_plan(st, BOX, clock.t, 0.2)
    ctl.rejoin(clock.t, plan)
    step(clock, coord, ctls, plan.t_back + 0.2)
    assert ctl.state is ControllerState.REJOINED
    assert not any(r.lane == "B" for r in table.all())
    assert robots["R1"].state().y == pytest.approx(Y_A, abs=1e-6)


def test_deadlock_of_facing_robots_is_resolved_without_an_alert():
    clock, coord, robots, ctls, _ = fleet(("R1", "A", 10.0, 0.0, 1), ("R2", "A", 12.0, 0.0, -1))
    step(clock, coord, ctls, CFG.deadlock_time + 0.5)
    assert coord.history or coord.standoffs
    so = (coord.standoffs + coord.history)[0]
    assert so.members <= {"R1", "R2"} and "deadlock" in coord.reports[0].message


def test_unknown_object_holds_the_robot_until_it_is_gone():
    clock, coord, robots, ctls, _ = fleet(("R1", "A", 10.0, 1.0, 1))
    person = TrackedObject(12.0, Y_A, 0.0, 0.0, 0.3, 0.3, "p1")
    present = [True]

    def tracks(t):
        return {"R1": [person] if present[0] else []}

    while not coord.standoffs and clock.t < 3.0:
        clock.t += 0.05
        coord.tick(clock.t, tracks(clock.t))
    so = coord.standoffs[0]
    assert so.unknown == {"track:p1"}
    for _ in range(40):
        clock.t += 0.05
        coord.tick(clock.t, tracks(clock.t))
    assert so.phase is StandoffPhase.WAIT_UNKNOWN and robots["R1"].state().v == 0.0
    stopped_at = robots["R1"].state().s
    assert stopped_at + 0.45 + CFG.min_gap < person.s - 0.15  # stopped short of the person
    present[0] = False
    for _ in range(60):
        clock.t += 0.05
        coord.tick(clock.t, tracks(clock.t))
    assert not coord.standoffs and robots["R1"].state().v > 0.5


# ---------------------------------------------------------- graded response (Plan A, feature B)

def test_small_timing_error_is_avoided_by_orca_without_a_brake():
    from sim.aisle_sim import SCENARIOS, brakes, run

    base = run(SCENARIOS["orca_small_error"], quiet=True, orca=False)
    orca = run(SCENARIOS["orca_small_error"], quiet=True)
    assert brakes(base) >= 1 and brakes(orca) == 0
    assert not orca.collisions and orca.fleet_delay < base.fleet_delay


def test_hard_alarm_always_brakes_even_with_orca():
    cfg = EmergencyConfig()
    clock, coord, robots, ctls, _ = fleet(("R1", "A", 10.0, 1.0, 1), ("R2", "A", 11.3, 0.0, -1))
    for m in coord.members.values():
        m.orca = True
    a = collision_risk(robot(10.0, 1.0), band(), 2.0, robot(11.3, 0.0), 2.0, cfg, "R1", "R2")
    assert a.probability >= cfg.p_hard
    clock.t = 0.05
    coord.tick(clock.t, alerts=[a])
    assert coord.standoffs  # braked, not "avoided"
