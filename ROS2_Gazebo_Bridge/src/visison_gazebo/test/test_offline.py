"""ROS-free checks of the Gazebo scenario pieces (run with plain pytest)."""

import math

import pytest

from visison import (
    Action, BypassPlanner, EmergencyConfig, InMemoryReservationTable, RobotState,
    displacement_trajectory, plan_chain_reverse,
)
from visison_gazebo.perception import detect_in_lane, detect_objects, drop_near, scan_to_world
from visison_gazebo.scenario import (
    AISLE, BOX_CENTER_S, CONFIG, FALLEN_BOX, LANE_A_Y, LANE_B_Y, R2_PARK_S, SPEC,
    fleet_layout, r1_reservations, r2_schedule,
)
from visison_gazebo.tracking import UnicycleTracker, reference


@pytest.mark.parametrize("arrival", [10.0, 30.0, 50.0, 90.0])
def test_r2_passes_the_box_at_the_requested_time(arrival):
    sch = r2_schedule(arrival)
    assert sch.profile().time_at(-BOX_CENTER_S) == pytest.approx(arrival, abs=1e-6)
    assert sch.spawn_s <= R2_PARK_S + 1e-9


def decide(arrival, s=12.0, t=13.0):
    table = InMemoryReservationTable()
    table.add(*r1_reservations(), *r2_schedule(arrival).reservations())
    state = RobotState("R1", s, SPEC.limits.v_max, "A", 1, LANE_A_Y)
    return BypassPlanner(AISLE, SPEC, CONFIG).evaluate(state, FALLEN_BOX, t, table)


def test_default_arrival_bypasses_now_and_early_arrival_waits():
    assert decide(50.0).action is Action.BYPASS_NOW
    assert decide(30.0).action is Action.WAIT_THEN_BYPASS


def simulate_unicycle(traj, t0, t1, dt=0.01, a_max=1.0, alpha_max=2.0, v_reverse=0.1):
    """Kinematic unicycle with the diff-drive plugin's acceleration limits."""
    tr = UnicycleTracker(v_max=SPEC.limits.v_max * 1.25, v_reverse=v_reverse)
    x, y, th, _, _ = reference(traj, t0)
    v = traj.sample(t0).v
    w, worst, t = 0.0, 0.0, t0
    while t < t1:
        xr, yr, thr, vr, wr = reference(traj, t)
        vc, wc = tr.command(x, y, th, xr, yr, thr, vr, wr)
        v += max(-a_max * dt, min(a_max * dt, vc - v))
        w += max(-alpha_max * dt, min(alpha_max * dt, wc - w))
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th += w * dt
        t += dt
        worst = max(worst, math.hypot(xr - x, yr - y))
    return worst


def test_tracker_follows_the_bypass_within_a_few_centimetres():
    d = decide(50.0)
    plan = d.plan
    err = simulate_unicycle(plan.trajectory, 13.0, plan.t_back + 5)
    assert err < 0.08


def test_lidar_detects_the_front_face_of_the_box():
    # synthetic scan from R1 at s=12: rays hitting the plane x=20 for y in [0.1, 1.1]
    x, y = 12.0, LANE_A_Y
    angle_min, inc, n = -math.pi, 2 * math.pi / 360, 360
    ranges = []
    for i in range(n):
        a = angle_min + i * inc
        r = math.inf
        if math.cos(a) > 1e-6:
            r_hit = (FALLEN_BOX.s_min - (x + 0.1)) / math.cos(a)
            if FALLEN_BOX.y_min <= y + r_hit * math.sin(a) <= FALLEN_BOX.y_max:
                r = r_hit
        ranges.append(r)
    pts = scan_to_world(ranges, angle_min, inc, 0.12, 12.0, x, y, 0.0)
    ob = detect_in_lane(pts, x, +1, AISLE.lane_bounds("A"), sensing_range=10.0, assumed_depth=1.0)
    assert ob is not None
    assert ob.s_min == pytest.approx(FALLEN_BOX.s_min, abs=1e-6)
    assert ob.s_max == pytest.approx(FALLEN_BOX.s_min + 1.0)
    assert ob.y_min >= FALLEN_BOX.y_min and ob.y_max <= FALLEN_BOX.y_max
    assert detect_in_lane(pts, x, +1, AISLE.lane_bounds("A"), sensing_range=5.0) is None


def synthetic_scan(x, y, boxes, n=720):
    """Ranges from a lidar at (x + 0.1, y), heading +x, to axis-aligned boxes (x0, x1, y0, y1)."""
    sx, sy = x + 0.1, y
    angle_min, inc = -math.pi, 2 * math.pi / n
    ranges = []
    for i in range(n):
        a = angle_min + i * inc
        c, s = math.cos(a), math.sin(a)
        best = math.inf
        for x0, x1, y0, y1 in boxes:  # slab intersection
            t0, t1 = 0.0, math.inf
            for o, d, lo, hi in ((sx, c, x0, x1), (sy, s, y0, y1)):
                if abs(d) < 1e-12:
                    if not lo <= o <= hi:
                        t0, t1 = 1.0, 0.0
                    continue
                ta, tb = sorted(((lo - o) / d, (hi - o) / d))
                t0, t1 = max(t0, ta), min(t1, tb)
            if t0 <= t1:
                best = min(best, t0)
        ranges.append(best)
    return scan_to_world(ranges, angle_min, inc, 0.12, 12.0, x, y, 0.0)


def test_detect_objects_finds_the_oncoming_robot_and_drop_near_removes_it():
    # R1 halfway past the box (from s=18 the box would hide R2: occlusion is real)
    r2 = (24.75, 25.25, LANE_B_Y - 0.245, LANE_B_Y + 0.245)
    pts = synthetic_scan(19.0, LANE_B_Y - 0.4, [r2, (FALLEN_BOX.s_min, FALLEN_BOX.s_max, 0.1, 1.1)])
    objs = detect_objects(pts, (0.0, 2 * AISLE.lane_width))
    near_r2 = [o for o in objs if abs(o[1] - LANE_B_Y) < 0.3]
    assert len(near_r2) == 1
    s, y, length, width = near_r2[0]
    assert r2[0] - 0.05 <= s - length / 2 and s + length / 2 <= r2[1] + 0.05
    assert any(abs(o[0] - FALLEN_BOX.s_min) < 0.5 for o in objs)  # the box is a cluster too
    rest = drop_near(pts, [(25.0, LANE_B_Y, 0.25, 0.245)])
    assert not any(abs(o[1] - LANE_B_Y) < 0.3 for o in detect_objects(rest, (0.0, 2 * AISLE.lane_width)))


@pytest.mark.parametrize("n_b,n_a", [(1, 0), (3, 0), (4, 2)])
def test_fleet_layout_is_consistent_for_n_robots(n_b, n_a):
    layout = fleet_layout(n_b, n_a, r2_arrival=50.0, r2_early=10.0)
    ids = [p.robot_id for p in layout]
    assert len(ids) == len(set(ids)) == 1 + n_a + n_b and ids[0] == "R1"
    for lane in ("A", "B"):
        spawns = sorted(p.spawn_s for p in layout if p.lane == lane)
        ends = sorted(p.end_s for p in layout if p.lane == lane)
        assert all(b - a >= 2.0 for a, b in zip(spawns, spawns[1:]))  # nobody spawns on top of another
        assert all(b - a >= 2.0 for a, b in zip(ends, ends[1:]))  # nor parks on top of another
    for k, p in enumerate(q for q in layout if q.lane == "B"):
        assert p.reserved.profile().time_at(-BOX_CENTER_S) == pytest.approx(50.0 + k * 2.5 / 0.8, abs=1e-6)
        assert p.schedule.profile().time_at(-BOX_CENTER_S) == pytest.approx(40.0 + k * 2.5 / 0.8, abs=1e-6)
        assert p.reservations() and all(r.robot_id == p.robot_id for r in p.reservations())


def test_default_layout_is_the_original_two_robot_scene():
    r1, r2 = fleet_layout()
    assert (r1.robot_id, r1.spawn_s, r1.end_s) == ("R1", 2.0, 38.0)
    assert (r2.robot_id, r2.end_s) == ("R2", -1.0) and r2.schedule == r2.reserved == r2_schedule(50.0)


def test_chain_reverse_fits_the_gazebo_robots():
    r1 = RobotState("R1", 20.4, 0.0, "A", 1, LANE_B_Y)
    others = {f"R{i}": (RobotState(f"R{i}", 21.6 + 2.5 * (i - 2), 0.0, "B", -1, LANE_B_Y), SPEC) for i in (2, 3, 4)}
    rp = plan_chain_reverse(BypassPlanner(AISLE, SPEC, CONFIG), r1, FALLEN_BOX, others, 0.0, EmergencyConfig())
    assert rp.feasible and "R2" in rp.moves and max(rp.moves.values()) < 3.0


def test_tracker_reverses_along_a_back_off_trajectory():
    st = RobotState("R2", 22.0, 0.0, "B", -1, LANE_B_Y)
    traj = displacement_trajectory(AISLE, st, SPEC, 1.5, +1, 0.0, 0.2)  # back off towards +s
    assert traj.sample(1.0).v < 0
    err = simulate_unicycle(traj, 0.0, traj.profile.t_end + 2.0, v_reverse=0.3)
    assert err < 0.05


def test_gazebo_drive_model_through_the_arbiter_respects_its_limits():
    """A robot stopped ahead and a slower leader, with the real amr_sim limits.

    Every command is executable and R1 never closes in. (A head-on robot that does
    not cooperate cannot be avoided by speed alone in a lane: that is the emergency
    layer's job, not ORCA's.)
    """
    from visison import Command, Kind, MotionArbiter, MotionIntent, OrcaAgent, OrcaMode, Profile, Source, Trajectory
    from visison import validate_command
    from visison_gazebo.scenario import MODEL

    dt = 0.05
    for v_other, s_other, d_other in ((0.0, 6.0, -1), (0.3, 4.0, +1)):
        ref = Trajectory(Profile(0.0, 0.0, 0.8), +1, LANE_A_Y, [], SPEC.lane_change)
        state = {"x": 0.0, "v": 0.8, "other": s_other}
        arb = MotionArbiter("R1", MODEL, dt, tracker=UnicycleTracker(v_max=1.0, v_reverse=0.3), orca_enabled=True,
                            neighbours=lambda rid, t, i: (
                                OrcaAgent("R1", (state["x"], LANE_A_Y), (state["v"], 0.0), 0.5),
                                [OrcaAgent("R2", (state["other"], LANE_A_Y), (d_other * v_other, 0.0), 0.5, False)],
                                [], OrcaMode.PATH_SPEED_ONLY))
        arb.request(MotionIntent(Source.PLANNER, Kind.FOLLOW, ref), 0.0)
        cur, closest = Command(0.8, 0.0), math.inf
        for k in range(int(15 / dt)):
            cmd = arb.step(k * dt, (state["x"], LANE_A_Y, 0.0), cur)
            assert not validate_command(MODEL, cur, cmd, dt)
            state["x"] += cmd.v * dt
            state["v"], cur = cmd.v, cmd
            state["other"] += d_other * v_other * dt
            closest = min(closest, abs(state["other"] - state["x"]))
        assert closest >= 1.0 - 0.05  # the two 0.5 m discs never overlapped
