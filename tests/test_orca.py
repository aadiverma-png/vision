import math
import random

import pytest

from visison import (
    Command,
    DiffDrive,
    Line,
    OrcaAgent,
    OrcaConfig,
    OrcaMode,
    orca_1d,
    orca_half_plane,
    orca_solve,
    validate_command,
)
from visison.orca import linear_program_2d, linear_program_3d, wall_half_planes

DD = DiffDrive(track_width=0.4, v_wheel_max=1.2, a_acc=0.8, a_dec=1.2, alpha_max=3.0, w_max=1.5)
CFG = OrcaConfig()


def test_line_from_inequality_keeps_the_feasible_side_on_the_left():
    ln = Line.from_inequality((1.0, 0.0), 2.0)  # x <= 2
    assert ln.violation((1.0, 5.0)) < 0 and ln.violation((3.0, -1.0)) > 0
    assert ln.violation((2.0, 7.0)) == pytest.approx(0.0, abs=1e-12)


def test_half_plane_cut_off_circle_matches_closed_form():
    # 4 m apart, closing at 1 m/s, combined radius 1: no contact within tau = 2 s, so the
    # constraint is the cut-off circle: centre p_rel / tau = (2, 0), radius r / tau = 0.5
    a = OrcaAgent("a", (0.0, 0.0), (1.0, 0.0), 0.5)
    b = OrcaAgent("b", (4.0, 0.0), (0.0, 0.0), 0.5)
    line, u = orca_half_plane(a, b, tau=2.0, dt=0.05, responsibility=0.5)
    assert u == pytest.approx((0.5, 0.0))  # from v_rel = (1, 0) to the circle's edge at (1.5, 0)
    assert line.point == pytest.approx((1.25, 0.0))  # own velocity + half of u
    assert line.violation((1.2, 0.3)) <= 0 and line.violation((1.3, 0.0)) > 0  # x <= 1.25


def test_half_plane_leg_projection_blocks_a_collision_course():
    a = OrcaAgent("a", (0.0, 0.0), (1.0, 0.0), 0.5)
    b = OrcaAgent("b", (4.0, 0.0), (-1.5, 0.0), 0.5)
    line, u = orca_half_plane(a, b, 2.0, 0.05, 0.5)
    assert line.violation(a.p_dot) > 0  # the current velocity collides within tau
    assert math.hypot(*u) > 0


def test_lp_matches_brute_force():
    rng = random.Random(4)
    for _ in range(40):
        lines = [Line.from_inequality((rng.uniform(-1, 1), rng.uniform(-1, 1)), rng.uniform(0.1, 1.0))
                 for _ in range(4)]
        opt = (rng.uniform(-2, 2), rng.uniform(-2, 2))
        res, n = linear_program_2d(lines, 5.0, opt)
        if n < len(lines):
            continue
        grid = [(x / 40, y / 40) for x in range(-200, 201) for y in range(-200, 201)]
        feas = [c for c in grid if all(ln.violation(c) <= 0 for ln in lines) and math.hypot(*c) <= 5.0]
        best = min(feas, key=lambda c: math.dist(c, opt))
        assert all(ln.violation(res) <= 1e-7 for ln in lines)
        assert math.dist(res, opt) <= math.dist(best, opt) + 1e-9


def test_3d_program_keeps_hard_lines_and_minimises_the_worst_violation():
    hard = [Line.from_inequality((1.0, 0.0), 1.0)]  # x <= 1 must hold
    soft = [Line.from_inequality((-1.0, 0.0), -2.0), Line.from_inequality((0.0, 1.0), -1.0)]  # x >= 2, y <= -1
    lines = hard + soft
    res, n = linear_program_2d(lines, 10.0, (0.0, 0.0))
    assert n < len(lines)
    res = linear_program_3d(lines, 1, n, 10.0, res)
    assert hard[0].violation(res) <= 1e-9
    # x >= 2 can at best be missed by 1 (x = 1); the program reaches exactly that optimum
    assert max(ln.violation(res) for ln in soft) == pytest.approx(1.0, abs=1e-6)


def test_non_cooperative_neighbour_leaves_all_the_avoidance_to_us():
    a = OrcaAgent("a", (0.0, 0.0), (1.0, 0.0), 0.5)
    b = OrcaAgent("b", (3.0, 0.0), (-1.0, 0.0), 0.5)
    half, u = orca_half_plane(a, b, 2.0, 0.05, 0.5)
    full, u2 = orca_half_plane(a, b, 2.0, 0.05, 1.0)
    assert u == u2
    assert full.point[0] == pytest.approx(half.point[0] + 0.5 * u[0])


def test_walls_bound_lateral_motion_and_never_push_an_outside_robot():
    walls = wall_half_planes((0.0, 0.6), 0.3, (0.0, 1.2), tau_obs=1.0)
    assert all(w.violation((0.0, 0.0)) <= 0 for w in walls)
    assert walls[0].violation((0.0, 0.5)) > 0  # moving sideways faster than the room allows
    outside = wall_half_planes((0.0, 1.1), 0.3, (0.0, 1.2), tau_obs=1.0)
    assert outside[0].violation((0.0, 0.0)) <= 0  # standing still is fine, no forced motion


def test_speed_only_mode_keeps_the_path():
    ego = OrcaAgent("e", (0.15, 0.0), (0.8, 0.0), 0.6)
    ahead = OrcaAgent("o", (2.0, 0.0), (0.0, 0.0), 0.6, cooperative=False)
    pref = Command(0.8, 0.4)
    r = orca_solve(DD, Command(0.8, 0.4), pref, 0.0, ego, [ahead], [], CFG, 0.05, OrcaMode.PATH_SPEED_ONLY)
    assert r.cmd.v < pref.v
    assert r.cmd.w / r.cmd.v == pytest.approx(pref.w / pref.v)  # same curvature: same path


def test_braking_mode_off_leaves_the_command_alone():
    pref = Command(0.3, 0.0)
    r = orca_solve(DD, pref, pref, 0.0, OrcaAgent("e", (0, 0), (0.3, 0), 0.5), [], [], CFG, 0.05, OrcaMode.OFF)
    assert r.cmd == pref and r.feasible


def test_orca_1d_matches_the_closing_speed_rule():
    # 2 m gap, tau 2 s: closing speed may be at most 1 m/s; the excess is shared
    assert orca_1d(1.5, 1.5, 2.0, 0.0, 2.0) == pytest.approx(1.5 - 0.5 * 0.5)
    assert orca_1d(1.0, 1.0, 4.0, 0.0, 2.0) == 1.0  # no conflict within tau
    assert orca_1d(1.5, 1.5, 2.0, 0.0, 2.0, responsibility=1.0) == pytest.approx(1.0)


def test_head_on_pair_in_open_space_passes_without_contact_and_within_limits():
    dt, cfg = 0.05, OrcaConfig()
    R = DD.effective_radius(0.6, 0.5, 0.05)
    robots = [[0.0, 0.0, 0.0, Command(0.8, 0.0)], [10.0, 0.0, math.pi, Command(0.8, 0.0)]]
    goals = [(10.0, 0.0), (0.0, 0.0)]
    closest = math.inf
    for _ in range(600):
        agents = [OrcaAgent(str(i), (x + DD.D * math.cos(th), y + DD.D * math.sin(th)),
                            (c.v * math.cos(th), c.v * math.sin(th)), R) for i, (x, y, th, c) in enumerate(robots)]
        new = []
        for i, (x, y, th, c) in enumerate(robots):
            gx, gy = goals[i]
            err = math.atan2(math.sin(math.atan2(gy - y, gx - x) - th), math.cos(math.atan2(gy - y, gx - x) - th))
            pref = Command(0.8 if math.hypot(gx - x, gy - y) > 0.3 else 0.0, 2.0 * err)
            r = orca_solve(DD, c, pref, th, agents[i], [agents[1 - i]], [], cfg, dt)
            assert not validate_command(DD, c, r.cmd, dt)
            new.append([x + r.cmd.v * math.cos(th) * dt, y + r.cmd.v * math.sin(th) * dt, th + r.cmd.w * dt, r.cmd])
        robots = new
        closest = min(closest, math.dist(robots[0][:2], robots[1][:2]))
    assert closest >= 2 * R - 0.05  # effective discs never overlapped (up to one step of motion)
    assert math.dist(robots[0][:2], goals[0]) < 0.35 and math.dist(robots[1][:2], goals[1]) < 0.35
