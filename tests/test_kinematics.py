import math
import random

import pytest

from visison.kinematics import (
    HolonomicLaneChange,
    InfeasibleMotion,
    MotionLimits,
    Profile,
    QuinticLaneChange,
    trapezoid,
    travel_time,
)

LIM = MotionLimits(v_max=1.5, accel=0.8, decel=1.2)


def integrate(v0, pieces, n=4000):
    """Euler-integrate (accel, duration) pieces; returns distance, final and peak speed."""
    u, v, vmax = 0.0, v0, v0
    for a, dt in pieces:
        h = dt / n
        for _ in range(n):
            u += v * h + 0.5 * a * h * h
            v += a * h
            vmax = max(vmax, v)
    return u, v, vmax


@pytest.mark.parametrize("d,v0,v_end", [(10, 0, 0), (0.5, 0, 0), (3, 1.0, 0.3), (8, 1.5, 1.0), (2, 0.2, 1.2)])
def test_trapezoid_covers_distance_and_hits_speeds(d, v0, v_end):
    pieces = trapezoid(d, v0, v_end, LIM.v_max, LIM.accel, LIM.decel)
    u, v, vmax = integrate(v0, pieces)
    assert u == pytest.approx(d, abs=1e-6)
    assert v == pytest.approx(v_end, abs=1e-6)
    assert vmax <= LIM.v_max + 1e-9
    assert all(dt >= 0 for _, dt in pieces)


def test_trapezoid_rest_to_rest_matches_closed_form():
    d = 20.0
    expected = d / LIM.v_max + LIM.v_max / (2 * LIM.accel) + LIM.v_max / (2 * LIM.decel)
    assert travel_time(d, 0, 0, LIM) == pytest.approx(expected)


def test_trapezoid_triangular_peak_below_vmax():
    d = 0.5
    pieces = trapezoid(d, 0, 0, LIM.v_max, LIM.accel, LIM.decel)
    vp = LIM.accel * pieces[0][1]
    assert vp == pytest.approx(math.sqrt(2 * LIM.accel * LIM.decel * d / (LIM.accel + LIM.decel)))
    assert pieces[1][1] == pytest.approx(0)


@pytest.mark.parametrize("d,v0,v_end", [(0.1, 1.5, 0), (0.1, 0, 1.5), (-1, 0, 0), (1, 2.0, 0)])
def test_trapezoid_infeasible(d, v0, v_end):
    with pytest.raises(InfeasibleMotion):
        trapezoid(d, v0, v_end, LIM.v_max, LIM.accel, LIM.decel)


def test_profile_time_at_inverts_state():
    rng = random.Random(1)
    p = Profile(2.0, 1.0, 0.4).move(6.0, LIM, v_end=0.0).hold(3.0).change_speed(1.0, LIM.accel, LIM.decel).cruise(2.0)
    for _ in range(200):
        t = rng.uniform(p.t0, p.t_end + 2)
        u, v = p.state(t)
        if v > 1e-3:  # the inverse is unique only while moving
            assert p.time_at(u) == pytest.approx(t, abs=1e-6)
    assert p.time_at(p.u0 - 1) == p.t0


def test_profile_never_reaches_point_beyond_stop():
    p = Profile(0, 0, 1.0).move(2.0, LIM, v_end=0.0)
    assert p.time_at(2.0 + 1e-3) == math.inf


def num_derivative(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_quintic_duration_gives_max_lateral_accel():
    model, w = QuinticLaneChange(max_lateral_accel=0.5), 1.2
    T = model.duration(w)

    def y(t):
        return w * model.progress(w, t / T)

    peak = max(abs(num_derivative(lambda s: num_derivative(y, s), T * i / 400)) for i in range(1, 400))
    assert peak == pytest.approx(0.5, rel=1e-3)
    assert model.progress(w, 0) == 0 and model.progress(w, 1) == 1
    assert num_derivative(y, 1e-3) == pytest.approx(0, abs=1e-4)


@pytest.mark.parametrize("w", [0.2, 1.2, 5.0])
def test_holonomic_respects_limits(w):
    model = HolonomicLaneChange(max_lateral_speed=0.6, max_lateral_accel=0.4)
    T = model.duration(w)

    def y(t):
        return w * model.progress(w, t / T)

    ts = [T * i / 2000 for i in range(1, 2000)]
    assert max(abs(num_derivative(y, t)) for t in ts) <= 0.6 + 1e-6
    assert model.progress(w, 0) == 0
    assert model.progress(w, 1) == pytest.approx(1)
    ys = [model.progress(w, i / 100) for i in range(101)]
    assert ys == sorted(ys)
