import math
import random

from visison.intervals import Interval, earliest_free_point, merge, overlaps


def test_merge_fuses_overlapping_touching_and_nested():
    ivs = [(5, 6), (1, 3), (3, 4), (2, 2.5), (7, 7), (8, 9), (8.5, 10)]
    assert merge(ivs) == [Interval(1, 4), Interval(5, 6), Interval(8, 10)]


def test_merge_drops_empty():
    assert merge([(1, 1), (2, 1)]) == []


def test_overlaps_is_open():
    assert not overlaps(Interval(0, 1), Interval(1, 2))
    assert overlaps(Interval(0, 1.0001), Interval(1, 2))


def test_earliest_free_point_cases():
    f = [Interval(1, 3), Interval(3, 5), Interval(7, 8)]
    assert earliest_free_point(f, 0) == 0
    assert earliest_free_point(f, 1) == 1  # boundary of an open interval is free
    assert earliest_free_point(f, 2) == 5  # touching blocks are fused
    assert earliest_free_point(f, 7.5) == 8
    assert earliest_free_point(f, 2, t_until=4.9) is None
    assert earliest_free_point([], 3) == 3
    assert earliest_free_point([Interval(-math.inf, math.inf)], 0) == math.inf


def brute(forbidden, t_from, step=1e-3):
    t = t_from
    while any(iv.lo < t < iv.hi for iv in forbidden):
        t += step
    return t


def test_earliest_free_point_matches_brute_force():
    rng = random.Random(7)
    for _ in range(300):
        f = []
        for _ in range(rng.randint(0, 8)):
            lo = rng.uniform(0, 20)
            f.append(Interval(lo, lo + rng.uniform(0.1, 4)))
        t0 = rng.uniform(0, 15)
        exact = earliest_free_point(f, t0)
        assert not any(iv.lo < exact < iv.hi for iv in f)
        assert abs(brute(f, t0) - exact) < 2e-3  # random floats never touch exactly
