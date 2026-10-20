"""Interval algebra for the space-time window search.

Intervals are *open* sets ``(lo, hi)``: two intervals that only touch do not
conflict. Merging is conservative: touching intervals are fused, so an
isolated instant is never offered as a usable window.
"""

from __future__ import annotations

import math
from typing import Iterable, NamedTuple


class Interval(NamedTuple):
    lo: float
    hi: float

    @property
    def length(self) -> float:
        return self.hi - self.lo


def overlaps(a: Interval, b: Interval) -> bool:
    """True if the open intervals ``a`` and ``b`` share at least one point."""
    return a.lo < b.hi and b.lo < a.hi


def merge(intervals: Iterable[Interval]) -> list[Interval]:
    """Sort and fuse overlapping or touching intervals. Empty ones are dropped."""
    merged: list[Interval] = []
    for iv in sorted(Interval(*iv) for iv in intervals if iv[1] > iv[0]):
        if merged and iv.lo <= merged[-1].hi:
            if iv.hi > merged[-1].hi:
                merged[-1] = Interval(merged[-1].lo, iv.hi)
        else:
            merged.append(iv)
    return merged


def earliest_free_point(
    forbidden: Iterable[Interval], t_from: float, t_until: float = math.inf
) -> float | None:
    """Smallest ``t >= t_from`` outside every forbidden interval.

    Returns ``None`` if that point would be later than ``t_until``.
    Runs in O(n log n) because of the sort inside :func:`merge`.
    """
    t = t_from
    for iv in merge(forbidden):
        if iv.hi <= t:
            continue
        if iv.lo < t:  # t lies inside (lo, hi): jump to the end of the block
            t = iv.hi
        else:  # merged blocks are disjoint, so nothing later can cover t
            break
    return t if t <= t_until else None
