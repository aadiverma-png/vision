"""Aisle geometry in lane coordinates.

The aisle frame uses ``s``, the distance along the aisle in metres, and ``y``,
the lateral position in metres. Lane ``A`` covers ``y`` in ``[0, w]`` and
lane ``B`` covers ``[w, 2w]``. Each lane is split along ``s`` into
reservation cells of length ``cell_size``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Aisle:
    lane_width: float
    length: float
    cell_size: float = 0.5
    lanes: tuple[str, str] = ("A", "B")
    aisle_id: str = "aisle"

    @property
    def n_cells(self) -> int:
        return math.ceil(self.length / self.cell_size)

    def lane_bounds(self, lane: str) -> tuple[float, float]:
        i = self.lanes.index(lane)
        return i * self.lane_width, (i + 1) * self.lane_width

    def lane_center(self, lane: str) -> float:
        lo, hi = self.lane_bounds(lane)
        return 0.5 * (lo + hi)

    def other_lane(self, lane: str) -> str:
        return self.lanes[1 - self.lanes.index(lane)]

    def cells(self, s_lo: float, s_hi: float) -> range:
        """Indices of the cells that intersect ``(s_lo, s_hi)``."""
        k0 = max(math.floor(s_lo / self.cell_size), 0)
        k1 = min(math.ceil(s_hi / self.cell_size), self.n_cells)
        return range(k0, max(k0, k1))

    def cell_bounds(self, k: int) -> tuple[float, float]:
        return k * self.cell_size, min((k + 1) * self.cell_size, self.length)


@dataclass(frozen=True)
class Footprint:
    length: float  # along the direction of travel
    width: float


@dataclass(frozen=True)
class Obstacle:
    """Axis-aligned bounding box of a detected obstacle, in the aisle frame."""

    s_min: float
    s_max: float
    y_min: float
    y_max: float
    obstacle_id: str = "obstacle"
    confidence: float = 1.0

    def overlaps_lateral(self, y_lo: float, y_hi: float) -> bool:
        return self.y_min < y_hi and y_lo < self.y_max

    def overlaps_longitudinal(self, s_lo: float, s_hi: float) -> bool:
        return self.s_min < s_hi and s_lo < self.s_max


def squeeze_offset(
    aisle: Aisle, lane: str, obstacle: Obstacle, footprint: Footprint, clearance: float
) -> float | None:
    """Lateral centre that lets the robot pass without leaving its lane.

    A squeeze is possible when the free width beside the obstacle is at
    least ``footprint.width + 2 * clearance``. The robot then drives in the
    middle of the widest gap. Returns ``None`` if no gap is wide enough.
    """
    lo, hi = aisle.lane_bounds(lane)
    gaps = [(obstacle.y_min - lo, lo), (hi - obstacle.y_max, obstacle.y_max)]
    need = footprint.width + 2 * clearance
    width, start = max(gaps)
    return start + width / 2 if width >= need else None


@dataclass(frozen=True)
class OrientedBox:
    cx: float
    cy: float
    length: float
    width: float
    heading: float = 0.0

    def axes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        c, s = math.cos(self.heading), math.sin(self.heading)
        return (c, s), (-s, c)

    def corners(self) -> list[tuple[float, float]]:
        (ax, ay), (bx, by) = self.axes()
        hl, hw = self.length / 2, self.width / 2
        return [
            (self.cx + i * hl * ax + j * hw * bx, self.cy + i * hl * ay + j * hw * by)
            for i, j in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ]

    @classmethod
    def of_obstacle(cls, ob: Obstacle) -> OrientedBox:
        return cls(
            (ob.s_min + ob.s_max) / 2, (ob.y_min + ob.y_max) / 2,
            ob.s_max - ob.s_min, ob.y_max - ob.y_min,
        )


def boxes_overlap(a: OrientedBox, b: OrientedBox) -> bool:
    """Separating-axis test for two rectangles. Touching does not count."""
    ca, cb = a.corners(), b.corners()
    for ax, ay in (*a.axes(), *b.axes()):
        pa = [x * ax + y * ay for x, y in ca]
        pb = [x * ax + y * ay for x, y in cb]
        if max(pa) <= min(pb) or max(pb) <= min(pa):
            return False
    return True
