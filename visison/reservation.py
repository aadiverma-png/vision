"""Space-time reservations: the interface and a reference in-memory table.

A reservation says that ``robot_id`` may occupy cell ``cell`` of ``lane``
during the open time interval ``(start, end)``. Any fleet manager can be
connected by implementing :class:`ReservationProvider`. The only strict
requirement is that ``try_reserve`` is atomic (all claims granted or none).
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, Protocol, Sequence

from .geometry import Aisle
from .kinematics import Profile


@dataclass(frozen=True)
class Reservation:
    robot_id: str
    lane: str
    cell: int
    start: float
    end: float

    def conflicts_with(self, other: Reservation) -> bool:
        return (
            self.robot_id != other.robot_id
            and self.lane == other.lane
            and self.cell == other.cell
            and self.start < other.end
            and other.start < self.end
        )


class ReservationProvider(Protocol):
    def query(
        self, lane: str, cells: Iterable[int], t_from: float, exclude_robot: str | None = None
    ) -> list[Reservation]:
        """Reservations on ``cells`` of ``lane`` that end after ``t_from``."""
        ...

    def try_reserve(self, claims: Sequence[Reservation]) -> bool:
        """Grant every claim, or none of them if any claim conflicts."""
        ...

    def release(self, claims: Sequence[Reservation]) -> None:
        ...

    def shift(
        self, robot_id: str, delay: float, after: float, keep: Iterable[Reservation] = ()
    ) -> list[Reservation]:
        """Delay a robot's reservations that start at or after ``after``.

        Reservations listed in ``keep`` are not moved. Returns the other
        robots' reservations that now conflict, so the fleet manager can
        resolve them.
        """
        ...


class InMemoryReservationTable:
    """Thread-safe reference implementation. Useful for simulation and tests."""

    def __init__(self) -> None:
        self._cells: dict[tuple[str, int], list[Reservation]] = defaultdict(list)
        self._lock = threading.Lock()

    def add(self, *reservations: Reservation) -> None:
        with self._lock:
            for r in reservations:
                self._cells[(r.lane, r.cell)].append(r)

    def all(self) -> list[Reservation]:
        with self._lock:
            return [r for rs in self._cells.values() for r in rs]

    def query(self, lane, cells, t_from, exclude_robot=None):
        with self._lock:
            return [
                r
                for k in cells
                for r in self._cells.get((lane, k), ())
                if r.end > t_from and r.robot_id != exclude_robot
            ]

    def try_reserve(self, claims):
        with self._lock:
            for c in claims:
                if any(c.conflicts_with(r) for r in self._cells.get((c.lane, c.cell), ())):
                    return False
            for c in claims:
                self._cells[(c.lane, c.cell)].append(c)
            return True

    def release(self, claims):
        with self._lock:
            for c in claims:
                bucket = self._cells.get((c.lane, c.cell))
                if bucket and c in bucket:
                    bucket.remove(c)

    def shift(self, robot_id, delay, after, keep=()):
        keep = set(keep)
        moved: list[Reservation] = []
        with self._lock:
            for bucket in self._cells.values():
                for i, r in enumerate(bucket):
                    if r.robot_id == robot_id and r.start >= after and r not in keep:
                        bucket[i] = replace(r, start=r.start + delay, end=r.end + delay)
                        moved.append(bucket[i])
            return [
                o
                for m in moved
                for o in self._cells[(m.lane, m.cell)]
                if m.conflicts_with(o)
            ]


def sweep_occupancy(
    aisle: Aisle,
    profile: Profile,
    direction: int,
    u_from: float,
    u_to: float,
    half_length: float,
) -> list[tuple[int, float, float]]:
    """When a moving footprint covers each cell: a list of ``(cell, t_in, t_out)``.

    The robot centre travels from ``u_from`` to ``u_to`` along ``profile``,
    in travel-frame coordinates where ``s = direction * u``. Its footprint
    covers ``[u - half_length, u + half_length]``. A cell ``(ulo, uhi)`` is
    entered when ``u`` reaches ``max(u_from, ulo - h)`` and left when ``u``
    reaches ``min(u_to, uhi + h)``. Both times come from inverting the profile.
    """
    h = half_length
    s_lo, s_hi = sorted((direction * (u_from - h), direction * (u_to + h)))
    out = []
    for k in aisle.cells(s_lo, s_hi):
        lo, hi = aisle.cell_bounds(k)
        ulo, uhi = sorted((direction * lo, direction * hi))
        u_in, u_out = max(u_from, ulo - h), min(u_to, uhi + h)
        if u_out > u_in:
            out.append((k, profile.time_at(u_in), profile.time_at(u_out)))
    return out


def reservations_for_path(
    robot_id: str,
    aisle: Aisle,
    lane: str,
    profile: Profile,
    direction: int,
    u_from: float,
    u_to: float,
    half_length: float,
) -> list[Reservation]:
    """Turn a planned drive along one lane into cell reservations."""
    return [
        Reservation(robot_id, lane, k, t_in, t_out)
        for k, t_in, t_out in sweep_occupancy(aisle, profile, direction, u_from, u_to, half_length)
    ]
