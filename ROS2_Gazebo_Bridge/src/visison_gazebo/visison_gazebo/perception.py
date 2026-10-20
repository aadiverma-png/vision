"""Turn a 2-D lidar scan into obstacles and tracked objects.

* :func:`detect_in_lane` finds a static obstacle in the robot's own lane.
  The lidar only sees the obstacle's front face, so the far edge is not
  known. It is set to ``s_min + assumed_depth``, which is a conservative guess.
* :func:`detect_objects` clusters every return inside the aisle (both
  lanes) into ``(s, y, length, width)`` measurements for the
  ``visison.ObjectTracker``. That tracker feeds the emergency collision prediction.

Returns that belong to known robots (fleet poses) can be removed with
:func:`drop_near`, so another robot is never mistaken for a fallen box.
"""

from __future__ import annotations

import math

from visison import Obstacle


def scan_to_world(
    ranges, angle_min: float, angle_increment: float, range_min: float, range_max: float,
    x: float, y: float, th: float, sensor_offset: tuple[float, float] = (0.1, 0.0),
) -> list[tuple[float, float]]:
    """Valid scan returns as ``(x, y)`` points in the world/aisle frame."""
    c, s = math.cos(th), math.sin(th)
    sx, sy = x + c * sensor_offset[0] - s * sensor_offset[1], y + s * sensor_offset[0] + c * sensor_offset[1]
    pts = []
    for i, r in enumerate(ranges):
        if not (range_min < r < range_max) or math.isinf(r) or math.isnan(r):
            continue
        a = th + angle_min + i * angle_increment
        pts.append((sx + r * math.cos(a), sy + r * math.sin(a)))
    return pts


def drop_near(
    points: list[tuple[float, float]], boxes: list[tuple[float, float, float, float]], pad: float = 0.15,
) -> list[tuple[float, float]]:
    """Points outside every axis-aligned ``(x, y, half_length, half_width)`` box plus ``pad``."""
    return [
        (px, py) for px, py in points
        if not any(abs(px - x) <= hl + pad and abs(py - y) <= hw + pad for x, y, hl, hw in boxes)
    ]


def _clusters(points: list[tuple[float, float]], gap: float) -> list[list[tuple[float, float]]]:
    """Split scan-ordered points where neighbouring returns are more than ``gap`` apart."""
    out: list[list[tuple[float, float]]] = []
    for p in points:
        if out and math.dist(out[-1][-1], p) <= gap:
            out[-1].append(p)
        else:
            out.append([p])
    if len(out) > 1 and math.dist(out[0][0], out[-1][-1]) <= gap:
        out[0] = out.pop() + out[0]  # the scan wraps around at +-pi
    return out


def drop_walls(
    points: list[tuple[float, float]], min_length: float = 1.2, max_width: float = 0.25, gap: float = 0.3,
) -> list[tuple[float, float]]:
    """Remove long, thin runs of returns along the aisle: shelf faces, not obstacles.

    With a little odometry heading drift, a shelf face a few metres away
    appears to lie inside the lane. A robot or a fallen box is never such a
    line, so dropping those clusters removes the ghosts.
    """
    out = []
    for c in _clusters(points, gap):
        xs, ys = [p[0] for p in c], [p[1] for p in c]
        if max(xs) - min(xs) >= min_length and max(ys) - min(ys) <= max_width:
            continue
        out += c
    return out


def detect_objects(
    points: list[tuple[float, float]],
    y_bounds: tuple[float, float],
    edge_margin: float = 0.1,
    cluster_gap: float = 0.3,
    min_points: int = 3,
) -> list[tuple[float, float, float, float]]:
    """Clusters of returns between the aisle's side edges, as ``(s, y, length, width)``.

    ``points`` must be in scan order (as :func:`scan_to_world` returns them), so
    neighbouring beams that hit the same body are consecutive.
    """
    lo, hi = y_bounds[0] + edge_margin, y_bounds[1] - edge_margin
    inside = [(px, py) for px, py in points if lo < py < hi]
    out = []
    for c in _clusters(inside, cluster_gap):
        if len(c) < min_points:
            continue
        xs, ys = [p[0] for p in c], [p[1] for p in c]
        out.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                    max(max(xs) - min(xs), 0.05), max(max(ys) - min(ys), 0.05)))
    return out


def lane_points(
    points: list[tuple[float, float]],
    robot_s: float,
    direction: int,
    lane_bounds: tuple[float, float],
    sensing_range: float,
    edge_margin: float = 0.05,
) -> list[tuple[float, float, float]]:
    """Points inside the lane and ahead within range, as ``(distance, x, y)`` sorted by distance."""
    lo, hi = lane_bounds[0] + edge_margin, lane_bounds[1] - edge_margin
    return sorted(
        (direction * (px - robot_s), px, py)
        for px, py in points
        if lo < py < hi and 0.0 < direction * (px - robot_s) <= sensing_range
    )


def detect_in_lane(
    points: list[tuple[float, float]],
    robot_s: float,
    direction: int,
    lane_bounds: tuple[float, float],
    sensing_range: float,
    assumed_depth: float = 1.0,
    edge_margin: float = 0.05,
    min_points: int = 3,
    cluster_gap: float = 0.5,
    obstacle_id: str = "obstacle",
) -> Obstacle | None:
    """Nearest cluster of points ahead of the robot inside its lane, as an Obstacle."""
    ahead = lane_points(points, robot_s, direction, lane_bounds, sensing_range, edge_margin)
    if not ahead:
        return None
    cluster = [ahead[0]]
    for p in ahead[1:]:
        if p[0] - cluster[-1][0] > cluster_gap:
            break
        cluster.append(p)
    if len(cluster) < min_points:
        return None
    xs = [p[1] for p in cluster]
    ys = [p[2] for p in cluster]
    if direction > 0:
        s_min = min(xs)
        s_max = max(max(xs), s_min + assumed_depth)
    else:
        s_max = max(xs)
        s_min = min(min(xs), s_max - assumed_depth)
    return Obstacle(s_min, s_max, min(ys), max(ys), obstacle_id=obstacle_id)
