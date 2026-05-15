"""Pure geometric helpers used across the vector pipeline.

These functions are intentionally side-effect free and do not depend on any
global configuration, numpy, or OpenCV. They can be imported from anywhere in
the package without triggering heavy dependencies.

Exports
-------
- ``distance``: Euclidean distance between two 2D points.
- ``midpoint``: midpoint of two 2D points.
- ``point_line_distance``: perpendicular distance from a point to a segment.
- ``rdp``: Ramer-Douglas-Peucker polyline simplification.
- ``quadratic_flatness`` / ``cubic_flatness``: Bezier flatness estimators.
- ``flatten_quadratic`` / ``flatten_cubic``: adaptive Bezier flattening.
- ``sanitize_stroke``: drop duplicate adjacent points from a stroke.
"""

from __future__ import annotations

import math


Point = tuple[float, float]
EPS = 1.0e-9


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a: Point, b: Point) -> Point:
    """Midpoint of a segment (a, b)."""
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def point_line_distance(point: Point, start: Point, end: Point) -> float:
    """Perpendicular distance from ``point`` to the segment [start, end].

    Uses the clamped projection parameter to handle very short segments without
    a divide-by-zero.
    """
    if distance(start, end) <= EPS:
        return distance(point, start)
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj = (x1 + t * dx, y1 + t * dy)
    return distance(point, proj)


def rdp(points: list[Point], epsilon: float) -> list[Point]:
    """Ramer-Douglas-Peucker simplification with tolerance ``epsilon``.

    Returns a new list; the input is not mutated.
    """
    if len(points) <= 2:
        return points
    start = points[0]
    end = points[-1]
    max_dist = -1.0
    split_index = -1
    for index in range(1, len(points) - 1):
        dist = point_line_distance(points[index], start, end)
        if dist > max_dist:
            max_dist = dist
            split_index = index
    if max_dist <= epsilon or split_index < 0:
        return [start, end]
    left = rdp(points[: split_index + 1], epsilon)
    right = rdp(points[split_index:], epsilon)
    return left[:-1] + right


def quadratic_flatness(p0: Point, p1: Point, p2: Point) -> float:
    """How far the quadratic control point ``p1`` lies from chord (p0, p2)."""
    return point_line_distance(p1, p0, p2)


def cubic_flatness(p0: Point, p1: Point, p2: Point, p3: Point) -> float:
    """Max perpendicular distance of inner control points to chord (p0, p3)."""
    return max(point_line_distance(p1, p0, p3), point_line_distance(p2, p0, p3))


def flatten_quadratic(p0: Point, p1: Point, p2: Point,
                       tol: float, depth: int = 0) -> list[Point]:
    """Adaptive midpoint subdivision of a quadratic Bezier.

    Returns a polyline approximation whose chord deviation never exceeds
    ``tol``. Recursion bounded at depth 12 to avoid pathological inputs.
    """
    if depth >= 12 or quadratic_flatness(p0, p1, p2) <= tol:
        return [p0, p2]
    p01 = midpoint(p0, p1)
    p12 = midpoint(p1, p2)
    p012 = midpoint(p01, p12)
    left = flatten_quadratic(p0, p01, p012, tol, depth + 1)
    right = flatten_quadratic(p012, p12, p2, tol, depth + 1)
    return left[:-1] + right


def flatten_cubic(p0: Point, p1: Point, p2: Point, p3: Point,
                   tol: float, depth: int = 0) -> list[Point]:
    """Adaptive midpoint subdivision of a cubic Bezier.

    Returns a polyline approximation whose chord deviation never exceeds
    ``tol``. Recursion bounded at depth 12 to avoid pathological inputs.
    """
    if depth >= 12 or cubic_flatness(p0, p1, p2, p3) <= tol:
        return [p0, p3]
    p01 = midpoint(p0, p1)
    p12 = midpoint(p1, p2)
    p23 = midpoint(p2, p3)
    p012 = midpoint(p01, p12)
    p123 = midpoint(p12, p23)
    p0123 = midpoint(p012, p123)
    left = flatten_cubic(p0, p01, p012, p0123, tol, depth + 1)
    right = flatten_cubic(p0123, p123, p23, p3, tol, depth + 1)
    return left[:-1] + right


def sanitize_stroke(points: list[Point]) -> tuple[Point, ...] | None:
    """Drop duplicate adjacent points; return ``None`` if <2 distinct points."""
    deduped: list[Point] = []
    for point in points:
        if not deduped:
            deduped.append(point)
            continue
        if distance(deduped[-1], point) <= EPS:
            continue
        deduped.append(point)
    if len(deduped) < 2:
        return None
    return tuple(deduped)


__all__ = [
    'Point',
    'EPS',
    'distance',
    'midpoint',
    'point_line_distance',
    'rdp',
    'quadratic_flatness',
    'cubic_flatness',
    'flatten_quadratic',
    'flatten_cubic',
    'sanitize_stroke',
]
