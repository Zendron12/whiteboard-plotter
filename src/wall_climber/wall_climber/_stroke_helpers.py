"""Pure stroke-level geometry helpers used by ``vector_pipeline``.

These helpers operate on tuples of 2D points (``tuple[tuple[float, float],
...]``) and are independent of any module configuration. They were extracted
from ``vector_pipeline.py`` so the (still large) pipeline module can stay
focused on text/SVG/image specific orchestration.

Behaviour is identical to the inline helpers that previously lived inside
``vector_pipeline``; the pipeline now keeps thin wrapper aliases that
delegate to this module.
"""

from __future__ import annotations

import math

from wall_climber._geometry_helpers import (
    EPS,
    distance,
    rdp,
    sanitize_stroke,
)


Point = tuple[float, float]


def stroke_length(points: tuple[Point, ...]) -> float:
    """Return the polyline length of ``points``."""
    return sum(
        distance(points[index - 1], points[index])
        for index in range(1, len(points))
    )


def stroke_heading(
    points: tuple[Point, ...], *, from_start: bool,
) -> float | None:
    """Return the heading (radians) of the first or last non-degenerate edge."""
    if len(points) < 2:
        return None
    pairs = (
        zip(points[:-1], points[1:])
        if from_start
        else zip(reversed(points[1:]), reversed(points[:-1]))
    )
    for start, end in pairs:
        if distance(start, end) <= EPS:
            continue
        return math.atan2(end[1] - start[1], end[0] - start[0])
    return None


def heading_delta_deg(first: float | None, second: float | None) -> float:
    """Smallest absolute angle (degrees) between two headings, or 0 if unset."""
    if first is None or second is None:
        return 0.0
    delta = math.atan2(math.sin(second - first), math.cos(second - first))
    return abs(math.degrees(delta))


def heading_change_deg(
    prev_point: Point, point: Point, next_point: Point,
) -> float:
    """Absolute heading change (degrees) at ``point`` along a polyline."""
    before = math.atan2(
        point[1] - prev_point[1], point[0] - prev_point[0],
    )
    after = math.atan2(
        next_point[1] - point[1], next_point[0] - point[0],
    )
    delta = math.atan2(math.sin(after - before), math.cos(after - before))
    return abs(math.degrees(delta))


def stroke_has_preserved_short_feature(
    points: tuple[Point, ...], min_feature_len: float,
) -> bool:
    """Return True when the stroke is short enough that all points should stay."""
    if len(points) <= 2:
        return True
    if stroke_length(points) <= max(min_feature_len * 3.0, 1.0e-6):
        return True
    first_seg = distance(points[0], points[1])
    last_seg = distance(points[-2], points[-1])
    return first_seg <= min_feature_len or last_seg <= min_feature_len


def protected_stroke_indices(
    points: tuple[Point, ...],
    min_feature_len: float,
    *,
    corner_threshold_deg: float = 22.0,
) -> tuple[int, ...]:
    """Return the sorted indices of points that must survive simplification.

    Endpoints, corners with angle change >= ``corner_threshold_deg``, and the
    endpoints of any sub-feature shorter than ``min_feature_len`` are kept.
    """
    if len(points) <= 2:
        return tuple(range(len(points)))
    protected = {0, len(points) - 1}
    if stroke_has_preserved_short_feature(points, min_feature_len):
        protected.update(range(len(points)))
        return tuple(sorted(protected))
    for index in range(1, len(points) - 1):
        change = heading_change_deg(
            points[index - 1], points[index], points[index + 1],
        )
        if change >= corner_threshold_deg:
            protected.add(index)
    for index in range(1, len(points)):
        if distance(points[index - 1], points[index]) <= min_feature_len:
            protected.add(index - 1)
            protected.add(index)
    return tuple(sorted(protected))


def simplify_stroke_preserving_features(
    points: tuple[Point, ...],
    simplify_epsilon: float,
    min_feature_len: float,
) -> tuple[Point, ...]:
    """RDP-simplify the stroke without crossing protected indices."""
    if simplify_epsilon <= 0.0 or len(points) <= 2:
        return points
    protected = protected_stroke_indices(points, min_feature_len)
    if len(protected) >= len(points):
        return points
    simplified: list[Point] = [points[0]]
    for start_index, end_index in zip(protected[:-1], protected[1:]):
        span = list(points[start_index:end_index + 1])
        if len(span) <= 2:
            reduced = span
        else:
            reduced = rdp(span, simplify_epsilon)
        simplified.extend(reduced[1:])
    sanitized = sanitize_stroke(simplified)
    return sanitized if sanitized is not None else points


def interpolate_along_stroke(
    points: tuple[Point, ...], distance_along: float,
) -> Point:
    """Return the point ``distance_along`` metres into the polyline."""
    if distance_along <= 0.0:
        return points[0]
    traveled = 0.0
    for index in range(1, len(points)):
        start = points[index - 1]
        end = points[index]
        segment_length = distance(start, end)
        if segment_length <= EPS:
            continue
        next_traveled = traveled + segment_length
        if distance_along <= next_traveled + EPS:
            ratio = (distance_along - traveled) / segment_length
            return (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
        traveled = next_traveled
    return points[-1]


def resample_stroke_preserving_features(
    points: tuple[Point, ...],
    resample_step_m: float,
    min_feature_len: float,
) -> tuple[Point, ...]:
    """Re-sample the polyline at ``resample_step_m`` between protected indices."""
    if resample_step_m <= 0.0 or len(points) <= 2:
        return points
    if stroke_has_preserved_short_feature(points, min_feature_len):
        return points
    protected = protected_stroke_indices(points, min_feature_len)
    if len(protected) >= len(points):
        return points
    resampled: list[Point] = [points[0]]
    for start_index, end_index in zip(protected[:-1], protected[1:]):
        span = points[start_index:end_index + 1]
        span_length = stroke_length(span)
        if span_length <= resample_step_m + EPS:
            if distance(resampled[-1], span[-1]) > EPS:
                resampled.append(span[-1])
            continue
        target = resample_step_m
        while target < span_length - EPS:
            point = interpolate_along_stroke(span, target)
            if distance(resampled[-1], point) > EPS:
                resampled.append(point)
            target += resample_step_m
        if distance(resampled[-1], span[-1]) > EPS:
            resampled.append(span[-1])
    sanitized = sanitize_stroke(resampled)
    return sanitized if sanitized is not None else points


__all__ = [
    'stroke_length',
    'stroke_heading',
    'heading_delta_deg',
    'heading_change_deg',
    'stroke_has_preserved_short_feature',
    'protected_stroke_indices',
    'simplify_stroke_preserving_features',
    'interpolate_along_stroke',
    'resample_stroke_preserving_features',
]
